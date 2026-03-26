/// Shadow stack for GC root tracking in compiled JIT code.
///
/// RPython reference: rpython/jit/backend/llsupport/gc.py GcRootMap_shadowstack
///
/// Two stacks:
/// 1. GcRef shadow stack — individual GC refs (legacy, for non-jitframe roots)
/// 2. JitFrame shadow stack — jitframe pointers (RPython _call_header_shadowstack)
///
/// Protocol for jitframe shadow stack (assembler.py:1122-1136):
///   Entry: push_jf(jf_ptr)   — _call_header_shadowstack
///   Per-call: push_gcmap writes jf_gcmap; pop_gcmap clears it
///   GC: walk_jf_roots → read jf_gcmap → trace ref slots
///   Exit: pop_jf_to(depth)   — _call_footer_shadowstack
use std::cell::RefCell;

use majit_ir::GcRef;

/// Maximum shadow stack depth. RPython uses a fixed-size stack.
const MAX_SHADOW_STACK_DEPTH: usize = 8192;

thread_local! {
    /// Thread-local shadow stack for individual GcRef roots.
    static SHADOW_STACK: RefCell<ShadowStack> = RefCell::new(ShadowStack::new());

    /// Thread-local jitframe shadow stack.
    /// RPython: _call_header_shadowstack pushes [category, jf_ptr] pairs.
    /// category=1 is the is_minor marker (incminimark.py optimization).
    /// GC walks this to find jitframes and trace their ref slots via jf_gcmap.
    static JF_SHADOW_STACK: RefCell<Vec<JfShadowEntry>> = RefCell::new(Vec::with_capacity(16));
}

/// The shadow stack itself.
struct ShadowStack {
    entries: Vec<GcRef>,
}

impl ShadowStack {
    fn new() -> Self {
        ShadowStack {
            entries: Vec::with_capacity(64),
        }
    }
}

// ── GcRef shadow stack (individual refs) ─────────────────────────

/// Push a GC reference onto the shadow stack.
pub fn push(gcref: GcRef) -> usize {
    SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        let depth = ss.entries.len();
        assert!(depth < MAX_SHADOW_STACK_DEPTH, "shadow stack overflow");
        ss.entries.push(gcref);
        depth
    })
}

/// Pop entries from the shadow stack back to the given depth.
pub fn pop_to(depth: usize) -> Vec<GcRef> {
    SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        ss.entries.split_off(depth)
    })
}

/// Get a GcRef at the given index.
pub fn get(index: usize) -> GcRef {
    SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        ss.entries[index]
    })
}

/// Walk all entries on the GcRef shadow stack.
pub fn walk_roots(mut visitor: impl FnMut(&mut GcRef)) {
    SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        for entry in ss.entries.iter_mut() {
            if !entry.is_null() {
                visitor(entry);
            }
        }
    });
}

/// Current depth of the GcRef shadow stack.
pub fn depth() -> usize {
    SHADOW_STACK.with(|ss| ss.borrow().entries.len())
}

/// Clear both shadow stacks.
pub fn clear() {
    SHADOW_STACK.with(|ss| ss.borrow_mut().entries.clear());
    JF_SHADOW_STACK.with(|ss| ss.borrow_mut().clear());
}

// ── JitFrame shadow stack (assembler.py:1122-1136) ───────────────

/// RPython shadow stack entry: [category, jf_ptr].
/// assembler.py:1125: MOV [ebx], 1  — the `1` is_minor marker.
/// assembler.py:1126: MOV [ebx+WORD], ebp — the jf_ptr.
///
/// `jf_ptr` is a GcRef: the GC may copy the jitframe during minor
/// collection and update jf_ptr in-place (RPython root_walker semantics).
#[derive(Clone, Copy)]
struct JfShadowEntry {
    /// is_minor marker. RPython uses 1 to indicate jitframe entry.
    _category: usize,
    /// GcRef to the jitframe. Updated by the GC when the jitframe is
    /// copied from nursery to old gen (root_walker semantics).
    jf_ptr: GcRef,
}

/// Push a jitframe GcRef onto the jitframe shadow stack.
///
/// RPython _call_header_shadowstack (assembler.py:1122-1128):
///   MOV [shadowstack_top], 1       // is_minor marker
///   MOV [shadowstack_top+WORD], ebp  // jf_ptr
///   ADD shadowstack_top, 2*WORD
///
/// Returns the depth for later pop.
pub fn push_jf(jf_ptr: GcRef) -> usize {
    JF_SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        let depth = ss.len();
        assert!(depth < MAX_SHADOW_STACK_DEPTH, "jf shadow stack overflow");
        ss.push(JfShadowEntry {
            _category: 1,
            jf_ptr,
        });
        depth
    })
}

/// Pop jitframe entries back to the given depth.
///
/// RPython _call_footer_shadowstack (assembler.py:1130-1136):
///   SUB [rootstacktop], 2*WORD
pub fn pop_jf_to(depth: usize) {
    JF_SHADOW_STACK.with(|ss| {
        ss.borrow_mut().truncate(depth);
    });
}

/// Current depth of the jitframe shadow stack.
pub fn jf_depth() -> usize {
    JF_SHADOW_STACK.with(|ss| ss.borrow().len())
}

/// Walk jitframe shadow stack entries as GC roots.
///
/// Each jf_ptr is exposed as `&mut GcRef`. The GC treats it like any
/// other root: if it points into the nursery, the jitframe is copied
/// to old gen and the GcRef is updated in place.
///
/// The jitframe's internal ref slots are NOT traced here — that is
/// handled by `jitframe_custom_trace` via Phase 2 (remembered set +
/// custom_trace), exactly as in RPython where `root_walker.walk_roots()`
/// copies the jitframe, and then `jitframe_trace` (custom_trace hook)
/// traces the gcmap-indicated ref slots during Phase 2.
pub fn walk_jf_roots(mut visitor: impl FnMut(&mut GcRef)) {
    JF_SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        for entry in ss.iter_mut() {
            if !entry.jf_ptr.is_null() {
                visitor(&mut entry.jf_ptr);
            }
        }
    });
}

/// Read the jf_ptr of the top shadow stack entry.
///
/// RPython _reload_frame_if_necessary (assembler.py:405-412):
///   MOV ecx, [rootstacktop]
///   MOV ebp, [ecx - WORD]        // ebp = *(top - WORD) = jf_ptr
///
/// After a collecting call, the GC may have copied the jitframe. The
/// shadow stack entry has been updated. Compiled code reloads jf_ptr
/// from here to get the (possibly new) address.
pub fn jf_top_ptr() -> GcRef {
    JF_SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        ss.last().map(|e| e.jf_ptr).unwrap_or(GcRef::NULL)
    })
}

// ── Extern "C" interface for compiled code ──────────────────────

/// Push a GC reference from compiled code.
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_push(gcref_raw: i64) -> i64 {
    let gcref = GcRef(gcref_raw as usize);
    push(gcref) as i64
}

/// Pop shadow stack to depth and return the value at given index.
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_pop_and_get(depth: i64, index: i64) -> i64 {
    SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        let idx = index as usize;
        if idx < ss.entries.len() {
            ss.entries[idx].0 as i64
        } else {
            0
        }
    })
}

/// Set shadow stack depth (truncate).
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_set_depth(new_depth: i64) {
    SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        ss.entries.truncate(new_depth as usize);
    });
}

/// Read the top jf_ptr from the jitframe shadow stack.
///
/// Called from compiled code as `_reload_frame_if_necessary`
/// (assembler.py:405-412): after a collecting call, the GC may have
/// moved the jitframe. Reload jf_ptr from the shadow stack.
/// Takes a dummy argument because Cranelift's call_indirect on aarch64
/// has issues with 0-arg + return-value signatures.
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_get_top_jf_ptr(_dummy: i64) -> i64 {
    jf_top_ptr().0 as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop_roundtrip() {
        clear();
        let a = GcRef(0x1000);
        let b = GcRef(0x2000);
        let depth = push(a);
        assert_eq!(depth, 0);
        push(b);
        assert_eq!(super::depth(), 2);

        let popped = pop_to(depth);
        assert_eq!(popped.len(), 2);
        assert_eq!(popped[0], a);
        assert_eq!(popped[1], b);
        assert_eq!(super::depth(), 0);
    }

    #[test]
    fn test_walk_roots_updates() {
        clear();
        push(GcRef(0x1000));
        push(GcRef(0x2000));

        // Simulate GC moving objects
        walk_roots(|gcref| {
            gcref.0 += 0x100;
        });

        assert_eq!(get(0), GcRef(0x1100));
        assert_eq!(get(1), GcRef(0x2100));
        clear();
    }

    #[test]
    fn test_extern_c_interface() {
        clear();
        let depth = majit_shadow_stack_push(0x3000);
        assert_eq!(depth, 0);
        majit_shadow_stack_push(0x4000);

        let val = majit_shadow_stack_pop_and_get(0, 0);
        assert_eq!(val, 0x3000);
        let val = majit_shadow_stack_pop_and_get(0, 1);
        assert_eq!(val, 0x4000);

        majit_shadow_stack_set_depth(0);
        assert_eq!(super::depth(), 0);
    }

    #[test]
    fn test_jf_shadow_stack_push_pop() {
        clear();
        assert_eq!(jf_depth(), 0);
        let depth = push_jf(GcRef(0x1000));
        assert_eq!(depth, 0);
        assert_eq!(jf_depth(), 1);
        push_jf(GcRef(0x2000));
        assert_eq!(jf_depth(), 2);
        pop_jf_to(depth);
        assert_eq!(jf_depth(), 0);
    }

    #[test]
    fn test_walk_jf_roots_updates_gcref() {
        clear();
        push_jf(GcRef(0x1000));
        push_jf(GcRef(0x2000));

        // Simulate GC moving jitframes (root_walker semantics)
        walk_jf_roots(|gcref| {
            gcref.0 += 0x100;
        });

        // After walk, top entry should be updated
        assert_eq!(jf_top_ptr(), GcRef(0x2100));
        pop_jf_to(0);
    }

    #[test]
    fn test_jf_top_ptr_reload() {
        clear();
        push_jf(GcRef(0xABCD));
        assert_eq!(jf_top_ptr(), GcRef(0xABCD));
        assert_eq!(majit_jf_shadow_stack_get_top_jf_ptr(), 0xABCD);

        // Simulate GC updating the entry
        walk_jf_roots(|gcref| {
            gcref.0 = 0xDEAD;
        });
        assert_eq!(jf_top_ptr(), GcRef(0xDEAD));
        assert_eq!(majit_jf_shadow_stack_get_top_jf_ptr(), 0xDEAD as i64);
        pop_jf_to(0);
    }
}
