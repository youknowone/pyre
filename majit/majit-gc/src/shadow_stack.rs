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

    /// Thread-local stack of blackhole interpreter register banks.
    /// blackhole.py BlackholeInterpreter.registers_r parity: each active
    /// blackhole frame's ref register file is a GC root range. The GC must
    /// trace these slots during minor collection so nursery objects held only
    /// by a blackhole register survive across collecting calls.
    static BH_REGS_STACK: RefCell<Vec<BhRegsEntry>> = RefCell::new(Vec::with_capacity(16));
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

/// Read the GC-updated jf_ptr from the shadow stack at the given depth.
///
/// RPython _call_footer_shadowstack (assembler.py:1130-1136):
///   MOV eax, [rootstacktop - WORD]  // read GC-updated jf_ptr
///   SUB rootstacktop, 2*WORD
///
/// The GC may have forwarded the nursery jitframe to old gen and updated
/// the shadow stack entry in place. This returns the forwarded address.
pub fn peek_jf(depth: usize) -> GcRef {
    JF_SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        ss[depth].jf_ptr
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

// ── Blackhole register bank shadow stack ────────────────────────
//
// blackhole.py:840 BlackholeInterpreter.registers_r parity:
// the ref register file is part of the GC's root set during the
// blackhole interpreter's lifetime. RPython's GC scans the
// blackhole interpreter's RPython-managed Box arrays directly;
// pyre stores raw i64 ptrs in Vec<i64> so we register the active
// register banks with the collector explicitly.

/// One active blackhole register bank.
///
/// `regs_ptr` points at the start of a `Vec<i64>` slot range; `regs_len`
/// is the number of slots (working registers AND constants — constants
/// are pre-existing old-gen pointers and pass through `copy_nursery_object`
/// untouched). `tmpreg_ptr` points at the temporary ref register that
/// holds in-flight call return values.
#[derive(Clone, Copy)]
struct BhRegsEntry {
    regs_ptr: *mut i64,
    regs_len: usize,
    tmpreg_ptr: *mut i64,
}

// `*mut i64` is not `Send` by default. The thread-local storage means we
// only ever access entries from the thread that pushed them, so this
// `unsafe impl` is sound.
unsafe impl Send for BhRegsEntry {}

/// Push a blackhole interpreter's register bank onto the GC root stack.
///
/// The caller must ensure the slice remains valid until the matching
/// `pop_bh_regs_to(depth)` is called. Returns the previous depth so the
/// caller can restore it on exit (RPython _call_header_shadowstack /
/// _call_footer_shadowstack pattern).
///
/// # Safety
/// `regs` must remain alive and pinned until pop.
pub unsafe fn push_bh_regs(regs: &mut [i64], tmpreg: &mut i64) -> usize {
    BH_REGS_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        let depth = ss.len();
        assert!(
            depth < MAX_SHADOW_STACK_DEPTH,
            "blackhole regs stack overflow"
        );
        ss.push(BhRegsEntry {
            regs_ptr: regs.as_mut_ptr(),
            regs_len: regs.len(),
            tmpreg_ptr: tmpreg as *mut i64,
        });
        depth
    })
}

/// Pop blackhole register entries back to the given depth.
pub fn pop_bh_regs_to(depth: usize) {
    let _ = BH_REGS_STACK.try_with(|ss| {
        ss.borrow_mut().truncate(depth);
    });
}

/// Walk all active blackhole register banks as GC roots.
///
/// Each i64 slot is exposed as `&mut GcRef`. Slots holding non-nursery
/// pointers (constants, old-gen, NULL) are passed through unchanged by
/// `copy_nursery_object` semantics in the visitor.
pub fn walk_bh_regs(mut visitor: impl FnMut(&mut GcRef)) {
    BH_REGS_STACK.with(|ss| {
        let ss = ss.borrow();
        for entry in ss.iter() {
            // SAFETY: the BlackholeInterpreter that pushed this entry is on
            // the call stack above us (we are inside its run() body via a
            // collecting call). The Vec<i64> backing storage is pinned for
            // the lifetime of that frame.
            let slots = unsafe { std::slice::from_raw_parts_mut(entry.regs_ptr, entry.regs_len) };
            for slot in slots.iter_mut() {
                let gcref = unsafe { &mut *(slot as *mut i64 as *mut GcRef) };
                visitor(gcref);
            }
            // tmpreg_r holds in-flight call return values between
            // `let result = call_int_function(...)` and the subsequent
            // `self.registers_r[dst] = result;` store. RPython's blackhole
            // interpreter holds this in an RPython-managed slot so it is
            // automatically a root.
            let tmp = unsafe { &mut *(entry.tmpreg_ptr as *mut GcRef) };
            visitor(tmp);
        }
    });
}

/// Current depth of the blackhole register bank stack.
pub fn bh_regs_depth() -> usize {
    BH_REGS_STACK.with(|ss| ss.borrow().len())
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
pub extern "C" fn majit_shadow_stack_pop_and_get(_depth: i64, index: i64) -> i64 {
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
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_get_top_jf_ptr() -> i64 {
    jf_top_ptr().0 as i64
}

/// _call_header_shadowstack (assembler.py:1122-1128) parity.
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_push(jf_ptr_raw: i64) -> i64 {
    push_jf(crate::GcRef(jf_ptr_raw as usize)) as i64
}

/// _call_footer_shadowstack (assembler.py:1130-1136) parity.
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_pop_to(depth: i64) {
    pop_jf_to(depth as usize);
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
