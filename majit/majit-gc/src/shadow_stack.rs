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

/// JitFrame layout constants for gcmap walking.
/// These match the #[repr(C)] JitFrame struct in jitframe.rs.
///
/// JitFrame header: [jf_frame_info, jf_descr, jf_force_descr, jf_gcmap,
///                    jf_savedata, jf_guard_exc, jf_forward]
/// Then: [jf_frame_length, jf_frame_items...]
const JF_GCMAP_BYTE_OFS: usize = 24; // offset_of!(JitFrame, jf_gcmap)
const JF_FRAME_ITEM0_BYTE_OFS: usize = 64; // JITFRAME_FIXED_SIZE(56) + length(8)
const WORD: usize = 8;

/// RPython shadow stack entry: [category, jf_ptr].
/// assembler.py:1125: MOV [ebx], 1  — the `1` is_minor marker.
/// assembler.py:1126: MOV [ebx+WORD], ebp — the jf_ptr.
#[derive(Clone, Copy)]
struct JfShadowEntry {
    /// is_minor marker. RPython uses 1 to indicate jitframe entry.
    /// incminimark.py uses this to skip old-gen jitframes during minor GC.
    _category: usize,
    jf_ptr: *mut u8,
    /// When true, this jitframe is GC-managed (old-gen allocated with header).
    /// walk_jf_roots skips these entries — Phase 2 traces them via
    /// remembered set + custom_trace (Option C from the plan).
    gc_managed: bool,
}

/// Push a jitframe pointer onto the jitframe shadow stack.
///
/// RPython _call_header_shadowstack (assembler.py:1122-1128):
///   MOV [shadowstack_top], 1       // is_minor marker
///   MOV [shadowstack_top+WORD], ebp  // jf_ptr
///   ADD shadowstack_top, 2*WORD
///
/// `gc_managed`: when true, this jitframe is GC-allocated (old-gen with header).
/// walk_jf_roots will skip gc_managed entries — they are traced via
/// remembered set + custom_trace in Phase 2 instead.
///
/// Returns the depth for later pop.
pub fn push_jf(jf_ptr: *mut u8, gc_managed: bool) -> usize {
    JF_SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        let depth = ss.len();
        assert!(depth < MAX_SHADOW_STACK_DEPTH, "jf shadow stack overflow");
        ss.push(JfShadowEntry {
            _category: 1,
            jf_ptr,
            gc_managed,
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

/// Collect GC-managed jitframe payload addresses from the shadow stack.
///
/// Returns a list of (obj_addr) for gc_managed entries. These must be
/// treated as roots during major GC marking to prevent sweep from
/// freeing live jitframes.
pub fn gc_managed_jf_addrs() -> Vec<usize> {
    JF_SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        ss.iter()
            .filter(|e| e.gc_managed && !e.jf_ptr.is_null())
            .map(|e| e.jf_ptr as usize)
            .collect()
    })
}

/// Walk jitframe shadow stack entries, tracing ref slots via jf_gcmap.
///
/// For each jf_ptr on the stack:
///   1. Read jf_gcmap pointer from jf_ptr + JF_GCMAP_BYTE_OFS
///   2. If gcmap is non-null, read bitmap [length, data...]
///   3. For each set bit, call visitor on the corresponding jf_frame slot
///
/// This is the pyre equivalent of RPython's root_walker.walk_roots()
/// combined with jitframe_trace (jitframe.py:104-136).
pub fn walk_jf_roots(mut visitor: impl FnMut(&mut GcRef)) {
    JF_SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        for entry in ss.iter() {
            let jf_ptr = entry.jf_ptr;
            if jf_ptr.is_null() {
                continue;
            }
            // GC-managed (old-gen) jitframes are traced via remembered set +
            // custom_trace in Phase 2. Skip here to prevent double-update
            // conflict (both walk_jf_roots and trace_and_update_object
            // trying to copy the same nursery ref).
            if entry.gc_managed {
                continue;
            }
            // Read jf_gcmap pointer
            let gcmap_ptr = unsafe { *(jf_ptr.add(JF_GCMAP_BYTE_OFS) as *const *const u8) };
            if gcmap_ptr.is_null() {
                continue;
            }
            // gcmap format: [length: isize, data: usize[length]]
            let gcmap_lgt = unsafe { *(gcmap_ptr as *const isize) };
            let frame_items = unsafe { jf_ptr.add(JF_FRAME_ITEM0_BYTE_OFS) };
            let mut no: isize = 0;
            while no < gcmap_lgt {
                let cur = unsafe { *(gcmap_ptr.add(WORD + WORD * no as usize) as *const usize) };
                let mut bitindex: usize = 0;
                while bitindex < 64 {
                    if cur & (1usize << bitindex) != 0 {
                        let index = no as usize * WORD * 8 + bitindex;
                        let slot_ptr = unsafe { frame_items.add(WORD * index) as *mut GcRef };
                        let gcref = unsafe { *slot_ptr };
                        if !gcref.is_null() {
                            unsafe { visitor(&mut *slot_ptr) };
                        }
                    }
                    bitindex += 1;
                }
                no += 1;
            }
        }
    });
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
        let depth = push_jf(0x1000 as *mut u8, false);
        assert_eq!(depth, 0);
        assert_eq!(jf_depth(), 1);
        push_jf(0x2000 as *mut u8, false);
        assert_eq!(jf_depth(), 2);
        pop_jf_to(depth);
        assert_eq!(jf_depth(), 0);
    }

    #[test]
    fn test_walk_jf_roots_with_gcmap() {
        clear();
        // Create a fake jitframe with gcmap pointer.
        // Layout: 64 bytes header + frame items
        let mut jf_buf = vec![0u8; 128];
        let jf_ptr = jf_buf.as_mut_ptr();

        // Set up a gcmap: [length=1, bitmap=0b0010] — bit 1 set
        // (bit 1 → frame_items[1] at offset 64 + 8 = 72).
        let gcmap: [isize; 2] = [1, 0b0010];
        // Write gcmap pointer to jf_gcmap field (offset 24)
        unsafe {
            *(jf_ptr.add(JF_GCMAP_BYTE_OFS) as *mut *const u8) = gcmap.as_ptr() as *const u8;
        }
        // Write a GcRef at frame item[1] (offset 64 + 8 = 72)
        unsafe {
            *(jf_ptr.add(JF_FRAME_ITEM0_BYTE_OFS + 8) as *mut usize) = 0xABCD;
        }

        push_jf(jf_ptr, false);

        let mut found = Vec::new();
        walk_jf_roots(|gcref| {
            found.push(gcref.0);
        });
        assert_eq!(found, vec![0xABCD]);

        pop_jf_to(0);
    }
}
