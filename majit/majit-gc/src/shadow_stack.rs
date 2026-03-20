/// Shadow stack for GC root tracking in compiled JIT code.
///
/// RPython reference: rpython/jit/backend/llsupport/gc.py GcRootMap_shadowstack
///
/// When compiled code calls a function that may trigger GC, it must
/// ensure all live GC references are visible to the collector. The
/// shadow stack is a thread-local stack of GcRef slots that the GC
/// walks during collection.
///
/// Protocol:
/// 1. Before CALL: push live GC refs to shadow stack
/// 2. CALL executes (may trigger GC)
/// 3. GC walks shadow stack, traces and possibly moves objects
/// 4. After CALL: pop from shadow stack, reload (pointers may have moved)
use std::cell::RefCell;

use majit_ir::GcRef;

/// Maximum shadow stack depth. RPython uses a fixed-size stack.
const MAX_SHADOW_STACK_DEPTH: usize = 8192;

thread_local! {
    /// Thread-local shadow stack: stores mutable pointers to GcRef slots.
    ///
    /// RPython gc.py: gc_adr_of_root_stack_top — returns the current
    /// top of the shadow stack. The compiled code pushes/pops entries.
    static SHADOW_STACK: RefCell<ShadowStack> = RefCell::new(ShadowStack::new());
}

/// The shadow stack itself.
struct ShadowStack {
    /// Stack of GcRef values (not pointers — owned copies).
    /// Compiled code pushes GcRef values before a call and pops after.
    /// The GC updates these values in-place if objects move.
    entries: Vec<GcRef>,
}

impl ShadowStack {
    fn new() -> Self {
        ShadowStack {
            entries: Vec::with_capacity(64),
        }
    }
}

/// Push a GC reference onto the shadow stack.
///
/// Called by compiled code before a CALL that may trigger GC.
/// Returns the index (depth) for later pop.
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
///
/// Called by compiled code after a CALL returns.
/// Returns the (possibly updated) GcRef values — the GC may have
/// moved the objects during the call.
pub fn pop_to(depth: usize) -> Vec<GcRef> {
    SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        ss.entries.split_off(depth)
    })
}

/// Get a GcRef at the given index (for reload after CALL).
///
/// After a CALL, the compiled code must reload GC references because
/// the GC may have moved objects. This reads from the shadow stack
/// where the GC has updated the pointers.
pub fn get(index: usize) -> GcRef {
    SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        ss.entries[index]
    })
}

/// Walk all entries on the shadow stack, calling the visitor for each.
///
/// Used by the GC during collection to trace live references.
/// The visitor may update the GcRef (if the object is moved).
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

/// Current depth of the shadow stack.
pub fn depth() -> usize {
    SHADOW_STACK.with(|ss| ss.borrow().entries.len())
}

/// Clear the shadow stack (used in tests or when JIT state is reset).
pub fn clear() {
    SHADOW_STACK.with(|ss| ss.borrow_mut().entries.clear());
}

// ── Extern "C" interface for compiled code ──────────────────────

/// Push a GC reference from compiled code.
///
/// Compiled code calls this before a CALL that may trigger GC.
/// The GcRef is passed as a raw i64 (pointer value).
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_push(gcref_raw: i64) -> i64 {
    let gcref = GcRef(gcref_raw as usize);
    push(gcref) as i64
}

/// Pop shadow stack to depth and return the value at given index.
///
/// Compiled code calls this after a CALL returns to reload a GC ref.
/// Returns the (possibly updated) GcRef as raw i64.
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
}
