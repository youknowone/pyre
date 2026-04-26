//! `push_roots` / `pop_roots` parity scaffold — Phase 2c of Task #145.
//!
//! Mirrors RPython's `framework.py:803-853 gct_fv_gc_malloc`, which
//! brackets every GC malloc call with `push_roots(hop)` and
//! `pop_roots(hop, livevars)`:
//!
//! ```text
//! self.push_roots(hop)
//! v_alloc = hop.genop("direct_call", [malloc_fast_ptr,
//!                                     c_const_gc, c_type_id, c_size,
//!                                     c_false, c_false, c_false],
//!                     resulttype=llmemory.GCREF)
//! self.pop_roots(hop, livevars)
//! ```
//!
//! `push_roots` collects the values of every live GC pointer in the
//! caller's frame into shadow-stack slots so the moving collector can
//! see them as roots. `pop_roots` reads the (possibly-relocated)
//! values back out after the allocation may have triggered
//! collection.
//!
//! In Rust the natural shape is RAII: [`push_roots`] returns a
//! [`RootScope`] guard; dropping the guard executes the equivalent of
//! `pop_roots`. Callers wrap an allocation site like this:
//!
//! ```ignore
//! let _roots = pyre_object::gc_roots::push_roots();
//! pyre_object::gc_roots::pin_root(some_pyobject_ref);
//! let p = pyre_object::lltype::malloc_typed(value);
//! // _roots is dropped here; the pin is rewound, mirroring
//! // pop_roots(livevars).
//! ```
//!
//! ## Phase plan
//!
//! - **Phase 2a** — no-op stub. The API surface existed but the body
//!   was empty.
//! - **Phase 2b** — TLS shadow-stack body. [`push_roots`] snapshots
//!   the thread-local shadow stack length into a [`RootScope`];
//!   [`pin_root`] appends a [`PyObjectRef`]; [`Drop`] truncates the
//!   stack back to the saved length. Mirrors
//!   `rpython/memory/gctransform/shadowstack.py walk_stack_root`.
//! - **Phase 2c (this commit)** — expose [`walk_shadow_stack`]
//!   so the backend GC can visit pinned roots during nursery
//!   collection. `pyre-jit::eval` registers a thin
//!   `pyre-object`-to-`majit-gc` adapter through
//!   `majit_gc::shadow_stack::register_extra_root_walker`; pinned
//!   pointers are now observable to the active `MiniMarkGC`
//!   instance and survive across collections.

use std::cell::RefCell;

use crate::pyobject::PyObjectRef;

thread_local! {
    /// Thread-local shadow stack — pyre's mirror of RPython's
    /// `framework.shadowstack` (`shadowstack.py`). Append-only across
    /// each [`RootScope`]'s lifetime; the matching [`Drop`] truncates
    /// back to the saved length.
    ///
    /// `RefCell` (rather than `Cell` of a raw pointer) is deliberate:
    /// every public entry point goes through `borrow` / `borrow_mut`
    /// so re-entrant access from within a [`Drop`] would surface as a
    /// `BorrowMutError` rather than silently corrupting the stack.
    static SHADOW_STACK: RefCell<Vec<PyObjectRef>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard returned by [`push_roots`].
///
/// Holding a `RootScope` corresponds to RPython's `livevars` list
/// living across the bracketed allocation call. Dropping the scope
/// executes the equivalent of `pop_roots(hop, livevars)` by rewinding
/// the shadow stack to the length captured at construction time.
#[must_use = "RootScope is the live-roots guard; dropping it pops the roots — \
              binding it to `_roots` makes the scope explicit, while binding \
              to `_` would pop immediately and defeat the bracket"]
pub struct RootScope {
    /// Length of [`SHADOW_STACK`] at the moment [`push_roots`] was
    /// called. Phase 2c will replace this with a backend
    /// `RootSet::Handle` once the active GC consumes the stack.
    save_point: usize,
}

impl RootScope {
    /// Internal constructor. Callers should always go through
    /// [`push_roots`] so the API surface stays uniform across phases.
    #[inline]
    fn new() -> Self {
        Self {
            save_point: shadow_stack_len(),
        }
    }
}

impl Drop for RootScope {
    /// Truncate the shadow stack to the length captured at
    /// construction time, mirroring `pop_roots`'s discard of the
    /// bracketed `livevars` slots.
    #[inline]
    fn drop(&mut self) {
        SHADOW_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            // `truncate` is a no-op if `save_point >= len()`, which is
            // the steady-state case for an empty bracket.
            stack.truncate(self.save_point);
        });
    }
}

/// Open a `push_roots(hop)` bracket. Drop the returned guard to
/// execute the matching `pop_roots(hop, livevars)`. See the module
/// docstring for the multi-phase plan.
#[inline]
pub fn push_roots() -> RootScope {
    RootScope::new()
}

/// Pin a [`PyObjectRef`] as a live GC root for the duration of the
/// surrounding [`RootScope`]. Mirrors RPython's per-livevar
/// `setarrayitem(rootstk, idx, gcref)` emitted by
/// `gct_fv_gc_malloc.push_roots`.
///
/// Callers should always pair this with a guard from [`push_roots`]
/// so the matching [`Drop`] truncates the entry. Pinning without a
/// guard is a leak from the GC's perspective once Phase 2c lands.
#[inline]
pub fn pin_root(root: PyObjectRef) {
    SHADOW_STACK.with(|s| s.borrow_mut().push(root));
}

/// Current length of the thread-local shadow stack. Used by
/// [`RootScope::new`] to capture the save-point and by tests to
/// observe pin/pop behaviour.
#[inline]
pub fn shadow_stack_len() -> usize {
    SHADOW_STACK.with(|s| s.borrow().len())
}

/// Read a single shadow-stack slot by index, panicking if the index
/// is out of bounds. Used by tests and ad-hoc host-side debugging to
/// confirm the slot contents survive across nested brackets — the GC
/// itself uses [`walk_shadow_stack`] for the collection-time visit.
#[inline]
pub fn shadow_stack_get(index: usize) -> PyObjectRef {
    SHADOW_STACK.with(|s| s.borrow()[index])
}

/// Visit every pinned root in the shadow stack with mutable access.
/// `pyre-jit/src/eval.rs` registers a thin adapter through
/// `majit_gc::shadow_stack::register_extra_root_walker` that forwards
/// `&mut PyObjectRef` slots into the GC's `&mut GcRef` visitor; the
/// two are layout-compatible because `GcRef` is
/// `#[repr(transparent)]` over `usize` and `PyObjectRef =
/// *mut PyObject` is also pointer-sized.
///
/// The visitor receives mutable references so a moving collector can
/// rewrite slot contents post-relocation. Re-entrant access from
/// within the visitor would surface as a `BorrowMutError` — the
/// `RefCell` is the safety net that catches an accidentally
/// allocating walker.
#[inline]
pub fn walk_shadow_stack(mut visitor: impl FnMut(&mut PyObjectRef)) {
    SHADOW_STACK.with(|s| {
        let mut stack = s.borrow_mut();
        for slot in stack.iter_mut() {
            visitor(slot);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(addr: usize) -> PyObjectRef {
        addr as PyObjectRef
    }

    /// The scope guard is returned by value and dropped at end of
    /// the binding. This is the shape callers will rely on across
    /// every phase.
    #[test]
    fn push_roots_returns_a_drop_guard() {
        let _roots = push_roots();
    }

    /// In Phase 2b `RootScope` carries a single `usize` save-point.
    /// Phase 2c will swap this for a backend `RootSet::Handle` (still
    /// pointer-sized on 64-bit hosts); this test traps unintended
    /// growth beyond that.
    #[test]
    fn root_scope_payload_is_one_word() {
        assert_eq!(
            std::mem::size_of::<RootScope>(),
            std::mem::size_of::<usize>()
        );
    }

    /// `pin_root` appends, the matching `Drop` truncates back to the
    /// pre-bracket length. Mirrors the per-call invariant of
    /// `push_roots(livevars) ... pop_roots(livevars)` in
    /// `framework.py:803-853`.
    #[test]
    fn pin_root_pops_when_scope_drops() {
        let before = shadow_stack_len();
        {
            let _roots = push_roots();
            pin_root(dummy(0xDEAD_BEEF));
            pin_root(dummy(0xCAFE_F00D));
            assert_eq!(shadow_stack_len(), before + 2);
            assert_eq!(shadow_stack_get(before) as usize, 0xDEAD_BEEF);
            assert_eq!(shadow_stack_get(before + 1) as usize, 0xCAFE_F00D);
        }
        assert_eq!(shadow_stack_len(), before);
    }

    /// Nested brackets pop in LIFO order — RPython allows nested
    /// `push_roots` for cases like Container::new(member_alloc()).
    /// Each `Drop` rewinds to its own save-point regardless of inner
    /// pins.
    #[test]
    fn nested_root_scopes_pop_in_lifo_order() {
        let before = shadow_stack_len();
        let outer = push_roots();
        pin_root(dummy(1));
        let mid_outer_len = shadow_stack_len();
        assert_eq!(mid_outer_len, before + 1);
        {
            let _inner = push_roots();
            pin_root(dummy(2));
            pin_root(dummy(3));
            assert_eq!(shadow_stack_len(), mid_outer_len + 2);
        }
        // _inner dropped — back to the outer's view.
        assert_eq!(shadow_stack_len(), mid_outer_len);
        drop(outer);
        assert_eq!(shadow_stack_len(), before);
    }

    /// Empty bracket is a no-op against the stack length, both at
    /// `push` and at `Drop`. This protects callers that wrap an
    /// allocation that does not need to pin any of its inputs.
    #[test]
    fn empty_bracket_does_not_touch_the_stack() {
        let before = shadow_stack_len();
        {
            let _roots = push_roots();
            assert_eq!(shadow_stack_len(), before);
        }
        assert_eq!(shadow_stack_len(), before);
    }

    /// `walk_shadow_stack` exposes every pinned slot with mutable
    /// access, in insertion order. Mirrors what the GC walker sees
    /// during collection — a moving collector that updates a slot
    /// must observe the write through the next `shadow_stack_get`.
    #[test]
    fn walk_shadow_stack_visits_each_slot_mutably() {
        let _roots = push_roots();
        pin_root(dummy(0x1));
        pin_root(dummy(0x2));
        pin_root(dummy(0x3));

        // Read pass: collect the addresses we see.
        let mut seen = Vec::new();
        walk_shadow_stack(|slot| seen.push(*slot as usize));
        // Trim to the entries we just pinned to avoid false matches
        // against entries pinned by a concurrent test (single-threaded
        // here, but the contract is per-thread).
        let tail = &seen[seen.len() - 3..];
        assert_eq!(tail, &[0x1, 0x2, 0x3]);

        // Write pass: a moving collector might rewrite slots — confirm
        // the writes stick.
        walk_shadow_stack(|slot| {
            if (*slot as usize) == 0x2 {
                *slot = dummy(0xAA);
            }
        });
        assert_eq!(shadow_stack_get(shadow_stack_len() - 2) as usize, 0xAA);
    }
}
