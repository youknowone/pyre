//! `push_roots` / `pop_roots` parity scaffold.
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

#[cfg(debug_assertions)]
use std::cell::Cell;
use std::cell::UnsafeCell;
use std::marker::PhantomData;

use crate::pyobject::PyObjectRef;

thread_local! {
    /// Thread-local shadow stack — pyre's mirror of RPython's
    /// `framework.shadowstack` (`shadowstack.py`). Append-only across
    /// each [`RootScope`]'s lifetime; the matching [`Drop`] truncates
    /// back to the saved length.
    ///
    /// Access is raw, matching RPython's base/top shadow-stack shape. Debug
    /// builds check overlapping ordinary accesses through
    /// `SHADOW_STACK_ACCESS_DEPTH` and separately reject mutation during a
    /// root walk through `SHADOW_STACK_WALK_IN_PROGRESS`; release builds pay
    /// no bookkeeping cost.
    static SHADOW_STACK: UnsafeCell<Vec<PyObjectRef>> =
        const { UnsafeCell::new(Vec::new()) };

    /// Signed debug borrow depth for [`SHADOW_STACK`]: non-negative values
    /// count shared accesses, while `-1` marks an exclusive access.
    #[cfg(debug_assertions)]
    static SHADOW_STACK_ACCESS_DEPTH: Cell<i32> = const { Cell::new(0) };

    /// Whether this thread is currently driving a shadow-stack root walk.
    #[cfg(debug_assertions)]
    static SHADOW_STACK_WALK_IN_PROGRESS: Cell<bool> = const { Cell::new(false) };
}

#[cfg(debug_assertions)]
struct ShadowStackAccess<'a> {
    depth: &'a Cell<i32>,
    previous: i32,
}

#[cfg(debug_assertions)]
impl<'a> ShadowStackAccess<'a> {
    #[inline(always)]
    fn shared(depth: &'a Cell<i32>) -> Self {
        let previous = depth.get();
        debug_assert!(
            previous >= 0,
            "shared shadow-stack access re-entered an exclusive access"
        );
        depth.set(previous + 1);
        Self { depth, previous }
    }

    #[inline(always)]
    fn exclusive(depth: &'a Cell<i32>) -> Self {
        let previous = depth.get();
        debug_assert_eq!(previous, 0, "exclusive shadow-stack access was re-entered");
        depth.set(-1);
        Self { depth, previous }
    }
}

#[cfg(debug_assertions)]
impl Drop for ShadowStackAccess<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.depth.set(self.previous);
    }
}

#[cfg(debug_assertions)]
struct ShadowStackWalk<'a> {
    in_progress: &'a Cell<bool>,
}

#[cfg(debug_assertions)]
impl<'a> ShadowStackWalk<'a> {
    fn new(in_progress: &'a Cell<bool>) -> Self {
        // Arm the flag outside the assertion: the walk must be marked whether
        // or not assertions are compiled in, and an assertion that doubles as
        // the only writer stops arming the moment it is disabled.
        let was_walking = in_progress.replace(true);
        debug_assert!(!was_walking, "a shadow-stack root walk was re-entered");
        Self { in_progress }
    }
}

#[cfg(debug_assertions)]
impl Drop for ShadowStackWalk<'_> {
    fn drop(&mut self) {
        self.in_progress.set(false);
    }
}

#[cfg(debug_assertions)]
#[inline(always)]
fn assert_shadow_stack_not_walking() {
    SHADOW_STACK_WALK_IN_PROGRESS.with(|in_progress| {
        debug_assert!(
            !in_progress.get(),
            "the shadow stack was mutated while a root walk was in progress"
        );
    });
}

#[inline(always)]
fn with_shadow_stack<R>(f: impl FnOnce(&Vec<PyObjectRef>) -> R) -> R {
    SHADOW_STACK.with(|cell| {
        #[cfg(debug_assertions)]
        {
            SHADOW_STACK_ACCESS_DEPTH.with(|depth| {
                let _access = ShadowStackAccess::shared(depth);
                // SAFETY: the debug-only access guard rejects overlap with an
                // exclusive access. In release this is the raw, thread-local
                // root-stack access shape used upstream.
                f(unsafe { &*cell.get() })
            })
        }
        #[cfg(not(debug_assertions))]
        {
            // SAFETY: the stack is thread-local, and callers must not
            // re-enter an access that holds an exclusive reference.
            f(unsafe { &*cell.get() })
        }
    })
}

#[inline(always)]
fn with_shadow_stack_mut<R>(f: impl FnOnce(&mut Vec<PyObjectRef>) -> R) -> R {
    SHADOW_STACK.with(|cell| {
        #[cfg(debug_assertions)]
        {
            SHADOW_STACK_ACCESS_DEPTH.with(|depth| {
                let _access = ShadowStackAccess::exclusive(depth);
                // SAFETY: the debug-only access guard rejects every
                // overlapping access. Release preserves the same
                // non-re-entrancy invariant without runtime bookkeeping.
                f(unsafe { &mut *cell.get() })
            })
        }
        #[cfg(not(debug_assertions))]
        {
            // SAFETY: the stack is thread-local, and callers must not
            // overlap this exclusive access with another stack access.
            f(unsafe { &mut *cell.get() })
        }
    })
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
    /// called. Will be replaced with a backend
    /// `RootSet::Handle` once the active GC consumes the stack.
    save_point: usize,
    /// A root bracket belongs to the thread whose shadow stack supplied this
    /// save point. Moving it could truncate an unrelated thread's stack.
    _not_send: PhantomData<*const ()>,
}

impl RootScope {
    /// Internal constructor. Callers should always go through
    /// [`push_roots`] so the API surface stays uniform across phases.
    #[inline]
    fn new() -> Self {
        Self {
            save_point: shadow_stack_len(),
            _not_send: PhantomData,
        }
    }
}

impl Drop for RootScope {
    /// Truncate the shadow stack to the length captured at
    /// construction time, mirroring `pop_roots`'s discard of the
    /// bracketed `livevars` slots.
    #[inline]
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        with_shadow_stack_mut(|stack| {
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
/// guard is a leak from the GC's perspective once the backend GC consumes the stack.
///
/// Pushes onto the thread-local `SHADOW_STACK`, a runtime-mutable root the
/// tracer cannot type; the JIT residualises the call instead of tracing into
/// it (`@dont_look_inside`, `rlib/jit.py:139`), the `shadow_stack_len` twin.
#[majit_macros::dont_look_inside]
pub fn pin_root(root: PyObjectRef) {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    // Publish the raw value first.  `try_gc_current_object_address` is itself
    // a GC operation and may wait behind another thread's collection; the
    // fresh allocation must already be visible to that collector before we
    // enter the query safepoint.  RPython's push_roots writes the livevar to
    // the shadow stack before calling the allocation/GC slow path for the
    // same reason.
    let index = with_shadow_stack_mut(|stack| {
        let index = stack.len();
        stack.push(root);
        index
    });
    // A foreign mutator may have completed a nursery collection after the
    // caller copied this GCREF but before it entered this explicit root
    // bracket.  RPython's `_trace_drag_out` always rewrites a root that names
    // an already-forwarded nursery object; normalize the host-side copy at
    // the same boundary so the shadow stack never gains a forwarding stub.
    let normalized = crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
    // The slot already holds `root`, so a query that found no forwarding stub
    // has nothing to write back.  That is the steady state — nothing collected
    // between the push above and the query — and the write-back is not free:
    // each `with` re-checks the thread-local's initialization state (the
    // `_tlv_get_addr` result itself is common-subexpression-eliminated across
    // the two accesses, the state load is not), and pinning sits on the
    // interpreter's allocation and call paths.
    if normalized != root {
        with_shadow_stack_mut(|stack| stack[index] = normalized);
    }
}

/// Publish a complete translated livevar set before performing any
/// forwarding query.  RPython's `push_roots(hop)` writes every live GCREF to
/// the root stack as one pre-allocation phase; calling [`pin_root`] repeatedly
/// would query after the first write and let a foreign collection run before
/// later values were visible.
///
/// Returns the first shadow-stack index occupied by `roots`.
#[majit_macros::dont_look_inside]
pub fn pin_roots(roots: &[PyObjectRef]) -> usize {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    let base = with_shadow_stack_mut(|stack| {
        let base = stack.len();
        stack.extend_from_slice(roots);
        base
    });
    for index in base..base + roots.len() {
        // Read the slot each time: a collection triggered by an earlier query
        // may already have rewritten every published root in place.
        let root = with_shadow_stack(|stack| stack[index]);
        let current = crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
        // Same write-back economy as `pin_root`: the slot already holds `root`,
        // so a query that found no forwarding stub has nothing to store, and
        // skipping it saves a second thread-local resolve per livevar.
        if current != root {
            with_shadow_stack_mut(|stack| stack[index] = current);
        }
    }
    base
}

/// Current length of the thread-local shadow stack. Used by
/// [`RootScope::new`] to capture the save-point and by tests to
/// observe pin/pop behaviour.
///
/// Reads the thread-local `SHADOW_STACK`, a runtime-mutable root the
/// tracer cannot type; the JIT residualises the read instead of tracing
/// into it (`@dont_look_inside`, `rlib/jit.py:139`). The attribute is a
/// tracing-policy marker only — it leaves the host backend free to inline
/// this body, exactly as the RPython decorator leaves the C backend free.
#[majit_macros::dont_look_inside]
pub fn shadow_stack_len() -> usize {
    with_shadow_stack(Vec::len)
}

/// Read a single shadow-stack slot by index, panicking if the index
/// is out of bounds.  RPython's `pop_roots` reads the root-stack slot
/// directly: a moving collection has already rewritten it in place through
/// [`walk_shadow_stack`] (or [`walk_shadow_stack_area`] for another mutator).
/// The initial [`pin_root`] still normalizes a copied pointer that may have
/// become stale before it was published.
///
/// This is the `pop_roots` half of the bracket — the read back of a pinned
/// livevar after a call that may have collected — so it sits on the ordinary
/// call path (`call_function_impl_result`, the inline-call and specialize
/// resume paths, `#[pyre_class]` method bodies), not just in tests.
///
/// Reads the thread-local `SHADOW_STACK` the tracer cannot type; the JIT
/// residualises the read instead of tracing into it (`@dont_look_inside`,
/// `rlib/jit.py:139`), the [`shadow_stack_len`] twin.
#[majit_macros::dont_look_inside]
pub fn shadow_stack_get(index: usize) -> PyObjectRef {
    // A plain slot read: the slot is already a registered root, so whatever
    // moved it has rewritten it here.  `init_gc_subsystem` registers this
    // thread's stack as a mutator root area (`capture_shadow_stack_area`)
    // together with `register_mutator`, before the thread runs any user code,
    // and [`walk_shadow_stack`] hands the collector `&mut` slots — so a
    // relocation between the pin and this read updates the slot in place and
    // no forwarding stub can be observed here.
    //
    // Only [`pin_root`] normalizes, and it must: the value it receives is
    // copied from outside the bracket, so a foreign mutator can have collected
    // after that copy and before the pin.  That is also the pin's safepoint;
    // a read allocates nothing and has no reason to park.
    with_shadow_stack(|stack| stack[index])
}

/// Copy a contiguous range of rooted values out of the shadow stack.
///
/// RPython's `pop_roots` reloads all livevars from the root stack as one
/// generated block after the potentially-collecting call.  Host-side callers
/// that need several adjacent roots should use this equivalent bulk shape
/// instead of repeatedly entering TLS through [`shadow_stack_get`].
///
/// Panics when `base..base + dst.len()` is outside the live root stack, just
/// like indexing the corresponding slots individually.
#[majit_macros::dont_look_inside]
pub fn shadow_stack_copy_range(base: usize, dst: &mut [PyObjectRef]) {
    with_shadow_stack(|stack| {
        dst.copy_from_slice(&stack[base..base + dst.len()]);
    });
}

/// Overwrite a single shadow-stack slot by index, panicking if the index
/// is out of bounds. A slot whose contents change over a bracket's lifetime
/// is written here rather than re-pinned, mirroring `gc_save_root`'s
/// overwrite of an already allocated slot (`shadowcolor.py:126-129`).
///
/// Writes the thread-local `SHADOW_STACK` the tracer cannot type; the JIT
/// residualises the write instead of tracing into it (`@dont_look_inside`,
/// `rlib/jit.py:139`), the [`shadow_stack_get`] twin.
#[majit_macros::dont_look_inside]
pub fn shadow_stack_set(index: usize, root: PyObjectRef) {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    // Publish the raw value first, for the reason [`pin_root`] gives: the
    // `try_gc_current_object_address` query is itself a GC operation and may
    // wait behind another thread's collection, so the value must already be
    // visible to that collector before we enter the query safepoint.
    with_shadow_stack_mut(|stack| stack[index] = root);
    // Then normalize a GCREF the caller may have copied before a foreign
    // mutator's nursery collection, so the slot never keeps a forwarding stub.
    let root = crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
    with_shadow_stack_mut(|stack| stack[index] = root);
}

/// Visit every pinned root in the shadow stack with mutable access.
/// `pyre-jit/src/eval.rs` registers a thin adapter through
/// `majit_gc::shadow_stack::register_extra_root_walker` that forwards
/// `&mut PyObjectRef` slots into the GC's `&mut GcRef` visitor; the
/// two are layout-compatible because `GcRef` is
/// `#[repr(transparent)]` over `usize` and `PyObjectRef =
/// *mut PyObject` is also pointer-sized.
///
/// The visitor receives mutable references so a moving collector can rewrite
/// slot contents post-relocation. Each slot is re-derived from the stack after
/// the previous visitor call, so a re-entrant push cannot leave the walk using
/// a pointer into an old allocation. Debug builds additionally reject shadow-
/// stack mutation by an accidentally allocating walker; release builds match
/// upstream's raw root-stack access without that diagnostic.
#[inline]
pub fn walk_shadow_stack(mut visitor: impl FnMut(&mut PyObjectRef)) {
    SHADOW_STACK.with(|cell| {
        let cell = cell as *const UnsafeCell<Vec<PyObjectRef>>;
        #[cfg(debug_assertions)]
        SHADOW_STACK_WALK_IN_PROGRESS.with(|in_progress| {
            let _walk = ShadowStackWalk::new(in_progress);
            // SAFETY: `cell` is this thread's live shadow-stack cell.
            unsafe { walk_shadow_stack_cell(cell, &mut visitor) };
        });
        #[cfg(not(debug_assertions))]
        // SAFETY: `cell` is this thread's live shadow-stack cell.
        unsafe {
            walk_shadow_stack_cell(cell, &mut visitor);
        }
    });
}

/// Capture the current thread's pinned-root stack for a foreign STW walk.
pub fn capture_shadow_stack_area() -> *const () {
    SHADOW_STACK.with(|s| s as *const _ as *const ())
}

/// Visit a captured thread's pinned-root stack without consulting caller TLS.
/// Like [`walk_shadow_stack`], this re-derives each slot after the previous
/// visitor call so a push cannot strand the walk in a reallocated buffer.
/// Debug builds reject shadow-stack mutation for the duration of the walk.
///
/// # Safety
/// `data` must come from [`capture_shadow_stack_area`], and the owning thread
/// must be quiesced for the duration of the walk.
pub unsafe fn walk_shadow_stack_area(data: *const (), mut visitor: impl FnMut(&mut PyObjectRef)) {
    let cell = data as *const UnsafeCell<Vec<PyObjectRef>>;
    #[cfg(debug_assertions)]
    SHADOW_STACK_WALK_IN_PROGRESS.with(|in_progress| {
        // For a self-collection this protects the stack being walked. A
        // foreign walk marks the collector thread instead, which is harmless:
        // the owning mutator is quiesced and performs no stack accesses.
        let _walk = ShadowStackWalk::new(in_progress);
        // SAFETY: the caller guarantees that `cell` is a captured shadow stack
        // whose owning thread is quiesced for this walk.
        unsafe { walk_shadow_stack_cell(cell, &mut visitor) };
    });
    #[cfg(not(debug_assertions))]
    // SAFETY: the caller guarantees that `cell` is a captured shadow stack
    // whose owning thread is quiesced for this walk.
    unsafe {
        walk_shadow_stack_cell(cell, &mut visitor);
    }
}

/// Walk one captured shadow-stack cell without retaining a reference to its
/// `Vec` across a visitor call.
///
/// # Safety
/// `cell` must remain a valid shadow-stack cell for the duration of the walk,
/// and no thread other than a re-entrant visitor may mutate it.
unsafe fn walk_shadow_stack_cell(
    cell: *const UnsafeCell<Vec<PyObjectRef>>,
    visitor: &mut impl FnMut(&mut PyObjectRef),
) {
    // The interval is fixed at entry: `walk_stack_root` (`shadowstack.py:43-46`)
    // receives `start` and `addr` as arguments and runs `while addr != start`,
    // so a root pushed mid-walk is not part of that walk. Re-reading the length
    // each iteration instead would extend the walk over roots pinned by the
    // visitor — and pinning during a walk is precisely what the debug guard
    // above declares illegal, so there is nothing to serve by diverging here.
    let end = {
        // SAFETY: no reference from this block outlives it.
        let stack = unsafe { &*(*cell).get() };
        stack.len()
    };
    let mut index = 0;
    while index < end {
        // Re-derive the slot address every iteration rather than holding one
        // `iter_mut()` cursor: a re-entrant push can still reallocate the
        // buffer, which would strand a cached pointer in the old allocation.
        // Only pushes can occur, so `end` stays within bounds.
        let slot = {
            // SAFETY: the caller guarantees exclusive access except for a
            // possible re-entrant mutation by `visitor`. No reference from a
            // previous iteration is live here, and this temporary reference
            // ends before `visitor` is called.
            let stack = unsafe { &mut *(*cell).get() };
            // Return only the slot address so the `Vec` reference is not held
            // across the visitor, which may push and reallocate its buffer.
            unsafe { stack.as_mut_ptr().add(index) }
        };
        // SAFETY: `slot` names the live element at `index`. The reference is
        // handed directly to the visitor and is not used after it returns.
        visitor(unsafe { &mut *slot });
        index += 1;
    }
}

// ── Prebuilt-root write tracking ────────────────────────────────────
//
// incminimark.py:106-114 `GCFLAG_TRACK_YOUNG_PTRS` / 339-344
// `old_objects_pointing_to_young` / 355 `prebuilt_root_objects`: an old or
// prebuilt object is NOT scanned during a minor collection unless the write
// barrier recorded a store into it since the previous minor collection; a
// major collection always traces the prebuilt roots.
//
// pyre's Box-immortal interpreter structures (module dicts / cells, heap-type
// namespace dicts, method caches, function fields) bypass the compiled-code
// write barrier — their stores go through Rust runtime helpers — so the
// band-aid root walks in `pyre-interpreter::eval::walk_pyframe_roots` rescan
// them at EVERY minor collection.  This flag is the write-barrier bit for
// that whole prebuilt family, one global bit instead of a per-object flag
// (coarser than upstream: any single store rescans everything on the next
// minor collection — still sound, just less selective).  Every mutation
// helper that can store a (possibly nursery-young) `PyObjectRef` into one of
// those walked structures must call [`mark_prebuilt_roots_dirty`]; the minor
// walk clears the bit after it runs (the walk itself promotes every young
// value it reaches, so a clean bit again means "no young pointers inside").
use std::sync::atomic::{AtomicBool, Ordering};

static PREBUILT_ROOTS_DIRTY: AtomicBool = AtomicBool::new(true);

/// Write-barrier hook for the prebuilt/Box-immortal root family
/// (incminimark `remember_young_pointer` analog).  Call from every helper
/// that stores a `PyObjectRef` into a structure reached only by the
/// `walk_pyframe_roots` band-aid walks.
///
/// Sets the static `PREBUILT_ROOTS_DIRTY` bit the tracer cannot model; the
/// JIT residualises the call instead of tracing into it (`@dont_look_inside`,
/// `rlib/jit.py:139`), the `pin_root` twin.
#[majit_macros::dont_look_inside]
pub fn mark_prebuilt_roots_dirty() {
    PREBUILT_ROOTS_DIRTY.store(true, Ordering::Relaxed);
}

/// Whether any prebuilt-family store happened since the last completed
/// minor-collection walk (incminimark: "is this object in
/// `old_objects_pointing_to_young`?").
#[inline]
pub fn prebuilt_roots_dirty() -> bool {
    PREBUILT_ROOTS_DIRTY.load(Ordering::Relaxed)
}

/// Clear the write-tracking bit after a minor-collection walk promoted every
/// young value reachable from the prebuilt family (incminimark: the minor
/// collection empties `old_objects_pointing_to_young` and re-sets
/// GCFLAG_TRACK_YOUNG_PTRS).  Must only be called by the walker, after the
/// walk completed, during a stop-the-world collection.
#[inline]
pub fn clear_prebuilt_roots_dirty() {
    PREBUILT_ROOTS_DIRTY.store(false, Ordering::Relaxed);
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

    /// Currently `RootScope` carries a single `usize` save-point.
    /// This will be swapped for a backend `RootSet::Handle` (still
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

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "the shadow stack was mutated while a root walk was in progress")]
    fn walk_shadow_stack_area_rejects_reentrant_push_in_debug_builds() {
        let _roots = push_roots();
        pin_root(dummy(0x1));
        let area = capture_shadow_stack_area();
        // SAFETY: `area` is this thread's shadow stack, and this synchronous
        // self-walk keeps it live for the duration of the call.
        unsafe { walk_shadow_stack_area(area, |_| pin_root(dummy(0x2))) };
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn walk_shadow_stack_survives_push_that_reallocates() {
        let _roots = push_roots();
        pin_root(dummy(0x11));
        pin_root(dummy(0x22));

        let (capacity_before, additional) =
            with_shadow_stack(|stack| (stack.capacity(), stack.capacity() - stack.len() + 1));
        let mut pushed = false;
        let mut seen = Vec::new();
        walk_shadow_stack(|slot| {
            seen.push(*slot as usize);
            if !pushed {
                pushed = true;
                for offset in 0..additional {
                    pin_root(dummy(0x1000 + offset));
                }
            }
        });

        assert!(pushed);
        // The visitor's pins really did move the buffer.
        assert!(with_shadow_stack(Vec::capacity) > capacity_before);
        // `0x22` is the discriminator: it is read on the iteration AFTER the
        // reallocation, so getting it right means the walk followed the buffer
        // to its new address instead of reading the freed one. The walk stops
        // at the two roots that were live at entry — the interval is fixed
        // there, as in `walk_stack_root` (`shadowstack.py:43-46`), so the
        // roots the visitor pinned are not part of this walk.
        assert_eq!(seen, vec![0x11, 0x22]);
        assert_eq!(shadow_stack_get(0) as usize, 0x11);
        assert_eq!(shadow_stack_get(1) as usize, 0x22);
    }
}
