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

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::Cell;
use std::marker::PhantomData;

use crate::pyobject::PyObjectRef;

/// `shadowstack.py:281 root_stack_depth` — the slot count `ShadowStackPool`
/// raw-mallocs for a thread's root stack before any growth.  `majit-gc`'s
/// jitframe root stack starts from the same constant.
const DEFAULT_ROOT_STACK_DEPTH: usize = 163840;

/// `gcdata.root_stack_base` / `root_stack_top` (`shadowstack.py:75-88`), with
/// the `limit` cell `majit-gc`'s jitframe root stack also keeps so a push can
/// see it has reached `root_stack_depth` without a separate depth load.
///
/// Deliberately holds no `Vec` and no `Box`.  A growable container puts drop
/// glue on the thread-local, and a thread-local carrying a destructor costs a
/// registration-state check on every access — `pin_root` and
/// `shadow_stack_get` sit on the interpreter's allocation and call paths, so
/// that check is not free: dropping it measures `.8711` / `.8727` on the
/// any/all rooting probes with a pure-integer control at exactly `1.0000`.
/// The buffer is raw-allocated once instead, as `_prepare_unused_stack` does
/// (`shadowstack.py:344-349`), and released by [`RootStackOwner`].
struct RootStack {
    base: Cell<*mut PyObjectRef>,
    top: Cell<*mut PyObjectRef>,
    limit: Cell<*mut PyObjectRef>,
}

impl RootStack {
    const fn new() -> Self {
        Self {
            base: Cell::new(std::ptr::null_mut()),
            top: Cell::new(std::ptr::null_mut()),
            limit: Cell::new(std::ptr::null_mut()),
        }
    }

    /// Slots currently in use — `root_stack_top - root_stack_base`.
    #[inline(always)]
    fn len(&self) -> usize {
        let base = self.base.get();
        if base.is_null() {
            return 0;
        }
        // SAFETY: `top` is derived from `base` by pointer arithmetic within
        // the one live allocation, so the offset is in range.
        unsafe { self.top.get().offset_from(base) as usize }
    }

    /// Slot count the current buffer can hold — `root_stack_depth`.
    fn capacity(&self) -> usize {
        let base = self.base.get();
        if base.is_null() {
            return 0;
        }
        // SAFETY: `limit` is the one-past-the-end pointer of the allocation
        // `base` starts.
        unsafe { self.limit.get().offset_from(base) as usize }
    }

    fn layout_for(depth: usize) -> Layout {
        Layout::array::<PyObjectRef>(depth).expect("root stack layout")
    }

    /// `ShadowStackPool.increase_root_stack_depth` (`shadowstack.py:351-364`):
    /// allocate a larger buffer, copy the used portion over, and re-derive
    /// base/top; the depth can never shrink.  Unlike upstream this also serves
    /// as the initial `_prepare_unused_stack`, so a null base is the
    /// "no buffer yet" case rather than a separate pooled one.
    #[cold]
    fn grow(&self, new_depth: usize) {
        let old_base = self.base.get();
        let used = self.len();
        let new_depth = new_depth.max(DEFAULT_ROOT_STACK_DEPTH).max(used);
        if new_depth <= self.capacity() {
            return; // can't easily decrease the size
        }
        // Registering this thread's free-at-exit hook is what keeps the hot
        // `ROOT_STACK` key destructor-free; touching it here rather than on
        // every push keeps that cost off the push path.
        let _ = ROOT_STACK_OWNER.try_with(|_| ());
        // SAFETY: a non-zero array layout; `alloc_zeroed` gives null slots,
        // which the walkers read as "no root" exactly like a null GCREF.
        let new_base = unsafe { alloc_zeroed(Self::layout_for(new_depth)) } as *mut PyObjectRef;
        assert!(!new_base.is_null(), "root stack allocation failed");
        if !old_base.is_null() {
            // SAFETY: `used` slots are live in the old buffer and the new
            // allocation is at least that large; the two never overlap.
            unsafe { std::ptr::copy_nonoverlapping(old_base, new_base, used) };
            // SAFETY: the old buffer came from this same allocator with the
            // layout its own capacity names, and nothing references it now —
            // callers hold indices, not slot addresses, across a push.
            unsafe { dealloc(old_base as *mut u8, Self::layout_for(self.capacity())) };
        }
        self.base.set(new_base);
        // SAFETY: both offsets are inside (or one past the end of) the new
        // allocation.
        unsafe {
            self.top.set(new_base.add(used));
            self.limit.set(new_base.add(new_depth));
        }
    }

    /// `incr_stack(1)` (`shadowstack.py:80-84`) — return the slot that was the
    /// top and advance past it.  Upstream's bump is unchecked because the
    /// native stack check fires first, with `root_stack_depth` scaled to the
    /// recursion limit.  pyre pins one slot per argument as well as per live
    /// variable, so a single variadic call can outrun any fixed headroom; the
    /// limit test grows instead of trusting that margin.
    #[inline(always)]
    fn incr_stack(&self) -> *mut PyObjectRef {
        let top = self.top.get();
        if top == self.limit.get() {
            self.grow(self.capacity() * 2);
            return self.incr_stack();
        }
        // SAFETY: `top` is below `limit`, so it names a live slot and the
        // advanced pointer is at most one past the end.
        self.top.set(unsafe { top.add(1) });
        top
    }

    /// `decr_stack` to an absolute depth — the `pop_roots` rewind.
    #[inline(always)]
    fn truncate(&self, len: usize) {
        if len < self.len() {
            // SAFETY: `len` is below the live length, so the slot is inside
            // the allocation.
            self.top.set(unsafe { self.base.get().add(len) });
        }
    }

    /// Address of slot `index`, panicking exactly like indexing did.
    #[inline(always)]
    fn slot(&self, index: usize) -> *mut PyObjectRef {
        assert!(index < self.len(), "shadow-stack index out of range");
        // SAFETY: bounds-checked against the live length just above.
        unsafe { self.base.get().add(index) }
    }
}

/// Frees [`ROOT_STACK`]'s buffer when the thread exits.  It lives in its own
/// thread-local key, touched only from `RootStack::grow`, so the key on the
/// push/read path keeps no destructor of its own.
struct RootStackOwner;

impl Drop for RootStackOwner {
    fn drop(&mut self) {
        // `ROOT_STACK` has no destructor, so it is never torn down and this
        // access cannot fail; `try_with` keeps that from being load-bearing.
        let _ = ROOT_STACK.try_with(|stack| {
            let base = stack.base.get();
            if base.is_null() {
                return;
            }
            let layout = RootStack::layout_for(stack.capacity());
            stack.base.set(std::ptr::null_mut());
            stack.top.set(std::ptr::null_mut());
            stack.limit.set(std::ptr::null_mut());
            // SAFETY: the buffer came from this allocator with this layout and
            // the cells that named it have just been cleared.
            unsafe { dealloc(base as *mut u8, layout) };
        });
    }
}

thread_local! {
    /// Thread-local shadow stack — pyre's mirror of RPython's
    /// `framework.shadowstack` (`shadowstack.py`). Append-only across
    /// each [`RootScope`]'s lifetime; the matching [`Drop`] rewinds `top`
    /// back to the saved length.
    ///
    /// Access is raw, matching RPython's base/top shadow-stack shape. Debug
    /// builds reject mutation during a root walk through
    /// `SHADOW_STACK_WALK_IN_PROGRESS`; release builds pay no bookkeeping.
    static ROOT_STACK: RootStack = const { RootStack::new() };

    /// Registered on the first `grow`; see [`RootStackOwner`].
    static ROOT_STACK_OWNER: RootStackOwner = const { RootStackOwner };

    /// Whether this thread is currently driving a shadow-stack root walk.
    #[cfg(debug_assertions)]
    static SHADOW_STACK_WALK_IN_PROGRESS: Cell<bool> = const { Cell::new(false) };
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

/// Run `f` against this thread's root stack.  The borrow guards the old
/// `Vec` shape needed are gone with it: base/top are `Cell`s, so no
/// reference is held across a call that could push.
#[inline(always)]
fn with_shadow_stack<R>(f: impl FnOnce(&RootStack) -> R) -> R {
    ROOT_STACK.with(f)
}

/// `increase_root_stack_depth(new_depth)` (`rlib/rgc.py:775-779` →
/// `shadowstack.py:351-364`).  `sys.setrecursionlimit` scales the root stack
/// with the limit at `pypy/module/sys/vm.py:97`; the depth can only grow.
pub fn increase_root_stack_depth(new_depth: usize) {
    with_shadow_stack(|stack| stack.grow(new_depth));
}

/// `root_stack_depth` — the slot count this thread's root stack can hold.
pub fn root_stack_depth() -> usize {
    with_shadow_stack(|stack| {
        let capacity = stack.capacity();
        if capacity == 0 {
            DEFAULT_ROOT_STACK_DEPTH
        } else {
            capacity
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
    /// Length of [`ROOT_STACK`] at the moment [`push_roots`] was
    /// called. Will be replaced with a backend
    /// `RootSet::Handle` once the active GC consumes the stack.
    save_point: usize,
    /// Resolved thread-local stack cell. RPython keeps the shadow-stack
    /// base/top address live for the whole root bracket; caching the cell here
    /// avoids resolving TLS again for every generated slot write/read. It names
    /// the [`RootStack`] and not its buffer, so a `grow` under this bracket
    /// leaves it valid.
    stack_slot: *const RootStack,
    /// A root bracket belongs to the thread whose shadow stack supplied this
    /// save point. Moving it could truncate an unrelated thread's stack.
    _not_send: PhantomData<*const ()>,
}

impl RootScope {
    /// Internal constructor. Callers should always go through
    /// [`push_roots`] so the API surface stays uniform across phases.
    #[inline]
    fn new() -> Self {
        let (save_point, stack_slot) =
            with_shadow_stack(|stack| (stack.len(), stack as *const RootStack));
        Self {
            save_point,
            stack_slot,
            _not_send: PhantomData,
        }
    }

    /// First slot owned by this root bracket.
    #[inline]
    pub fn base(&self) -> usize {
        self.save_point
    }

    /// Scope-local [`pin_root`] using the already-resolved root-stack cell.
    #[majit_macros::dont_look_inside]
    pub fn pin_root(&self, root: PyObjectRef) {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        // SAFETY: `stack_slot` is this thread's root-stack cell, alive for the
        // bracket; `incr_stack` returns the slot it just claimed.
        let index = unsafe {
            let stack = &*self.stack_slot;
            let index = stack.len();
            *stack.incr_stack() = root;
            index
        };
        let normalized =
            crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
        if normalized != root {
            // SAFETY: same cell, and `index` is still live.
            unsafe { *(*self.stack_slot).slot(index) = normalized };
        }
    }

    /// Scope-local [`shadow_stack_get`] using the cached cell.
    #[majit_macros::dont_look_inside]
    pub fn get(&self, index: usize) -> PyObjectRef {
        // SAFETY: same cell; `slot` bounds-checks `index`.
        unsafe { *(*self.stack_slot).slot(index) }
    }

    /// Scope-local [`publish_roots`] using the cached cell.
    #[majit_macros::dont_look_inside]
    pub fn publish(&self, roots: &[PyObjectRef]) -> usize {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        // SAFETY: this thread's cell, alive for the bracket; `incr_stack`
        // returns the slot it just claimed.
        unsafe {
            let stack = &*self.stack_slot;
            let base = stack.len();
            for &root in roots {
                *stack.incr_stack() = root;
            }
            base
        }
    }

    /// Scope-local [`normalize_roots`] using the cached cell.
    #[majit_macros::dont_look_inside]
    pub fn normalize(&self, base: usize, len: usize) {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        for index in base..base + len {
            // Re-read the slot each time, as [`normalize_roots`] does: a
            // collection triggered by an earlier query may already have
            // rewritten every published root in place.
            // SAFETY: `publish` claimed every index in this range.
            let root = unsafe { *(*self.stack_slot).slot(index) };
            let current =
                crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
            if current != root {
                // SAFETY: same slot, still live.
                unsafe { *(*self.stack_slot).slot(index) = current };
            }
        }
    }

    /// Scope-local [`shadow_stack_set`] using the cached cell — the write a
    /// slot whose contents change over the bracket takes on each update.
    #[majit_macros::dont_look_inside]
    pub fn set(&self, index: usize, root: PyObjectRef) {
        #[cfg(debug_assertions)]
        assert_shadow_stack_not_walking();
        // Publish the raw value before the query, for the reason [`pin_root`]
        // gives: the query is itself a GC operation that may park behind
        // another thread's collection.
        // SAFETY: same cell; `slot` bounds-checks `index`.
        unsafe { *(*self.stack_slot).slot(index) = root };
        let normalized =
            crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
        if normalized != root {
            // SAFETY: same slot, still live.
            unsafe { *(*self.stack_slot).slot(index) = normalized };
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
        // `truncate` is a no-op if `save_point >= len()`, which is
        // the steady-state case for an empty bracket.
        // SAFETY: `stack_slot` is this thread's root-stack cell; `_not_send`
        // keeps the bracket on the thread that resolved it.
        unsafe { (*self.stack_slot).truncate(self.save_point) };
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
/// Pushes onto the thread-local `ROOT_STACK`, a runtime-mutable root the
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
    let index = with_shadow_stack(|stack| {
        let index = stack.len();
        // SAFETY: `incr_stack` returns the slot it just claimed.
        unsafe { *stack.incr_stack() = root };
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
    // between the push above and the query — and skipping it keeps a second
    // thread-local resolve off a path that sits on the interpreter's
    // allocation and call paths.
    if normalized != root {
        with_shadow_stack(|stack| {
            // SAFETY: `index` was live when claimed and only this thread can
            // have shortened the stack, which it has not.
            unsafe { *stack.slot(index) = normalized }
        });
    }
}

/// Publish a complete translated livevar set before performing any
/// forwarding query.  RPython's `push_roots(hop)` writes every live GCREF to
/// the root stack as one pre-allocation phase; calling [`pin_root`] repeatedly
/// would query after the first write and let a foreign collection run before
/// later values were visible.
///
/// Returns the first shadow-stack index occupied by `roots`.
///
/// A livevar set that does not sit in one slice publishes each slice with
/// [`publish_roots`] and normalizes the whole range once with
/// [`normalize_roots`]. Calling this function per slice would reopen the
/// window it exists to close: the first call's queries are safepoints, and
/// the later slices are not yet on the stack when they run.
#[majit_macros::dont_look_inside]
pub fn pin_roots(roots: &[PyObjectRef]) -> usize {
    let base = publish_roots(roots);
    normalize_roots(base, roots.len());
    base
}

/// The write half of `push_roots(hop)`: store `roots` in fresh root-stack
/// slots and return the first index, querying the GC for nothing.
///
/// Split out for the callers whose livevar set spans several slices — they run
/// every `publish_roots` before the first [`normalize_roots`], so no value is
/// still invisible to a foreign collector once this mutator starts entering
/// safepoints.
#[majit_macros::dont_look_inside]
pub fn publish_roots(roots: &[PyObjectRef]) -> usize {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    with_shadow_stack(|stack| {
        let base = stack.len();
        for &root in roots {
            // SAFETY: `incr_stack` returns the slot it just claimed.
            unsafe { *stack.incr_stack() = root };
        }
        base
    })
}

/// Resolve forwarding across the `len` published slots starting at `base` —
/// the half of the pin that is itself a GC operation. See [`pin_root`] for why
/// a value copied into the bracket from outside it can already name a
/// forwarded nursery object.
#[majit_macros::dont_look_inside]
pub fn normalize_roots(base: usize, len: usize) {
    #[cfg(debug_assertions)]
    assert_shadow_stack_not_walking();
    for index in base..base + len {
        // Read the slot each time: a collection triggered by an earlier query
        // may already have rewritten every published root in place.
        // SAFETY: every index in this range was claimed just above.
        let root = with_shadow_stack(|stack| unsafe { *stack.slot(index) });
        let current = crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
        // Same write-back economy as `pin_root`: the slot already holds `root`,
        // so a query that found no forwarding stub has nothing to store, and
        // skipping it saves a second thread-local resolve per livevar.
        if current != root {
            // SAFETY: same slot, still live.
            with_shadow_stack(|stack| unsafe { *stack.slot(index) = current });
        }
    }
}

/// Current length of the thread-local shadow stack. Used by
/// [`RootScope::new`] to capture the save-point and by tests to
/// observe pin/pop behaviour.
///
/// Reads the thread-local `ROOT_STACK`, a runtime-mutable root the
/// tracer cannot type; the JIT residualises the read instead of tracing
/// into it (`@dont_look_inside`, `rlib/jit.py:139`). The attribute is a
/// tracing-policy marker only — it leaves the host backend free to inline
/// this body, exactly as the RPython decorator leaves the C backend free.
#[majit_macros::dont_look_inside]
pub fn shadow_stack_len() -> usize {
    with_shadow_stack(RootStack::len)
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
/// Reads the thread-local `ROOT_STACK` the tracer cannot type; the JIT
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
    // SAFETY: `slot` bounds-checks `index` against the live length.
    with_shadow_stack(|stack| unsafe { *stack.slot(index) })
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
        assert!(
            base + dst.len() <= stack.len(),
            "shadow-stack range out of bounds"
        );
        if dst.is_empty() {
            // An empty range starting one past the last live slot is in
            // bounds — `&stack[len..len]` was — but `slot(len)` is not, and
            // `copy_nonoverlapping` wants a non-null source even for a
            // zero-length copy.  Neither applies when there is nothing to
            // copy.
            return;
        }
        // SAFETY: the whole range was just bounds-checked, and `dst` is a
        // distinct caller-owned slice.
        unsafe { std::ptr::copy_nonoverlapping(stack.slot(base), dst.as_mut_ptr(), dst.len()) };
    });
}

/// Overwrite a single shadow-stack slot by index, panicking if the index
/// is out of bounds. A slot whose contents change over a bracket's lifetime
/// is written here rather than re-pinned, mirroring `gc_save_root`'s
/// overwrite of an already allocated slot (`shadowcolor.py:126-129`).
///
/// Writes the thread-local `ROOT_STACK` the tracer cannot type; the JIT
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
    // SAFETY: `slot` bounds-checks `index` against the live length.
    with_shadow_stack(|stack| unsafe { *stack.slot(index) = root });
    // Then normalize a GCREF the caller may have copied before a foreign
    // mutator's nursery collection, so the slot never keeps a forwarding stub.
    let root = crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
    // SAFETY: same slot, still live.
    with_shadow_stack(|stack| unsafe { *stack.slot(index) = root });
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
    ROOT_STACK.with(|stack| {
        let cell = stack as *const RootStack;
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
    // The captured address is the `RootStack` cell itself, never its buffer:
    // `grow` replaces the buffer, so a foreign walker must re-read base/top —
    // the same rule `majit-gc`'s jitframe root stack states for the
    // `root_stack_top` address it hands to compiled code.
    ROOT_STACK.with(|s| s as *const _ as *const ())
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
    let cell = data as *const RootStack;
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

/// Walk one captured root stack, re-reading `base` for every slot.
///
/// # Safety
/// `cell` must remain a valid root-stack cell for the duration of the walk,
/// and no thread other than a re-entrant visitor may mutate it.
unsafe fn walk_shadow_stack_cell(
    cell: *const RootStack,
    visitor: &mut impl FnMut(&mut PyObjectRef),
) {
    // The interval is fixed at entry: `walk_stack_root` (`shadowstack.py:43-46`)
    // receives `start` and `addr` as arguments and runs `while addr != start`,
    // so a root pushed mid-walk is not part of that walk. Re-reading the length
    // each iteration instead would extend the walk over roots pinned by the
    // visitor — and pinning during a walk is precisely what the debug guard
    // above declares illegal, so there is nothing to serve by diverging here.
    // SAFETY: the caller guarantees `cell` names a live root stack.
    let stack = unsafe { &*cell };
    let end = stack.len();
    let mut index = 0;
    while index < end {
        // Re-read `base` every iteration rather than caching it: a re-entrant
        // push that hits the limit calls `grow`, which moves the buffer and
        // would strand a cached pointer in the freed one. Only pushes can
        // occur, so `end` stays within bounds.
        // SAFETY: `index < end <= len()`, so the slot is live.
        let mut root = unsafe { *stack.base.get().add(index) };
        visitor(&mut root);
        // Store back through a re-read `base` rather than handing the visitor
        // a `&mut` into the buffer: a re-entrant push that grows the stack
        // frees the buffer the slot was read from, and a visitor that writes
        // its reference after allocating would then write into the freed one.
        // `grow` copies the live prefix, so `index` still names this slot.
        // SAFETY: `index < end <= len()`, so the slot is live.
        unsafe { *stack.base.get().add(index) = root };
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

    /// `RootScope` carries the saved top plus the resolved root-stack cell,
    /// matching RPython's base/top pair. Trap unintended growth beyond those
    /// two words.
    #[test]
    fn root_scope_payload_is_two_words() {
        assert_eq!(
            std::mem::size_of::<RootScope>(),
            2 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn scope_local_pin_and_get_share_the_cached_stack_cell() {
        let before = shadow_stack_len();
        {
            let roots = push_roots();
            assert_eq!(roots.base(), before);
            roots.pin_root(dummy(0x1234));
            roots.pin_root(dummy(0x5678));
            assert_eq!(roots.get(before) as usize, 0x1234);
            assert_eq!(roots.get(before + 1) as usize, 0x5678);
        }
        assert_eq!(shadow_stack_len(), before);
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

    /// An empty range is in bounds anywhere up to and including the live
    /// length — `&stack[len..len]` was, and `pop_roots` is emitted even for
    /// zero livevars (`shadowstack.py:37-40`). The top-of-stack case is the
    /// one a per-slot bounds check gets wrong.
    #[test]
    fn copy_range_accepts_an_empty_range_at_the_top_of_the_stack() {
        let _roots = push_roots();
        pin_root(dummy(0x1));
        shadow_stack_copy_range(shadow_stack_len(), &mut []);
    }

    /// Past the live length it is out of bounds even when empty, which is
    /// also what indexing did.
    #[test]
    #[should_panic(expected = "shadow-stack range out of bounds")]
    fn copy_range_rejects_an_empty_range_past_the_top_of_the_stack() {
        let _roots = push_roots();
        pin_root(dummy(0x1));
        shadow_stack_copy_range(shadow_stack_len() + 1, &mut []);
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
    fn walk_shadow_stack_survives_push_that_grows_the_buffer() {
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
        // The visitor's pins really did hit the limit and move the buffer.
        assert!(with_shadow_stack(RootStack::capacity) > capacity_before);
        // `0x22` is the discriminator: it is read on the iteration AFTER the
        // grow, so getting it right means the walk followed the buffer
        // to its new address instead of reading the freed one. The walk stops
        // at the two roots that were live at entry — the interval is fixed
        // there, as in `walk_stack_root` (`shadowstack.py:43-46`), so the
        // roots the visitor pinned are not part of this walk.
        assert_eq!(seen, vec![0x11, 0x22]);
        assert_eq!(shadow_stack_get(0) as usize, 0x11);
        assert_eq!(shadow_stack_get(1) as usize, 0x22);
    }

    /// The read twin above only proves the walk *reads* from the relocated
    /// buffer. A moving collector also *writes* the forwarded address back,
    /// and that write happens after the visitor has had its chance to grow
    /// the stack — so the store has to resolve the buffer address again
    /// rather than reuse the one the slot was read from.
    #[cfg(not(debug_assertions))]
    #[test]
    fn walk_shadow_stack_write_back_follows_a_grow_inside_the_visitor() {
        let _roots = push_roots();
        pin_root(dummy(0x11));
        pin_root(dummy(0x22));

        let additional = with_shadow_stack(|stack| stack.capacity() - stack.len() + 1);
        let mut pushed = false;
        walk_shadow_stack(|slot| {
            if !pushed {
                pushed = true;
                for offset in 0..additional {
                    pin_root(dummy(0x1000 + offset));
                }
            }
            // Forward the root the way a moving collector would, *after* the
            // buffer has already moved under this slot.
            *slot = dummy(*slot as usize + 0xA0_0000);
        });

        assert!(pushed);
        // Slot 0 is the discriminator: it was read from the old buffer, and
        // the grow happened before the visitor forwarded it. The write is
        // only observable here if it went to the buffer the stack owns now.
        // Slot 1 was read after the grow, so it lands correctly either way.
        assert_eq!(shadow_stack_get(0) as usize, 0xA0_0011);
        assert_eq!(shadow_stack_get(1) as usize, 0xA0_0022);
    }
}
