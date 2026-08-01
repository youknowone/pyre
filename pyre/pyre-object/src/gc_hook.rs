//! Process-global GC allocation hook for host-side Python object allocators.
//!
//! `pyre-object` sits below `majit-gc` in the dependency graph and must
//! not depend on it for the GC itself. Host-side allocators (`w_int_new`,
//! `w_float_new`, …) that want to route through the real GC instead of
//! `Box::into_raw` go through the callback registered here. `pyre-jit::eval`
//! installs the concrete trampoline on `JitDriver` init so the callback
//! reaches the backend-owned GC allocator via `majit_gc` hooks.
//!
//! The GC is a process-global singleton (`gc_sync`), so these hooks are
//! process-global fn-pointer cells ([`majit_gc::hook_cell::FnPtrCell`] via
//! [`majit_gc::global_hook`]) rather than per-thread: one install publishes
//! the same pointer to every thread, and a collector running on an arbitrary
//! thread must see it even if that thread never ran the install path.
//!
//! Callers use [`try_gc_alloc`] which returns `None` when no hook is
//! installed — they fall back to the `Box::into_raw` path in
//! that case. Incremental migration drops the `Box::into_raw`
//! fallback at each call site as the hook's reliability is verified
//! under the full bench suite.
//!
//! Layering: this module defines the function-pointer slots only. Wire-up
//! lives in `pyre-jit`.

use crate::PyObjectRef;

/// Per-`#[pyre_class]`-type registry of inline `PyObjectRef` field offsets
/// (the descriptor's `gc_ptr_offsets`), keyed by the type's static `PyType`
/// pointer cast to `usize`.
///
/// Populated once at single-threaded JIT-driver init (`register_pyre_class`
/// in `pyre-jit/src/eval.rs`).  Read during GC marking by the interpreter's
/// generic immortal-root walker to force-forward the managed children of a
/// `malloc_typed`-immortal object the marker skips.  The stored slice is the
/// descriptor's `&'static [usize]`, so a lookup copies a fat pointer with no
/// allocation.
static PYRE_CLASS_GC_OFFSETS: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<usize, &'static [usize]>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

/// Register a type's inline `gc_ptr_offsets` under its `PyType` pointer.
/// Called only at single-threaded init; a poisoned lock silently drops the
/// insert (init is the sole writer, so this cannot happen in practice).
pub fn register_pyre_class_offsets(pytype_ptr: usize, offsets: &'static [usize]) {
    if let Ok(mut map) = PYRE_CLASS_GC_OFFSETS.write() {
        map.insert(pytype_ptr, offsets);
    }
}

/// Look up the registered inline `PyObjectRef` field offsets for `ob_type`.
/// Returns `None` for a null `ob_type` or an unregistered type.  Takes a
/// process-global read lock — cheap and alloc-free, safe while the collector
/// is marking; a poisoned lock reads as `None`.
///
/// # Safety
/// `ob_type` must be null or point to a valid, `'static` `PyType`.
pub unsafe fn offsets_for_pytype(
    ob_type: *const crate::pyobject::PyType,
) -> Option<&'static [usize]> {
    if ob_type.is_null() {
        return None;
    }
    PYRE_CLASS_GC_OFFSETS
        .read()
        .ok()?
        .get(&(ob_type as usize))
        .copied()
}

/// Signature of the host-side GC allocation callback.
///
/// `type_id` is the backend-registered GC type id (same id used by
/// JIT-compiled `NewWithVtable`). `payload_size` is the number of
/// payload bytes requested. The callback returns an uninitialised
/// pointer to managed memory of exactly that size, ready for raw
/// field writes. On allocation failure the callback returns
/// `std::ptr::null_mut()`.
pub type GcAllocHookFn = fn(type_id: u32, payload_size: usize) -> *mut u8;

majit_gc::global_hook!(static GC_ALLOC_HOOK: GcAllocHookFn);
majit_gc::global_hook!(static GC_ALLOC_STABLE_HOOK: GcAllocHookFn);

/// `malloc_fast` companion of [`GcAllocHookFn`].
///
/// `gct_fv_gc_malloc` (`framework.py:820-838`) resolves the allocated type's
/// finalizer and weakref properties while transforming the malloc site, and
/// calls the `inline=True` copy of `malloc_fixedsize` annotated
/// `s_False, s_False, s_False` (`framework.py:361-382`) whenever both are
/// false — which folds incminimark.py:687-693's two registrations out of the
/// body. A pyre call site with a statically known type id is in the same
/// position, so it selects the folded entry point the same way.
///
/// # Safety
/// The caller passes a `type_id` naming a type with no destructor and no
/// weakref flag.
pub type GcAllocFastHookFn = unsafe fn(type_id: u32, payload_size: usize) -> *mut u8;

majit_gc::global_hook!(static GC_ALLOC_FAST_HOOK: GcAllocFastHookFn);

/// Placement-reporting companion of [`GcAllocHookFn`] for no-collect
/// allocations that may spill from the nursery to old-gen.
pub type GcAllocWithPlacementHookFn =
    unsafe fn(type_id: u32, payload_size: usize, needs_write_barrier: *mut bool) -> *mut u8;

majit_gc::global_hook!(
    static GC_ALLOC_WITH_PLACEMENT_HOOK: GcAllocWithPlacementHookFn
);

majit_gc::global_hook!(
    static GC_ALLOC_FAST_WITH_PLACEMENT_HOOK: GcAllocWithPlacementHookFn
);

/// Install the allocation callback. Overwrites any previously-installed hook.
pub fn register_gc_alloc_hook(hook: GcAllocHookFn) {
    GC_ALLOC_HOOK.set(Some(hook));
}

/// Remove the callback. Subsequent [`try_gc_alloc`] returns `None` until a
/// new hook is registered.
pub fn clear_gc_alloc_hook() {
    GC_ALLOC_HOOK.set(None);
}

/// Attempt a GC allocation via the installed hook. Returns `None`
/// when no hook is installed, or `Some(null)` when the hook itself
/// returned null.
#[inline]
pub fn try_gc_alloc(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    GC_ALLOC_HOOK.get().map(|f| f(type_id, payload_size))
}

/// Install the `malloc_fast` allocation callback.
pub fn register_gc_alloc_fast_hook(hook: GcAllocFastHookFn) {
    GC_ALLOC_FAST_HOOK.set(Some(hook));
}

/// Remove the `malloc_fast` allocation callback.
pub fn clear_gc_alloc_fast_hook() {
    GC_ALLOC_FAST_HOOK.set(None);
}

/// [`try_gc_alloc`] for a statically-known type that carries neither a
/// finalizer nor the weakref flag. Falls back to the general hook when no
/// backend installed the folded one — the general body is always a correct
/// implementation of the folded one.
///
/// # Safety
/// `type_id` must name a type with no destructor and no weakref flag.
#[inline]
pub unsafe fn try_gc_alloc_fast(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    if let Some(f) = GC_ALLOC_FAST_HOOK.get() {
        return Some(unsafe { f(type_id, payload_size) });
    }
    try_gc_alloc(type_id, payload_size)
}

pub fn register_gc_alloc_with_placement_hook(hook: GcAllocWithPlacementHookFn) {
    GC_ALLOC_WITH_PLACEMENT_HOOK.set(Some(hook));
}

pub fn clear_gc_alloc_with_placement_hook() {
    GC_ALLOC_WITH_PLACEMENT_HOOK.set(None);
}

/// Attempt a no-collect allocation and report whether a fresh GC-reference
/// field needs a creation barrier. Backends without placement reporting fall
/// back to the ordinary allocation hook and conservatively keep the barrier.
#[inline]
pub fn try_gc_alloc_with_placement(
    type_id: u32,
    payload_size: usize,
    needs_write_barrier: &mut bool,
) -> Option<*mut u8> {
    if let Some(f) = GC_ALLOC_WITH_PLACEMENT_HOOK.get() {
        return Some(unsafe { f(type_id, payload_size, needs_write_barrier as *mut bool) });
    }
    *needs_write_barrier = true;
    try_gc_alloc(type_id, payload_size)
}

/// Install the `malloc_fast` placement-reporting allocation callback.
pub fn register_gc_alloc_fast_with_placement_hook(hook: GcAllocWithPlacementHookFn) {
    GC_ALLOC_FAST_WITH_PLACEMENT_HOOK.set(Some(hook));
}

/// Remove the `malloc_fast` placement-reporting allocation callback.
pub fn clear_gc_alloc_fast_with_placement_hook() {
    GC_ALLOC_FAST_WITH_PLACEMENT_HOOK.set(None);
}

/// [`try_gc_alloc_with_placement`] for a statically-known type that carries
/// neither a finalizer nor the weakref flag.
///
/// # Safety
/// `type_id` must name a type with no destructor and no weakref flag.
#[inline]
pub unsafe fn try_gc_alloc_fast_with_placement(
    type_id: u32,
    payload_size: usize,
    needs_write_barrier: &mut bool,
) -> Option<*mut u8> {
    if let Some(f) = GC_ALLOC_FAST_WITH_PLACEMENT_HOOK.get() {
        return Some(unsafe { f(type_id, payload_size, needs_write_barrier as *mut bool) });
    }
    try_gc_alloc_with_placement(type_id, payload_size, needs_write_barrier)
}

/// Install the stable (old-gen) allocation callback for this thread.
///
/// Used by host-side allocators (`w_int_new`, `w_float_new`, …)
/// whose callers hold the returned pointer on the Rust stack across
/// subsequent allocations without registering it as a GC root
/// The backend routes this to an old-gen allocator
/// whose returned pointer is stable across minor and major
/// collections (MiniMark mark-sweep does not move old-gen objects).
pub fn register_gc_alloc_stable_hook(hook: GcAllocHookFn) {
    GC_ALLOC_STABLE_HOOK.set(Some(hook));
}

/// Remove the stable-allocation callback.
pub fn clear_gc_alloc_stable_hook() {
    GC_ALLOC_STABLE_HOOK.set(None);
}

/// Attempt a stable-address GC allocation via the installed hook.
/// See [`register_gc_alloc_stable_hook`] for semantics. Returns
/// `None` when no hook is installed.
#[inline]
pub fn try_gc_alloc_stable(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    GC_ALLOC_STABLE_HOOK.get().map(|f| f(type_id, payload_size))
}

/// Null-collapsing view of [`try_gc_alloc_stable`] for JIT-traced callers.
///
/// Returns `null` both when no hook is installed and when the hook itself
/// returned `null`; every traced caller already treats those two cases
/// identically (fall back to `malloc_typed`), so the `Option` discriminant
/// carries no information the caller reads.  Residualising this primitive
/// (`@dont_look_inside`, `rlib/jit.py:139`) keeps the process-global hook
/// dispatch out of the trace — a `*mut u8` return has no discriminant to
/// erase, unlike the `Option<*mut u8>` accessor.
#[majit_macros::dont_look_inside]
pub fn try_gc_alloc_stable_raw(type_id: u32, payload_size: usize) -> *mut u8 {
    try_gc_alloc_stable(type_id, payload_size).unwrap_or(core::ptr::null_mut())
}

majit_gc::global_hook!(static GC_ALLOC_COLLECTING_HOOK: GcAllocHookFn);

/// Install the *collecting* nursery allocation callback.
///
/// Unlike [`register_gc_alloc_hook`] (no-collect), the backend routes this to a
/// nursery allocator that runs a minor collection when the nursery is full. Only
/// for callers that hold no unrooted GC pointer across the allocation and run at
/// a JIT safepoint (gcmap-rooted) — i.e. the elidable bigint payload helpers.
pub fn register_gc_alloc_collecting_hook(hook: GcAllocHookFn) {
    GC_ALLOC_COLLECTING_HOOK.set(Some(hook));
}

/// Remove the collecting-allocation callback.
pub fn clear_gc_alloc_collecting_hook() {
    GC_ALLOC_COLLECTING_HOOK.set(None);
}

/// Attempt a collecting GC nursery allocation via the installed hook. Returns
/// `None` when no collecting hook is installed (callers fall back to the
/// no-collect [`try_gc_alloc`]).
#[inline]
pub fn try_gc_alloc_collecting(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    GC_ALLOC_COLLECTING_HOOK
        .get()
        .map(|f| f(type_id, payload_size))
}

/// Root-aware collecting allocation callback. `root` is a caller-owned slot
/// containing the one GC child manufactured before its parent allocation.
pub type GcAllocCollectingRootedHookFn = unsafe fn(
    type_id: u32,
    payload_size: usize,
    root: *mut *mut u8,
    needs_write_barrier: *mut bool,
) -> *mut u8;

majit_gc::global_hook!(
    static GC_ALLOC_COLLECTING_ROOTED_HOOK: GcAllocCollectingRootedHookFn
);

majit_gc::global_hook!(
    static GC_ALLOC_FAST_COLLECTING_ROOTED_HOOK: GcAllocCollectingRootedHookFn
);

pub fn register_gc_alloc_collecting_rooted_hook(hook: GcAllocCollectingRootedHookFn) {
    GC_ALLOC_COLLECTING_ROOTED_HOOK.set(Some(hook));
}

pub fn clear_gc_alloc_collecting_rooted_hook() {
    GC_ALLOC_COLLECTING_ROOTED_HOOK.set(None);
}

/// Install the `malloc_fast` rooted collecting allocation callback.
pub fn register_gc_alloc_fast_collecting_rooted_hook(hook: GcAllocCollectingRootedHookFn) {
    GC_ALLOC_FAST_COLLECTING_ROOTED_HOOK.set(Some(hook));
}

/// Remove the `malloc_fast` rooted collecting allocation callback.
pub fn clear_gc_alloc_fast_collecting_rooted_hook() {
    GC_ALLOC_FAST_COLLECTING_ROOTED_HOOK.set(None);
}

/// Attempt a collecting allocation while preserving `root`.
///
/// Backends with the rooted fast path register the slot only if the nursery
/// bump fails. The fallback retains the previous conservative behavior for
/// backends that expose only the ordinary collecting hook.
///
/// # Safety
/// `root` must remain a valid mutable GC-pointer slot until this call returns.
/// `needs_write_barrier` must remain a valid mutable `bool` slot.
#[inline]
pub unsafe fn try_gc_alloc_collecting_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut *mut u8,
    needs_write_barrier: *mut bool,
) -> Option<*mut u8> {
    if let Some(f) = GC_ALLOC_COLLECTING_ROOTED_HOOK.get() {
        return Some(unsafe { f(type_id, payload_size, root, needs_write_barrier) });
    }
    let collecting = GC_ALLOC_COLLECTING_HOOK.get()?;
    // Backends without placement reporting retain the conservative creation
    // barrier. MiniMark's rooted hook clears this for a nursery result.
    unsafe { *needs_write_barrier = true };
    let pinned = unsafe { try_gc_add_root(root) };
    let result = collecting(type_id, payload_size);
    if pinned {
        try_gc_remove_root(root);
    }
    Some(result)
}

/// [`try_gc_alloc_collecting_rooted`] for a statically-known type that carries
/// neither a finalizer nor the weakref flag.
///
/// # Safety
/// Same contract as [`try_gc_alloc_collecting_rooted`], plus `type_id` must
/// name a type with no destructor and no weakref flag.
#[inline]
pub unsafe fn try_gc_alloc_fast_collecting_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut *mut u8,
    needs_write_barrier: *mut bool,
) -> Option<*mut u8> {
    if let Some(f) = GC_ALLOC_FAST_COLLECTING_ROOTED_HOOK.get() {
        return Some(unsafe { f(type_id, payload_size, root, needs_write_barrier) });
    }
    unsafe { try_gc_alloc_collecting_rooted(type_id, payload_size, root, needs_write_barrier) }
}

/// Signature of the host-side full-collection callback. Used by
/// `pypy/module/gc/interp_gc.py:7-26 collect` ports — i.e. user-level
/// `gc.collect()` reaches the live GC through this hook.
pub type GcCollectHookFn = fn();

majit_gc::global_hook!(static GC_COLLECT_HOOK: GcCollectHookFn);

/// Install the full-collection callback. Overwrites any previously-installed
/// hook.
pub fn register_gc_collect_hook(hook: GcCollectHookFn) {
    GC_COLLECT_HOOK.set(Some(hook));
}

/// Remove the full-collection callback.
pub fn clear_gc_collect_hook() {
    GC_COLLECT_HOOK.set(None);
}

/// Trigger a full mark-sweep collection via the installed hook. No-op
/// when no hook is installed.
#[majit_macros::dont_look_inside]
pub fn try_gc_collect() {
    if let Some(f) = GC_COLLECT_HOOK.get() {
        f();
    }
}

/// Signature of the host-side non-moving old-gen-only major callback.
/// Reclaims stable-allocated interp int/float without moving the nursery, so
/// the interpreter safepoint can drive it under an active JIT (non-empty
/// nursery) — unlike [`try_gc_collect`], whose embedded minor would relocate a
/// Rust-stack nursery `PyObjectRef` that has no shadowstack root.
pub type GcCollectOldgenHookFn = fn();

majit_gc::global_hook!(static GC_COLLECT_OLDGEN_HOOK: GcCollectOldgenHookFn);

/// Install the non-moving-major callback.
pub fn register_gc_collect_oldgen_hook(hook: GcCollectOldgenHookFn) {
    GC_COLLECT_OLDGEN_HOOK.set(Some(hook));
}

/// Remove the non-moving-major callback.
pub fn clear_gc_collect_oldgen_hook() {
    GC_COLLECT_OLDGEN_HOOK.set(None);
}

/// Trigger a non-moving old-gen-only major collection via the installed hook.
/// No-op when no hook is installed.
///
/// Reads the runtime-mutable `GC_COLLECT_OLDGEN_HOOK` fn-pointer cell, not a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the [`try_gc_owns_object`] twin). A `()`
/// return has no discriminant to erase and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn try_gc_collect_oldgen() {
    if let Some(f) = GC_COLLECT_OLDGEN_HOOK.get() {
        f();
    }
}

/// Signature of the host-side automatic major-progress toggle callback.
pub type GcSetEnabledHookFn = fn(bool);

majit_gc::global_hook!(static GC_SET_ENABLED_HOOK: GcSetEnabledHookFn);

/// Install the automatic major-progress toggle callback.
pub fn register_gc_set_enabled_hook(hook: GcSetEnabledHookFn) {
    GC_SET_ENABLED_HOOK.set(Some(hook));
}

/// Remove the automatic major-progress toggle callback.
pub fn clear_gc_set_enabled_hook() {
    GC_SET_ENABLED_HOOK.set(None);
}

/// Toggle automatic major-collection progress via the installed hook. No-op
/// when no hook is installed.
#[inline]
pub fn try_gc_set_enabled(enabled: bool) {
    if let Some(f) = GC_SET_ENABLED_HOOK.get() {
        f(enabled);
    }
}

/// RPython `gc_fq_register` / `gc_fq_next_dead` hooks.  The trigger is the
/// translated FinalizerQueue handler and only schedules interpreter work.
pub type GcFinalizerTriggerFn = fn();
pub type GcRegisterFinalizerHookFn = fn(usize, PyObjectRef, GcFinalizerTriggerFn);
pub type GcFinalizerNextDeadHookFn = fn(usize) -> PyObjectRef;

majit_gc::global_hook!(static GC_REGISTER_FINALIZER_HOOK: GcRegisterFinalizerHookFn);
majit_gc::global_hook!(static GC_FINALIZER_NEXT_DEAD_HOOK: GcFinalizerNextDeadHookFn);

pub fn register_gc_finalizer_hooks(
    register: GcRegisterFinalizerHookFn,
    next_dead: GcFinalizerNextDeadHookFn,
) {
    GC_REGISTER_FINALIZER_HOOK.set(Some(register));
    GC_FINALIZER_NEXT_DEAD_HOOK.set(Some(next_dead));
}

pub fn try_gc_register_finalizer(fq_index: usize, obj: PyObjectRef, trigger: GcFinalizerTriggerFn) {
    if let Some(register) = GC_REGISTER_FINALIZER_HOOK.get() {
        register(fq_index, obj, trigger);
    }
}

pub fn try_gc_finalizer_next_dead(fq_index: usize) -> PyObjectRef {
    GC_FINALIZER_NEXT_DEAD_HOOK
        .get()
        .map(|next_dead| next_dead(fq_index))
        .unwrap_or(crate::PY_NULL)
}

/// `ObjSpace.allocate_instance`-level hook.  pyre-object owns allocation but
/// pyre-interpreter owns MRO lookup and can therefore decide `hasuserdel`.
pub type MaybeRegisterFinalizerHookFn = fn(PyObjectRef);

majit_gc::global_hook!(static MAYBE_REGISTER_FINALIZER_HOOK: MaybeRegisterFinalizerHookFn);

pub fn register_maybe_finalizer_hook(hook: MaybeRegisterFinalizerHookFn) {
    MAYBE_REGISTER_FINALIZER_HOOK.set(Some(hook));
}

// `dont_look_inside`: the host finalizer-registration hook is a
// process-global fn-pointer cell (`MAYBE_REGISTER_FINALIZER_HOOK`) whose
// dispatch stays opaque to the JIT; calls residualize via the registered
// fnaddr. The `try_gc_owns_object` twin.
#[majit_macros::dont_look_inside]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn maybe_register_finalizer(obj: PyObjectRef) {
    if let Some(hook) = MAYBE_REGISTER_FINALIZER_HOOK.get() {
        hook(obj);
    }
}

/// Signature of the host-side `threshold_reached` callback
/// (incminimark.py:1288-1290).
pub type GcMajorThresholdReachedHookFn = fn() -> bool;

majit_gc::global_hook!(static GC_MAJOR_THRESHOLD_REACHED_HOOK: GcMajorThresholdReachedHookFn);

/// Install the next-major-threshold callback.
pub fn register_gc_major_threshold_reached_hook(hook: GcMajorThresholdReachedHookFn) {
    GC_MAJOR_THRESHOLD_REACHED_HOOK.set(Some(hook));
}

/// Remove the next-major-threshold callback.
pub fn clear_gc_major_threshold_reached_hook() {
    GC_MAJOR_THRESHOLD_REACHED_HOOK.set(None);
}

/// Whether the collector has reached the threshold it set for its next major
/// collection, via the installed hook. `false` when none is installed, so an
/// interpreter with no collector behind it never asks for one.
///
/// Reads the runtime-mutable `GC_MAJOR_THRESHOLD_REACHED_HOOK` fn-pointer cell,
/// not a build-time constant, so the JIT residualizes the call instead of
/// tracing into it (`@dont_look_inside`, the [`try_gc_collect_oldgen`] twin).
/// The `-> bool` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn try_gc_major_threshold_reached() -> bool {
    match GC_MAJOR_THRESHOLD_REACHED_HOOK.get() {
        Some(f) => f(),
        None => false,
    }
}

/// Signature of the host-side root-register callbacks.
/// `slot` is a pointer to a slot holding a `PyObjectRef`
/// (equivalently `*mut u8`); the GC treats it as a live root until
/// [`try_gc_remove_root`] is called with the same pointer.
///
/// Used around host-side allocator calls that may trigger a minor
/// collection — the nursery-moving collector needs the caller's slot
/// registered so the live pointer is traced and updated.
///
/// RPython accomplishes this automatically via its GC transform
/// pass (shadowstack save/restore around safepoints). pyre has no
/// such pass, so root registration is explicit at the call site.
/// TODO: this is a known deviation from RPython.
pub type GcAddRootHookFn = unsafe fn(slot: *mut *mut u8);
pub type GcRemoveRootHookFn = fn(slot: *mut *mut u8);

majit_gc::global_hook!(static GC_ADD_ROOT_HOOK: GcAddRootHookFn);
majit_gc::global_hook!(static GC_REMOVE_ROOT_HOOK: GcRemoveRootHookFn);

/// Install the root-register / remove callbacks.
pub fn register_gc_root_hooks(add: GcAddRootHookFn, remove: GcRemoveRootHookFn) {
    GC_REMOVE_ROOT_HOOK.set(Some(remove));
    GC_ADD_ROOT_HOOK.set(Some(add));
}

/// Remove the root-register callbacks.
pub fn clear_gc_root_hooks() {
    GC_ADD_ROOT_HOOK.set(None);
    GC_REMOVE_ROOT_HOOK.set(None);
}

/// Register `slot` as a live GC root via the installed callback.
/// Returns `true` when the callback was invoked.
///
/// # Safety
/// Caller must keep `slot` valid until [`try_gc_remove_root`] is
/// called with the same pointer.
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — the `try_gc_write_barrier`
// twin; calls residualize via the registered fnaddr (`rlib/jit.py:139`).
#[majit_macros::dont_look_inside]
pub unsafe fn try_gc_add_root(slot: *mut *mut u8) -> bool {
    match GC_ADD_ROOT_HOOK.get() {
        Some(f) => {
            unsafe { f(slot) };
            true
        }
        None => false,
    }
}

/// Remove a previously-registered root via the installed callback.
/// Returns `true` when the callback was invoked.
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — the `try_gc_add_root` twin;
// calls residualize via the registered fnaddr (`rlib/jit.py:139`).
#[majit_macros::dont_look_inside]
pub fn try_gc_remove_root(slot: *mut *mut u8) -> bool {
    match GC_REMOVE_ROOT_HOOK.get() {
        Some(f) => {
            f(slot);
            true
        }
        None => false,
    }
}

/// Signature of the host-side "is GC-managed" predicate. Callers
/// (host-side allocators with mixed `try_gc_alloc_stable` /
/// `std::alloc` allocation paths during the L1/L2 stepping-stone
/// window) use this to discriminate GC-managed blocks from
/// `std::alloc`-backed ones at dealloc time.
pub type GcOwnsObjectHookFn = fn(addr: usize) -> bool;

majit_gc::global_hook!(static GC_OWNS_OBJECT_HOOK: GcOwnsObjectHookFn);

/// Install the GC-ownership predicate. Overwrites any previously-
/// installed hook.
pub fn register_gc_owns_object_hook(hook: GcOwnsObjectHookFn) {
    GC_OWNS_OBJECT_HOOK.set(Some(hook));
}

/// Remove the GC-ownership predicate.
pub fn clear_gc_owns_object_hook() {
    GC_OWNS_OBJECT_HOOK.set(None);
}

/// Whether `addr` lies inside the active backend's managed GC heap.
/// Returns `false` when no hook is installed — callers treat that as
/// "no GC owns this pointer" and fall through to their non-GC
/// dealloc path. This is the host-side mirror of
/// `majit_gc::gc_owns_object`.
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — traces never look inside a
// GC ownership check (the backend GC rewrite owns that concern); calls
// residualize via the registered fnaddr. The `try_gc_write_barrier` twin.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_owns_object(addr: *mut u8) -> bool {
    match GC_OWNS_OBJECT_HOOK.get() {
        Some(f) => f(addr as usize),
        None => false,
    }
}

/// Return the current address for `addr` without registering it as a root.
/// When the active GC does not know the object, the address is unchanged.
///
/// Unlike the allocation and barrier hooks around it this one calls
/// `majit_gc` outright.  The module rule those hooks encode is that
/// `pyre-object` must not bind itself to a GC *implementation*;
/// `gc_current_object_address` is not one — it is a backend-agnostic query
/// that dispatches internally through `gc_is_nursery_object`'s own published
/// bounds and hook, and answers `addr` for every address no GC owns,
/// including before any GC exists.  Routing it through a second fn-pointer
/// cell therefore bought nothing but an opaque call: every root pin and every
/// root reload pays it, and behind it sits a two-word range compare that
/// wants to be inlined into the caller.
#[inline]
pub fn try_gc_current_object_address(addr: *mut u8) -> *mut u8 {
    majit_gc::gc_current_object_address(addr as usize) as *mut u8
}

/// minimark.py:1900-1915 `identityhash` hook.
/// Returns a GC-move-stable address for the given object.
pub type GcIdentityHashHookFn = fn(obj_addr: usize) -> usize;

majit_gc::global_hook!(static GC_IDENTITY_HASH_HOOK: GcIdentityHashHookFn);

pub fn register_gc_identity_hash_hook(hook: GcIdentityHashHookFn) {
    GC_IDENTITY_HASH_HOOK.set(Some(hook));
}

pub fn clear_gc_identity_hash_hook() {
    GC_IDENTITY_HASH_HOOK.set(None);
}

/// Return a stable identity hash for `obj_addr`.  When the hook is
/// installed, nursery objects get a shadow-based stable address;
/// old-gen objects return their own address.  When no hook is
/// installed, returns `obj_addr` unchanged (pre-GC fallback).
#[inline]
pub fn gc_identity_hash(obj_addr: usize) -> usize {
    match GC_IDENTITY_HASH_HOOK.get() {
        Some(f) => f(obj_addr),
        None => obj_addr,
    }
}

/// Signature of the host-side write barrier callback. `obj` is the
/// GC-managed object whose field is being updated with a possible young
/// pointer. The backend decides whether `obj` is old enough to require
/// remembering.
pub type GcWriteBarrierHookFn = fn(obj: *mut u8);

majit_gc::global_hook!(static GC_WRITE_BARRIER_HOOK: GcWriteBarrierHookFn);
majit_gc::global_hook!(static GC_WRITE_BARRIER_MANAGED_HOOK: GcWriteBarrierHookFn);

/// Install the write-barrier callback.
pub fn register_gc_write_barrier_hook(hook: GcWriteBarrierHookFn) {
    GC_WRITE_BARRIER_HOOK.set(Some(hook));
}

/// Install the callback for objects returned by a managed allocation hook.
pub fn register_gc_write_barrier_managed_hook(hook: GcWriteBarrierHookFn) {
    GC_WRITE_BARRIER_MANAGED_HOOK.set(Some(hook));
}

/// Remove the write-barrier callback.
pub fn clear_gc_write_barrier_hook() {
    GC_WRITE_BARRIER_HOOK.set(None);
}

pub fn clear_gc_write_barrier_managed_hook() {
    GC_WRITE_BARRIER_MANAGED_HOOK.set(None);
}

/// Run the active GC write barrier for `obj` when one is installed.
// `dont_look_inside`: host hook dispatch (a process-global
// atomic fn-pointer cell) stays opaque to the JIT — traces never look inside a
// write barrier (the backend GC rewrite owns that concern); calls
// residualize via the registered fnaddr.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_write_barrier(obj: *mut u8) -> bool {
    match GC_WRITE_BARRIER_HOOK.get() {
        Some(f) => {
            f(obj);
            true
        }
        None => false,
    }
}

/// Run the active barrier for a pointer returned by a managed allocation
/// hook.  The backend may omit the general hybrid-heap ownership lookup.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_write_barrier_managed(obj: *mut u8) -> bool {
    match GC_WRITE_BARRIER_MANAGED_HOOK.get() {
        Some(f) => {
            f(obj);
            true
        }
        None => try_gc_write_barrier(obj),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The hook cells are process-global, and every allocator in this crate
    // reads them with no gate — `malloc_typed_managed`, `alloc_dict_object`,
    // `w_weakref_new`, `alloc_items_block_gc`,
    // `try_alloc_typed_items_block_nursery`, `alloc_rbigint_nursery_impl`, …
    // So from `register_*` until the matching `clear_*` an installed mock
    // answers *every* thread in this binary, and libtest runs the other tests
    // on their own threads concurrently. That shapes the mocks twice over:
    //
    //   * a mock must return memory the caller can build its object in.
    //     A synthetic pointer value took the process down:
    //     `try_alloc_typed_items_block_nursery` writes the capacity header
    //     through whatever the hook returns, so a concurrent
    //     `try_gc_alloc(tid, 24)` stored to address 0x18.
    //   * a mock must record its arguments per thread. A concurrent allocation
    //     calls the same mock with its own type id and size, and a shared cell
    //     would carry that call's values into the installing test's assertions.
    //
    // The lock below only serializes the installers against each other; that is
    // what keeps the "unregistered/cleared -> None/false" assertions sound.
    static HOOK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn hook_test_guard() -> std::sync::MutexGuard<'static, ()> {
        HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the mock allocator was last asked for on this thread, and what it
    /// handed back.
    #[derive(Clone, Copy)]
    struct MockCall {
        type_id: u32,
        payload_size: usize,
        ptr: *mut u8,
    }

    thread_local! {
        static LAST_MOCK_CALL: std::cell::Cell<Option<MockCall>> =
            const { std::cell::Cell::new(None) };
    }

    fn last_mock_call() -> MockCall {
        LAST_MOCK_CALL
            .with(|cell| cell.get())
            .expect("the mock allocator was not invoked on this thread")
    }

    /// Alignment of the mock's blocks.
    ///
    /// `ItemsBlock` / `TypedItemsBlock` are the only payloads this crate hands
    /// back to `std::alloc::dealloc` (`dealloc_items_block` /
    /// `dealloc_typed_items_block`, reached here because no owns-object hook is
    /// installed), and both lead with a `usize` capacity header — so matching
    /// that alignment makes those frees exact instead of layout-mismatched.
    /// Every other payload routed through these hooks is a `repr(C)` struct of
    /// pointer / `usize` / `u64` / `f64` fields, which this covers too.
    const MOCK_ALIGN: usize = std::mem::align_of::<usize>();

    const _: () = assert!(MOCK_ALIGN == std::mem::align_of::<crate::object_array::ItemsBlock>());
    const _: () =
        assert!(MOCK_ALIGN == std::mem::align_of::<crate::object_array::TypedItemsBlock>());

    /// Real, zeroed memory of exactly `payload_size` bytes — what a GC
    /// allocator returns — or null when the request cannot be served, which is
    /// the failure [`GcAllocHookFn`] specifies. Reporting that faithfully is
    /// load-bearing: `RBigInt::lshift(i64::MAX)` asks for a block no allocator
    /// can hand out and expects `RBigIntError::Memory` back.
    ///
    /// A block whose caller does not free it stays leaked for the rest of the
    /// run, which is what the collector this stands in for would do with it.
    fn mock_alloc(payload_size: usize) -> *mut u8 {
        let Ok(layout) = std::alloc::Layout::from_size_align(payload_size.max(1), MOCK_ALIGN)
        else {
            return std::ptr::null_mut();
        };
        unsafe { std::alloc::alloc_zeroed(layout) }
    }

    fn mock_hook(type_id: u32, payload_size: usize) -> *mut u8 {
        let ptr = mock_alloc(payload_size);
        LAST_MOCK_CALL.with(|cell| {
            cell.set(Some(MockCall {
                type_id,
                payload_size,
                ptr,
            }))
        });
        ptr
    }

    fn null_hook(_type_id: u32, _payload_size: usize) -> *mut u8 {
        std::ptr::null_mut()
    }

    unsafe fn nursery_rooted_hook(
        type_id: u32,
        payload_size: usize,
        _root: *mut *mut u8,
        needs_write_barrier: *mut bool,
    ) -> *mut u8 {
        unsafe { *needs_write_barrier = false };
        mock_hook(type_id, payload_size)
    }

    unsafe fn nursery_placement_hook(
        type_id: u32,
        payload_size: usize,
        needs_write_barrier: *mut bool,
    ) -> *mut u8 {
        unsafe { *needs_write_barrier = false };
        mock_hook(type_id, payload_size)
    }

    #[test]
    fn returns_none_when_unregistered() {
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_hook();
        assert!(try_gc_alloc(1, 16).is_none());
    }

    #[test]
    fn invokes_registered_hook_with_args() {
        let _hook_lock = hook_test_guard();
        register_gc_alloc_hook(mock_hook);
        let ptr = try_gc_alloc(7, 24);
        let call = last_mock_call();
        assert_eq!(ptr, Some(call.ptr));
        assert_eq!(call.type_id, 7);
        assert_eq!(call.payload_size, 24);
        clear_gc_alloc_hook();
    }

    #[test]
    fn clear_removes_hook() {
        let _hook_lock = hook_test_guard();
        register_gc_alloc_hook(mock_hook);
        assert!(try_gc_alloc(1, 8).is_some());
        clear_gc_alloc_hook();
        assert!(try_gc_alloc(1, 8).is_none());
    }

    #[test]
    fn hook_returning_null_propagates_some_null() {
        let _hook_lock = hook_test_guard();
        register_gc_alloc_hook(null_hook);
        let ptr = try_gc_alloc(1, 8);
        assert!(ptr.is_some());
        assert!(ptr.unwrap().is_null());
        clear_gc_alloc_hook();
    }

    #[test]
    fn managed_bigint_digits_do_not_fall_back_to_raw_after_hook_failure() {
        let _hook_lock = hook_test_guard();
        register_gc_alloc_hook(null_hook);
        let block = unsafe {
            crate::object_array::try_alloc_typed_items_block_nursery(
                4,
                crate::object_array::GC_INT_ARRAY_GC_TYPE_ID,
            )
        };
        assert!(block.is_none());
        clear_gc_alloc_hook();
    }

    // Per-thread for the same reason the allocation record is: while these
    // hooks are installed, `try_gc_alloc_collecting_rooted`'s fallback pins and
    // unpins the caller's slot, and that runs on whichever thread allocates.
    thread_local! {
        static LAST_ROOT_PTR: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static REMOVE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    unsafe fn mock_add_root(slot: *mut *mut u8) {
        LAST_ROOT_PTR.with(|cell| cell.set(slot as usize));
    }
    fn mock_remove_root(slot: *mut *mut u8) {
        let _ = slot;
        REMOVE_CALLS.with(|cell| cell.set(cell.get() + 1));
    }

    fn mock_write_barrier(_obj: *mut u8) {}

    #[test]
    fn root_hooks_register_and_remove_round_trip() {
        let _hook_lock = hook_test_guard();
        clear_gc_root_hooks();
        let mut slot: *mut u8 = std::ptr::null_mut();
        assert!(!unsafe { try_gc_add_root(&mut slot as *mut *mut u8) });
        assert!(!try_gc_remove_root(&mut slot as *mut *mut u8));

        LAST_ROOT_PTR.with(|cell| cell.set(0));
        REMOVE_CALLS.with(|cell| cell.set(0));
        register_gc_root_hooks(mock_add_root, mock_remove_root);

        let slot_ptr = &mut slot as *mut *mut u8;
        assert!(unsafe { try_gc_add_root(slot_ptr) });
        assert_eq!(LAST_ROOT_PTR.with(|cell| cell.get()), slot_ptr as usize);
        assert!(try_gc_remove_root(slot_ptr));
        assert_eq!(REMOVE_CALLS.with(|cell| cell.get()), 1);

        clear_gc_root_hooks();
        assert!(!unsafe { try_gc_add_root(slot_ptr) });
        assert!(!try_gc_remove_root(slot_ptr));
    }

    #[test]
    fn stable_hook_is_independent_from_nursery_hook() {
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_hook();
        clear_gc_alloc_stable_hook();
        assert!(try_gc_alloc(1, 8).is_none());
        assert!(try_gc_alloc_stable(1, 8).is_none());

        register_gc_alloc_hook(mock_hook);
        // Stable hook still not installed.
        assert!(try_gc_alloc(1, 8).is_some());
        assert!(try_gc_alloc_stable(1, 8).is_none());

        register_gc_alloc_stable_hook(mock_hook);
        let ptr = try_gc_alloc_stable(3, 32);
        let call = last_mock_call();
        assert_eq!(ptr, Some(call.ptr));
        assert_eq!(call.type_id, 3);
        assert_eq!(call.payload_size, 32);

        clear_gc_alloc_hook();
        clear_gc_alloc_stable_hook();
    }

    #[test]
    fn no_collect_placement_hook_has_conservative_fallback() {
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_hook();
        clear_gc_alloc_with_placement_hook();
        let mut needs_write_barrier = false;

        register_gc_alloc_hook(mock_hook);
        let result = try_gc_alloc_with_placement(7, 24, &mut needs_write_barrier);
        assert_eq!(result, Some(last_mock_call().ptr));
        assert!(needs_write_barrier);

        register_gc_alloc_with_placement_hook(nursery_placement_hook);
        let result = try_gc_alloc_with_placement(7, 24, &mut needs_write_barrier);
        assert_eq!(result, Some(last_mock_call().ptr));
        assert!(!needs_write_barrier);

        clear_gc_alloc_hook();
        clear_gc_alloc_with_placement_hook();
    }

    #[test]
    fn rooted_collecting_hook_reports_placement_conservatively() {
        let _hook_lock = hook_test_guard();
        clear_gc_alloc_collecting_hook();
        clear_gc_alloc_collecting_rooted_hook();
        clear_gc_root_hooks();
        let mut root = std::ptr::null_mut();
        let mut needs_write_barrier = false;

        register_gc_alloc_collecting_hook(mock_hook);
        let result =
            unsafe { try_gc_alloc_collecting_rooted(7, 24, &mut root, &mut needs_write_barrier) };
        assert_eq!(result, Some(last_mock_call().ptr));
        assert!(needs_write_barrier);

        register_gc_alloc_collecting_rooted_hook(nursery_rooted_hook);
        needs_write_barrier = true;
        let result =
            unsafe { try_gc_alloc_collecting_rooted(7, 24, &mut root, &mut needs_write_barrier) };
        assert_eq!(result, Some(last_mock_call().ptr));
        assert!(!needs_write_barrier);

        clear_gc_alloc_collecting_hook();
        clear_gc_alloc_collecting_rooted_hook();
    }

    #[test]
    fn write_barrier_hook_registers_invokes_and_clears() {
        let _hook_lock = hook_test_guard();
        clear_gc_write_barrier_hook();
        let obj = 0x1000usize as *mut u8;
        assert!(!try_gc_write_barrier(obj));

        register_gc_write_barrier_hook(mock_write_barrier);
        // `try_gc_write_barrier` returns true iff it dispatched to the installed
        // hook, so the return value alone proves invocation. We deliberately do
        // NOT assert on a shared call counter: the write-barrier hook cell is
        // process-global and its many ungated production callers (list / tuple /
        // dict / descriptor / …) fire the same hook from concurrent tests, which
        // would perturb any global counter and reintroduce a parallel-test flake.
        assert!(try_gc_write_barrier(obj));

        clear_gc_write_barrier_hook();
        assert!(!try_gc_write_barrier(obj));
    }
}
