//! Thread-local GC allocation hook for host-side Python object allocators.
//!
//! `pyre-object` sits below `majit-gc` in the dependency graph and must
//! not depend on it. Host-side allocators (`w_int_new`, `w_float_new`,
//! …) that want to route through the real GC instead of `Box::into_raw`
//! go through the callback registered here. `pyre-jit::eval` installs
//! the concrete trampoline on `JitDriver` init so the callback reaches
//! the backend-owned GC allocator via `majit_gc` TLS hooks.
//!
//! Callers use [`try_gc_alloc`] which returns `None` when no hook is
//! installed — they fall back to the `Box::into_raw` path in
//! that case. Incremental migration drops the `Box::into_raw`
//! fallback at each call site as the hook's reliability is verified
//! under the full bench suite.
//!
//! Layering: this module has no external dependencies. It defines the
//! function-pointer slot only. Wire-up lives in `pyre-jit`.

use std::cell::Cell;

/// Signature of the host-side GC allocation callback.
///
/// `type_id` is the backend-registered GC type id (same id used by
/// JIT-compiled `NewWithVtable`). `payload_size` is the number of
/// payload bytes requested. The callback returns an uninitialised
/// pointer to managed memory of exactly that size, ready for raw
/// field writes. On allocation failure the callback returns
/// `std::ptr::null_mut()`.
pub type GcAllocHookFn = fn(type_id: u32, payload_size: usize) -> *mut u8;

thread_local! {
    static GC_ALLOC_HOOK: Cell<Option<GcAllocHookFn>> = const { Cell::new(None) };
    static GC_ALLOC_STABLE_HOOK: Cell<Option<GcAllocHookFn>> = const { Cell::new(None) };
}

/// Install the allocation callback for this thread. Overwrites any
/// previously-installed hook.
pub fn register_gc_alloc_hook(hook: GcAllocHookFn) {
    GC_ALLOC_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the callback on this thread. Subsequent [`try_gc_alloc`]
/// returns `None` until a new hook is registered.
pub fn clear_gc_alloc_hook() {
    GC_ALLOC_HOOK.with(|cell| cell.set(None));
}

/// Attempt a GC allocation via the installed hook. Returns `None`
/// when no hook is installed on this thread, or `Some(null)` when the
/// hook itself returned null.
#[inline]
pub fn try_gc_alloc(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    GC_ALLOC_HOOK.with(|cell| cell.get().map(|f| f(type_id, payload_size)))
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
    GC_ALLOC_STABLE_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the stable-allocation callback on this thread.
pub fn clear_gc_alloc_stable_hook() {
    GC_ALLOC_STABLE_HOOK.with(|cell| cell.set(None));
}

/// Attempt a stable-address GC allocation via the installed hook.
/// See [`register_gc_alloc_stable_hook`] for semantics. Returns
/// `None` when no hook is installed on this thread.
#[inline]
pub fn try_gc_alloc_stable(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    GC_ALLOC_STABLE_HOOK.with(|cell| cell.get().map(|f| f(type_id, payload_size)))
}

/// Null-collapsing view of [`try_gc_alloc_stable`] for JIT-traced callers.
///
/// Returns `null` both when no hook is installed and when the hook itself
/// returned `null`; every traced caller already treats those two cases
/// identically (fall back to `malloc_typed`), so the `Option` discriminant
/// carries no information the caller reads.  Residualising this primitive
/// (`@dont_look_inside`, `rlib/jit.py:139`) keeps the thread-local hook
/// dispatch out of the trace — a `*mut u8` return has no discriminant to
/// erase, unlike the `Option<*mut u8>` accessor.
#[majit_macros::dont_look_inside]
pub fn try_gc_alloc_stable_raw(type_id: u32, payload_size: usize) -> *mut u8 {
    try_gc_alloc_stable(type_id, payload_size).unwrap_or(core::ptr::null_mut())
}

thread_local! {
    static GC_ALLOC_COLLECTING_HOOK: Cell<Option<GcAllocHookFn>> = const { Cell::new(None) };
}

/// Install the *collecting* nursery allocation callback for this thread.
///
/// Unlike [`register_gc_alloc_hook`] (no-collect), the backend routes this to a
/// nursery allocator that runs a minor collection when the nursery is full. Only
/// for callers that hold no unrooted GC pointer across the allocation and run at
/// a JIT safepoint (gcmap-rooted) — i.e. the elidable bigint payload helpers.
pub fn register_gc_alloc_collecting_hook(hook: GcAllocHookFn) {
    GC_ALLOC_COLLECTING_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the collecting-allocation callback on this thread.
pub fn clear_gc_alloc_collecting_hook() {
    GC_ALLOC_COLLECTING_HOOK.with(|cell| cell.set(None));
}

/// Attempt a collecting GC nursery allocation via the installed hook. Returns
/// `None` when no collecting hook is installed (callers fall back to the
/// no-collect [`try_gc_alloc`]).
#[inline]
pub fn try_gc_alloc_collecting(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    GC_ALLOC_COLLECTING_HOOK.with(|cell| cell.get().map(|f| f(type_id, payload_size)))
}

/// Signature of the host-side memory-pressure callback: charge `bytes` of
/// off-heap, GC-invisible payload (a bignum's external limb `Vec`).
pub type GcChargeMemoryPressureFn = fn(bytes: usize);

thread_local! {
    static GC_CHARGE_MEMORY_PRESSURE_HOOK: Cell<Option<GcChargeMemoryPressureFn>> =
        const { Cell::new(None) };
}

/// Install the memory-pressure callback for this thread.
pub fn register_gc_charge_memory_pressure_hook(hook: GcChargeMemoryPressureFn) {
    GC_CHARGE_MEMORY_PRESSURE_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the memory-pressure callback on this thread.
pub fn clear_gc_charge_memory_pressure_hook() {
    GC_CHARGE_MEMORY_PRESSURE_HOOK.with(|cell| cell.set(None));
}

/// Charge `bytes` of off-heap memory pressure via the installed hook (no-op when
/// none is installed, e.g. bare unit tests or backends without a generational GC).
/// Only the bignum collecting-alloc site calls this, from a gcmap-rooted residual
/// call where a forced minor is safe.
#[inline]
pub fn try_gc_charge_memory_pressure(bytes: usize) {
    GC_CHARGE_MEMORY_PRESSURE_HOOK.with(|cell| {
        if let Some(f) = cell.get() {
            f(bytes);
        }
    })
}

/// Signature of the host-side old-gen external-byte callback: add `bytes` of
/// `obj_addr`'s off-heap payload to the major-collection threshold's external
/// total when the object is old-gen.
pub type GcChargeOldgenExternalFn = fn(obj_addr: usize, bytes: usize);

thread_local! {
    static GC_CHARGE_OLDGEN_EXTERNAL_HOOK: Cell<Option<GcChargeOldgenExternalFn>> =
        const { Cell::new(None) };
}

/// Install the old-gen external-byte callback for this thread.
pub fn register_gc_charge_oldgen_external_hook(hook: GcChargeOldgenExternalFn) {
    GC_CHARGE_OLDGEN_EXTERNAL_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the old-gen external-byte callback on this thread.
pub fn clear_gc_charge_oldgen_external_hook() {
    GC_CHARGE_OLDGEN_EXTERNAL_HOOK.with(|cell| cell.set(None));
}

/// Charge `bytes` of `obj_addr`'s off-heap payload against the major threshold
/// via the installed hook when the object is old-gen (no-op when none is
/// installed). Unlike [`try_gc_charge_memory_pressure`] this never forces a
/// minor, so it is safe after allocating an unrooted payload: a directly-old-gen
/// bignum's limb `Vec` would otherwise stay invisible to the threshold until
/// the next major's `recompute_oldgen_external_bytes`.
#[inline]
pub fn try_gc_charge_oldgen_external(obj_addr: usize, bytes: usize) {
    GC_CHARGE_OLDGEN_EXTERNAL_HOOK.with(|cell| {
        if let Some(f) = cell.get() {
            f(obj_addr, bytes);
        }
    })
}

/// Signature of the host-side full-collection callback. Used by
/// `pypy/module/gc/interp_gc.py:7-26 collect` ports — i.e. user-level
/// `gc.collect()` reaches the live GC through this hook.
pub type GcCollectHookFn = fn();

thread_local! {
    static GC_COLLECT_HOOK: Cell<Option<GcCollectHookFn>> = const { Cell::new(None) };
}

/// Install the full-collection callback for this thread. Overwrites
/// any previously-installed hook.
pub fn register_gc_collect_hook(hook: GcCollectHookFn) {
    GC_COLLECT_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the full-collection callback on this thread.
pub fn clear_gc_collect_hook() {
    GC_COLLECT_HOOK.with(|cell| cell.set(None));
}

/// Trigger a full mark-sweep collection via the installed hook. No-op
/// when no hook is installed on this thread.
#[inline]
pub fn try_gc_collect() {
    GC_COLLECT_HOOK.with(|cell| {
        if let Some(f) = cell.get() {
            f();
        }
    });
}

/// Signature of the host-side non-moving old-gen-only major callback.
/// Reclaims stable-allocated interp int/float without moving the nursery, so
/// the interpreter safepoint can drive it under an active JIT (non-empty
/// nursery) — unlike [`try_gc_collect`], whose embedded minor would relocate a
/// Rust-stack nursery `PyObjectRef` that has no shadowstack root.
pub type GcCollectOldgenHookFn = fn();

thread_local! {
    static GC_COLLECT_OLDGEN_HOOK: Cell<Option<GcCollectOldgenHookFn>> = const { Cell::new(None) };
}

/// Install the non-moving-major callback for this thread.
pub fn register_gc_collect_oldgen_hook(hook: GcCollectOldgenHookFn) {
    GC_COLLECT_OLDGEN_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the non-moving-major callback on this thread.
pub fn clear_gc_collect_oldgen_hook() {
    GC_COLLECT_OLDGEN_HOOK.with(|cell| cell.set(None));
}

/// Trigger a non-moving old-gen-only major collection via the installed hook.
/// No-op when no hook is installed on this thread.
#[inline]
pub fn try_gc_collect_oldgen() {
    GC_COLLECT_OLDGEN_HOOK.with(|cell| {
        if let Some(f) = cell.get() {
            f();
        }
    });
}

/// Signature of the host-side heap-stats callback returning
/// `(oldgen_total, nursery_used)`. Used by the interpreter GC safepoint
/// (`crate::gc_interp`) to gate a collection on an empty nursery.
pub type GcHeapStatsHookFn = fn() -> (usize, usize);

thread_local! {
    static GC_HEAP_STATS_HOOK: Cell<Option<GcHeapStatsHookFn>> = const { Cell::new(None) };
}

/// Install the heap-stats callback for this thread.
pub fn register_gc_heap_stats_hook(hook: GcHeapStatsHookFn) {
    GC_HEAP_STATS_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the heap-stats callback on this thread.
pub fn clear_gc_heap_stats_hook() {
    GC_HEAP_STATS_HOOK.with(|cell| cell.set(None));
}

/// Bytes currently used in the active GC's nursery, via the installed
/// hook. `0` when no hook is installed (treated as "nursery empty").
#[inline]
pub fn try_gc_nursery_used() -> usize {
    GC_HEAP_STATS_HOOK.with(|cell| match cell.get() {
        Some(f) => f().1,
        None => 0,
    })
}

/// Bytes currently held in the active GC's old generation, via the installed
/// heap-stats hook. `0` when no hook is installed. The interpreter safepoint
/// reads this to gate a non-moving major on old-gen growth.
#[inline]
pub fn try_gc_oldgen_total() -> usize {
    GC_HEAP_STATS_HOOK.with(|cell| match cell.get() {
        Some(f) => f().0,
        None => 0,
    })
}

/// Signature of the host-side "is the JIT-frame shadow stack empty"
/// callback. Used by the interpreter GC safepoint to avoid collecting
/// while a compiled trace is suspended (its jitframe roots can be
/// mis-mapped from a nested interpreter collection).
pub type GcJitframeEmptyHookFn = fn() -> bool;

thread_local! {
    static GC_JITFRAME_EMPTY_HOOK: Cell<Option<GcJitframeEmptyHookFn>> = const { Cell::new(None) };
}

/// Install the jitframe-shadow-stack-empty callback for this thread.
pub fn register_gc_jitframe_empty_hook(hook: GcJitframeEmptyHookFn) {
    GC_JITFRAME_EMPTY_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the jitframe-shadow-stack-empty callback on this thread.
pub fn clear_gc_jitframe_empty_hook() {
    GC_JITFRAME_EMPTY_HOOK.with(|cell| cell.set(None));
}

/// Whether no compiled trace is suspended (jitframe shadow stack empty),
/// via the installed hook. `true` when no hook is installed (no JIT →
/// no jitframes).
#[inline]
pub fn try_gc_jitframe_empty() -> bool {
    GC_JITFRAME_EMPTY_HOOK.with(|cell| match cell.get() {
        Some(f) => f(),
        None => true,
    })
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

thread_local! {
    static GC_ADD_ROOT_HOOK: Cell<Option<GcAddRootHookFn>> = const { Cell::new(None) };
    static GC_REMOVE_ROOT_HOOK: Cell<Option<GcRemoveRootHookFn>> = const { Cell::new(None) };
}

/// Install the root-register / remove callbacks for this thread.
pub fn register_gc_root_hooks(add: GcAddRootHookFn, remove: GcRemoveRootHookFn) {
    GC_ADD_ROOT_HOOK.with(|cell| cell.set(Some(add)));
    GC_REMOVE_ROOT_HOOK.with(|cell| cell.set(Some(remove)));
}

/// Remove the root-register callbacks on this thread.
pub fn clear_gc_root_hooks() {
    GC_ADD_ROOT_HOOK.with(|cell| cell.set(None));
    GC_REMOVE_ROOT_HOOK.with(|cell| cell.set(None));
}

/// Register `slot` as a live GC root via the installed callback.
/// Returns `true` when the callback was invoked.
///
/// # Safety
/// Caller must keep `slot` valid until [`try_gc_remove_root`] is
/// called with the same pointer.
// `dont_look_inside`: host hook dispatch (`thread_local!` `Cell`
// indirection) stays opaque to the JIT — the `try_gc_write_barrier`
// twin; calls residualize via the registered fnaddr (`rlib/jit.py:139`).
#[majit_macros::dont_look_inside]
pub unsafe fn try_gc_add_root(slot: *mut *mut u8) -> bool {
    GC_ADD_ROOT_HOOK.with(|cell| match cell.get() {
        Some(f) => {
            unsafe { f(slot) };
            true
        }
        None => false,
    })
}

/// Remove a previously-registered root via the installed callback.
/// Returns `true` when the callback was invoked.
// `dont_look_inside`: host hook dispatch (`thread_local!` `Cell`
// indirection) stays opaque to the JIT — the `try_gc_add_root` twin;
// calls residualize via the registered fnaddr (`rlib/jit.py:139`).
#[majit_macros::dont_look_inside]
pub fn try_gc_remove_root(slot: *mut *mut u8) -> bool {
    GC_REMOVE_ROOT_HOOK.with(|cell| match cell.get() {
        Some(f) => {
            f(slot);
            true
        }
        None => false,
    })
}

/// Signature of the host-side "is GC-managed" predicate. Callers
/// (host-side allocators with mixed `try_gc_alloc_stable` /
/// `std::alloc` allocation paths during the L1/L2 stepping-stone
/// window) use this to discriminate GC-managed blocks from
/// `std::alloc`-backed ones at dealloc time.
pub type GcOwnsObjectHookFn = fn(addr: usize) -> bool;
pub type GcCurrentObjectAddressHookFn = fn(addr: usize) -> usize;

thread_local! {
    static GC_OWNS_OBJECT_HOOK: Cell<Option<GcOwnsObjectHookFn>> = const { Cell::new(None) };
    static GC_CURRENT_OBJECT_ADDRESS_HOOK: Cell<Option<GcCurrentObjectAddressHookFn>> =
        const { Cell::new(None) };
}

/// Install the GC-ownership predicate. Overwrites any previously-
/// installed hook.
pub fn register_gc_owns_object_hook(hook: GcOwnsObjectHookFn) {
    GC_OWNS_OBJECT_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the GC-ownership predicate on this thread.
pub fn clear_gc_owns_object_hook() {
    GC_OWNS_OBJECT_HOOK.with(|cell| cell.set(None));
}

/// Install the non-rooting current-address lookup hook.
pub fn register_gc_current_object_address_hook(hook: GcCurrentObjectAddressHookFn) {
    GC_CURRENT_OBJECT_ADDRESS_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the current-address lookup hook on this thread.
pub fn clear_gc_current_object_address_hook() {
    GC_CURRENT_OBJECT_ADDRESS_HOOK.with(|cell| cell.set(None));
}

/// Whether `addr` lies inside the active backend's managed GC heap.
/// Returns `false` when no hook is installed — callers treat that as
/// "no GC owns this pointer" and fall through to their non-GC
/// dealloc path. This is the host-side mirror of
/// `majit_gc::gc_owns_object`.
#[inline]
pub fn try_gc_owns_object(addr: *mut u8) -> bool {
    GC_OWNS_OBJECT_HOOK.with(|cell| match cell.get() {
        Some(f) => f(addr as usize),
        None => false,
    })
}

/// Return the current address for `addr` without registering it as a root.
/// When no hook is installed, or the active GC does not know the object, the
/// address is unchanged.
#[inline]
pub fn try_gc_current_object_address(addr: *mut u8) -> *mut u8 {
    GC_CURRENT_OBJECT_ADDRESS_HOOK.with(|cell| match cell.get() {
        Some(f) => f(addr as usize) as *mut u8,
        None => addr,
    })
}

/// minimark.py:1900-1915 `identityhash` hook.
/// Returns a GC-move-stable address for the given object.
pub type GcIdentityHashHookFn = fn(obj_addr: usize) -> usize;

thread_local! {
    static GC_IDENTITY_HASH_HOOK: Cell<Option<GcIdentityHashHookFn>> = const { Cell::new(None) };
}

pub fn register_gc_identity_hash_hook(hook: GcIdentityHashHookFn) {
    GC_IDENTITY_HASH_HOOK.with(|cell| cell.set(Some(hook)));
}

pub fn clear_gc_identity_hash_hook() {
    GC_IDENTITY_HASH_HOOK.with(|cell| cell.set(None));
}

/// Return a stable identity hash for `obj_addr`.  When the hook is
/// installed, nursery objects get a shadow-based stable address;
/// old-gen objects return their own address.  When no hook is
/// installed, returns `obj_addr` unchanged (pre-GC fallback).
#[inline]
pub fn gc_identity_hash(obj_addr: usize) -> usize {
    GC_IDENTITY_HASH_HOOK.with(|cell| match cell.get() {
        Some(f) => f(obj_addr),
        None => obj_addr,
    })
}

/// Signature of the host-side write barrier callback. `obj` is the
/// GC-managed object whose field is being updated with a possible young
/// pointer. The backend decides whether `obj` is old enough to require
/// remembering.
pub type GcWriteBarrierHookFn = fn(obj: *mut u8);

thread_local! {
    static GC_WRITE_BARRIER_HOOK: Cell<Option<GcWriteBarrierHookFn>> = const { Cell::new(None) };
}

/// Install the write-barrier callback for this thread.
pub fn register_gc_write_barrier_hook(hook: GcWriteBarrierHookFn) {
    GC_WRITE_BARRIER_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the write-barrier callback on this thread.
pub fn clear_gc_write_barrier_hook() {
    GC_WRITE_BARRIER_HOOK.with(|cell| cell.set(None));
}

/// Run the active GC write barrier for `obj` when one is installed.
// `dont_look_inside`: host hook dispatch (`thread_local!` `Cell`
// indirection) stays opaque to the JIT — traces never look inside a
// write barrier (the backend GC rewrite owns that concern); calls
// residualize via the registered fnaddr.
#[majit_macros::dont_look_inside]
pub extern "C" fn try_gc_write_barrier(obj: *mut u8) -> bool {
    GC_WRITE_BARRIER_HOOK.with(|cell| match cell.get() {
        Some(f) => {
            f(obj);
            true
        }
        None => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LAST_TYPE_ID: AtomicUsize = AtomicUsize::new(0);
    static LAST_SIZE: AtomicUsize = AtomicUsize::new(0);

    fn mock_hook(type_id: u32, payload_size: usize) -> *mut u8 {
        LAST_TYPE_ID.store(type_id as usize, Ordering::Relaxed);
        LAST_SIZE.store(payload_size, Ordering::Relaxed);
        // Return a non-null dummy pointer. Tests don't dereference it.
        payload_size as *mut u8
    }

    fn null_hook(_type_id: u32, _payload_size: usize) -> *mut u8 {
        std::ptr::null_mut()
    }

    #[test]
    fn returns_none_when_unregistered() {
        clear_gc_alloc_hook();
        assert!(try_gc_alloc(1, 16).is_none());
    }

    #[test]
    fn invokes_registered_hook_with_args() {
        register_gc_alloc_hook(mock_hook);
        let ptr = try_gc_alloc(7, 24);
        assert!(ptr.is_some());
        assert_eq!(ptr.unwrap() as usize, 24);
        assert_eq!(LAST_TYPE_ID.load(Ordering::Relaxed), 7);
        assert_eq!(LAST_SIZE.load(Ordering::Relaxed), 24);
        clear_gc_alloc_hook();
    }

    #[test]
    fn clear_removes_hook() {
        register_gc_alloc_hook(mock_hook);
        assert!(try_gc_alloc(1, 8).is_some());
        clear_gc_alloc_hook();
        assert!(try_gc_alloc(1, 8).is_none());
    }

    #[test]
    fn hook_returning_null_propagates_some_null() {
        register_gc_alloc_hook(null_hook);
        let ptr = try_gc_alloc(1, 8);
        assert!(ptr.is_some());
        assert!(ptr.unwrap().is_null());
        clear_gc_alloc_hook();
    }

    static LAST_ROOT_PTR: AtomicUsize = AtomicUsize::new(0);
    static REMOVE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static WRITE_BARRIER_CALLS: AtomicUsize = AtomicUsize::new(0);

    unsafe fn mock_add_root(slot: *mut *mut u8) {
        LAST_ROOT_PTR.store(slot as usize, Ordering::Relaxed);
    }
    fn mock_remove_root(slot: *mut *mut u8) {
        let _ = slot;
        REMOVE_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    fn mock_write_barrier(obj: *mut u8) {
        let _ = obj;
        WRITE_BARRIER_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn root_hooks_register_and_remove_round_trip() {
        clear_gc_root_hooks();
        let mut slot: *mut u8 = std::ptr::null_mut();
        assert!(!unsafe { try_gc_add_root(&mut slot as *mut *mut u8) });
        assert!(!try_gc_remove_root(&mut slot as *mut *mut u8));

        LAST_ROOT_PTR.store(0, Ordering::Relaxed);
        REMOVE_CALLS.store(0, Ordering::Relaxed);
        register_gc_root_hooks(mock_add_root, mock_remove_root);

        let slot_ptr = &mut slot as *mut *mut u8;
        assert!(unsafe { try_gc_add_root(slot_ptr) });
        assert_eq!(LAST_ROOT_PTR.load(Ordering::Relaxed), slot_ptr as usize);
        assert!(try_gc_remove_root(slot_ptr));
        assert_eq!(REMOVE_CALLS.load(Ordering::Relaxed), 1);

        clear_gc_root_hooks();
        assert!(!unsafe { try_gc_add_root(slot_ptr) });
        assert!(!try_gc_remove_root(slot_ptr));
    }

    #[test]
    fn stable_hook_is_independent_from_nursery_hook() {
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
        assert!(ptr.is_some());
        assert_eq!(ptr.unwrap() as usize, 32);
        assert_eq!(LAST_TYPE_ID.load(Ordering::Relaxed), 3);
        assert_eq!(LAST_SIZE.load(Ordering::Relaxed), 32);

        clear_gc_alloc_hook();
        clear_gc_alloc_stable_hook();
    }

    #[test]
    fn write_barrier_hook_registers_invokes_and_clears() {
        clear_gc_write_barrier_hook();
        let obj = 0x1000usize as *mut u8;
        assert!(!try_gc_write_barrier(obj));

        WRITE_BARRIER_CALLS.store(0, Ordering::Relaxed);
        register_gc_write_barrier_hook(mock_write_barrier);
        assert!(try_gc_write_barrier(obj));
        assert_eq!(WRITE_BARRIER_CALLS.load(Ordering::Relaxed), 1);

        clear_gc_write_barrier_hook();
        assert!(!try_gc_write_barrier(obj));
    }
}
