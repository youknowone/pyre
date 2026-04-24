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
//! installed — they fall back to the `Box::into_raw` Phase 1 path in
//! that case. Session-by-session migration drops the `Box::into_raw`
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
}
