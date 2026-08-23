//! select module — PyPy: pypy/module/select/
//!
//! `select.select` and the `select.poll` object live in `interp_select`;
//! the `select.kqueue` / `select.kevent` objects (macOS/BSD) live in
//! `interp_kqueue` / `interp_kevent` because each `#[pyre_class]` emits a
//! module-scoped `type_object()` that would otherwise collide.  epoll
//! (Linux) is not implemented yet.

#[cfg(all(target_os = "macos", feature = "host_env"))]
pub mod interp_kevent;
#[cfg(all(target_os = "macos", feature = "host_env"))]
pub mod interp_kqueue;

crate::pyre_module_init!(interp_select);

/// The `select.kevent` type object, or null on a platform without kqueue.
///
/// The type is not acceptable as a base class and its constructor still binds
/// keywords — `kqueue_event_init` parses the six-name kwlist, and
/// `interp_kqueue.py W_Kevent.descr__init__` names the same six — so the call
/// path has to name it among the non-base types that take keywords.
pub fn kevent_type() -> pyre_object::PyObjectRef {
    #[cfg(all(target_os = "macos", feature = "host_env"))]
    {
        interp_kevent::type_object()
    }
    #[cfg(not(all(target_os = "macos", feature = "host_env")))]
    {
        pyre_object::PY_NULL
    }
}
