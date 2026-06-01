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
