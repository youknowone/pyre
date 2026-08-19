//! _socket module — PyPy: pypy/module/_socket/
//!
//! Provides the lowest-level socket API exposed to Python.  The
//! interp_socket submodule carries the W_Socket class implementation
//! plus address conversion / IDNA / error mapping helpers, and
//! rsocket_rffi carries the host socket layer both platforms reach it
//! through.  It exists only where such a layer does: a target that is
//! neither POSIX nor Windows keeps the module's platform-neutral surface
//! (the error classes, the byte-order helpers, the default timeout) and
//! nothing that would need a descriptor behind it.

#[cfg(any(unix, windows))]
pub(crate) mod rsocket_rffi;

crate::pyre_module_init!(interp_socket);
