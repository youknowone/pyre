//! _socket module — PyPy: pypy/module/_socket/
//!
//! Provides the lowest-level socket API exposed to Python.  The
//! interp_socket submodule carries the W_Socket class implementation
//! plus address conversion / IDNA / error mapping helpers, and
//! rsocket_rffi carries the host socket layer both platforms reach it
//! through.  A target with no such layer still carries the module: what it
//! lacks it lacks entry point by entry point, the way a build whose C library
//! has the headers but not the calls lacks them, and `interp_socket_wasm`
//! publishes the part that is left -- the type `socket.py` subclasses and the
//! numbers it reads.

// The text half of the address converters compiles everywhere so its corpus
// runs with the unit tests, on hosts whose entry points reach libc instead.
#[cfg(any(test, not(any(unix, windows))))]
mod inet_text;
#[cfg(not(any(unix, windows)))]
mod interp_socket_wasm;
#[cfg(any(unix, windows))]
pub(crate) mod rsocket_rffi;

crate::pyre_module_init!(interp_socket);
