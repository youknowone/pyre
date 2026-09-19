//! _socket module — PyPy: pypy/module/_socket/
//!
//! Provides the lowest-level socket API exposed to Python.  The
//! interp_socket submodule carries the W_Socket class implementation
//! plus address conversion / IDNA / error mapping helpers.  The host
//! socket layer is `pyre_interpreter::rsocket_rffi` (`rpython/rlib/_rsocket_rffi.py`).
//! A target with no such layer still carries the module: what it
//! lacks it lacks entry point by entry point, the way a build whose C library
//! has the headers but not the calls lacks them, and `interp_socket_wasm`
//! publishes the part that is left -- the type `socket.py` subclasses and the
//! numbers it reads.

// The text half of the address converters compiles everywhere so its corpus
// runs with the unit tests, on hosts whose entry points reach libc instead.
#[cfg(any(test, not(any(unix, windows))))]
use rustpython_common::inet;
#[cfg(not(any(unix, windows)))]
mod interp_socket_wasm;

pyre_interpreter::pyre_module_init!(interp_socket);
