//! _socket module — PyPy: pypy/module/_socket/
//!
//! Provides the lowest-level socket API exposed to Python.  The
//! interp_socket submodule carries the W_Socket class implementation
//! plus address conversion / IDNA / error mapping helpers.

pub mod interp_socket;

pub use interp_socket::register_module as init;
