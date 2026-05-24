//! _socket module — PyPy: pypy/module/_socket/
//!
//! Provides the lowest-level socket API exposed to Python.  The
//! interp_socket submodule carries the W_Socket class implementation
//! plus address conversion / IDNA / error mapping helpers; moduledef
//! wires the resulting names into the ns dict at import time.

pub mod interp_socket;
pub mod moduledef;
