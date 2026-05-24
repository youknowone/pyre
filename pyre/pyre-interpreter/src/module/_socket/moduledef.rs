//! _socket module definition — PyPy: pypy/module/_socket/moduledef.py.
//!
//! `init` is the entry point invoked by importing.rs and is responsible
//! for populating the module namespace with constants, error classes,
//! module-level functions and the `socket` type.  The per-name
//! registration logic lives in `super::interp_socket::register_module`
//! and `super::interp_socket::register_socket_type` so this file stays
//! a thin glue layer matching the PyPy moduledef.py pattern.

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_socket::register_module(ns);
}
