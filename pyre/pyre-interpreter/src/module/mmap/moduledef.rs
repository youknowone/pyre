//! mmap module definition — PyPy: pypy/module/mmap/moduledef.py
//!
//! Glue layer that delegates to `interp_mmap::register_module` for the
//! actual name registration.

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_mmap::register_module(ns);
}
