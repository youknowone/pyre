//! _ctypes module definition — PyPy: pypy/module/_ctypes/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_ctypes::register_module(ns);
}
