//! _abc module definition — PyPy: pypy/module/_abc/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_abc::register_module(ns);
}
