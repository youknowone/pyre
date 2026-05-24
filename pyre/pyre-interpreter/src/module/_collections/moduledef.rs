//! _collections module definition — PyPy: pypy/module/_collections/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_collections::register_module(ns);
}
