//! gc module definition — PyPy: pypy/module/gc/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_gc::register_module(ns);
}
