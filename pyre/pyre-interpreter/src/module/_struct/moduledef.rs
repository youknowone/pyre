//! _struct module definition — PyPy: pypy/module/struct/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_struct::register_module(ns);
}
