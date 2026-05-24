//! select module definition — PyPy: pypy/module/select/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_select::register_module(ns);
}
