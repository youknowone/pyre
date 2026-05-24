//! _imp module definition — PyPy: pypy/module/imp/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_imp::register_module(ns);
}
