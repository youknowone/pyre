//! unicodedata module definition — PyPy: pypy/module/unicodedata/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_unicodedata::register_module(ns);
}
