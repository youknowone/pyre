//! _locale module definition — PyPy: pypy/module/_locale/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_locale::register_module(ns);
}
