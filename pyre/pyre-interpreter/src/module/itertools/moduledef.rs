//! itertools module definition — PyPy: pypy/module/itertools/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_itertools::register_module(ns);
}
