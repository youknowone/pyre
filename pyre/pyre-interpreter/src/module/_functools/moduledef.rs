//! _functools module definition — PyPy: pypy/module/_functools/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_functools::register_module(ns);
}
