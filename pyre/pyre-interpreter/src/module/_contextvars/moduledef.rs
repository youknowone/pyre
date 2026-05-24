//! _contextvars module definition — PyPy: pypy/module/_contextvars/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_contextvars::register_module(ns);
}
