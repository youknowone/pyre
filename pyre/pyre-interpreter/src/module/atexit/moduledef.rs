//! atexit module definition — PyPy: pypy/module/atexit/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_atexit::register_module(ns);
}
