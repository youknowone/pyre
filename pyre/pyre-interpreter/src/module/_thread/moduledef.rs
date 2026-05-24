//! _thread module definition — PyPy: pypy/module/thread/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_thread::register_module(ns);
}
