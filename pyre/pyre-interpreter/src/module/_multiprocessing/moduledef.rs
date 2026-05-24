//! _multiprocessing module definition — PyPy: pypy/module/_multiprocessing/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_multiprocessing::register_module(ns);
}
