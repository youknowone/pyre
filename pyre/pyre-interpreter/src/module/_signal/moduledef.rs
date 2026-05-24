//! _signal module definition — PyPy: pypy/module/signal/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_signal::register_module(ns);
}
