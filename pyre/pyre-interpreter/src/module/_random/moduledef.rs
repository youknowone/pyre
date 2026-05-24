//! _random module definition — PyPy: pypy/module/_random/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_random::register_module(ns);
}
