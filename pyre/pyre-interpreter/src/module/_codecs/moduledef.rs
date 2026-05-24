//! _codecs module definition — PyPy: pypy/module/_codecs/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_codecs::register_module(ns);
}
