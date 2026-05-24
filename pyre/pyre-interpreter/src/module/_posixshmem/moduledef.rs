//! _posixshmem module definition — PyPy: pypy/module/_posixshmem/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_posixshmem::register_module(ns);
}
