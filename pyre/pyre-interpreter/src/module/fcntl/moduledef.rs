//! fcntl module definition — PyPy: pypy/module/fcntl/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_fcntl::register_module(ns);
}
