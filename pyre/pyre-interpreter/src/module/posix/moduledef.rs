//! posix module definition — PyPy: pypy/module/posix/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_posix::register_module(ns);
}
