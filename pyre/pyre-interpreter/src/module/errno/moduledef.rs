//! errno module definition — PyPy: pypy/module/errno/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_errno::register_module(ns);
}
