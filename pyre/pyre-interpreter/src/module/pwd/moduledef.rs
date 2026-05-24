//! pwd module definition — PyPy: pypy/module/pwd/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_pwd::register_module(ns);
}
