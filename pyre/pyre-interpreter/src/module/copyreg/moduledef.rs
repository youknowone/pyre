//! copyreg module definition — PyPy: pypy/module/copyreg/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_copyreg::register_module(ns);
}
