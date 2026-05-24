//! faulthandler module definition — PyPy: pypy/module/faulthandler/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_faulthandler::register_module(ns);
}
