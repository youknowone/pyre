//! grp module definition — PyPy: lib_pypy/grp.py (via cffi).

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_grp::register_module(ns);
}
