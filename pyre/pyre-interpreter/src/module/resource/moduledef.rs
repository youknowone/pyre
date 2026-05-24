//! resource module definition — PyPy: lib_pypy/resource.py (via cffi).

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_resource::register_module(ns);
}
