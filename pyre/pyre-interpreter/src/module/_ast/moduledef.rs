//! _ast module definition — PyPy: pypy/module/_ast/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_ast::register_module(ns);
}
