//! _opcode module definition — PyPy: pypy/module/_opcode/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_opcode::register_module(ns);
}
