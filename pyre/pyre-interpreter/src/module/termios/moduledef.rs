//! termios module definition — PyPy: pypy/module/termios/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_termios::register_module(ns);
}
