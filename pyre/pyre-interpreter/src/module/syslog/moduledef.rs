//! syslog module definition — PyPy: pypy/module/syslog/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    super::interp_syslog::register_module(ns);
}
