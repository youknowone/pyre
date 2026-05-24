//! importlib submodule entry points — one per builtin-registry name.

use crate::DictStorage;

pub fn init_pkg(ns: &mut DictStorage) {
    super::interp_importlib::register_pkg(ns);
}

pub fn init_util(ns: &mut DictStorage) {
    super::interp_importlib::register_util(ns);
}

pub fn init_abc(ns: &mut DictStorage) {
    super::interp_importlib::register_abc(ns);
}

pub fn init_machinery(ns: &mut DictStorage) {
    super::interp_importlib::register_machinery(ns);
}
