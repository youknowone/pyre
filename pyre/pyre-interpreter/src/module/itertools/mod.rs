//! itertools module — PyPy: pypy/module/itertools/

pub mod interp_itertools;
pub use interp_itertools::register_module as init;
