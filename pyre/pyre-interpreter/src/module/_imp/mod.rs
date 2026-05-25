//! _imp module — PyPy: pypy/module/imp/
//!
//! Minimal subset required by importlib._bootstrap to decide which
//! loader handles a name.

pub mod interp_imp;
pub use interp_imp::register_module as init;
