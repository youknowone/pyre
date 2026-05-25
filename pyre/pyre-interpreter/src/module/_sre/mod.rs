//! _sre module — PyPy: pypy/module/_sre/
//!
//! SRE bytecode interpreter bridge.  `interp_sre.py` shape is preserved
//! in `interp_sre.rs`; mod.rs is the declarative entry point.

pub mod interp_sre;

pub use interp_sre::register_module as init;
