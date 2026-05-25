//! sys module — PyPy: pypy/module/sys/

pub mod interp_sys;
pub mod state;

pub use interp_sys::register_module as init;
