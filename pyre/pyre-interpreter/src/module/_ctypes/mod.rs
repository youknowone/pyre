//! _ctypes module — PyPy: pypy/module/_rawffi/, pypy/module/_ctypes/
//!
//! Slice C1: dlopen / dlsym / dlclose + size/align/memmove constants.  The
//! full c_int / Structure / CFUNCTYPE / Pointer machinery still requires
//! libffi-style argument marshalling and per-instance heap state.

pub mod interp_ctypes;
pub use interp_ctypes::register_module as init;
