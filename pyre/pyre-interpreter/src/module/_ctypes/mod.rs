//! _ctypes module — PyPy: `pypy/module/_rawffi/` plus `lib_pypy/_ctypes/`.
//!
//! Slice C1: dlopen / dlsym / dlclose + size/align/memmove constants.  The
//! full c_int / Structure / CFUNCTYPE / Pointer machinery still requires
//! libffi-style argument marshalling and per-instance heap state.

crate::pyre_module_init!(interp_ctypes);
