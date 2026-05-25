//! faulthandler module — PyPy: pypy/module/faulthandler/
//!
//! Pyre has no Python-level traceback machinery yet, so signal handlers
//! write a short "Fatal Python error: <name>" line to fd 2 and restore the
//! default disposition + reraise the signal so the process dies normally.

crate::pyre_module_init!(interp_faulthandler);
