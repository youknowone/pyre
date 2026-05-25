//! posix module — PyPy: pypy/module/posix/
//!
//! Provides the minimal surface that os.py module init needs to succeed
//! plus the host_env-backed implementations of the calls pyre actually
//! exercises.  The shared `stat_result` builtin type lives here too.

crate::pyre_module_init!(interp_posix);
