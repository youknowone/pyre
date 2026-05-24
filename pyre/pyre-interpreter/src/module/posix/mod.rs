//! posix module — PyPy: pypy/module/posix/
//!
//! Provides the minimal surface that os.py module init needs to succeed
//! plus the host_env-backed implementations of the calls pyre actually
//! exercises.  The shared `stat_result` builtin type lives here too.

pub mod interp_posix;
pub mod moduledef;
