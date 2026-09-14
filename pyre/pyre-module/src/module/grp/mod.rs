//! grp module — PyPy: `lib_pypy/grp.py` via `_pwdgrp_cffi`; pyre takes
//! `Modules/grpmodule.c` shape since pyre has no app-level stdlib.
//!
//! getgrgid / getgrnam / getgrall return 4-tuples
//! `(gr_name, gr_passwd, gr_gid, gr_mem)` matching CPython.
//!
//! The module is registered only on unix: a host without the user/group
//! database has no `grp` module for `import grp` to find, which is what the
//! stdlib's `try/except ImportError` callers expect.

pub mod grp;

pub use grp::register_module as init;
