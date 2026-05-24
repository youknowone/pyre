//! grp module — PyPy: `lib_pypy/grp.py` via `_pwdgrp_cffi`; pyre takes
//! `Modules/grpmodule.c` shape since pyre has no app-level stdlib.
//!
//! getgrgid / getgrnam / getgrall return 4-tuples
//! `(gr_name, gr_passwd, gr_gid, gr_mem)` matching CPython.

pub mod interp_grp;
pub mod moduledef;
