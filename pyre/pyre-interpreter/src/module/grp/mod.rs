//! grp module — PyPy: `lib_pypy/grp.py` via `_pwdgrp_cffi`; pyre takes
//! `Modules/grpmodule.c` shape since pyre has no app-level stdlib.
//!
//! getgrgid / getgrnam / getgrall return 4-tuples
//! `(gr_name, gr_passwd, gr_gid, gr_mem)` matching CPython.
//!
//! `register_module` is `#[cfg(unix)]`; on Windows the module dict stays
//! empty so `import grp` still resolves to the builtin module object.

pub mod interp_grp;

#[cfg(unix)]
pub use interp_grp::register_module as init;

#[cfg(not(unix))]
pub fn init(_ns: &mut crate::DictStorage) {}
