//! pwd module — PyPy: `pypy/module/pwd/`
//!
//! getpwuid / getpwnam / getpwall return 7-tuples with the
//! `(pw_name, pw_passwd, pw_uid, pw_gid, pw_gecos, pw_dir, pw_shell)`
//! layout.  `struct_passwd` / `struct_pwent` share identity matching
//! `app_pwd.py:1-21`.
//!
//! The module is registered only on unix: a host without the user/group
//! database has no `pwd` module for `import pwd` to find, which is what the
//! stdlib's `try/except ImportError` callers expect.

pub mod interp_pwd;

pub use interp_pwd::register_module as init;
