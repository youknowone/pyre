//! pwd module — PyPy: `pypy/module/pwd/`
//!
//! getpwuid / getpwnam / getpwall return 7-tuples with the
//! `(pw_name, pw_passwd, pw_uid, pw_gid, pw_gecos, pw_dir, pw_shell)`
//! layout.  `struct_passwd` / `struct_pwent` share identity matching
//! `app_pwd.py:1-21`.

pub mod interp_pwd;
pub mod moduledef;
