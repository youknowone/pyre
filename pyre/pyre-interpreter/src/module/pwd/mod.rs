//! pwd module — PyPy: `pypy/module/pwd/`
//!
//! getpwuid / getpwnam / getpwall return 7-tuples with the
//! `(pw_name, pw_passwd, pw_uid, pw_gid, pw_gecos, pw_dir, pw_shell)`
//! layout.  `struct_passwd` / `struct_pwent` share identity matching
//! `app_pwd.py:1-21`.
//!
//! `register_module` is `#[cfg(unix)]`; on Windows the module dict stays
//! empty so `import pwd` still resolves to the builtin module object.

pub mod interp_pwd;

#[cfg(unix)]
pub use interp_pwd::register_module as init;

#[cfg(not(unix))]
pub fn init(_ns: pyre_object::PyObjectRef) {}
