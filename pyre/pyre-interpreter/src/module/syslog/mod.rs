//! syslog module — PyPy: pypy/module/syslog/
//!
//! openlog / syslog / closelog / setlogmask backed by
//! `rustpython_host_env::syslog`.  Unix-only.

pub mod interp_syslog;
pub use interp_syslog::register_module as init;
