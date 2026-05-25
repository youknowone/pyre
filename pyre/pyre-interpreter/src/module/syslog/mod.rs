//! syslog module — PyPy: pypy/module/syslog/
//!
//! openlog / syslog / closelog / setlogmask backed by
//! `rustpython_host_env::syslog`.  Unix-only.

crate::pyre_module_init!(interp_syslog);
