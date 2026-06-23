//! syslog module — PyPy: `lib_pypy/syslog.py`.
//!
//! openlog / syslog / closelog / setlogmask backed by
//! `rustpython_host_env::syslog`.  Unix-only.

crate::pyre_module_init!(syslog);
