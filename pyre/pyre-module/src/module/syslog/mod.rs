//! syslog module — PyPy: `lib_pypy/syslog.py`.
//!
//! openlog / syslog / closelog / setlogmask backed by
//! `rustpython_host_env::syslog`.  Unix-only.

pyre_interpreter::pyre_module_init!(syslog);
