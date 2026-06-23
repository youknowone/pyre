//! resource module — `lib_pypy/resource.py` (PyPy keeps it app-level via
//! `_resource_cffi`); pyre takes Modules/resource.c shape since pyre has
//! no app-level stdlib.
//!
//! Exposes getrusage / getrlimit / setrlimit plus the standard RUSAGE_*
//! / RLIMIT_* constants and `struct_rusage`.

crate::pyre_module_init!(resource);
