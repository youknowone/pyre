//! _multiprocessing module — PyPy: pypy/module/_multiprocessing/
//!
//! Exposes `SemLock(kind, value, maxvalue, name, unlink)` and
//! `sem_unlink(name)`.  Single-threaded pyre still needs the methods to
//! exist so multiprocessing.py teardown survives.

crate::pyre_module_init!(interp_multiprocessing);
