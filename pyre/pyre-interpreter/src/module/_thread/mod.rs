//! _thread module — PyPy: pypy/module/thread/
//!
//! Single-threaded pyre: Lock / RLock state lives in the instance dict
//! as `_locked_count`; allocate_lock / start_new_thread / etc. are
//! stubs.

crate::pyre_module_init!(interp_thread);
