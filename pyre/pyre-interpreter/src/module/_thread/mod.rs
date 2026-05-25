//! _thread module — PyPy: pypy/module/thread/
//!
//! Single-threaded pyre: Lock / RLock state lives in the instance dict
//! as `_locked_count`; allocate_lock / start_new_thread / etc. are
//! stubs.

pub mod interp_thread;
pub use interp_thread::register_module as init;
