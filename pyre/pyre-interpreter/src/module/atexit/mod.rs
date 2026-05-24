//! atexit module — PyPy: pypy/module/atexit/
//!
//! Stub: single-threaded pyre doesn't actually run the registered
//! callbacks on shutdown yet; `register` accepts any callable and returns
//! it so `@atexit.register` decorators work.

pub mod interp_atexit;
pub mod moduledef;
