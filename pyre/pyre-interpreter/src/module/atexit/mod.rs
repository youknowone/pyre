//! atexit module — PyPy: `pypy/module/atexit/`.
//!
//! Stub surface — pyre is single-threaded and does not currently run
//! registered callbacks on shutdown.  `register` accepts any callable
//! and returns it so `@atexit.register` decorator syntax works; the
//! other names are accepted but inert.

use pyre_object::*;

crate::py_module! {
    "atexit",
    functions: {
        // Return the function so `@atexit.register` works.
        "register"        / * = |args| Ok(args.first().copied().unwrap_or(w_none())),
        "unregister"      / 1 = |_| Ok(w_none()),
        "_run_exitfuncs"  / 0 = |_| Ok(w_none()),
        "_clear"          / 0 = |_| Ok(w_none()),
        "_ncallbacks"     / 0 = |_| Ok(w_int_new(0)),
    },
}
