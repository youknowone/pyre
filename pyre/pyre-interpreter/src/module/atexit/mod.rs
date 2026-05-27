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
        // `interp_atexit.py register(callable, *args, **kw)` — first arg
        // is required and must be callable; result is the callable so
        // `@atexit.register` decorator syntax keeps working.
        "register" / * = |args| {
            let f = match args.first() {
                Some(&f) => f,
                None => return Err(crate::PyError::type_error(
                    "register() takes at least 1 argument (0 given)",
                )),
            };
            if !crate::baseobjspace::callable_w(f) {
                return Err(crate::PyError::type_error(
                    "the first argument must be callable",
                ));
            }
            Ok(f)
        },
        "unregister"      / 1 = |_| Ok(w_none()),
        "_run_exitfuncs"  / 0 = |_| Ok(w_none()),
        "_clear"          / 0 = |_| Ok(w_none()),
        "_ncallbacks"     / 0 = |_| Ok(w_int_new(0)),
    },
}
