//! atexit module — PyPy: `pypy/module/atexit/`.
//!
//! Stub surface — pyre is single-threaded and does not currently run
//! registered callbacks on shutdown.  `register` accepts any callable
//! and returns it so `@atexit.register` decorator syntax works; the
//! other names are accepted but inert.

use pyre_object::*;

fn register(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Return the function so `@atexit.register` works.
    Ok(args.first().copied().unwrap_or(w_none()))
}

fn noop(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn ncallbacks(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_int_new(0))
}

crate::py_module! {
    "atexit",
    functions: {
        "register"        / * = register,
        "unregister"      / 1 = noop,
        "_run_exitfuncs"  / 0 = noop,
        "_clear"          / 0 = noop,
        "_ncallbacks"     / 0 = ncallbacks,
    },
}
