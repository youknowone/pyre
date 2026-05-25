//! atexit module — PyPy: `pypy/module/atexit/`.
//!
//! Stub surface — pyre is single-threaded and does not currently run
//! registered callbacks on shutdown.  `register` accepts any callable
//! and returns it so `@atexit.register` decorator syntax works; the
//! other names are accepted but inert.

crate::py_module! {
    "atexit",
    interpleveldefs: {
        "register" => crate::make_builtin_function("register", |args| {
            // Return the function so `@atexit.register` works.
            Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
        }),
        "unregister" => crate::make_builtin_function_with_arity(
            "unregister", |_| Ok(pyre_object::w_none()), 1),
        "_run_exitfuncs" => crate::make_builtin_function_with_arity(
            "_run_exitfuncs", |_| Ok(pyre_object::w_none()), 0),
        "_clear" => crate::make_builtin_function_with_arity(
            "_clear", |_| Ok(pyre_object::w_none()), 0),
        "_ncallbacks" => crate::make_builtin_function_with_arity(
            "_ncallbacks", |_| Ok(pyre_object::w_int_new(0)), 0),
    }
}
