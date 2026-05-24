//! atexit implementation — PyPy: pypy/module/atexit/interp_atexit.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// atexit stub — PyPy: pypy/module/atexit/. Single-threaded pyre doesn't
/// actually run the registered callbacks on shutdown yet; `register` accepts
/// any callable and returns it so `@atexit.register` decorators work.
pub fn register_module(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "register",
        crate::make_builtin_function("register", |args| {
            // Return the function so `@atexit.register` decorator form works.
            Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
        }),
    );
    crate::dict_storage_store(
        ns,
        "unregister",
        crate::make_builtin_function_with_arity("unregister", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "_run_exitfuncs",
        crate::make_builtin_function_with_arity("_run_exitfuncs", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "_clear",
        crate::make_builtin_function_with_arity("_clear", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "_ncallbacks",
        crate::make_builtin_function_with_arity(
            "_ncallbacks",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
}
