//! copyreg module — PyPy: `pypy/module/copyreg/`.
//!
//! Stub surface — pyre does not support pickle.  `copyreg.pickle` is a
//! no-op that returns None; `copyreg.dispatch_table` is an empty dict
//! so callers' `dispatch_table[type]` lookups raise KeyError as they
//! would on CPython with no registered reducer.

crate::py_module! {
    "copyreg",
    interpleveldefs: {
        // `copyreg.pickle(type, reduce_func, constructor=None)` —
        // register a pickle reducer.  Stub ignores the call.
        "pickle" => crate::make_builtin_function_with_arity(
            "pickle",
            |_| Ok(pyre_object::w_none()),
            3,
        ),
        "dispatch_table" => pyre_object::w_dict_new(),
    }
}
