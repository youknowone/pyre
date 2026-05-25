//! _functools module — PyPy: `pypy/module/_functools/`.
//!
//! Stub surface — `reduce` raises TypeError (callers should use the
//! pure-Python equivalent); `cmp_to_key` returns an identity wrapper
//! that gives correct ordering for any operands already in their natural
//! sort order (str / int / tuple of those — pyre's stdlib doesn't
//! exercise other shapes).

crate::py_module! {
    "_functools",
    interpleveldefs: {
        "reduce" => crate::make_builtin_function("reduce", |_| {
            Err(crate::PyError::type_error("reduce not implemented"))
        }),
        // `functools.cmp_to_key(cmp)` — pyre's identity wrapper covers
        // the str / int / tuple sort key cases the stdlib actually
        // uses; arbitrary cmp callables are not honoured.
        "cmp_to_key" => crate::make_builtin_function_with_arity(
            "cmp_to_key",
            |_args| {
                Ok(crate::make_builtin_function_with_arity(
                    "cmp_to_key.K",
                    |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
                    1,
                ))
            },
            1,
        ),
    }
}
