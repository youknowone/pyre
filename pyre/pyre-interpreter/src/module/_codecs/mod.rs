//! _codecs module — PyPy: `pypy/module/_codecs/`.
//!
//! Stub providing lookup_error / register_error and encode / decode
//! identity shells — enough for codecs.py module init to complete.
//! Real codec dispatch is not modelled.

crate::py_module! {
    "_codecs",
    interpleveldefs: {
        // `lookup_error(name)` returns a pass-through handler that
        // never fires because the pure-Python stdlib paths pyre
        // exercises do not encounter encoding errors yet.
        "lookup_error" => crate::make_builtin_function_with_arity(
            "lookup_error",
            |_| {
                Ok(crate::make_builtin_function_with_arity(
                    "error_handler",
                    |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
                    1,
                ))
            },
            1,
        ),
        "register_error" => crate::make_builtin_function_with_arity(
            "register_error", |_| Ok(pyre_object::w_none()), 2),
        "register" => crate::make_builtin_function_with_arity(
            "register", |_| Ok(pyre_object::w_none()), 1),
        "lookup" => crate::make_builtin_function_with_arity(
            "lookup", |_| Ok(pyre_object::w_none()), 1),
        // encode / decode / _forget_codec — return input unchanged.
        // Matches PyPy `_codecs.encode` when the codec is the identity.
        "encode" => crate::make_builtin_function_with_arity(
            "encode",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
        "decode" => crate::make_builtin_function_with_arity(
            "decode",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
        "_forget_codec" => crate::make_builtin_function_with_arity(
            "_forget_codec",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
        "charmap_build" => crate::make_builtin_function_with_arity(
            "charmap_build", |_| Ok(pyre_object::w_dict_new()), 1),
    }
}
