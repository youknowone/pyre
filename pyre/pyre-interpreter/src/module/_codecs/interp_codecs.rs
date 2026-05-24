//! _codecs implementation — PyPy: pypy/module/_codecs/interp_codecs.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// _codecs stub — PyPy: pypy/module/_codecs/
///
/// Provides lookup_error/register_error and encode/decode no-op stubs so
/// codecs.py module init runs to completion.
pub fn register_module(ns: &mut DictStorage) {
    // lookup_error(name) — returns an error handler for the given error
    // strategy. Pyre returns a pass-through lambda that never fires because
    // we don't encounter encoding errors in the pure-Python stdlib paths
    // we exercise so far.
    crate::dict_storage_store(
        ns,
        "lookup_error",
        crate::make_builtin_function_with_arity(
            "lookup_error",
            |_| {
                Ok(crate::make_builtin_function_with_arity(
                    "error_handler",
                    |args| {
                        Ok(if args.is_empty() {
                            pyre_object::w_none()
                        } else {
                            args[0]
                        })
                    },
                    1,
                ))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "register_error",
        crate::make_builtin_function_with_arity("register_error", |_| Ok(pyre_object::w_none()), 2),
    );
    crate::dict_storage_store(
        ns,
        "register",
        crate::make_builtin_function_with_arity("register", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "lookup",
        crate::make_builtin_function_with_arity("lookup", |_| Ok(pyre_object::w_none()), 1),
    );
    // encode/decode — return input unchanged. Matches PyPy _codecs.encode
    // when the codec is the identity.
    let identity = crate::make_builtin_function_with_arity(
        "identity",
        |args| {
            Ok(if args.is_empty() {
                pyre_object::w_none()
            } else {
                args[0]
            })
        },
        1,
    );
    crate::dict_storage_store(ns, "encode", identity);
    crate::dict_storage_store(ns, "decode", identity);
    crate::dict_storage_store(ns, "_forget_codec", identity);
    crate::dict_storage_store(
        ns,
        "charmap_build",
        crate::make_builtin_function_with_arity(
            "charmap_build",
            |_| Ok(pyre_object::w_dict_new()),
            1,
        ),
    );
}
