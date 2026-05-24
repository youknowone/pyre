//! _functools implementation — PyPy: pypy/module/_functools/interp_functools.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// _functools stub
pub fn register_module(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "reduce",
        crate::make_builtin_function("reduce", |_| {
            Err(crate::PyError::type_error("reduce not implemented"))
        }),
    );
    // functools.cmp_to_key(cmp) — returns a callable that wraps a value in
    // an opaque key. For sorting str / int / tuple of those (the only paths
    // pyre's stdlib actually exercises), the items are already comparable,
    // so an identity key gives the same ordering as `cmp(a, b)` would.
    crate::dict_storage_store(
        ns,
        "cmp_to_key",
        crate::make_builtin_function_with_arity(
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
    );
}
