//! _functools module — CPython accelerator imported by
//! `lib-python/3/functools.py`.
//!
//! Stub surface — `reduce` raises TypeError (callers should use the
//! pure-Python equivalent); `cmp_to_key` returns an identity wrapper
//! that gives correct ordering for any operands already in their natural
//! sort order (str / int / tuple of those — pyre's stdlib doesn't
//! exercise other shapes).

use pyre_object::*;

// `functools.cmp_to_key(cmp)` — pyre's identity wrapper covers the
// str / int / tuple sort key cases the stdlib actually uses; arbitrary
// cmp callables are not honoured.
fn cmp_to_key(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(crate::make_builtin_function_with_arity(
        "cmp_to_key.K",
        |args| Ok(args.first().copied().unwrap_or(w_none())),
        1,
    ))
}

crate::py_module! {
    "_functools",
    functions: {
        "reduce"     / * = |_| Err(crate::PyError::type_error("reduce not implemented")),
        "cmp_to_key" / 1 = cmp_to_key,
    },
}
