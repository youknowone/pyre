//! _codecs module — PyPy: `pypy/module/_codecs/`.
//!
//! Stub providing lookup_error / register_error and encode / decode
//! identity shells — enough for codecs.py module init to complete.
//! Real codec dispatch is not modelled.

use pyre_object::*;

// `lookup_error(name)` returns a pass-through handler that never fires
// because the pure-Python stdlib paths pyre exercises do not encounter
// encoding errors yet.
fn lookup_error(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(crate::make_builtin_function_with_arity(
        "error_handler",
        |args| Ok(args.first().copied().unwrap_or(w_none())),
        1,
    ))
}

crate::py_module! {
    "_codecs",
    functions: {
        "lookup_error"   / 1 = lookup_error,
        "register_error" / 2 = |_| Ok(w_none()),
        "register"       / 1 = |_| Ok(w_none()),
        "lookup"         / 1 = |_| Ok(w_none()),
        // encode / decode / _forget_codec return input unchanged — matches
        // PyPy `_codecs.encode` when the codec is the identity.
        "encode"         / 1 = |args| Ok(args.first().copied().unwrap_or(w_none())),
        "decode"         / 1 = |args| Ok(args.first().copied().unwrap_or(w_none())),
        "_forget_codec"  / 1 = |args| Ok(args.first().copied().unwrap_or(w_none())),
        "charmap_build"  / 1 = |_| Ok(w_dict_new()),
    },
}
