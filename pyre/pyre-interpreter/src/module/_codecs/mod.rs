//! _codecs module — PyPy: `pypy/module/_codecs/`.
//!
//! Text codecs (`encode` / `decode`) delegate to `str.encode` /
//! `bytes.decode`, which cover `PyCodec_Encode` / `PyCodec_Decode` for the
//! text path. The codec registry (`register` / `lookup`) and error
//! handlers remain stubs; binary transform codecs are not modelled.

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
    inline_functions: {
        // `encode(obj, encoding='utf-8', errors='strict')` — text path of
        // `PyCodec_Encode`: a str is encoded via `str.encode`; anything
        // else passes through unchanged.
        fn encode(
            obj: PyObjectRef,
            #[default(w_str_new("utf-8"))] encoding: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            if unsafe { is_str(obj) } {
                let m = crate::baseobjspace::getattr_str(obj, "encode")?;
                return crate::call::call_function_impl_result(m, &[encoding, errors]);
            }
            Ok(obj)
        }
        // `decode(obj, encoding='utf-8', errors='strict')` — text path of
        // `PyCodec_Decode`: bytes / bytearray decode via `.decode`;
        // anything else passes through unchanged.
        fn decode(
            obj: PyObjectRef,
            #[default(w_str_new("utf-8"))] encoding: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            if unsafe { is_bytes(obj) || is_bytearray(obj) } {
                let m = crate::baseobjspace::getattr_str(obj, "decode")?;
                return crate::call::call_function_impl_result(m, &[encoding, errors]);
            }
            Ok(obj)
        }
    },
    functions: {
        "lookup_error"     / 1 = lookup_error,
        "register_error"   / 2 = |_| Ok(w_none()),
        "_unregister_error" / 1 = |_| Ok(w_bool_from(false)),
        "register"       / 1 = |_| Ok(w_none()),
        "lookup"         / 1 = |_| Ok(w_none()),
        "_forget_codec"  / 1 = |args| Ok(args.first().copied().unwrap_or(w_none())),
        "charmap_build"  / 1 = |_| Ok(w_dict_new()),
    },
}
