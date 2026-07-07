//! binascii module — PyPy: `pypy/module/binascii/`.
//!
//! base64 / hex / uu / quoted-printable / crc conversions. The byte transforms
//! are a deliberate duplication of RustPython's verified `binascii` core, ported
//! verbatim into [`transforms`] (pure `&[u8]` in / `Vec<u8>` out) and kept
//! outside the LLBC extraction; this module is the W_Root argument/error glue.

// Verbatim vendored transform core. `rlecode_hqx` / `rledecode_hqx` live here
// for completeness but are not part of the 3.14 module surface (removed with
// binhex), so they stay unexposed.
#[allow(dead_code)]
mod transforms;

use pyre_object::*;

/// Accept a str (ASCII) or any bytes-like and surface the raw bytes.
fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    unsafe {
        if is_str(obj) {
            Ok(w_str_get_value(obj).as_bytes().to_vec())
        } else if bytesobject::is_bytes_like(obj) {
            Ok(bytesobject::bytes_like_data(obj).to_vec())
        } else {
            Err(crate::PyError::type_error(
                "argument should be bytes, buffer or ASCII string",
            ))
        }
    }
}

// ── errors ──────────────────────────────────────────────────────────────

/// Build a `binascii.Error` (a `ValueError` subclass) carrying `msg`.
fn binascii_error(msg: impl Into<String>) -> crate::PyError {
    let msg = msg.into();
    let mut err = crate::PyError::value_error(msg.clone());
    if let Some(cls) = crate::builtins::lookup_exc_class("binascii.Error") {
        let args = [cls, w_str_new(&msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            err.exc_object = exc;
        }
    }
    err
}

/// Map a base64 decode failure to the exact `binascii.Error` message.
fn base64_decode_message(e: transforms::Base64DecodeError) -> String {
    use transforms::Base64DecodeError as E;
    match e {
        E::InvalidByte {
            index: 0,
            byte: transforms::PAD,
        } => "Leading padding not allowed".to_owned(),
        E::InvalidByte {
            byte: transforms::PAD,
            ..
        } => "Discontinuous padding not allowed".to_owned(),
        E::InvalidByte { .. } => "Only base64 data is allowed".to_owned(),
        E::InvalidLastSymbol {
            byte: transforms::PAD,
            ..
        } => "Excess data after padding".to_owned(),
        E::InvalidLastSymbol { index: length, .. } => format!(
            "Invalid base64-encoded string: number of data characters ({length}) cannot be 1 more than a multiple of 4"
        ),
        E::InvalidLength(_) => "Incorrect padding".to_owned(),
    }
}

/// Map a transform [`transforms::Error`] to the matching `binascii.Error`.
fn transform_error(e: transforms::Error) -> crate::PyError {
    let msg = match e {
        transforms::Error::OddLengthString => "Odd-length string".to_owned(),
        transforms::Error::NonHexadecimalDigit => "Non-hexadecimal digit found".to_owned(),
        transforms::Error::IllegalChar => "Illegal char".to_owned(),
        transforms::Error::TrailingGarbage => "Trailing garbage".to_owned(),
        transforms::Error::TooLong => "At most 45 bytes at once".to_owned(),
        transforms::Error::Base64(b) => base64_decode_message(b),
    };
    binascii_error(msg)
}

// ── argument helpers ────────────────────────────────────────────────────

/// Optional flag argument, read from `name=` or the positional slot `index`.
/// A missing or `None` value yields `default`; anything else is truth-tested.
fn arg_bool(
    pos: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    name: &str,
    index: usize,
    default: bool,
) -> bool {
    match crate::builtins::kwarg_get(kwargs, name).or_else(|| pos.get(index).copied()) {
        Some(o) if unsafe { is_none(o) } => default,
        Some(o) => crate::baseobjspace::is_true(o).unwrap_or(default),
        None => default,
    }
}

/// The `sep` / `bytes_per_sep` separator for `hexlify` / `b2a_hex`, validated
/// exactly as the C accelerator: length-1, ASCII.
fn arg_sep(
    pos: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
) -> Result<(Option<u8>, isize), crate::PyError> {
    let sep = match crate::builtins::kwarg_get(kwargs, "sep").or_else(|| pos.get(1).copied()) {
        Some(o) if !unsafe { is_none(o) } => {
            let bytes = as_bytes(o)?;
            if bytes.len() != 1 {
                return Err(crate::PyError::value_error("sep must be length 1."));
            }
            if !bytes[0].is_ascii() {
                return Err(crate::PyError::value_error("sep must be ASCII."));
            }
            Some(bytes[0])
        }
        _ => None,
    };
    let bytes_per_sep =
        match crate::builtins::kwarg_get(kwargs, "bytes_per_sep").or_else(|| pos.get(2).copied()) {
            Some(o) if !unsafe { is_none(o) } => crate::baseobjspace::int_w(o)? as isize,
            _ => 1,
        };
    Ok((sep, bytes_per_sep))
}

fn arg_u32(
    pos: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    name: &str,
    index: usize,
    default: u32,
) -> Result<u32, crate::PyError> {
    match crate::builtins::kwarg_get(kwargs, name).or_else(|| pos.get(index).copied()) {
        Some(o) if !unsafe { is_none(o) } => Ok(crate::baseobjspace::int_w(o)? as u32),
        _ => Ok(default),
    }
}

crate::py_module! {
    "binascii",
    exceptions: {
        // binascii.c — Error subclasses ValueError; Incomplete subclasses
        // Exception (NULL base).
        "Error" => crate::builtins::lookup_exc_class("ValueError").expect("ValueError installed"),
        "Incomplete" => crate::builtins::lookup_exc_class("Exception").expect("Exception installed"),
    },
    functions: {
        "b2a_hex" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let (sep, bytes_per_sep) = arg_sep(pos, kwargs)?;
            Ok(w_bytes_from_bytes(&transforms::hexlify(&data, sep, bytes_per_sep)))
        },
        "hexlify" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let (sep, bytes_per_sep) = arg_sep(pos, kwargs)?;
            Ok(w_bytes_from_bytes(&transforms::hexlify(&data, sep, bytes_per_sep)))
        },
        "a2b_hex" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let out = transforms::unhexlify(&data).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
        "unhexlify" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let out = transforms::unhexlify(&data).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
        "crc32" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let init = arg_u32(pos, kwargs, "crc", 1, 0)?;
            Ok(w_int_new(transforms::crc32(&data, init) as i64))
        },
        "crc_hqx" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let init = match crate::builtins::kwarg_get(kwargs, "crc").or_else(|| pos.get(1).copied()) {
                Some(o) => crate::baseobjspace::int_w(o)? as u32,
                None => return Err(crate::PyError::type_error(
                    "crc_hqx() missing required argument 'crc' (pos 2)",
                )),
            };
            Ok(w_int_new(transforms::crc_hqx(&data, init) as i64))
        },
        "a2b_base64" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let strict_mode = arg_bool(pos, kwargs, "strict_mode", 1, false);
            let out = transforms::a2b_base64(&data, strict_mode).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
        "b2a_base64" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let newline = arg_bool(pos, kwargs, "newline", 1, true);
            Ok(w_bytes_from_bytes(&transforms::b2a_base64(&data, newline)))
        },
        "a2b_qp" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let header = arg_bool(pos, kwargs, "header", 1, false);
            Ok(w_bytes_from_bytes(&transforms::a2b_qp(&data, header)))
        },
        "b2a_qp" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let quotetabs = arg_bool(pos, kwargs, "quotetabs", 1, false);
            let istext = arg_bool(pos, kwargs, "istext", 2, true);
            let header = arg_bool(pos, kwargs, "header", 3, false);
            Ok(w_bytes_from_bytes(&transforms::b2a_qp(&data, quotetabs, istext, header)))
        },
        "a2b_uu" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let out = transforms::a2b_uu(&data).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
        "b2a_uu" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let backtick = arg_bool(pos, kwargs, "backtick", 1, false);
            let out = transforms::b2a_uu(&data, backtick).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
    },
}
