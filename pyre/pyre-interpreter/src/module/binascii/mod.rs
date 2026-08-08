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

/// `PyArg_UnpackTuple` for the entries that declare no keyword at all
/// (`crc32(data, crc=0, /)`, `crc_hqx(data, crc, /)`).  A keyword is refused
/// under the module-qualified name; the arity message names the function bare
/// and is the only one in this module that carries no `()`.
fn unpack_positional<'a>(
    args: &'a [PyObjectRef],
    fn_name: &str,
    min: usize,
    max: usize,
) -> Result<&'a [PyObjectRef], crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::real_kwarg_count(kwargs) != 0 {
        return Err(crate::PyError::type_error(format!(
            "binascii.{fn_name}() takes no keyword arguments"
        )));
    }
    if pos.len() < min {
        return Err(crate::PyError::type_error(format!(
            "{fn_name} expected {}{min} argument{}, got {}",
            if min == max { "" } else { "at least " },
            if min == 1 { "" } else { "s" },
            pos.len(),
        )));
    }
    if pos.len() > max {
        return Err(crate::PyError::type_error(format!(
            "{fn_name} expected {}{max} arguments, got {}",
            if min == max { "" } else { "at most " },
            pos.len(),
        )));
    }
    Ok(pos)
}

/// The entries whose `data` is positional-only and whose remaining slots are
/// keyword-only (`a2b_base64`, `b2a_base64`, `b2a_uu`).  A positional-only
/// parameter is never reported by name, so both an omitted and a surplus
/// positional read as the one positional slot; `_PyArg_UnpackKeywords` holds
/// an unrecognized keyword back until then, which is why `f(data=…)` is a
/// missing positional rather than an unexpected keyword.
fn arg_data_posonly(
    args: &[PyObjectRef],
    fn_name: &str,
    kwonly: &[&str],
) -> Result<(PyObjectRef, Option<PyObjectRef>), crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        fn_name,
        pos.len(),
        crate::builtins::real_kwarg_count(kwargs),
        1,
        1,
        kwonly.len(),
    )?;
    if pos.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "{fn_name}() takes exactly 1 positional argument (0 given)"
        )));
    }
    if let Some(dict) = kwargs {
        for (key, _) in unsafe { pyre_object::w_dict_str_entries_wtf8(dict) }.iter() {
            // A keyword can be any `str`, so the name is compared and reported
            // as the WTF-8 it is: `format!` would fold a surrogate to U+FFFD.
            let named = key.as_str().ok();
            if named != Some("__pyre_kw__") && !named.is_some_and(|n| kwonly.contains(&n)) {
                let mut msg = rustpython_wtf8::Wtf8Buf::from_string(format!(
                    "{fn_name}() got an unexpected keyword argument '"
                ));
                msg.push_wtf8(key);
                msg.push_str("'");
                return Err(crate::PyError::type_error(msg));
            }
        }
    }
    Ok((pos[0], kwargs))
}

/// A keyword-only flag: absent or `None` keeps `default`.
fn kwonly_bool(kwargs: Option<PyObjectRef>, name: &str, default: bool) -> bool {
    slot_bool(
        crate::builtins::kwarg_get(kwargs, name).unwrap_or(PY_NULL),
        default,
    )
}

/// A slot [`crate::builtins::bind_builtin_kwargs`] resolved: `PY_NULL` for an
/// omitted optional argument, and `None` for one passed explicitly, both keep
/// `default`.
fn slot_bool(w: PyObjectRef, default: bool) -> bool {
    if w.is_null() || unsafe { is_none(w) } {
        return default;
    }
    crate::baseobjspace::is_true(w).unwrap_or(default)
}

/// `ascii_buffer_converter` — accept a str (ASCII) or any bytes-like and
/// surface the raw bytes.  Only the `a2b_*` decoders take a str source.
fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    unsafe {
        if is_str(obj) {
            // The ASCII check runs on the raw buffer: a lone surrogate is
            // non-ASCII, so it takes the same rejection as any other
            // non-ASCII character.
            let s = w_str_get_wtf8(obj);
            if !s.as_bytes().is_ascii() {
                return Err(crate::PyError::value_error(
                    "string argument should contain only ASCII characters",
                ));
            }
            Ok(s.as_bytes().to_vec())
        } else {
            match crate::typedef::buffer_as_bytes_like(obj)? {
                Some(src) => Ok(bytesobject::bytes_like_data(src).to_vec()),
                None => Err(crate::PyError::type_error(
                    "argument should be bytes, buffer or ASCII string",
                )),
            }
        }
    }
}

/// The `Py_buffer` converter — the `b2a_*` encoders and the checksums take a
/// bytes-like source only, so a str of any kind is rejected by its type.
fn as_buffer_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    match unsafe { crate::typedef::buffer_as_bytes_like(obj) }? {
        Some(src) => Ok(unsafe { bytesobject::bytes_like_data(src) }.to_vec()),
        _ => Err(crate::PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            crate::baseobjspace::object_functionstr_type_name(obj)
        ))),
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

/// The `sep` / `bytes_per_sep` separator slots of `hexlify` / `b2a_hex`,
/// validated exactly as the C accelerator: length-1, ASCII.
fn sep_args(
    w_sep: PyObjectRef,
    w_bytes_per_sep: PyObjectRef,
) -> Result<(Option<u8>, isize), crate::PyError> {
    let sep = if w_sep.is_null() || unsafe { is_none(w_sep) } {
        None
    } else {
        let bytes = as_bytes(w_sep)?;
        if bytes.len() != 1 {
            return Err(crate::PyError::value_error("sep must be length 1."));
        }
        if !bytes[0].is_ascii() {
            return Err(crate::PyError::value_error("sep must be ASCII."));
        }
        Some(bytes[0])
    };
    let bytes_per_sep = if w_bytes_per_sep.is_null() || unsafe { is_none(w_bytes_per_sep) } {
        1
    } else {
        crate::builtins::space_index_w(w_bytes_per_sep)? as isize
    };
    Ok((sep, bytes_per_sep))
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
        // `(data, sep=<unrepresentable>, bytes_per_sep=1)` — three
        // positional-or-keyword slots.
        "b2a_hex" / * = |args| {
            let scope = crate::builtins::bind_builtin_kwargs(
                args, &["data", "sep", "bytes_per_sep"], &[true, false, false], "b2a_hex")?;
            let data = as_buffer_bytes(scope[0])?;
            let (sep, bytes_per_sep) = sep_args(scope[1], scope[2])?;
            Ok(w_bytes_from_bytes(&transforms::hexlify(&data, sep, bytes_per_sep)))
        },
        "hexlify" / * = |args| {
            let scope = crate::builtins::bind_builtin_kwargs(
                args, &["data", "sep", "bytes_per_sep"], &[true, false, false], "hexlify")?;
            let data = as_buffer_bytes(scope[0])?;
            let (sep, bytes_per_sep) = sep_args(scope[1], scope[2])?;
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
        // `(data, crc=0, /)` and `(data, crc, /)` — positional-only throughout.
        "crc32" / * = |args| {
            let pos = unpack_positional(args, "crc32", 1, 2)?;
            let data = as_buffer_bytes(pos[0])?;
            let init = match pos.get(1) {
                Some(&o) => crate::baseobjspace::int_w(o)? as u32,
                None => 0,
            };
            Ok(w_int_new(transforms::crc32(&data, init) as i64))
        },
        "crc_hqx" / * = |args| {
            let pos = unpack_positional(args, "crc_hqx", 2, 2)?;
            let data = as_buffer_bytes(pos[0])?;
            let init = crate::baseobjspace::int_w(pos[1])? as u32;
            Ok(w_int_new(transforms::crc_hqx(&data, init) as i64))
        },
        // `(data, /, *, flag=…)` — one positional-only slot, the rest
        // keyword-only.
        "a2b_base64" / * = |args| {
            let (w_data, kwargs) = arg_data_posonly(args, "a2b_base64", &["strict_mode"])?;
            let data = as_bytes(w_data)?;
            let strict_mode = kwonly_bool(kwargs, "strict_mode", false);
            let out = transforms::a2b_base64(&data, strict_mode).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
        "b2a_base64" / * = |args| {
            let (w_data, kwargs) = arg_data_posonly(args, "b2a_base64", &["newline"])?;
            let data = as_buffer_bytes(w_data)?;
            let newline = kwonly_bool(kwargs, "newline", true);
            Ok(w_bytes_from_bytes(&transforms::b2a_base64(&data, newline)))
        },
        // `(data, header=False)` / `(data, quotetabs, istext, header)` — every
        // slot positional-or-keyword.
        "a2b_qp" / * = |args| {
            let scope = crate::builtins::bind_builtin_kwargs(
                args, &["data", "header"], &[true, false], "a2b_qp")?;
            let data = as_bytes(scope[0])?;
            let header = slot_bool(scope[1], false);
            Ok(w_bytes_from_bytes(&transforms::a2b_qp(&data, header)))
        },
        "b2a_qp" / * = |args| {
            let scope = crate::builtins::bind_builtin_kwargs(
                args,
                &["data", "quotetabs", "istext", "header"],
                &[true, false, false, false],
                "b2a_qp",
            )?;
            let data = as_buffer_bytes(scope[0])?;
            let quotetabs = slot_bool(scope[1], false);
            let istext = slot_bool(scope[2], true);
            let header = slot_bool(scope[3], false);
            Ok(w_bytes_from_bytes(&transforms::b2a_qp(&data, quotetabs, istext, header)))
        },
        "a2b_uu" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let out = transforms::a2b_uu(&data).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
        "b2a_uu" / * = |args| {
            let (w_data, kwargs) = arg_data_posonly(args, "b2a_uu", &["backtick"])?;
            let data = as_buffer_bytes(w_data)?;
            let backtick = kwonly_bool(kwargs, "backtick", false);
            let out = transforms::b2a_uu(&data, backtick).map_err(transform_error)?;
            Ok(w_bytes_from_bytes(&out))
        },
    },
}
