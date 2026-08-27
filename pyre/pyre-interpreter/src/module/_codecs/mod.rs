//! _codecs module — PyPy: `pypy/module/_codecs/`.
//!
//! Text codecs (`encode` / `decode`) delegate to `str.encode` /
//! `bytes.decode`, which cover `PyCodec_Encode` / `PyCodec_Decode` for the
//! text path. The codec registry (`register` / `lookup`) and error
//! handlers remain stubs; binary transform codecs are not modelled.

use std::sync::atomic::{AtomicPtr, Ordering};

use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

/// `_PyArg_BadArgument` — how a clinic-converted argument is reported.  A
/// one-argument function names no position; `None` names itself where every
/// other value names its type.
fn bad_arg(fname: &str, pos: Option<usize>, want: &str, w_obj: PyObjectRef) -> crate::PyError {
    let at = match pos {
        Some(n) => format!(" {n}"),
        None => String::new(),
    };
    crate::PyError::type_error(format!(
        "{fname}() argument{at} must be {want}, not {}",
        crate::type_methods::clinic_arg_type_name(w_obj)
    ))
}

/// The `Py_buffer` converter's own wording, which names neither the function
/// nor the position and quotes the type it was handed.
fn bad_buffer_arg(w_obj: PyObjectRef) -> crate::PyError {
    crate::PyError::type_error(format!(
        "a bytes-like object is required, not '{}'",
        crate::type_methods::arg_type_name(w_obj)
    ))
}

/// `str(accept={str, NoneType})` — a codec's handler name, which `None`
/// spells as `strict`.
fn codec_errors_arg(
    fname: &str,
    pos: usize,
    w_errors: PyObjectRef,
) -> Result<String, crate::PyError> {
    if unsafe { pyre_object::is_none(w_errors) } {
        Ok("strict".to_string())
    } else if unsafe { is_str(w_errors) } {
        Ok(crate::baseobjspace::str_utf8_w(w_errors)?.to_string())
    } else {
        Err(bad_arg(fname, Some(pos), "str or None", w_errors))
    }
}

struct CodecState {
    codec_search_path: PyObjectRef,
    codec_search_cache: PyObjectRef,
    codec_error_registry: PyObjectRef,
    codec_need_encodings: bool,
}

impl CodecState {
    fn new() -> Self {
        let mut state = Self {
            codec_search_path: w_list_new(Vec::new()),
            codec_search_cache: w_dict_new(),
            codec_error_registry: w_dict_new(),
            codec_need_encodings: true,
        };
        register_builtin_error_handlers(&mut state);
        state
    }
}

/// The registry is `space.fromcache(CodecState)` upstream — one instance per
/// interpreter, not per thread — so `codecs.register` / `register_error` and
/// the search cache stay visible to every reader.  Published once and never
/// replaced, which is what lets the collector read the slot without
/// coordinating with a mutator that is inside `with_codec_state`.
static CODEC_STATE: AtomicPtr<CodecState> = AtomicPtr::new(std::ptr::null_mut());

fn with_codec_state<R>(f: impl FnOnce(&mut CodecState) -> R) -> R {
    let mut ptr = CODEC_STATE.load(Ordering::Acquire);
    if ptr.is_null() {
        // `CodecState::new` allocates the registry objects and registers the
        // builtin error handlers, so build it before publishing the slot.
        let created = Box::into_raw(Box::new(CodecState::new()));
        ptr = match CODEC_STATE.compare_exchange(
            std::ptr::null_mut(),
            created,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => created,
            Err(existing) => {
                drop(unsafe { Box::from_raw(created) });
                existing
            }
        };
    }
    f(unsafe { &mut *ptr })
}

/// Forward the registry's Python objects.  Upstream reaches the same list and
/// dicts through the space's object graph; pyre holds them off-heap, so the
/// collector has to be handed them.
pub(crate) fn walk_codec_state_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let ptr = CODEC_STATE.load(Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    // A collection can run while the mutator is inside `with_codec_state`, so
    // reach the slots through raw pointers instead of a second `&mut`.
    unsafe {
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).codec_search_path));
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).codec_search_cache));
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).codec_error_registry));
    }
}

// PyPy `interp_codecs.py normalize`.
fn normalize(encoding: &str) -> String {
    let mut chars = String::new();
    let mut punct = false;
    for c in encoding.chars() {
        if c.is_alphanumeric() || c == '.' {
            if punct && !chars.is_empty() {
                chars.push('_');
            }
            if c.is_ascii() {
                chars.push(c.to_ascii_lowercase());
            }
            punct = false;
        } else {
            punct = true;
        }
    }
    chars
}

fn is_callable(obj: PyObjectRef) -> bool {
    // interp_codecs.py:151/663 `space.is_true(space.callable(...))`.
    !obj.is_null() && crate::baseobjspace::callable_w(obj)
}

struct CodecException {
    w_exc: PyObjectRef,
    w_obj: PyObjectRef,
    w_end: PyObjectRef,
    start: usize,
    end: usize,
    kind: Option<pyre_object::interp_exceptions::ExcKind>,
}

fn check_exception(w_exc: PyObjectRef) -> Result<CodecException, crate::PyError> {
    let map_attr_error = |err: crate::PyError| {
        if err.kind == crate::PyErrorKind::AttributeError {
            crate::PyError::type_error("wrong exception")
        } else {
            err
        }
    };
    let w_start = crate::baseobjspace::getattr_str(w_exc, "start").map_err(map_attr_error)?;
    let w_end = crate::baseobjspace::getattr_str(w_exc, "end").map_err(map_attr_error)?;
    let w_obj = crate::baseobjspace::getattr_str(w_exc, "object").map_err(map_attr_error)?;
    let start_i64 = crate::baseobjspace::int_w(w_start)?;
    let end_i64 = crate::baseobjspace::int_w(w_end)?;
    if end_i64 - start_i64 < 0
        || !(unsafe { crate::baseobjspace::isinstance_str_w(w_obj) }
            || unsafe { crate::baseobjspace::isinstance_bytes_w(w_obj) })
    {
        return Err(crate::PyError::type_error("wrong exception"));
    }
    let kind = if unsafe { pyre_object::is_exception(w_exc) } {
        Some(unsafe { pyre_object::interp_exceptions::w_exception_get_kind(w_exc) })
    } else {
        None
    };
    // Bounds are clamped like the C accessors so Rust slicing stays in range.
    let start = start_i64.max(0) as usize;
    let end = end_i64.max(start_i64.max(0)) as usize;
    Ok(CodecException {
        w_exc,
        w_obj,
        w_end,
        start,
        end,
        kind,
    })
}

fn codec_error_arg(args: &[PyObjectRef]) -> Result<CodecException, crate::PyError> {
    args.first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("error handler requires an exception"))
        .and_then(check_exception)
}

fn codec_result(replacement: PyObjectRef, position: PyObjectRef) -> PyObjectRef {
    w_tuple_new(vec![replacement, position])
}

fn strict_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    if unsafe { pyre_object::is_exception(exc.w_exc) } {
        Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) })
    } else {
        Err(crate::PyError::type_error(
            "codec must pass exception instance",
        ))
    }
}

fn ignore_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    Ok(codec_result(w_str_new(""), exc.w_end))
}

fn error_codepoints(exc: &CodecException) -> Result<Vec<u32>, crate::PyError> {
    if !unsafe { crate::baseobjspace::isinstance_str_w(exc.w_obj) } {
        return Err(crate::PyError::type_error(
            "don't know how to handle exception in error callback",
        ));
    }
    Ok(unsafe { w_str_get_wtf8(exc.w_obj) }
        .code_points()
        .skip(exc.start)
        .take(exc.end.saturating_sub(exc.start))
        .map(|cp| cp.to_u32())
        .collect())
}

fn raw_unicode_escape(code: u32) -> String {
    if code >= 0x10000 {
        format!("\\U{code:08x}")
    } else if code >= 0x100 {
        format!("\\u{code:04x}")
    } else {
        format!("\\x{code:02x}")
    }
}

fn replace_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    let size = exc.end - exc.start;
    let replacement = match exc.kind {
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) => "?".repeat(size),
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError) => "\u{fffd}".to_string(),
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError) => {
            "\u{fffd}".repeat(size)
        }
        _ => {
            return Err(crate::PyError::type_error(
                "don't know how to handle exception in error callback",
            ));
        }
    };
    Ok(codec_result(w_str_new(&replacement), exc.w_end))
}

fn xmlcharrefreplace_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    if exc.kind != Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) {
        return Err(crate::PyError::type_error(
            "don't know how to handle exception in error callback",
        ));
    }
    let replacement: String = error_codepoints(&exc)?
        .into_iter()
        .map(|code| format!("&#{code};"))
        .collect();
    Ok(codec_result(w_str_new(&replacement), exc.w_end))
}

fn backslashreplace_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    let replacement = match exc.kind {
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError)
        | Some(pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError) => {
            error_codepoints(&exc)?
                .into_iter()
                .map(raw_unicode_escape)
                .collect::<String>()
        }
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError) => {
            if !unsafe { pyre_object::is_bytes(exc.w_obj) } {
                return Err(crate::PyError::type_error("wrong exception"));
            }
            let data = unsafe { w_bytes_data(exc.w_obj) };
            let end = exc.end.min(data.len());
            let start = exc.start.min(end);
            data[start..end]
                .iter()
                .map(|&byte| raw_unicode_escape(byte as u32))
                .collect::<String>()
        }
        _ => {
            return Err(crate::PyError::type_error(
                "don't know how to handle exception in error callback",
            ));
        }
    };
    Ok(codec_result(w_str_new(&replacement), exc.w_end))
}

fn namereplace_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    if exc.kind != Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) {
        return Err(crate::PyError::type_error(
            "don't know how to handle exception in error callback",
        ));
    }
    let mut replacement = String::new();
    for code in error_codepoints(&exc)? {
        if let Some(name) =
            char::from_u32(code).and_then(crate::module::unicodedata::character_name)
        {
            replacement.push_str("\\N{");
            replacement.push_str(&name);
            replacement.push('}');
        } else {
            replacement.push_str(&raw_unicode_escape(code));
        }
    }
    Ok(codec_result(w_str_new(&replacement), exc.w_end))
}

#[derive(Clone, Copy)]
enum StandardEncoding {
    Utf8,
    Utf16Le,
    Utf16Be,
    Utf32Le,
    Utf32Be,
}

fn standard_encoding(name: &str) -> Option<(usize, StandardEncoding)> {
    let compact: String = name
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .flat_map(char::to_lowercase)
        .collect();
    match compact.as_str() {
        "utf8" | "cputf8" => Some((3, StandardEncoding::Utf8)),
        "utf16le" => Some((2, StandardEncoding::Utf16Le)),
        "utf16be" => Some((2, StandardEncoding::Utf16Be)),
        "utf16" if cfg!(target_endian = "little") => Some((2, StandardEncoding::Utf16Le)),
        "utf16" => Some((2, StandardEncoding::Utf16Be)),
        "utf32le" => Some((4, StandardEncoding::Utf32Le)),
        "utf32be" => Some((4, StandardEncoding::Utf32Be)),
        "utf32" if cfg!(target_endian = "little") => Some((4, StandardEncoding::Utf32Le)),
        "utf32" => Some((4, StandardEncoding::Utf32Be)),
        _ => None,
    }
}

fn exception_encoding(exc: &CodecException) -> Result<(usize, StandardEncoding), crate::PyError> {
    let w_encoding = crate::baseobjspace::getattr_str(exc.w_exc, "encoding")?;
    if !unsafe { is_str(w_encoding) } {
        return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
    }
    standard_encoding(crate::baseobjspace::str_utf8_w(w_encoding)?)
        .ok_or_else(|| unsafe { crate::PyError::from_exc_object(exc.w_exc) })
}

fn surrogatepass_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    match exc.kind {
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) => {
            let (_byte_len, encoding) = exception_encoding(&exc)?;
            let mut replacement = Vec::new();
            for code in error_codepoints(&exc)? {
                if !(0xD800..=0xDFFF).contains(&code) {
                    return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
                }
                match encoding {
                    StandardEncoding::Utf8 => replacement.extend_from_slice(&[
                        0xe0 | (code >> 12) as u8,
                        0x80 | ((code >> 6) & 0x3f) as u8,
                        0x80 | (code & 0x3f) as u8,
                    ]),
                    StandardEncoding::Utf16Le => {
                        replacement.extend_from_slice(&(code as u16).to_le_bytes())
                    }
                    StandardEncoding::Utf16Be => {
                        replacement.extend_from_slice(&(code as u16).to_be_bytes())
                    }
                    StandardEncoding::Utf32Le => replacement.extend_from_slice(&code.to_le_bytes()),
                    StandardEncoding::Utf32Be => replacement.extend_from_slice(&code.to_be_bytes()),
                }
            }
            Ok(codec_result(w_bytes_from_bytes(&replacement), exc.w_end))
        }
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError) => {
            let (byte_len, encoding) = exception_encoding(&exc)?;
            if !unsafe { pyre_object::is_bytes(exc.w_obj) } {
                return Err(crate::PyError::type_error("wrong exception"));
            }
            let data = unsafe { w_bytes_data(exc.w_obj) };
            if exc.start + byte_len > data.len() {
                return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
            }
            let bytes = &data[exc.start..exc.start + byte_len];
            let code = match encoding {
                StandardEncoding::Utf8 => {
                    if bytes[0] & 0xf0 != 0xe0 || bytes[1] & 0xc0 != 0x80 || bytes[2] & 0xc0 != 0x80
                    {
                        0
                    } else {
                        (((bytes[0] & 0x0f) as u32) << 12)
                            | (((bytes[1] & 0x3f) as u32) << 6)
                            | (bytes[2] & 0x3f) as u32
                    }
                }
                StandardEncoding::Utf16Le => u16::from_le_bytes(bytes.try_into().unwrap()) as u32,
                StandardEncoding::Utf16Be => u16::from_be_bytes(bytes.try_into().unwrap()) as u32,
                StandardEncoding::Utf32Le => u32::from_le_bytes(bytes.try_into().unwrap()),
                StandardEncoding::Utf32Be => u32::from_be_bytes(bytes.try_into().unwrap()),
            };
            if !(0xD800..=0xDFFF).contains(&code) {
                return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
            }
            let mut replacement = Wtf8Buf::new();
            replacement.push(CodePoint::from_u32(code).unwrap());
            Ok(codec_result(
                w_str_from_wtf8(replacement),
                w_int_new((exc.start + byte_len) as i64),
            ))
        }
        _ => Err(crate::PyError::type_error(
            "don't know how to handle exception in error callback",
        )),
    }
}

fn surrogateescape_errors(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = codec_error_arg(args)?;
    match exc.kind {
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError) => {
            let mut replacement = Vec::new();
            for code in error_codepoints(&exc)? {
                if !(0xDC80..=0xDCFF).contains(&code) {
                    return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
                }
                replacement.push((code - 0xDC00) as u8);
            }
            Ok(codec_result(w_bytes_from_bytes(&replacement), exc.w_end))
        }
        Some(pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError) => {
            if !unsafe { pyre_object::is_bytes(exc.w_obj) } {
                return Err(crate::PyError::type_error("wrong exception"));
            }
            let data = unsafe { w_bytes_data(exc.w_obj) };
            let mut replacement = Wtf8Buf::new();
            let mut consumed = 0usize;
            while consumed < 4
                && exc.start + consumed < exc.end
                && exc.start + consumed < data.len()
            {
                let byte = data[exc.start + consumed];
                if byte < 128 {
                    break;
                }
                replacement.push(CodePoint::from_u32(0xDC00 + byte as u32).unwrap());
                consumed += 1;
            }
            if consumed == 0 {
                return Err(unsafe { crate::PyError::from_exc_object(exc.w_exc) });
            }
            Ok(codec_result(
                w_str_from_wtf8(replacement),
                w_int_new((exc.start + consumed) as i64),
            ))
        }
        _ => Err(crate::PyError::type_error(
            "don't know how to handle exception in error callback",
        )),
    }
}

fn register_builtin_error_handlers(state: &mut CodecState) {
    let handlers: [(
        &str,
        fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
    ); 8] = [
        ("strict", strict_errors),
        ("ignore", ignore_errors),
        ("replace", replace_errors),
        ("xmlcharrefreplace", xmlcharrefreplace_errors),
        ("backslashreplace", backslashreplace_errors),
        ("surrogateescape", surrogateescape_errors),
        ("surrogatepass", surrogatepass_errors),
        ("namereplace", namereplace_errors),
    ];
    for (name, handler) in handlers {
        let w_handler = crate::make_builtin_function_with_arity(name, handler, 1);
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str(
                state.codec_error_registry,
                name,
                w_handler,
            );
        }
    }
}

/// `interp_codecs.py lookup_error`.  The direct codec loops implement
/// the eight built-ins themselves; custom handlers live in the same registry
/// dict PyPy uses and are returned verbatim.
pub(crate) fn validate_error_handler(errors: &str) -> Result<(), crate::PyError> {
    let found = with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(state.codec_error_registry, errors)
    });
    if found.is_some() {
        Ok(())
    } else {
        Err(crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown error handler name '{errors}'"),
        ))
    }
}

pub(crate) fn lookup_registered_error(errors: &str) -> Option<PyObjectRef> {
    with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(state.codec_error_registry, errors)
    })
}

fn lookup_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_errors) = args.first().copied() else {
        return Err(crate::PyError::type_error(
            "lookup_error() missing argument",
        ));
    };
    if !unsafe { is_str(w_errors) } {
        return Err(bad_arg("lookup_error", None, "str", w_errors));
    }
    let errors = crate::baseobjspace::str_utf8_w(w_errors)?;
    if let Some(w_handler) = with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(state.codec_error_registry, errors)
    }) {
        return Ok(w_handler);
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::LookupError,
        format!("unknown error handler name '{errors}'"),
    ))
}

fn register_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (Some(w_errors), Some(w_handler)) = (args.first().copied(), args.get(1).copied()) else {
        return Err(crate::PyError::type_error(
            "register_error() requires name and handler",
        ));
    };
    if !unsafe { is_str(w_errors) } {
        return Err(bad_arg("register_error", Some(1), "str", w_errors));
    }
    if !is_callable(w_handler) {
        return Err(crate::PyError::type_error("handler must be callable"));
    }
    let errors = crate::baseobjspace::str_utf8_w(w_errors)?;
    with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            state.codec_error_registry,
            errors,
            w_handler,
        );
    });
    Ok(w_none())
}

/// `_codecs__unregister_error` — drop a handler previously installed by
/// `register_error`.  Returns whether a handler of that name was registered,
/// so removing an unknown name is not an error.  The eight handlers installed
/// by [`register_builtin_error_handlers`] are refused: the codec loops reach
/// them by name, so un-registering one would leave those lookups unanswerable.
fn unregister_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_errors) = args.first().copied() else {
        return Err(crate::PyError::type_error(
            "_unregister_error() missing argument",
        ));
    };
    if !unsafe { is_str(w_errors) } {
        return Err(bad_arg("_unregister_error", None, "str", w_errors));
    }
    let errors = crate::baseobjspace::str_utf8_w(w_errors)?;
    if matches!(
        errors,
        "strict"
            | "ignore"
            | "replace"
            | "xmlcharrefreplace"
            | "backslashreplace"
            | "surrogateescape"
            | "surrogatepass"
            | "namereplace"
    ) {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("cannot un-register built-in error handler '{errors}'"),
        ));
    }
    let removed = with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_delitem_str(state.codec_error_registry, errors)
    });
    Ok(w_bool_from(removed))
}

fn register_codec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_search_function) = args.first().copied() else {
        return Err(crate::PyError::type_error("register() missing argument"));
    };
    if !is_callable(w_search_function) {
        return Err(crate::PyError::type_error("argument must be callable"));
    }
    // PyPy `interp_codecs.py register_codec`.
    with_codec_state(|state| unsafe {
        pyre_object::listobject::w_list_append(state.codec_search_path, w_search_function);
    });
    Ok(w_none())
}

fn unregister(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_search_function) = args.first().copied() else {
        return Err(crate::PyError::type_error("unregister() missing argument"));
    };
    // PyPy `interp_codecs.py unregister`: remove and clear cache;
    // return -1 when the search function was not present.
    with_codec_state(|state| {
        match crate::listobject::w_list_remove(state.codec_search_path, w_search_function) {
            Ok(()) => {
                unsafe { pyre_object::dictmultiobject::w_dict_clear(state.codec_search_cache) };
                Ok(w_int_new(0))
            }
            Err(_) => Ok(w_int_new(-1)),
        }
    })
}

fn ensure_encodings_imported(state: &mut CodecState) -> Result<(), crate::PyError> {
    if !state.codec_need_encodings {
        return Ok(());
    }
    // PyPy `_lookup_codec_loop`: import encodings once so it can register
    // `encodings.search_function` through this module's register().
    let ec = crate::call::getexecutioncontext();
    crate::importing::importhook("encodings", w_none(), w_none(), 0, ec)?;
    let _ = crate::importing::importhook("encodings.utf_8", w_none(), w_none(), 0, ec);
    state.codec_need_encodings = false;
    if unsafe { pyre_object::w_list_len(state.codec_search_path) } == 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::LookupError,
            "no codec search functions registered: can't find encoding",
        ));
    }
    Ok(())
}

fn lookup_codec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_encoding) = args.first().copied() else {
        return Err(crate::PyError::type_error("lookup() missing encoding"));
    };
    if !unsafe { is_str(w_encoding) } {
        return Err(bad_arg("lookup", None, "str", w_encoding));
    }
    let encoding = crate::baseobjspace::str_utf8_w(w_encoding)?.to_string();
    // PyPy's `space.text0_w` gateway rejects this before normalization.  A
    // NUL must not be folded away into a valid codec name.
    if encoding.contains('\0') {
        return Err(crate::PyError::value_error("embedded null character"));
    }
    let normalized_encoding = normalize(&encoding);

    with_codec_state(|state| {
        if let Some(w_result) = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(
                state.codec_search_cache,
                &normalized_encoding,
            )
        } {
            return Ok(w_result);
        }

        ensure_encodings_imported(state)?;
        let w_v = w_str_new(&normalized_encoding);
        let n = unsafe { pyre_object::w_list_len(state.codec_search_path) };
        for i in 0..n {
            let Some(w_search) =
                (unsafe { pyre_object::w_list_getitem(state.codec_search_path, i as i64) })
            else {
                continue;
            };
            let w_result = crate::call::call_function_impl_result(w_search, &[w_v])?;
            if unsafe { pyre_object::is_none(w_result) } {
                continue;
            }
            if !unsafe { pyre_object::is_tuple(w_result) }
                || unsafe { pyre_object::w_tuple_len(w_result) } != 4
            {
                return Err(crate::PyError::type_error(
                    "codec search functions must return 4-tuples",
                ));
            }
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(
                    state.codec_search_cache,
                    &normalized_encoding,
                    w_result,
                );
            }
            return Ok(w_result);
        }
        Err(crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown encoding: {encoding}"),
        ))
    })
}

pub(crate) fn lookup_text_codec(
    action: &str,
    encoding: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let w_codec_info = lookup_codec(&[w_str_new(encoding)])?;
    match crate::baseobjspace::getattr_str(w_codec_info, "_is_text_encoding") {
        Ok(w_flag) if !crate::baseobjspace::is_true(w_flag)? => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::LookupError,
                format!(
                    "'{encoding}' is not a text encoding; use codecs.{action}() to handle arbitrary codecs"
                ),
            ));
        }
        Ok(_) => {}
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    Ok(w_codec_info)
}

fn call_codec(
    w_coder: PyObjectRef,
    w_obj: PyObjectRef,
    action: &str,
    encoding: &str,
    errors: Option<&str>,
) -> Result<PyObjectRef, crate::PyError> {
    // PyPy `interp_codecs.py _call_codec`.
    let call = if let Some(errors) = errors {
        crate::call::call_function_impl_result(w_coder, &[w_obj, w_str_new(errors)])
    } else {
        crate::call::call_function_impl_result(w_coder, &[w_obj])
    };
    // A codec that raises gets one line of context naming the operation and
    // the encoding, and is re-raised otherwise unchanged:
    //
    //     _PyErr_FormatNote("%s with '%s' codec failed", "encoding", encoding);
    //
    // `interp_codecs.py _wrap_codec_error` instead rebuilds the error with the
    // original as `__cause__`, which is the pre-3.12 spelling of the same
    // context — `gh-102406` replaced the chaining with a PEP 678 note, and
    // pyre targets 3.14.  The message text is the one both produce.
    let w_res = match call {
        Ok(w_res) => w_res,
        Err(mut operr) => {
            crate::baseobjspace::add_internal_exception_note(
                &mut operr,
                &format!("{action} with '{encoding}' codec failed"),
            )?;
            return Err(operr);
        }
    };
    if !unsafe { pyre_object::is_tuple(w_res) } || unsafe { pyre_object::w_tuple_len(w_res) } != 2 {
        // The two messages are spelled differently upstream -- the decoder's
        // has no space after the comma -- and the difference is observable.
        let msg = if action.starts_with("en") {
            "encoder must return a tuple (object, integer)".to_string()
        } else if action.starts_with("de") {
            "decoder must return a tuple (object,integer)".to_string()
        } else {
            format!("{action} must return a tuple (object, integer)")
        };
        return Err(crate::PyError::type_error(msg));
    }
    Ok(unsafe { pyre_object::w_tuple_getitem(w_res, 0).unwrap_or_else(w_none) })
}

/// `PyCodec_Encode` / `PyCodec_Decode` — the arbitrary-object entry points.
///
/// Unlike `str.encode` / `bytes.decode` these place no restriction on either
/// side: the codec is looked up without the text-encoding test and its coder is
/// handed the object as it stands, so a codec answering with something other
/// than `bytes` (or `str`) is answered with, rather than reported.  An `errors`
/// argument that was not supplied is not invented either -- the coder is called
/// with one argument, which is what lets a coder taking only the object work.
fn codec_encode_or_decode(
    w_obj: PyObjectRef,
    w_encoding: PyObjectRef,
    w_errors: Option<PyObjectRef>,
    encode: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let name = if encode { "encode" } else { "decode" };
    if !unsafe { is_str(w_encoding) } {
        return Err(crate::PyError::type_error(format!(
            "{name}() argument 'encoding' must be str, not {}",
            crate::type_methods::clinic_arg_type_name(w_encoding)
        )));
    }
    // Only an omitted argument is absent: `None` is a value the coder is not
    // asked to accept, so it is refused the way any other non-`str` is.
    let errors = match w_errors {
        None => None,
        Some(w_errors) if unsafe { is_str(w_errors) } => {
            Some(crate::baseobjspace::str_utf8_w(w_errors)?.to_string())
        }
        Some(w_errors) => {
            return Err(crate::PyError::type_error(format!(
                "{name}() argument 'errors' must be str, not {}",
                crate::type_methods::clinic_arg_type_name(w_errors)
            )));
        }
    };
    // The lookup runs Python -- an uncached name imports its `encodings`
    // module and calls every registered search function -- so the object
    // outlives it on the shadow stack rather than in a plain local.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let encoding = crate::baseobjspace::str_utf8_w(w_encoding)?.to_string();
    let w_codec_info = lookup_codec(&[w_str_new(&encoding)])?;
    let w_coder = unsafe {
        pyre_object::w_tuple_getitem(w_codec_info, i64::from(!encode)).unwrap_or_else(w_none)
    };
    call_codec(
        w_coder,
        pyre_object::gc_roots::shadow_stack_get(sp),
        if encode { "encoding" } else { "decoding" },
        &encoding,
        errors.as_deref(),
    )
}

pub(crate) fn encode_text_codec(
    w_obj: PyObjectRef,
    encoding: &str,
    errors: &str,
) -> Result<PyObjectRef, crate::PyError> {
    // Rooted for the same window as `decode_text_codec`: the lookup runs
    // Python while `w_obj` is still whatever the caller handed over.
    let _roots = pyre_object::gc_roots::push_roots();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    let w_codec_info = lookup_text_codec("encode", encoding)?;
    if crate::importing::dev_mode_flag() {
        validate_error_handler(errors)?;
    }
    let w_encfunc = unsafe { pyre_object::w_tuple_getitem(w_codec_info, 0).unwrap_or_else(w_none) };
    let w_retval = call_codec(w_encfunc, w_obj, "encoding", encoding, Some(errors))?;
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_retval) } {
        let tname = unsafe { pyre_object::type_name_of(w_retval) };
        return Err(crate::PyError::type_error(format!(
            "'{encoding}' encoder returned '{tname}' instead of 'bytes'; use codecs.encode() to encode to arbitrary types"
        )));
    }
    Ok(w_retval)
}

pub(crate) fn decode_text_codec(
    w_obj: PyObjectRef,
    encoding: &str,
    errors: &str,
) -> Result<PyObjectRef, crate::PyError> {
    // The lookup runs Python: an uncached name imports its `encodings` module
    // and calls every registered search function, and only `call_codec` below
    // publishes `w_obj` as a root.  Callers reach here with a value that has
    // no heap edge — `decode_bytes_to_wtf8` copies the source bytes into a
    // fresh one — and old-gen buys immobility, not survival, so the copy is
    // swept mid-lookup and the decoder is handed a reused box.  Bytes do not
    // move, so the pin is for liveness alone and the value is used as it is.
    let _roots = pyre_object::gc_roots::push_roots();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    let w_codec_info = lookup_text_codec("decode", encoding)?;
    if crate::importing::dev_mode_flag() {
        validate_error_handler(errors)?;
    }
    let w_decfunc = unsafe { pyre_object::w_tuple_getitem(w_codec_info, 1).unwrap_or_else(w_none) };
    let w_retval = call_codec(w_decfunc, w_obj, "decoding", encoding, Some(errors))?;
    if !unsafe { pyre_object::is_str(w_retval) } {
        let tname = unsafe { pyre_object::type_name_of(w_retval) };
        return Err(crate::PyError::type_error(format!(
            "'{encoding}' decoder returned '{tname}' instead of 'str'; use codecs.decode() to decode to arbitrary types"
        )));
    }
    Ok(w_retval)
}

fn forget_codec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_encoding) = args.first().copied() else {
        return Ok(w_none());
    };
    if !unsafe { is_str(w_encoding) } {
        return Ok(w_none());
    }
    let normalized_encoding = normalize(crate::baseobjspace::str_utf8_w(w_encoding)?);
    with_codec_state(|state| {
        let w_cache = state.codec_search_cache;
        let w_key = w_str_new(&normalized_encoding);
        if unsafe { pyre_object::dictmultiobject::w_dict_lookup(w_cache, w_key).is_some() } {
            let _ = crate::baseobjspace::delitem(w_cache, w_key);
        }
    });
    Ok(w_none())
}

fn encode_with_name(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    fname: &str,
    encoding: &str,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(bad_arg(fname, Some(1), "str", w_obj));
    }
    let errors = codec_errors_arg(fname, 2, errors)?;
    // PyPy `make_encoder_wrapper`: convert to unicode, call unicodehelper
    // encoder, return `(bytes, unicode_length)`.
    let encode_method = crate::baseobjspace::getattr_str(w_obj, "encode")?;
    let encoded = crate::call::call_function_impl_result(
        encode_method,
        &[w_str_new(encoding), w_str_new(&errors)],
    )?;
    Ok(w_tuple_new(vec![
        encoded,
        w_int_new(unsafe { pyre_object::w_str_len(w_obj) } as i64),
    ]))
}

fn decode_with_name(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    fname: &str,
    encoding: &str,
) -> Result<PyObjectRef, crate::PyError> {
    // `make_decoder_wrapper`: decode a bytes buffer and return
    // `(unicode, bytes_consumed)`.  The input is unwrapped with `bufferstr`,
    // which reads any buffer -- a `memoryview` included -- and the decoding
    // itself then runs on the `newbytes` built from what it read.
    let data = decode_input_bytes(w_obj)?;
    let errors = codec_errors_arg(fname, 2, errors)?;
    let consumed = data.len();
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_bytes_from_bytes(&data));
    let decode_method =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(sp), "decode")?;
    let decoded = crate::call::call_function_impl_result(
        decode_method,
        &[w_str_new(encoding), w_str_new(&errors)],
    )?;
    Ok(w_tuple_new(vec![decoded, w_int_new(consumed as i64)]))
}

/// `bufferstr_w`: the read-only bytes of a decoder input.
///
/// The acquisition is `acquire_readbuf`'s single `PyBUF_SIMPLE` export, which
/// every exporter answers, so an `mmap`, a `ctypes` array and an object whose
/// `__buffer__` returns a view all decode alongside `bytes` and `bytearray`.
/// Only `BufferInterfaceNotFound` is spelled here, with the converter's own
/// wording.  A buffer the decoder may not read -- a strided `memoryview`, which
/// the request refuses because the bytes it would decode are not the ones the
/// object exposes -- has already said so with the acquisition's own error, and
/// reporting it as a wrong argument type instead names the wrong thing.
fn decode_input_bytes(w_obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    // The acquisition may run `__buffer__`, so the object naming the failure is
    // read back from the shadow stack rather than from the argument.
    let Some(buffer) = crate::baseobjspace::simple_buffer_bytes(w_obj)? else {
        return Err(bad_buffer_arg(pyre_object::gc_roots::shadow_stack_get(sp)));
    };
    let data = buffer.as_bytes().to_vec();
    buffer.release();
    Ok(data)
}

/// PyPy `interp_codecs.utf_{16,32}_ex_decode`: the three-value entry point
/// used by the stdlib incremental decoders.  Unlike `bytes.decode`, this must
/// leave an incomplete code unit unconsumed while `final` is false.
fn utf16_32_ex_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    byteorder: i64,
    w_final: PyObjectRef,
    is32: bool,
    fname: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let data = decode_input_bytes(w_obj)?;
    let errors = codec_errors_arg(fname, 2, errors)?;
    let fixed_be = match byteorder {
        0 => None,
        -1 => Some(false),
        _ => Some(true),
    };
    let codec = if is32 { "utf32" } else { "utf16" };
    let (decoded, consumed, bo) = crate::type_methods::decode_utf16_32_helper(
        &data,
        is32,
        fixed_be,
        codec,
        &errors,
        crate::baseobjspace::is_true(w_final)?,
    )?;
    Ok(w_tuple_new(vec![
        w_str_from_wtf8_managed(decoded),
        w_int_new(consumed as i64),
        w_int_new(bo as i64),
    ]))
}

/// PyPy `make_decoder_wrapper` for the two-value UTF-16/32 decoder entry
/// points.  These share the helper used by the `*_ex_decode` functions but
/// discard its byte-order result.
fn utf16_32_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    w_final: PyObjectRef,
    is32: bool,
    fixed_be: Option<bool>,
    codec: &str,
    fname: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let data = decode_input_bytes(w_obj)?;
    let errors = codec_errors_arg(fname, 2, errors)?;
    let (decoded, consumed, _) = crate::type_methods::decode_utf16_32_helper(
        &data,
        is32,
        fixed_be,
        codec,
        &errors,
        crate::baseobjspace::is_true(w_final)?,
    )?;
    Ok(w_tuple_new(vec![
        w_str_from_wtf8_managed(decoded),
        w_int_new(consumed as i64),
    ]))
}

/// PyPy `interp_codecs.utf_8_decode`, including its incremental consumed
/// position.  The stdlib `BufferedIncrementalDecoder` retains
/// `data[consumed:]` for the next call.
fn utf8_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    w_final: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let data = decode_input_bytes(w_obj)?;
    let errors = codec_errors_arg("utf_8_decode", 2, errors)?;
    // `interp_codecs.utf_8_decode`: `surrogatepass` is the one handler that
    // decodes a complete ED A0..BF 80..BF sequence in the state machine and
    // retains an incomplete one for the next chunk.
    let (decoded, consumed) = crate::typedef::decode_utf8_with_errors_incremental(
        &data,
        &errors,
        crate::baseobjspace::is_true(w_final)?,
        errors == "surrogatepass",
    )?;
    Ok(w_tuple_new(vec![
        w_str_from_wtf8_managed(decoded),
        w_int_new(consumed as i64),
    ]))
}

/// `charmapencode_lookup` — map one code point through the encoding table and
/// append its bytes to `out`.
///
/// Answers `false` for a character the table leaves undefined, which a missing
/// key and an explicit `None` value both express; the caller turns a run of
/// those into one error span.  A mapping that raises anything other than
/// `LookupError` / `KeyError` propagates, so a `__getitem__` of its own is not
/// mistaken for "undefined".
fn charmap_output(
    w_mapping: PyObjectRef,
    cp: u32,
    out: &mut Vec<u8>,
) -> Result<bool, crate::PyError> {
    if unsafe { is_str(w_mapping) } && cp as usize >= unsafe { w_str_len(w_mapping) } {
        return Ok(false);
    }
    // Minting the key can collect, and a table read can run a `__getitem__` of
    // its own, so the table is pinned and re-read rather than carried across
    // either in a plain local.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_mapping);
    let w_key = pyre_object::gc_roots::pin_root(w_int_new(cp as i64));
    let w_mapping = pyre_object::gc_roots::shadow_stack_get(sp);
    let w_ch = match crate::baseobjspace::getitem(w_mapping, w_key) {
        Ok(w_ch) => w_ch,
        Err(e)
            if matches!(
                e.kind,
                crate::PyErrorKind::LookupError | crate::PyErrorKind::KeyError
            ) =>
        {
            return Ok(false);
        }
        Err(e) => return Err(e),
    };
    // `Charmap_Encode.get` tests the table's value with `w_bytes`, so a
    // `bytearray` falls through to the type error below with everything else
    // the table is not allowed to give.
    if unsafe { pyre_object::is_bytes(w_ch) } {
        out.extend_from_slice(unsafe { pyre_object::bytesobject::w_bytes_data(w_ch) });
        Ok(true)
    } else if unsafe { pyre_object::is_int(w_ch) } {
        let x = unsafe { pyre_object::w_int_get_value(w_ch) };
        if !(0..256).contains(&x) {
            return Err(crate::PyError::type_error(
                "character mapping must be in range(256)",
            ));
        }
        out.push(x as u8);
        Ok(true)
    } else if unsafe { pyre_object::is_none(w_ch) } {
        Ok(false)
    } else {
        Err(crate::PyError::type_error(
            "character mapping must return integer, bytes or None, not str",
        ))
    }
}

/// `utf8_encode_charmap` — encode through a code-point-to-bytes table.
///
/// Characters the table leaves undefined are gathered into one run and handed
/// to the error handler registered under `errors`, so a handler sees a single
/// span rather than one call per character.  A `str` replacement is mapped
/// through the table in turn — that is what lets `"replace"` emit whatever the
/// table gives for `?` — and a replacement the table cannot encode reports the
/// span that was originally undefined, not the replacement's own position.
fn charmap_encode_impl(
    w_unicode: PyObjectRef,
    errors: PyObjectRef,
    w_mapping: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_none(w_mapping) } {
        return encode_with_name(w_unicode, errors, "charmap_encode", "latin-1");
    }
    if !unsafe { is_str(w_unicode) } {
        return Err(bad_arg("charmap_encode", Some(1), "str", w_unicode));
    }
    // The loop below runs a Python error handler, which can move the string
    // the name was read out of, so the name is owned rather than viewed.
    let errors_s = codec_errors_arg("charmap_encode", 2, errors)?;
    let cps: Vec<u32> = unsafe { w_str_get_wtf8(w_unicode) }
        .code_points()
        .map(|cp| cp.to_u32())
        .collect();
    let char_len = cps.len();
    let mut out = Vec::new();
    // The code points are copied out above, so only the two objects have to
    // survive the collections a table read or a handler call can trigger.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_unicode);
    let _ = pyre_object::gc_roots::pin_root(w_mapping);
    let mut i = 0usize;
    while i < char_len {
        if charmap_output(
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
            cps[i],
            &mut out,
        )? {
            i += 1;
            continue;
        }

        let start = i;
        let mut end = i + 1;
        let mut probe = Vec::new();
        while end < char_len {
            probe.clear();
            if charmap_output(
                pyre_object::gc_roots::shadow_stack_get(sp + 1),
                cps[end],
                &mut probe,
            )? {
                break;
            }
            end += 1;
        }

        match errors_s.as_str() {
            "strict" => {
                return Err(crate::typedef::unicode_encode_error(
                    "charmap",
                    pyre_object::gc_roots::shadow_stack_get(sp),
                    start as i64,
                    end as i64,
                    "character maps to <undefined>",
                ));
            }
            "ignore" => {
                i = end;
            }
            _ => {
                let (replacement, newpos) =
                    crate::type_methods::call_registered_encode_error_handler(
                        &errors_s,
                        "charmap",
                        pyre_object::gc_roots::shadow_stack_get(sp),
                        char_len,
                        start,
                        end,
                        "character maps to <undefined>",
                    )?;
                match replacement {
                    crate::type_methods::EncodeReplacement::Bytes(bytes) => {
                        out.extend_from_slice(&bytes);
                    }
                    crate::type_methods::EncodeReplacement::Str(replacement_cps) => {
                        for replacement_cp in replacement_cps {
                            if !charmap_output(
                                pyre_object::gc_roots::shadow_stack_get(sp + 1),
                                replacement_cp,
                                &mut out,
                            )? {
                                return Err(crate::typedef::unicode_encode_error(
                                    "charmap",
                                    pyre_object::gc_roots::shadow_stack_get(sp),
                                    start as i64,
                                    end as i64,
                                    "character maps to <undefined>",
                                ));
                            }
                        }
                    }
                }
                i = newpos;
            }
        }
    }
    // The reported count is the input's own length, which the handler cannot
    // change however far it moved the resume position.
    let char_count = unsafe { pyre_object::w_str_len(pyre_object::gc_roots::shadow_stack_get(sp)) };
    // Pinned because minting the bytes below can collect and move it.
    let _ = pyre_object::gc_roots::pin_root(w_int_new(char_count as i64));
    let w_encoded = w_bytes_from_bytes(&out);
    Ok(w_tuple_new(vec![
        w_encoded,
        pyre_object::gc_roots::shadow_stack_get(sp + 2),
    ]))
}

/// `charmapdecode_lookup` — read one byte's entry out of a decoding table.
///
/// Answers `None` for a byte the table has no entry for.  Minting the key can
/// collect and a table can carry a `__getitem__` of its own, so the table is
/// pinned across both and re-read rather than carried in a plain local.
fn charmap_decode_lookup(
    w_mapping: PyObjectRef,
    b: u8,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_mapping);
    let w_key = pyre_object::gc_roots::pin_root(w_int_new(b as i64));
    let w_mapping = pyre_object::gc_roots::shadow_stack_get(sp);
    match crate::baseobjspace::getitem(w_mapping, w_key) {
        Ok(w_ch) => Ok(Some(w_ch)),
        Err(e)
            if matches!(
                e.kind,
                crate::PyErrorKind::LookupError | crate::PyErrorKind::KeyError
            ) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

fn charmap_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    w_mapping: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_none(w_mapping) } {
        return decode_with_name(w_obj, errors, "charmap_decode", "latin-1");
    }
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_obj) } {
        return Err(bad_buffer_arg(w_obj));
    }
    // The loop below runs a table's own `__getitem__` and an error handler, so
    // a name viewed out of the string it was read from would not survive it.
    let errors_s = codec_errors_arg("charmap_decode", 2, errors)?;
    // A custom error handler may replace `exc.object`; decoding then resumes
    // from the new bytes (`data`).
    // The input is copied rather than viewed: those same calls can collect, and
    // a slice into the object's live buffer would not survive it moving.
    let mut data: Vec<u8> = unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) }.to_vec();
    // charmap_decode reports the number of input bytes consumed, which stays
    // the original length even if a handler replaces `exc.object`.
    let orig_len = data.len();
    // Only the table has to outlive those calls; a string table's code points
    // are copied out of it up front.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_mapping);
    let mapping_chars: Option<Vec<_>> = if unsafe { is_str(w_mapping) } {
        Some(
            unsafe { w_str_get_wtf8(w_mapping) }
                .code_points()
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];
        let mapped = if let Some(chars) = mapping_chars.as_ref() {
            chars.get(b as usize).copied().map(|cp| {
                let mut one = rustpython_wtf8::Wtf8Buf::new();
                one.push(cp);
                w_str_from_wtf8(one)
            })
        } else {
            charmap_decode_lookup(pyre_object::gc_roots::shadow_stack_get(sp), b)?
        };
        // A mapped char maps to itself unless it signals "undefined" (a missing
        // entry, the `￾` sentinel, or `None`).
        if let Some(w_ch) = mapped {
            if unsafe { is_str(w_ch) } {
                let s = unsafe { w_str_get_wtf8(w_ch) };
                if s.as_bytes() != "\u{FFFE}".as_bytes() {
                    out.push_wtf8(s);
                    i += 1;
                    continue;
                }
            } else if unsafe { pyre_object::is_int(w_ch) } {
                let x = unsafe { pyre_object::w_int_get_value(w_ch) };
                if !(0..=0x10FFFF).contains(&x) {
                    return Err(crate::PyError::type_error(
                        "character mapping must be in range(0x110000)",
                    ));
                }
                // The sentinel says "undefined" whichever way the table spells
                // it, so an integer entry falls through to the error handler
                // exactly as the one-character string does.
                if x != 0xFFFE {
                    out.push(rustpython_wtf8::CodePoint::from_u32(x as u32).unwrap());
                    i += 1;
                    continue;
                }
            } else if !unsafe { pyre_object::is_none(w_ch) } {
                return Err(crate::PyError::type_error(
                    "character mapping must return integer, None or str",
                ));
            }
        }
        // The byte maps to <undefined>: run the decode error handler over the
        // single byte at `i` (`str_decode_charmap` span `pos .. pos + 1`).
        match errors_s.as_str() {
            "ignore" => i += 1,
            "replace" => {
                out.push_char('\u{FFFD}');
                i += 1;
            }
            "backslashreplace" => {
                out.push_str(&format!("\\x{b:02x}"));
                i += 1;
            }
            "xmlcharrefreplace" | "namereplace" => {
                return Err(crate::typedef::decode_error_encode_only_handler());
            }
            "strict" => {
                return Err(crate::typedef::unicode_decode_error(
                    "charmap",
                    &data[..],
                    i,
                    i + 1,
                    "character maps to <undefined>",
                ));
            }
            _ => {
                let (np, nb) = crate::type_methods::call_registered_decode_error_handler(
                    &errors_s,
                    "charmap",
                    &data[..],
                    i,
                    i + 1,
                    "character maps to <undefined>",
                    &mut out,
                )?;
                if let Some(nb) = nb {
                    data = nb;
                }
                i = np;
            }
        }
    }
    Ok(w_tuple_new(vec![
        w_str_from_wtf8_managed(out),
        w_int_new(orig_len as i64),
    ]))
}

fn utf7_is_base64(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'+' || b == b'/'
}

fn utf7_to_base64(n: u32) -> u8 {
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(n & 0x3f) as usize]
}

fn utf7_from_base64(b: u8) -> u32 {
    match b {
        b'a'..=b'z' => (b - 71) as u32,
        b'A'..=b'Z' => (b - 65) as u32,
        b'0'..=b'9' => (b + 4) as u32,
        b'+' => 62,
        _ => 63,
    }
}

fn utf7_decode_direct(b: u8) -> bool {
    b <= 127 && b != b'+'
}

fn utf7_category(oc: u32) -> u8 {
    if oc > 127 {
        return 3;
    }
    let b = oc as u8;
    if matches!(b, b'\t' | b'\n' | b'\r' | b' ') {
        2
    } else if b.is_ascii_alphanumeric() || b"'(),-./:?".contains(&b) {
        0
    } else if b"!\"#$%&*;<=>@[]^_`{|}".contains(&b) {
        1
    } else {
        3
    }
}

fn utf7_encode_direct(oc: u32) -> bool {
    oc < 128 && oc > 0 && utf7_category(oc) != 3
}

fn utf7_encode_unit(out: &mut Vec<u8>, unit: u32, base64bits: &mut u32, base64buffer: &mut u32) {
    *base64bits += 16;
    *base64buffer = (*base64buffer << 16) | unit;
    while *base64bits >= 6 {
        out.push(utf7_to_base64(*base64buffer >> (*base64bits - 6)));
        *base64bits -= 6;
    }
    *base64buffer &= (1 << *base64bits) - 1;
}

fn utf7_encode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(bad_arg("utf_7_encode", Some(1), "str", w_obj));
    }
    let _errors = codec_errors_arg("utf_7_encode", 2, errors)?;
    // PyPy `unicodehelper.py:utf8_encode_utf_7`.
    let mut out = Vec::new();
    let mut in_shift = false;
    let mut base64bits = 0;
    let mut base64buffer = 0;
    for cp in unsafe { w_str_get_wtf8(w_obj) }.code_points() {
        let oc = cp.to_u32();
        if !in_shift {
            if oc == b'+' as u32 {
                out.extend_from_slice(b"+-");
            } else if utf7_encode_direct(oc) {
                out.push(oc as u8);
            } else {
                out.push(b'+');
                in_shift = true;
                if oc >= 0x10000 {
                    utf7_encode_unit(
                        &mut out,
                        0xd800 | ((oc - 0x10000) >> 10),
                        &mut base64bits,
                        &mut base64buffer,
                    );
                    utf7_encode_unit(
                        &mut out,
                        0xdc00 | ((oc - 0x10000) & 0x3ff),
                        &mut base64bits,
                        &mut base64buffer,
                    );
                } else {
                    utf7_encode_unit(&mut out, oc, &mut base64bits, &mut base64buffer);
                }
            }
        } else if utf7_encode_direct(oc) {
            if base64bits != 0 {
                out.push(utf7_to_base64(base64buffer << (6 - base64bits)));
                base64buffer = 0;
                base64bits = 0;
            }
            in_shift = false;
            if utf7_is_base64(oc as u8) || oc == b'-' as u32 {
                out.push(b'-');
            }
            out.push(oc as u8);
        } else if oc >= 0x10000 {
            utf7_encode_unit(
                &mut out,
                0xd800 | ((oc - 0x10000) >> 10),
                &mut base64bits,
                &mut base64buffer,
            );
            utf7_encode_unit(
                &mut out,
                0xdc00 | ((oc - 0x10000) & 0x3ff),
                &mut base64bits,
                &mut base64buffer,
            );
        } else {
            utf7_encode_unit(&mut out, oc, &mut base64bits, &mut base64buffer);
        }
    }
    if base64bits != 0 {
        out.push(utf7_to_base64(base64buffer << (6 - base64bits)));
    }
    if in_shift {
        out.push(b'-');
    }
    Ok(w_tuple_new(vec![
        w_bytes_from_bytes(&out),
        w_int_new(unsafe { pyre_object::w_str_len(w_obj) } as i64),
    ]))
}

/// Route a utf-7 decode error through the requested handler, shaped like
/// `unicode_escape_error`. Returns the resume position and, when a custom
/// handler replaced `exc.object`, the new input bytes to resume from.
fn utf7_decode_error(
    errors: &str,
    original: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    out: &mut rustpython_wtf8::Wtf8Buf,
) -> Result<(usize, Option<Vec<u8>>), crate::PyError> {
    match errors {
        "strict" => Err(crate::typedef::unicode_decode_error(
            "utf7", original, start, end, reason,
        )),
        "ignore" => Ok((end, None)),
        "replace" => {
            out.push_char('\u{FFFD}');
            Ok((end, None))
        }
        "backslashreplace" => {
            for &b in &original[start..end.min(original.len())] {
                out.push_str(&format!("\\x{b:02x}"));
            }
            Ok((end, None))
        }
        "xmlcharrefreplace" | "namereplace" => {
            Err(crate::typedef::decode_error_encode_only_handler())
        }
        _ => crate::type_methods::call_registered_decode_error_handler(
            errors, "utf7", original, start, end, reason, out,
        ),
    }
}

fn utf7_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    is_final: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_obj) } {
        return Err(bad_buffer_arg(w_obj));
    }
    // PyPy `unicodehelper.py:str_decode_utf_7`.
    let errors_s = codec_errors_arg("utf_7_decode", 2, errors)?;
    let errors_s = errors_s.as_str();
    // A custom error handler may replace `exc.object`; decoding then resumes
    // from the new bytes (`data`).
    let mut data: std::borrow::Cow<[u8]> =
        std::borrow::Cow::Borrowed(unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) });
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut pos = 0usize;
    let mut in_shift = false;
    let mut base64bits = 0u32;
    let mut base64buffer = 0u32;
    let mut surrogate = 0u32;
    // Output byte length captured when a shift opened (`shiftOutStartPos`),
    // used to back off an unterminated shift in a non-final chunk.
    let mut shift_out_start = 0usize;
    // Input position of the `+` that opened the current shift, used as the
    // start anchor for its error spans (`startinpos`).
    let mut startinpos = 0usize;
    while pos < data.len() {
        let ch = data[pos];
        if in_shift {
            if utf7_is_base64(ch) {
                base64buffer = (base64buffer << 6) | utf7_from_base64(ch);
                base64bits += 6;
                pos += 1;
                if base64bits >= 16 {
                    let out_ch = base64buffer >> (base64bits - 16);
                    base64bits -= 16;
                    base64buffer &= (1 << base64bits) - 1;
                    if surrogate != 0 {
                        if (0xdc00..=0xdfff).contains(&out_ch) {
                            let code = (((surrogate & 0x3ff) << 10) | (out_ch & 0x3ff)) + 0x10000;
                            out.push(rustpython_wtf8::CodePoint::from_u32(code).unwrap());
                            surrogate = 0;
                            continue;
                        }
                        out.push(rustpython_wtf8::CodePoint::from_u32(surrogate).unwrap());
                        surrogate = 0;
                    }
                    if (0xd800..=0xdbff).contains(&out_ch) {
                        surrogate = out_ch;
                    } else {
                        out.push(rustpython_wtf8::CodePoint::from_u32(out_ch).unwrap());
                    }
                }
            } else {
                // now leaving a base-64 section
                in_shift = false;
                if base64bits >= 6 {
                    // At least one base-64 character was seen but a whole
                    // unit was not: partial character. The terminating byte
                    // is consumed and folded into the error span.
                    pos += 1;
                    let (np, nb) = utf7_decode_error(
                        errors_s,
                        &data[..],
                        startinpos,
                        pos,
                        "partial character in shift sequence",
                        &mut out,
                    )?;
                    if let Some(nb) = nb {
                        data = std::borrow::Cow::Owned(nb);
                    }
                    pos = np;
                    continue;
                } else if base64bits > 0 && base64buffer != 0 {
                    // Leftover bits that should have been zero.
                    pos += 1;
                    let (np, nb) = utf7_decode_error(
                        errors_s,
                        &data[..],
                        startinpos,
                        pos,
                        "non-zero padding bits in shift sequence",
                        &mut out,
                    )?;
                    if let Some(nb) = nb {
                        data = std::borrow::Cow::Owned(nb);
                    }
                    pos = np;
                    continue;
                }
                if surrogate != 0 && utf7_decode_direct(ch) {
                    out.push(rustpython_wtf8::CodePoint::from_u32(surrogate).unwrap());
                }
                surrogate = 0;
                if ch == b'-' {
                    // '-' is absorbed; other terminating characters are preserved.
                    pos += 1;
                }
            }
        } else if ch == b'+' {
            startinpos = pos;
            pos += 1;
            if pos < data.len() && data[pos] == b'-' {
                pos += 1;
                out.push_char('+');
            } else if pos < data.len() && !utf7_is_base64(data[pos]) {
                let (np, nb) = utf7_decode_error(
                    errors_s,
                    &data[..],
                    startinpos,
                    startinpos + 2,
                    "ill-formed sequence",
                    &mut out,
                )?;
                if let Some(nb) = nb {
                    data = std::borrow::Cow::Owned(nb);
                }
                pos = np;
            } else {
                // begin base64-encoded section
                in_shift = true;
                surrogate = 0;
                shift_out_start = out.len();
                base64bits = 0;
                base64buffer = 0;
            }
        } else if utf7_decode_direct(ch) {
            out.push_char(ch as char);
            pos += 1;
        } else {
            startinpos = pos;
            pos += 1;
            let (np, nb) = utf7_decode_error(
                errors_s,
                &data[..],
                startinpos,
                pos,
                "unexpected special character",
                &mut out,
            )?;
            if let Some(nb) = nb {
                data = std::borrow::Cow::Owned(nb);
            }
            pos = np;
        }
    }
    // end of string
    let mut consumed = data.len();
    if in_shift && is_final {
        // in shift sequence with no more input to follow
        in_shift = false;
        if surrogate != 0 || base64bits >= 6 || (base64bits > 0 && base64buffer != 0) {
            // The handler pushes its replacement into `out` itself; the input
            // is fully consumed, so its returned position is not reused.
            let (_np, _nb) = utf7_decode_error(
                errors_s,
                &data[..],
                startinpos,
                pos,
                "unterminated shift sequence",
                &mut out,
            )?;
        }
    } else if in_shift {
        // Non-final chunk ending mid-shift: back off to the '+' that opened it.
        consumed = startinpos;
        out.truncate(shift_out_start);
    }
    Ok(w_tuple_new(vec![
        w_str_from_wtf8_managed(out),
        w_int_new(consumed as i64),
    ]))
}

fn push_ascii_hex_escape(out: &mut Vec<u8>, prefix: u8, cp: u32, digits: usize) {
    out.push(b'\\');
    out.push(prefix);
    for shift in (0..digits).rev() {
        out.push(b"0123456789abcdef"[((cp >> (shift * 4)) & 0xf) as usize]);
    }
}

fn unicode_escape_encode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(bad_arg("unicode_escape_encode", Some(1), "str", w_obj));
    }
    let _errors = codec_errors_arg("unicode_escape_encode", 2, errors)?;
    // PyPy `unicodehelper.py:utf8_encode_unicode_escape`.
    let mut out = Vec::new();
    for cp in unsafe { w_str_get_wtf8(w_obj) }.code_points() {
        match cp.to_u32() {
            0x5c => out.extend_from_slice(br"\\"),
            0x09 => out.extend_from_slice(br"\t"),
            0x0a => out.extend_from_slice(br"\n"),
            0x0d => out.extend_from_slice(br"\r"),
            0x20..=0x7e => out.push(cp.to_u32() as u8),
            c @ 0x00..=0xff => push_ascii_hex_escape(&mut out, b'x', c, 2),
            c @ 0x100..=0xffff => push_ascii_hex_escape(&mut out, b'u', c, 4),
            c => push_ascii_hex_escape(&mut out, b'U', c, 8),
        }
    }
    Ok(w_tuple_new(vec![
        w_bytes_from_bytes(&out),
        w_int_new(unsafe { pyre_object::w_str_len(w_obj) } as i64),
    ]))
}

fn unicode_escape_error(
    errors: &str,
    original: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    out: &mut rustpython_wtf8::Wtf8Buf,
) -> Result<(usize, Option<Vec<u8>>), crate::PyError> {
    match errors {
        "strict" => Err(crate::typedef::unicode_decode_error(
            "unicodeescape",
            original,
            start,
            end,
            reason,
        )),
        "ignore" => Ok((end, None)),
        "replace" => {
            out.push_char('\u{FFFD}');
            Ok((end, None))
        }
        "backslashreplace" => {
            for &b in &original[start..end.min(original.len())] {
                out.push_str(&format!("\\x{b:02x}"));
            }
            Ok((end, None))
        }
        "xmlcharrefreplace" | "namereplace" => {
            Err(crate::typedef::decode_error_encode_only_handler())
        }
        _ => crate::type_methods::call_registered_decode_error_handler(
            errors,
            "unicodeescape",
            original,
            start,
            end,
            reason,
            out,
        ),
    }
}

/// Route a unicode-escape decode error, rebinding `data` when the handler
/// replaced `exc.object`. `pos_delta` accumulates the buffer length change so
/// the reported consumed count stays relative to the original input
/// (`str_decode_unicode_escape`'s `pos_delta`). Returns the resume position
/// in the (possibly replaced) buffer.
fn unicode_escape_run_error(
    data: &mut std::borrow::Cow<[u8]>,
    out: &mut rustpython_wtf8::Wtf8Buf,
    pos_delta: &mut i64,
    start: usize,
    end: usize,
    reason: &str,
    errors: &str,
) -> Result<usize, crate::PyError> {
    let prelen = data.len();
    let (np, nb) = unicode_escape_error(errors, &data[..], start, end, reason, out)?;
    if let Some(b) = nb {
        *data = std::borrow::Cow::Owned(b);
        *pos_delta += prelen as i64 - data.len() as i64;
    }
    Ok(np)
}

/// `unicodehelper.py:hexescape` — `pos` points just past the `\x`/`\u`/`\U`
/// intro, so the escape's backslash is at `pos - 2`. Decodes `digits` hex
/// digits into a code point, or routes a truncated/illegal error. Returns the
/// resume position.
fn unicode_escape_hex(
    data: &mut std::borrow::Cow<[u8]>,
    out: &mut rustpython_wtf8::Wtf8Buf,
    pos_delta: &mut i64,
    pos: usize,
    digits: usize,
    message: &str,
    errors: &str,
) -> Result<usize, crate::PyError> {
    if pos + digits <= data.len()
        && data[pos..pos + digits]
            .iter()
            .all(|b| b.is_ascii_hexdigit())
    {
        let value = u32::from_str_radix(std::str::from_utf8(&data[pos..pos + digits]).unwrap(), 16)
            .unwrap();
        if let Some(cp) = rustpython_wtf8::CodePoint::from_u32(value) {
            out.push(cp);
            return Ok(pos + digits);
        }
        // A valid hex value outside the Unicode range: the whole escape span
        // (`pos - 2 .. pos + digits`) is reported.
        return unicode_escape_run_error(
            data,
            out,
            pos_delta,
            pos - 2,
            pos + digits,
            "illegal Unicode character",
            errors,
        );
    }
    // Too few digits, or a non-hex digit: the error span covers the run of hex
    // digits actually present after the intro.
    let mut endinpos = pos;
    while endinpos < data.len() && data[endinpos].is_ascii_hexdigit() {
        endinpos += 1;
    }
    unicode_escape_run_error(data, out, pos_delta, pos - 2, endinpos, message, errors)
}

/// The `DeprecationWarning` text an unrecognised escape earns while decoding.
///
/// `prefix` distinguishes the two decoders: the bytes-to-bytes transform names
/// the sequence as a `bytes` literal, the text one as a `str` literal.  The
/// wording stops after the "will not work in the future" sentence -- the longer
/// report carrying a "Did you mean" suggestion belongs to the compiler, which
/// has the surrounding literal to suggest a raw string for.
fn invalid_escape_warning(prefix: &str, sequence: &str, octal: bool) -> String {
    let kind = if octal {
        "an invalid octal escape sequence"
    } else {
        "an invalid escape sequence"
    };
    format!("{prefix}\"\\{sequence}\" is {kind}. Such sequences will not work in the future. ")
}

/// Acquire a backslash-escape decoder's input.  A `str` answers with its own
/// bytes; everything else is unwrapped the way the rest of the decoders in
/// this module unwrap theirs, so the accepted buffer shapes are the same set
/// and a strided `memoryview` is refused here as it is there.
fn escape_decoder_input(w_obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    if unsafe { is_str(w_obj) } {
        return Ok(unsafe { w_str_get_wtf8(w_obj) }.as_bytes().to_vec());
    }
    decode_input_bytes(w_obj)
}

fn unicode_escape_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    final_: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let initial = escape_decoder_input(w_obj)?;
    let errors_s = codec_errors_arg("unicode_escape_decode", 2, errors)?;
    let errors_s = errors_s.as_str();
    // `unicodehelper.py:str_decode_unicode_escape` (final=True). A custom error
    // handler may replace `exc.object`; decoding then resumes from the new
    // bytes (`data`), and `pos_delta` keeps the reported consumed count
    // relative to the original input length.
    let mut data: std::borrow::Cow<[u8]> = std::borrow::Cow::Owned(initial);
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut pos = 0usize;
    let mut pos_delta = 0i64;
    let mut first_escape_warning: Option<String> = None;
    while pos < data.len() {
        let ch = data[pos];
        if ch != b'\\' {
            out.push(rustpython_wtf8::CodePoint::from_u32(ch as u32).unwrap());
            pos += 1;
            continue;
        }
        let escape_start = pos;
        pos += 1;
        if pos >= data.len() {
            if !final_ {
                // More input may follow, so the backslash is left unconsumed
                // rather than reported: what it introduces is still unknown.
                pos = escape_start;
                break;
            }
            let end = data.len();
            pos = unicode_escape_run_error(
                &mut data,
                &mut out,
                &mut pos_delta,
                escape_start,
                end,
                "\\ at end of string",
                errors_s,
            )?;
            continue;
        }
        let ch = data[pos];
        pos += 1;
        match ch {
            b'\n' => {}
            b'\\' => out.push_char('\\'),
            b'\'' => out.push_char('\''),
            b'"' => out.push_char('"'),
            b'b' => out.push_char('\x08'),
            b'f' => out.push_char('\x0c'),
            b't' => out.push_char('\t'),
            b'n' => out.push_char('\n'),
            b'r' => out.push_char('\r'),
            b'v' => out.push_char('\x0b'),
            b'a' => out.push_char('\x07'),
            b'0'..=b'7' => {
                let octal_start = pos - 1;
                let mut value = (ch - b'0') as u32;
                for _ in 0..2 {
                    if pos < data.len() && matches!(data[pos], b'0'..=b'7') {
                        value = (value << 3) + (data[pos] - b'0') as u32;
                        pos += 1;
                    }
                }
                // Only three octal digits are read, so the largest escape is
                // `\777`; anything past `\377` leaves the byte range the
                // sequence is written to address.
                if value > 0o377 && first_escape_warning.is_none() {
                    first_escape_warning = Some(invalid_escape_warning(
                        "",
                        &String::from_utf8_lossy(&data[octal_start..pos]),
                        true,
                    ));
                }
                out.push(rustpython_wtf8::CodePoint::from_u32(value).unwrap());
            }
            b'x' | b'u' | b'U' => {
                let (digits, msg) = match ch {
                    b'x' => (2usize, "truncated \\xXX escape"),
                    b'u' => (4usize, "truncated \\uXXXX escape"),
                    _ => (8usize, "truncated \\UXXXXXXXX escape"),
                };
                if !final_ && pos + digits > data.len() {
                    // The escape runs off the end of this chunk; the digits it
                    // is missing can still arrive.  A sequence the chunk does
                    // decide -- four bytes that are not all hex -- is reported
                    // here whether or not more input follows.
                    pos = escape_start;
                    break;
                }
                pos = unicode_escape_hex(
                    &mut data,
                    &mut out,
                    &mut pos_delta,
                    pos,
                    digits,
                    msg,
                    errors_s,
                )?;
            }
            b'N' => {
                // `\N{NAME}` is resolved through the character database.  Only
                // a name that names one character resolves, so a named
                // sequence is reported as unknown; a name that is empty,
                // unterminated, or not introduced by a brace at all is
                // malformed instead, and each spelling reports its own span.
                let (msg, end) = if pos < data.len() && data[pos] == b'{' {
                    let name_start = pos + 1;
                    let mut look = name_start;
                    while look < data.len() && data[look] != b'}' {
                        look += 1;
                    }
                    if look >= data.len() {
                        if !final_ {
                            // The closing brace can arrive with the next chunk.
                            pos = escape_start;
                            break;
                        }
                        ("malformed \\N character escape", data.len())
                    } else if look == name_start {
                        // An empty name is not a name to look up.
                        ("malformed \\N character escape", name_start)
                    } else {
                        // Owned so the database read does not borrow `data`,
                        // which the error path below hands out mutably.
                        let name = String::from_utf8(data[name_start..look].to_vec()).ok();
                        match name
                            .as_deref()
                            .and_then(rustpython_unicode::lookup_character)
                        {
                            Some(ch) => {
                                out.push_char(ch);
                                pos = look + 1;
                                continue;
                            }
                            None => ("unknown Unicode character name", look + 1),
                        }
                    }
                } else if pos >= data.len() && !final_ {
                    // `\N` at the very end: the byte deciding whether a name
                    // follows has not arrived yet.
                    pos = escape_start;
                    break;
                } else {
                    ("malformed \\N character escape", pos)
                };
                pos = unicode_escape_run_error(
                    &mut data,
                    &mut out,
                    &mut pos_delta,
                    escape_start,
                    end,
                    msg,
                    errors_s,
                )?;
            }
            _ => {
                if first_escape_warning.is_none() {
                    first_escape_warning = Some(invalid_escape_warning(
                        "",
                        &char::from(ch).to_string(),
                        false,
                    ));
                }
                out.push_char('\\');
                out.push(rustpython_wtf8::CodePoint::from_u32(ch as u32).unwrap());
            }
        }
    }
    if let Some(message) = first_escape_warning {
        crate::warn::warn_deprecation(&message)?;
    }
    Ok(w_tuple_new(vec![
        w_str_from_wtf8_managed(out),
        w_int_new(pos as i64 + pos_delta),
    ]))
}

/// `unicode_escape_decode`'s raw counterpart: only `\uXXXX` and
/// `\UXXXXXXXX` are escapes, every other byte -- a lone backslash included --
/// standing for its own Latin-1 code point.
fn raw_unicode_escape_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    final_: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let data = escape_decoder_input(w_obj)?;
    let errors_s = codec_errors_arg("raw_unicode_escape_decode", 2, errors)?;
    let (out, consumed) =
        crate::type_methods::decode_raw_unicode_escape_stateful(&data, &errors_s, final_)?;
    Ok(w_tuple_new(vec![
        w_str_from_wtf8_managed(out),
        w_int_new(consumed as i64),
    ]))
}

/// `interp_codecs.py escape_decode` / `_PyString_DecodeEscape` — the
/// bytes-to-bytes Python string-literal escape transform used by protocol-0
/// pickle.
fn escape_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let data = if unsafe { is_str(w_obj) } {
        unsafe { w_str_get_wtf8(w_obj) }.as_bytes().to_vec()
    } else if let Some(src) = crate::typedef::buffer_as_bytes_like(w_obj)? {
        unsafe { pyre_object::bytesobject::bytes_like_data(src) }.to_vec()
    } else {
        return Err(bad_buffer_arg(w_obj));
    };
    let errors_s = codec_errors_arg("escape_decode", 2, errors)?;
    let errors_s = errors_s.as_str();

    let mut out = Vec::with_capacity(data.len());
    let mut pos = 0usize;
    let mut first_escape_warning: Option<String> = None;
    while pos < data.len() {
        if data[pos] != b'\\' {
            out.push(data[pos]);
            pos += 1;
            continue;
        }

        let escape_start = pos;
        pos += 1;
        if pos == data.len() {
            return Err(crate::PyError::value_error("Trailing \\ in string"));
        }
        let ch = data[pos];
        pos += 1;
        match ch {
            b'\n' => {}
            b'\\' => out.push(b'\\'),
            b'\'' => out.push(b'\''),
            b'"' => out.push(b'"'),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b't' => out.push(b'\t'),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b'v' => out.push(0x0b),
            b'a' => out.push(0x07),
            b'0'..=b'7' => {
                let octal_start = pos - 1;
                while pos < data.len()
                    && pos < octal_start + 3
                    && (b'0'..=b'7').contains(&data[pos])
                {
                    pos += 1;
                }
                let raw = data[octal_start..pos]
                    .iter()
                    .fold(0u16, |value, digit| value * 8 + (digit - b'0') as u16);
                if raw >= 256 && first_escape_warning.is_none() {
                    first_escape_warning = Some(invalid_escape_warning(
                        "b",
                        &String::from_utf8_lossy(&data[octal_start..pos]),
                        true,
                    ));
                }
                out.push(raw as u8);
            }
            b'x' => {
                let hi = data.get(pos).and_then(|byte| (*byte as char).to_digit(16));
                let lo = data
                    .get(pos + 1)
                    .and_then(|byte| (*byte as char).to_digit(16));
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi * 16 + lo) as u8);
                    pos += 2;
                } else {
                    match errors_s {
                        "strict" => {
                            return Err(crate::PyError::value_error(format!(
                                "invalid \\x escape at position {escape_start}"
                            )));
                        }
                        "replace" => out.push(b'?'),
                        "ignore" => {}
                        other => {
                            return Err(crate::PyError::value_error(format!(
                                "decoding error; unknown error handling code: {other}"
                            )));
                        }
                    }
                    if data.get(pos).is_some_and(u8::is_ascii_hexdigit) {
                        pos += 1;
                    }
                }
            }
            other => {
                out.push(b'\\');
                pos -= 1;
                if first_escape_warning.is_none() {
                    first_escape_warning = Some(invalid_escape_warning(
                        "b",
                        &char::from(other).to_string(),
                        false,
                    ));
                }
            }
        }
    }

    if let Some(message) = first_escape_warning {
        crate::warn::warn_deprecation(&message)?;
    }
    Ok(w_tuple_new(vec![
        w_bytes_from_bytes(&out),
        w_int_new(data.len() as i64),
    ]))
}

/// `interp_codecs.py escape_encode` / `string_escape_encode(data,
/// quote=False)` — the inverse bytes transform.
fn escape_encode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_bytes(w_obj) } {
        return Err(bad_arg("escape_encode", Some(1), "bytes", w_obj));
    }
    let _errors = codec_errors_arg("escape_encode", 2, errors)?;
    let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_obj) };
    let mut out = Vec::with_capacity(data.len());
    for byte in data {
        match *byte {
            b'\t' => out.extend_from_slice(b"\\t"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\'' => out.extend_from_slice(b"\\'"),
            0x20..=0x7e => out.push(*byte),
            value => out.extend_from_slice(format!("\\x{value:02x}").as_bytes()),
        }
    }
    Ok(w_tuple_new(vec![
        w_bytes_from_bytes(&out),
        w_int_new(data.len() as i64),
    ]))
}

fn charmap_build(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(chars) = args.first().copied() else {
        return Err(crate::PyError::type_error(
            "charmap_build() missing argument",
        ));
    };
    if !unsafe { is_str(chars) } {
        return Err(bad_arg("charmap_build", None, "str", chars));
    }

    // PyPy `interp_codecs.py charmap_build`: build a dict mapping
    // each Unicode codepoint in `chars` to its ordinal position.
    let w_charmap = w_dict_new();
    for (num, cp) in unsafe { w_str_get_wtf8(chars) }.code_points().enumerate() {
        unsafe {
            pyre_object::dictmultiobject::w_dict_store(
                w_charmap,
                w_int_new(cp.to_u32() as i64),
                w_int_new(num as i64),
            );
        }
    }
    Ok(w_charmap)
}

/// The `errors` argument the code page entry points share: `None` is
/// `strict`, a `str` is itself, and nothing else is accepted.
#[cfg(windows)]
fn code_page_errors(
    name: &str,
    position: usize,
    w_errors: PyObjectRef,
) -> Result<String, crate::PyError> {
    codec_errors_arg(name, position, w_errors)
}

/// The code page number argument.  A negative number names no code page and
/// is rejected before any conversion is attempted.
#[cfg(windows)]
fn code_page_number(w_code_page: PyObjectRef) -> Result<u32, crate::PyError> {
    let code_page = crate::baseobjspace::c_int_w(w_code_page)?;
    if code_page < 0 {
        return Err(crate::PyError::value_error("invalid code page number"));
    }
    Ok(code_page as u32)
}

/// `_codecs.code_page_encode` - `(bytes, characters consumed)`, where the
/// count is the whole string whatever the error handler did inside it.
#[cfg(windows)]
fn code_page_encode_impl(
    name: &str,
    position: usize,
    code_page: u32,
    w_str: PyObjectRef,
    w_errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_str) } {
        return Err(bad_arg(name, Some(position), "str", w_str));
    }
    let errors = code_page_errors(name, position + 1, w_errors)?;
    let bytes = crate::unicodehelper_win32::encode_code_page(code_page, w_str, &errors)?;
    Ok(w_tuple_new(vec![
        pyre_object::bytesobject::w_bytes_from_bytes(&bytes),
        w_int_new(unsafe { w_str_len(w_str) } as i64),
    ]))
}

/// `_codecs.code_page_decode` - `(str, bytes consumed)`.
#[cfg(windows)]
fn code_page_decode_impl(
    name: &str,
    position: usize,
    code_page: u32,
    w_data: PyObjectRef,
    w_errors: PyObjectRef,
    w_final: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let data = decode_input_bytes(w_data)?;
    let errors = code_page_errors(name, position + 1, w_errors)?;
    let is_final = crate::baseobjspace::is_true(w_final)?;
    let (text, consumed) =
        crate::unicodehelper_win32::decode_code_page(code_page, &data, &errors, is_final)?;
    // `final` is the caller promising that no continuation is coming, so the
    // whole buffer reads as consumed: nothing is being held back for one.
    let consumed = if is_final { data.len() } else { consumed };
    Ok(w_tuple_new(vec![
        w_str_from_wtf8_managed(text),
        w_int_new(consumed as i64),
    ]))
}

/// Strip the keyword marker these positional-only entry points cannot take.
/// A variadic builtin is handed the raw slice, with a keyword call's marker
/// dict as its last element; left in place it reads as one more positional
/// argument.
#[cfg(windows)]
fn code_page_positional<'a>(
    name: &str,
    args: &'a [PyObjectRef],
) -> Result<&'a [PyObjectRef], crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(format!(
            "_codecs.{name}() takes no keyword arguments"
        )));
    }
    Ok(positional)
}

/// Split `(str[, errors])` for the two encoders that name their own code page.
#[cfg(windows)]
fn fixed_code_page_encode(
    name: &str,
    code_page: u32,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let args = code_page_positional(name, args)?;
    match args {
        [w_str] => code_page_encode_impl(name, 1, code_page, *w_str, w_none()),
        [w_str, w_errors] => code_page_encode_impl(name, 1, code_page, *w_str, *w_errors),
        _ => Err(code_page_arity_error(name, 1, 2, args.len())),
    }
}

/// Split `(data[, errors[, final]])` for the two decoders that name their own
/// code page.
#[cfg(windows)]
fn fixed_code_page_decode(
    name: &str,
    code_page: u32,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let args = code_page_positional(name, args)?;
    match args {
        [w_data] => {
            code_page_decode_impl(name, 1, code_page, *w_data, w_none(), w_bool_from(false))
        }
        [w_data, w_errors] => {
            code_page_decode_impl(name, 1, code_page, *w_data, *w_errors, w_bool_from(false))
        }
        [w_data, w_errors, w_final] => {
            code_page_decode_impl(name, 1, code_page, *w_data, *w_errors, *w_final)
        }
        _ => Err(code_page_arity_error(name, 1, 3, args.len())),
    }
}

/// `_PyArg_CheckPositional` wording for a positional-only entry point whose
/// trailing arguments carry defaults.
#[cfg(windows)]
fn code_page_arity_error(name: &str, least: usize, most: usize, given: usize) -> crate::PyError {
    let bound = if given < least {
        let plural = if least == 1 { "" } else { "s" };
        format!("at least {least} argument{plural}")
    } else {
        format!("at most {most} arguments")
    };
    crate::PyError::type_error(format!("{name} expected {bound}, got {given}"))
}

crate::py_module! {
    "_codecs",
    inline_functions: {
        fn readbuffer_encode(
            data: PyObjectRef,
            #[default(w_none())] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // `Py_buffer(accept={str, buffer})`: a str contributes its own
            // UTF-8 bytes, anything else the buffer it exposes.
            let bytes = if unsafe { is_str(data) } {
                crate::baseobjspace::str_utf8_w(data)?.as_bytes().to_vec()
            } else {
                decode_input_bytes(data)?
            };
            let _errors = codec_errors_arg("readbuffer_encode", 2, errors)?;
            Ok(w_tuple_new(vec![
                pyre_object::bytesobject::w_bytes_from_bytes(&bytes),
                w_int_new(bytes.len() as i64),
            ]))
        }
        fn ascii_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "ascii_encode", "ascii")
        }
        fn ascii_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "ascii_decode", "ascii")
        }
        fn latin_1_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "latin_1_encode", "latin-1")
        }
        fn latin_1_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "latin_1_decode", "latin-1")
        }
        fn utf_8_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_8_encode", "utf-8")
        }
        fn utf_8_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf8_decode_impl(obj, errors, final_)
        }
        fn utf_16_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_16_encode", "utf-16")
        }
        fn utf_16_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(obj, errors, final_, false, None, "utf16", "utf_16_decode")
        }
        fn utf_16_ex_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(0i64)] byteorder: i64,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_ex_decode_impl(obj, errors, byteorder, final_, false, "utf_16_ex_decode")
        }
        fn utf_16_be_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_16_be_encode", "utf-16-be")
        }
        fn utf_16_be_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(
                obj,
                errors,
                final_,
                false,
                Some(true),
                "utf16-be",
                "utf_16_be_decode",
            )
        }
        fn utf_16_le_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_16_le_encode", "utf-16-le")
        }
        fn utf_16_le_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(
                obj,
                errors,
                final_,
                false,
                Some(false),
                "utf16-le",
                "utf_16_le_decode",
            )
        }
        fn utf_32_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_32_encode", "utf-32")
        }
        fn utf_32_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(obj, errors, final_, true, None, "utf32", "utf_32_decode")
        }
        fn utf_32_ex_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(0i64)] byteorder: i64,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_ex_decode_impl(obj, errors, byteorder, final_, true, "utf_32_ex_decode")
        }
        fn utf_32_be_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_32_be_encode", "utf-32-be")
        }
        fn utf_32_be_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(
                obj,
                errors,
                final_,
                true,
                Some(true),
                "utf32-be",
                "utf_32_be_decode",
            )
        }
        fn utf_32_le_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf_32_le_encode", "utf-32-le")
        }
        fn utf_32_le_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf16_32_decode_impl(
                obj,
                errors,
                final_,
                true,
                Some(false),
                "utf32-le",
                "utf_32_le_decode",
            )
        }
        fn raw_unicode_escape_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(
                obj,
                errors,
                "raw_unicode_escape_encode",
                "raw-unicode-escape",
            )
        }
        fn raw_unicode_escape_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(true))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            raw_unicode_escape_decode_impl(obj, errors, crate::baseobjspace::is_true(final_)?)
        }
        fn utf_7_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf7_encode_impl(obj, errors)
        }
        fn utf_7_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] is_final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf7_decode_impl(obj, errors, crate::baseobjspace::is_true(is_final)?)
        }
        fn unicode_escape_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            unicode_escape_encode_impl(obj, errors)
        }
        fn unicode_escape_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(true))] final_: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            unicode_escape_decode_impl(obj, errors, crate::baseobjspace::is_true(final_)?)
        }
        fn escape_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            escape_decode_impl(obj, errors)
        }
        fn escape_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            escape_encode_impl(obj, errors)
        }
        fn charmap_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_none())] mapping: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            charmap_encode_impl(obj, errors, mapping)
        }
        fn charmap_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_none())] mapping: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            charmap_decode_impl(obj, errors, mapping)
        }
        fn encode(
            obj: PyObjectRef,
            #[default(w_str_new("utf-8"))] encoding: PyObjectRef,
            errors: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            codec_encode_or_decode(obj, encoding, errors, true)
        }
        fn decode(
            obj: PyObjectRef,
            #[default(w_str_new("utf-8"))] encoding: PyObjectRef,
            errors: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            codec_encode_or_decode(obj, encoding, errors, false)
        }
    },
    functions: {
        "lookup_error"     / 1 = lookup_error,
        "register_error"   / 2 = register_error,
        "_unregister_error" / 1 = unregister_error,
        "register"       / 1 = register_codec,
        "unregister"     / 1 = unregister,
        "lookup"         / 1 = lookup_codec,
        "_forget_codec"  / 1 = forget_codec,
        "charmap_build"  / 1 = charmap_build,
    },
    extra_init: |ns| {
        // The code page codecs exist only where the code pages do; their
        // absence elsewhere is what makes `encodings/mbcs.py` an ImportError
        // rather than a codec that answers with the wrong bytes.
        #[cfg(windows)]
        {
            use windows_sys::Win32::Globalization::{CP_ACP, CP_OEMCP};

            fn mbcs_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                fixed_code_page_encode("mbcs_encode", CP_ACP, args)
            }
            fn mbcs_decode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                fixed_code_page_decode("mbcs_decode", CP_ACP, args)
            }
            fn oem_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                fixed_code_page_encode("oem_encode", CP_OEMCP, args)
            }
            fn oem_decode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                fixed_code_page_decode("oem_decode", CP_OEMCP, args)
            }
            fn code_page_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                let args = code_page_positional("code_page_encode", args)?;
                match args {
                    [w_cp, w_str] => code_page_encode_impl(
                        "code_page_encode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_str,
                        w_none(),
                    ),
                    [w_cp, w_str, w_errors] => code_page_encode_impl(
                        "code_page_encode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_str,
                        *w_errors,
                    ),
                    _ => Err(code_page_arity_error("code_page_encode", 2, 3, args.len())),
                }
            }
            fn code_page_decode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                let args = code_page_positional("code_page_decode", args)?;
                match args {
                    [w_cp, w_data] => code_page_decode_impl(
                        "code_page_decode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_data,
                        w_none(),
                        w_bool_from(false),
                    ),
                    [w_cp, w_data, w_errors] => code_page_decode_impl(
                        "code_page_decode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_data,
                        *w_errors,
                        w_bool_from(false),
                    ),
                    [w_cp, w_data, w_errors, w_final] => code_page_decode_impl(
                        "code_page_decode",
                        2,
                        code_page_number(*w_cp)?,
                        *w_data,
                        *w_errors,
                        *w_final,
                    ),
                    _ => Err(code_page_arity_error("code_page_decode", 2, 4, args.len())),
                }
            }

            for (name, entry) in [
                ("mbcs_encode", mbcs_encode as crate::BuiltinCodeFn),
                ("mbcs_decode", mbcs_decode),
                ("oem_encode", oem_encode),
                ("oem_decode", oem_decode),
                ("code_page_encode", code_page_encode),
                ("code_page_decode", code_page_decode),
            ] {
                crate::module_ns_store(
                    ns,
                    name,
                    crate::gateway::with_module(
                        "_codecs",
                        crate::make_module_builtin_function(name, entry),
                    ),
                );
            }
        }
        #[cfg(not(windows))]
        let _ = ns;
    },
}
