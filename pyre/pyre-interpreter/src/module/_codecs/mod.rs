//! _codecs module — PyPy: `pypy/module/_codecs/`.
//!
//! Text codecs (`encode` / `decode`) delegate to `str.encode` /
//! `bytes.decode`, which cover `PyCodec_Encode` / `PyCodec_Decode` for the
//! text path. The codec registry (`register` / `lookup`) and error
//! handlers remain stubs; binary transform codecs are not modelled.

use std::cell::Cell;

use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

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

thread_local! {
    static CODEC_STATE: Cell<*mut CodecState> = const { Cell::new(std::ptr::null_mut()) };
}

fn with_codec_state<R>(f: impl FnOnce(&mut CodecState) -> R) -> R {
    CODEC_STATE.with(|slot| {
        let mut ptr = slot.get();
        if ptr.is_null() {
            ptr = Box::into_raw(Box::new(CodecState::new()));
            slot.set(ptr);
        }
        f(unsafe { &mut *ptr })
    })
}

pub(crate) unsafe fn walk_codec_state_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    CODEC_STATE.with(|slot| {
        let ptr = slot.get();
        if ptr.is_null() {
            return;
        }
        let state = unsafe { &mut *ptr };
        visitor(&mut state.codec_search_path);
        visitor(&mut state.codec_search_cache);
        visitor(&mut state.codec_error_registry);
    });
}

pub(crate) fn capture_codec_state_root_area() -> *const () {
    CODEC_STATE.with(|state| state as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_codec_state_root_area`], and the owning
/// thread must be quiesced.
pub(crate) unsafe fn walk_codec_state_root_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    let state = unsafe { &*(data as *const Cell<*mut CodecState>) };
    let ptr = state.get();
    if ptr.is_null() {
        return;
    }
    let state = unsafe { &mut *ptr };
    visitor(&mut state.codec_search_path);
    visitor(&mut state.codec_search_cache);
    visitor(&mut state.codec_error_registry);
}

// PyPy `interp_codecs.py:166-190 normalize`.
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
    standard_encoding(unsafe { w_str_get_value(w_encoding) })
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

/// `interp_codecs.py:602-610 lookup_error`.  The direct codec loops implement
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
            format!("unknown error handler name {errors}"),
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
        return Err(crate::PyError::type_error(
            "lookup_error() argument must be str",
        ));
    }
    let errors = unsafe { w_str_get_value(w_errors) };
    if let Some(w_handler) = with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(state.codec_error_registry, errors)
    }) {
        return Ok(w_handler);
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::LookupError,
        format!("unknown error handler name {errors}"),
    ))
}

fn register_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (Some(w_errors), Some(w_handler)) = (args.first().copied(), args.get(1).copied()) else {
        return Err(crate::PyError::type_error(
            "register_error() requires name and handler",
        ));
    };
    if !unsafe { is_str(w_errors) } {
        return Err(crate::PyError::type_error(
            "register_error() argument 1 must be str",
        ));
    }
    if !is_callable(w_handler) {
        return Err(crate::PyError::type_error("argument must be callable"));
    }
    let errors = unsafe { w_str_get_value(w_errors) };
    with_codec_state(|state| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            state.codec_error_registry,
            errors,
            w_handler,
        );
    });
    Ok(w_none())
}

fn register_codec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_search_function) = args.first().copied() else {
        return Err(crate::PyError::type_error("register() missing argument"));
    };
    if !is_callable(w_search_function) {
        return Err(crate::PyError::type_error("argument must be callable"));
    }
    // PyPy `interp_codecs.py:143-155 register_codec`.
    with_codec_state(|state| unsafe {
        pyre_object::listobject::w_list_append(state.codec_search_path, w_search_function);
    });
    Ok(w_none())
}

fn unregister(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_search_function) = args.first().copied() else {
        return Err(crate::PyError::type_error("unregister() missing argument"));
    };
    // PyPy `interp_codecs.py:157-164 unregister`: remove and clear cache;
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
        return Err(crate::PyError::type_error("lookup() argument must be str"));
    }
    let encoding = unsafe { w_str_get_value(w_encoding) }.to_string();
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
    errors: Option<&str>,
) -> Result<PyObjectRef, crate::PyError> {
    // PyPy `interp_codecs.py:577-595 _call_codec`.
    let w_res = if let Some(errors) = errors {
        crate::call::call_function_impl_result(w_coder, &[w_obj, w_str_new(errors)])?
    } else {
        crate::call::call_function_impl_result(w_coder, &[w_obj])?
    };
    if !unsafe { pyre_object::is_tuple(w_res) } || unsafe { pyre_object::w_tuple_len(w_res) } != 2 {
        let msg = if action.starts_with("en") {
            "encoder must return a tuple (object, integer)".to_string()
        } else if action.starts_with("de") {
            "decoder must return a tuple (object, integer)".to_string()
        } else {
            format!("{action} must return a tuple (object, integer)")
        };
        return Err(crate::PyError::type_error(msg));
    }
    Ok(unsafe { pyre_object::w_tuple_getitem(w_res, 0).unwrap_or_else(w_none) })
}

pub(crate) fn encode_text_codec(
    w_obj: PyObjectRef,
    encoding: &str,
    errors: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let w_codec_info = lookup_text_codec("encode", encoding)?;
    if crate::importing::dev_mode_flag() {
        validate_error_handler(errors)?;
    }
    let w_encfunc = unsafe { pyre_object::w_tuple_getitem(w_codec_info, 0).unwrap_or_else(w_none) };
    let w_retval = call_codec(w_encfunc, w_obj, "encoding", Some(errors))?;
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_retval) } {
        let tname = unsafe { (*(*w_retval).ob_type).name };
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
    let w_codec_info = lookup_text_codec("decode", encoding)?;
    if crate::importing::dev_mode_flag() {
        validate_error_handler(errors)?;
    }
    let w_decfunc = unsafe { pyre_object::w_tuple_getitem(w_codec_info, 1).unwrap_or_else(w_none) };
    let w_retval = call_codec(w_decfunc, w_obj, "decoding", Some(errors))?;
    if !unsafe { pyre_object::is_str(w_retval) } {
        let tname = unsafe { (*(*w_retval).ob_type).name };
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
    let normalized_encoding = normalize(unsafe { w_str_get_value(w_encoding) });
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
    encoding: &str,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(crate::PyError::type_error("encoder argument must be str"));
    }
    // PyPy `make_encoder_wrapper`: convert to unicode, call unicodehelper
    // encoder, return `(bytes, unicode_length)`.
    let encode_method = crate::baseobjspace::getattr_str(w_obj, "encode")?;
    let encoded =
        crate::call::call_function_impl_result(encode_method, &[w_str_new(encoding), errors])?;
    Ok(w_tuple_new(vec![
        encoded,
        w_int_new(unsafe { pyre_object::w_str_len(w_obj) } as i64),
    ]))
}

fn decode_with_name(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    encoding: &str,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_obj) } {
        return Err(crate::PyError::type_error(
            "decoder argument must be bytes-like",
        ));
    }
    // PyPy `make_decoder_wrapper`: decode a bytes buffer and return
    // `(unicode, bytes_consumed)`.
    let consumed = unsafe { pyre_object::bytesobject::bytes_like_data(w_obj).len() };
    let decode_method = crate::baseobjspace::getattr_str(w_obj, "decode")?;
    let decoded =
        crate::call::call_function_impl_result(decode_method, &[w_str_new(encoding), errors])?;
    Ok(w_tuple_new(vec![decoded, w_int_new(consumed as i64)]))
}

fn charmap_encode_impl(
    w_unicode: PyObjectRef,
    errors: PyObjectRef,
    w_mapping: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_none(w_mapping) } {
        return encode_with_name(w_unicode, errors, "latin-1");
    }
    if !unsafe { is_str(w_unicode) } {
        return Err(crate::PyError::type_error(
            "charmap_encode() argument must be str",
        ));
    }
    let errors_s = if unsafe { is_str(errors) } {
        unsafe { w_str_get_value(errors) }
    } else {
        "strict"
    };
    let mut out = Vec::new();
    for cp in unsafe { w_str_get_wtf8(w_unicode) }.code_points() {
        let w_ch = match crate::baseobjspace::getitem(w_mapping, w_int_new(cp.to_u32() as i64)) {
            Ok(w_ch) => w_ch,
            Err(e)
                if matches!(
                    e.kind,
                    crate::PyErrorKind::LookupError | crate::PyErrorKind::KeyError
                ) =>
            {
                match errors_s {
                    "ignore" => continue,
                    "replace" => {
                        out.push(b'?');
                        continue;
                    }
                    _ => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::UnicodeEncodeError,
                            "character maps to <undefined>",
                        ));
                    }
                }
            }
            Err(e) => return Err(e),
        };
        if unsafe { pyre_object::bytesobject::is_bytes_like(w_ch) } {
            out.extend_from_slice(unsafe { pyre_object::bytesobject::bytes_like_data(w_ch) });
        } else if unsafe { pyre_object::is_int(w_ch) } {
            let x = unsafe { pyre_object::w_int_get_value(w_ch) };
            if !(0..256).contains(&x) {
                return Err(crate::PyError::type_error(
                    "character mapping must be in range(256)",
                ));
            }
            out.push(x as u8);
        } else if unsafe { pyre_object::is_none(w_ch) } {
            match errors_s {
                "ignore" => {}
                "replace" => out.push(b'?'),
                _ => {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::UnicodeEncodeError,
                        "character maps to <undefined>",
                    ));
                }
            }
        } else {
            return Err(crate::PyError::type_error(
                "character mapping must return integer, bytes or None, not str",
            ));
        }
    }
    Ok(w_tuple_new(vec![
        w_bytes_from_bytes(&out),
        w_int_new(unsafe { pyre_object::w_str_len(w_unicode) } as i64),
    ]))
}

fn charmap_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
    w_mapping: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_none(w_mapping) } {
        return decode_with_name(w_obj, errors, "latin-1");
    }
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_obj) } {
        return Err(crate::PyError::type_error(
            "charmap_decode() argument must be bytes-like",
        ));
    }
    let errors_s = if unsafe { is_str(errors) } {
        unsafe { w_str_get_value(errors) }
    } else {
        "strict"
    };
    // A custom error handler may replace `exc.object`; decoding then resumes
    // from the new bytes (`data`).
    let mut data: std::borrow::Cow<[u8]> =
        std::borrow::Cow::Borrowed(unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) });
    // charmap_decode reports the number of input bytes consumed, which stays
    // the original length even if a handler replaces `exc.object`.
    let orig_len = data.len();
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
            match crate::baseobjspace::getitem(w_mapping, w_int_new(b as i64)) {
                Ok(w_ch) => Some(w_ch),
                Err(e)
                    if matches!(
                        e.kind,
                        crate::PyErrorKind::LookupError | crate::PyErrorKind::KeyError
                    ) =>
                {
                    None
                }
                Err(e) => return Err(e),
            }
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
                out.push(rustpython_wtf8::CodePoint::from_u32(x as u32).unwrap());
                i += 1;
                continue;
            } else if !unsafe { pyre_object::is_none(w_ch) } {
                return Err(crate::PyError::type_error(
                    "character mapping must return integer, None or str",
                ));
            }
        }
        // The byte maps to <undefined>: run the decode error handler over the
        // single byte at `i` (`str_decode_charmap` span `pos .. pos + 1`).
        match errors_s {
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
                    errors_s,
                    "charmap",
                    &data[..],
                    i,
                    i + 1,
                    "character maps to <undefined>",
                    &mut out,
                )?;
                if let Some(nb) = nb {
                    data = std::borrow::Cow::Owned(nb);
                }
                i = np;
            }
        }
    }
    Ok(w_tuple_new(vec![
        w_str_from_wtf8(out),
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

fn utf7_encode_impl(w_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(crate::PyError::type_error(
            "utf_7_encode() argument must be str",
        ));
    }
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
        return Err(crate::PyError::type_error(
            "utf_7_decode() argument must be bytes-like",
        ));
    }
    // PyPy `unicodehelper.py:str_decode_utf_7`.
    let errors_s = if unsafe { is_str(errors) } {
        unsafe { w_str_get_value(errors) }
    } else {
        "strict"
    };
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
        w_str_from_wtf8(out),
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

fn unicode_escape_encode_impl(w_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_str(w_obj) } {
        return Err(crate::PyError::type_error(
            "unicode_escape_encode() argument must be str",
        ));
    }
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

fn unicode_escape_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let initial: Vec<u8> = if unsafe { pyre_object::bytesobject::is_bytes_like(w_obj) } {
        unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) }.to_vec()
    } else if unsafe { is_str(w_obj) } {
        unsafe { w_str_get_wtf8(w_obj) }.as_bytes().to_vec()
    } else {
        return Err(crate::PyError::type_error(
            "unicode_escape_decode() argument must be bytes-like or str",
        ));
    };
    let errors_s = if unsafe { is_str(errors) } {
        unsafe { w_str_get_value(errors) }
    } else {
        "strict"
    };
    // `unicodehelper.py:str_decode_unicode_escape` (final=True). A custom error
    // handler may replace `exc.object`; decoding then resumes from the new
    // bytes (`data`), and `pos_delta` keeps the reported consumed count
    // relative to the original input length.
    let mut data: std::borrow::Cow<[u8]> = std::borrow::Cow::Owned(initial);
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut pos = 0usize;
    let mut pos_delta = 0i64;
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
                let mut value = (ch - b'0') as u32;
                for _ in 0..2 {
                    if pos < data.len() && matches!(data[pos], b'0'..=b'7') {
                        value = (value << 3) + (data[pos] - b'0') as u32;
                        pos += 1;
                    }
                }
                out.push(rustpython_wtf8::CodePoint::from_u32(value).unwrap());
            }
            b'x' | b'u' | b'U' => {
                let (digits, msg) = match ch {
                    b'x' => (2usize, "truncated \\xXX escape"),
                    b'u' => (4usize, "truncated \\uXXXX escape"),
                    _ => (8usize, "truncated \\UXXXXXXXX escape"),
                };
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
                // pyre has no Unicode-name database, so a `\N` escape never
                // resolves; it is always reported as an error.
                let (msg, end) = if pos < data.len() && data[pos] == b'{' {
                    let mut look = pos + 1;
                    while look < data.len() && data[look] != b'}' {
                        look += 1;
                    }
                    if look < data.len() && data[look] == b'}' {
                        ("unknown Unicode character name", look + 1)
                    } else {
                        ("malformed \\N character escape", data.len())
                    }
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
                out.push_char('\\');
                out.push(rustpython_wtf8::CodePoint::from_u32(ch as u32).unwrap());
            }
        }
    }
    Ok(w_tuple_new(vec![
        w_str_from_wtf8(out),
        w_int_new(pos as i64 + pos_delta),
    ]))
}

fn charmap_build(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(chars) = args.first().copied() else {
        return Err(crate::PyError::type_error(
            "charmap_build() missing argument",
        ));
    };
    if !unsafe { is_str(chars) } {
        return Err(crate::PyError::type_error(
            "charmap_build() argument must be str",
        ));
    }

    // PyPy `interp_codecs.py:1006-1016 charmap_build`: build a dict mapping
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

crate::py_module! {
    "_codecs",
    inline_functions: {
        fn ascii_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "ascii")
        }
        fn ascii_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "ascii")
        }
        fn latin_1_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "latin-1")
        }
        fn latin_1_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "latin-1")
        }
        fn utf_8_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf-8")
        }
        fn utf_8_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "utf-8")
        }
        fn utf_16_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf-16")
        }
        fn utf_16_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "utf-16")
        }
        fn utf_16_be_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf-16-be")
        }
        fn utf_16_be_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "utf-16-be")
        }
        fn utf_16_le_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf-16-le")
        }
        fn utf_16_le_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "utf-16-le")
        }
        fn utf_32_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf-32")
        }
        fn utf_32_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "utf-32")
        }
        fn utf_32_be_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf-32-be")
        }
        fn utf_32_be_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "utf-32-be")
        }
        fn utf_32_le_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "utf-32-le")
        }
        fn utf_32_le_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "utf-32-le")
        }
        fn raw_unicode_escape_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            encode_with_name(obj, errors, "raw-unicode-escape")
        }
        fn raw_unicode_escape_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            decode_with_name(obj, errors, "raw-unicode-escape")
        }
        fn utf_7_encode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] _errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf7_encode_impl(obj)
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
            #[default(w_str_new("strict"))] _errors: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            unicode_escape_encode_impl(obj)
        }
        fn unicode_escape_decode(
            obj: PyObjectRef,
            #[default(w_str_new("strict"))] errors: PyObjectRef,
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            unicode_escape_decode_impl(obj, errors)
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
        "register_error"   / 2 = register_error,
        "_unregister_error" / 1 = |_| Ok(w_bool_from(false)),
        "register"       / 1 = register_codec,
        "unregister"     / 1 = unregister,
        "lookup"         / 1 = lookup_codec,
        "_forget_codec"  / 1 = forget_codec,
        "charmap_build"  / 1 = charmap_build,
    },
}
