//! _codecs module — PyPy: `pypy/module/_codecs/`.
//!
//! Text codecs (`encode` / `decode`) delegate to `str.encode` /
//! `bytes.decode`, which cover `PyCodec_Encode` / `PyCodec_Decode` for the
//! text path. The codec registry (`register` / `lookup`) and error
//! handlers remain stubs; binary transform codecs are not modelled.

use std::cell::Cell;

use pyre_object::*;

struct CodecState {
    codec_search_path: PyObjectRef,
    codec_search_cache: PyObjectRef,
    codec_error_registry: PyObjectRef,
    codec_need_encodings: bool,
}

impl CodecState {
    fn new() -> Self {
        Self {
            codec_search_path: w_list_new(Vec::new()),
            codec_search_cache: w_dict_new(),
            codec_error_registry: w_dict_new(),
            codec_need_encodings: true,
        }
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
    if obj.is_null() {
        return false;
    }
    unsafe {
        if crate::is_function(obj)
            || pyre_object::is_method(obj)
            || pyre_object::is_type(obj)
            || pyre_object::is_staticmethod(obj)
            || pyre_object::is_classmethod(obj)
        {
            return true;
        }
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            return crate::baseobjspace::lookup_in_type(w_type, "__call__").is_some();
        }
    }
    false
}

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
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) };
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
    for &b in data {
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
        let Some(w_ch) = mapped else {
            match errors_s {
                "ignore" => continue,
                "replace" => {
                    out.push_char('\u{FFFD}');
                    continue;
                }
                _ => {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::UnicodeDecodeError,
                        "character maps to <undefined>",
                    ));
                }
            }
        };
        if unsafe { is_str(w_ch) } {
            let s = unsafe { w_str_get_wtf8(w_ch) };
            if s.as_bytes() == "\u{FFFE}".as_bytes() {
                match errors_s {
                    "ignore" => continue,
                    "replace" => {
                        out.push_char('\u{FFFD}');
                        continue;
                    }
                    _ => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::UnicodeDecodeError,
                            "character maps to <undefined>",
                        ));
                    }
                }
            }
            out.push_wtf8(s);
        } else if unsafe { pyre_object::is_int(w_ch) } {
            let x = unsafe { pyre_object::w_int_get_value(w_ch) };
            if !(0..=0x10FFFF).contains(&x) {
                return Err(crate::PyError::type_error(
                    "character mapping must be in range(0x110000)",
                ));
            }
            out.push(rustpython_wtf8::CodePoint::from_u32(x as u32).unwrap());
        } else if unsafe { pyre_object::is_none(w_ch) } {
            match errors_s {
                "ignore" => {}
                "replace" => out.push_char('\u{FFFD}'),
                _ => {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::UnicodeDecodeError,
                        "character maps to <undefined>",
                    ));
                }
            }
        } else {
            return Err(crate::PyError::type_error(
                "character mapping must return integer, None or str",
            ));
        }
    }
    Ok(w_tuple_new(vec![
        w_str_from_wtf8(out),
        w_int_new(data.len() as i64),
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

fn utf7_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
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
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) };
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut pos = 0usize;
    let mut in_shift = false;
    let mut base64bits = 0u32;
    let mut base64buffer = 0u32;
    let mut surrogate = 0u32;
    let mut shift_out_start = 0usize;
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
                in_shift = false;
                if base64bits > 0 {
                    let bad = base64bits >= 6 || base64buffer != 0;
                    if bad {
                        if errors_s == "ignore" {
                            pos += 1;
                            continue;
                        }
                        return Err(crate::typedef::unicode_decode_error(
                            "utf7",
                            data,
                            pos.saturating_sub(1),
                            pos,
                            "partial character in shift sequence",
                        ));
                    }
                }
                if surrogate != 0 && utf7_decode_direct(ch) {
                    out.push(rustpython_wtf8::CodePoint::from_u32(surrogate).unwrap());
                }
                surrogate = 0;
                if ch == b'-' {
                    pos += 1;
                }
            }
        } else if ch == b'+' {
            pos += 1;
            if pos < data.len() && data[pos] == b'-' {
                pos += 1;
                out.push_char('+');
            } else if pos < data.len() && !utf7_is_base64(data[pos]) {
                if errors_s == "ignore" {
                    pos += 1;
                    continue;
                }
                return Err(crate::typedef::unicode_decode_error(
                    "utf7",
                    data,
                    pos.saturating_sub(1),
                    (pos + 1).min(data.len()),
                    "ill-formed sequence",
                ));
            } else {
                in_shift = true;
                surrogate = 0;
                shift_out_start = pos - 1;
                base64bits = 0;
                base64buffer = 0;
            }
        } else if utf7_decode_direct(ch) {
            out.push_char(ch as char);
            pos += 1;
        } else {
            if errors_s == "ignore" {
                pos += 1;
                continue;
            }
            return Err(crate::typedef::unicode_decode_error(
                "utf7",
                data,
                pos,
                (pos + 1).min(data.len()),
                "unexpected special character",
            ));
        }
    }
    if in_shift && (surrogate != 0 || base64bits >= 6 || (base64bits > 0 && base64buffer != 0)) {
        if errors_s != "ignore" {
            return Err(crate::typedef::unicode_decode_error(
                "utf7",
                data,
                shift_out_start,
                pos,
                "unterminated shift sequence",
            ));
        }
        pos = shift_out_start;
    }
    Ok(w_tuple_new(vec![
        w_str_from_wtf8(out),
        w_int_new(pos as i64),
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

fn hex_value(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u32),
        b'a'..=b'f' => Some((b - b'a' + 10) as u32),
        b'A'..=b'F' => Some((b - b'A' + 10) as u32),
        _ => None,
    }
}

fn unicode_escape_error(
    errors: &str,
    original: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    out: &mut rustpython_wtf8::Wtf8Buf,
) -> Result<(), crate::PyError> {
    match errors {
        "ignore" => Ok(()),
        "replace" => {
            out.push_char('\u{FFFD}');
            Ok(())
        }
        "backslashreplace" => {
            for &b in &original[start..end.min(original.len())] {
                out.push_str(&format!("\\x{b:02x}"));
            }
            Ok(())
        }
        _ => Err(crate::PyError::new(
            crate::PyErrorKind::UnicodeDecodeError,
            reason,
        )),
    }
}

fn unicode_escape_decode_impl(
    w_obj: PyObjectRef,
    errors: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let data: Vec<u8> = if unsafe { pyre_object::bytesobject::is_bytes_like(w_obj) } {
        unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) }.to_vec()
    } else if unsafe { is_str(w_obj) } {
        unsafe { w_str_get_wtf8(w_obj) }.as_bytes().to_vec()
    } else {
        return Err(crate::PyError::type_error(
            "unicode_escape_decode() argument must be bytes-like or str",
        ));
    };
    // PyPy `unicodehelper.py:str_decode_unicode_escape`.
    let errors_s = if unsafe { is_str(errors) } {
        unsafe { w_str_get_value(errors) }
    } else {
        "strict"
    };
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut pos = 0usize;
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
            unicode_escape_error(
                errors_s,
                &data,
                escape_start,
                data.len(),
                "\\ at end of string",
                &mut out,
            )?;
            break;
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
                let digits = match ch {
                    b'x' => 2,
                    b'u' => 4,
                    _ => 8,
                };
                let msg = match ch {
                    b'x' => "truncated \\xXX escape",
                    b'u' => "truncated \\uXXXX escape",
                    _ => "truncated \\UXXXXXXXX escape",
                };
                if pos + digits > data.len() {
                    unicode_escape_error(errors_s, &data, escape_start, data.len(), msg, &mut out)?;
                    pos = data.len();
                    continue;
                }
                let mut value = 0u32;
                let mut ok = true;
                for &b in &data[pos..pos + digits] {
                    if let Some(v) = hex_value(b) {
                        value = (value << 4) | v;
                    } else {
                        ok = false;
                        break;
                    }
                }
                let end = pos + digits;
                pos = end;
                if !ok || value > 0x10ffff {
                    unicode_escape_error(
                        errors_s,
                        &data,
                        escape_start,
                        end,
                        "illegal Unicode character",
                        &mut out,
                    )?;
                    continue;
                }
                out.push(rustpython_wtf8::CodePoint::from_u32(value).unwrap());
            }
            b'N' => {
                let mut end = pos;
                if pos < data.len() && data[pos] == b'{' {
                    end += 1;
                    while end < data.len() && data[end] != b'}' {
                        end += 1;
                    }
                    if end < data.len() && data[end] == b'}' {
                        end += 1;
                    }
                }
                unicode_escape_error(
                    errors_s,
                    &data,
                    escape_start,
                    end,
                    "unknown Unicode character name",
                    &mut out,
                )?;
                pos = end;
            }
            _ => {
                out.push_char('\\');
                out.push(rustpython_wtf8::CodePoint::from_u32(ch as u32).unwrap());
            }
        }
    }
    Ok(w_tuple_new(vec![
        w_str_from_wtf8(out),
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
            #[default(w_bool_from(false))] _final: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            utf7_decode_impl(obj, errors)
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
        "register_error"   / 2 = |_| Ok(w_none()),
        "_unregister_error" / 1 = |_| Ok(w_bool_from(false)),
        "register"       / 1 = register_codec,
        "unregister"     / 1 = unregister,
        "lookup"         / 1 = lookup_codec,
        "_forget_codec"  / 1 = forget_codec,
        "charmap_build"  / 1 = charmap_build,
    },
}
