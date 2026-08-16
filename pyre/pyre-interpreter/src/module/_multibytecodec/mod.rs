//! PyPy `_multibytecodec`: wrappers for its vendored cjkcodecs engine.

use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};
use std::ffi::c_void;

const MBERR_TOOFEW: isize = -2;
const MBERR_INTERNAL: isize = -3;
const MBERR_NOMEMORY: isize = -4;
const MBENC_FLUSH: isize = 1;
const MBENC_RESET: isize = 2;

#[cfg(windows)]
type CjkWchar = u16;
#[cfg(not(windows))]
type CjkWchar = u32;

unsafe extern "C" {
    fn pypy_cjkcodec_shift_jis() -> *const c_void;
    fn pypy_cjkcodec_cp932() -> *const c_void;
    fn pypy_cjkcodec_euc_jp() -> *const c_void;
    fn pypy_cjkcodec_shift_jis_2004() -> *const c_void;
    fn pypy_cjkcodec_euc_jis_2004() -> *const c_void;
    fn pypy_cjkcodec_euc_jisx0213() -> *const c_void;
    fn pypy_cjkcodec_shift_jisx0213() -> *const c_void;

    fn pypy_cjk_dec_new(codec: *const c_void) -> *mut c_void;
    fn pypy_cjk_dec_init(state: *mut c_void, input: *mut i8, len: isize) -> isize;
    fn pypy_cjk_dec_free(state: *mut c_void);
    fn pypy_cjk_dec_chunk(state: *mut c_void) -> isize;
    fn pypy_cjk_dec_outbuf(state: *mut c_void) -> *const CjkWchar;
    fn pypy_cjk_dec_outlen(state: *mut c_void) -> isize;
    fn pypy_cjk_dec_inbuf_remaining(state: *mut c_void) -> isize;
    fn pypy_cjk_dec_inbuf_consumed(state: *mut c_void) -> isize;
    fn pypy_cjk_dec_replace_on_error(
        state: *mut c_void,
        replacement: *const CjkWchar,
        replacement_len: isize,
        new_end: isize,
    ) -> isize;

    fn pypy_cjk_enc_new(codec: *const c_void) -> *mut c_void;
    fn pypy_cjk_enc_init(state: *mut c_void, input: *mut CjkWchar, len: isize) -> isize;
    fn pypy_cjk_enc_free(state: *mut c_void);
    fn pypy_cjk_enc_chunk(state: *mut c_void, flags: isize) -> isize;
    fn pypy_cjk_enc_reset(state: *mut c_void) -> isize;
    fn pypy_cjk_enc_outbuf(state: *mut c_void) -> *const i8;
    fn pypy_cjk_enc_outlen(state: *mut c_void) -> isize;
    fn pypy_cjk_enc_inbuf_remaining(state: *mut c_void) -> isize;
    fn pypy_cjk_enc_inbuf_consumed(state: *mut c_void) -> isize;
    fn pypy_cjk_enc_replace_on_error(
        state: *mut c_void,
        replacement: *const i8,
        replacement_len: isize,
        new_end: isize,
    ) -> isize;
}

fn codec_ptr(name: &str) -> Option<*const c_void> {
    // `c_codecs.py`'s complete `_codecs_jp` getter table.
    let ptr = unsafe {
        match name {
            "shift_jis" => pypy_cjkcodec_shift_jis(),
            "cp932" => pypy_cjkcodec_cp932(),
            "euc_jp" => pypy_cjkcodec_euc_jp(),
            "shift_jis_2004" => pypy_cjkcodec_shift_jis_2004(),
            "euc_jis_2004" => pypy_cjkcodec_euc_jis_2004(),
            "euc_jisx0213" => pypy_cjkcodec_euc_jisx0213(),
            "shift_jisx0213" => pypy_cjkcodec_shift_jisx0213(),
            _ => return None,
        }
    };
    (!ptr.is_null()).then_some(ptr)
}

struct DecState(*mut c_void);
impl Drop for DecState {
    fn drop(&mut self) {
        unsafe { pypy_cjk_dec_free(self.0) };
    }
}

struct EncState(*mut c_void);
impl Drop for EncState {
    fn drop(&mut self) {
        unsafe { pypy_cjk_enc_free(self.0) };
    }
}

#[cfg(windows)]
fn text_to_wchars(text: &Wtf8) -> (Vec<CjkWchar>, Vec<usize>) {
    let mut units = Vec::new();
    let mut boundaries = vec![0];
    for cp in text.code_points() {
        let value = cp.to_u32();
        if value <= 0xffff {
            units.push(value as u16);
        } else {
            let value = value - 0x10000;
            units.push(0xd800 | ((value >> 10) as u16));
            units.push(0xdc00 | ((value & 0x3ff) as u16));
        }
        boundaries.push(units.len());
    }
    (units, boundaries)
}

#[cfg(not(windows))]
fn text_to_wchars(text: &Wtf8) -> (Vec<CjkWchar>, Vec<usize>) {
    let units: Vec<u32> = text.code_points().map(|cp| cp.to_u32()).collect();
    let boundaries = (0..=units.len()).collect();
    (units, boundaries)
}

fn wchar_consumed_to_codepoints(boundaries: &[usize], consumed: usize) -> usize {
    match boundaries.binary_search(&consumed) {
        Ok(index) => index,
        Err(index) => index.saturating_sub(1),
    }
}

#[cfg(windows)]
fn wchars_to_text(units: &[CjkWchar]) -> Wtf8Buf {
    let mut out = Wtf8Buf::new();
    for decoded in char::decode_utf16(units.iter().copied()) {
        match decoded {
            Ok(ch) => out.push_char(ch),
            Err(err) => out.push(CodePoint::from_u32(err.unpaired_surrogate() as u32).unwrap()),
        }
    }
    out
}

#[cfg(not(windows))]
fn wchars_to_text(units: &[CjkWchar]) -> Wtf8Buf {
    let mut out = Wtf8Buf::new();
    for &unit in units {
        if let Some(cp) = CodePoint::from_u32(unit) {
            out.push(cp);
        }
    }
    out
}

fn decode_impl(
    name: &str,
    input: &[u8],
    errors: &str,
    final_input: bool,
) -> Result<(Wtf8Buf, usize), crate::PyError> {
    let codec = codec_ptr(name).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::LookupError,
            "no such codec is supported.",
        )
    })?;
    // `c_codecs.py:108-114` raises before binding the state, and
    // `pypy_cjk_dec_free` (multibytecodec.c:41) dereferences its argument, so
    // the null check has to happen before the RAII wrapper owns the pointer.
    let raw = unsafe { pypy_cjk_dec_new(codec) };
    if raw.is_null() {
        return Err(crate::PyError::memory_error(""));
    }
    let state = DecState(raw);
    let mut owned_input = input.to_vec();
    if unsafe {
        pypy_cjk_dec_init(
            state.0,
            owned_input.as_mut_ptr().cast::<i8>(),
            owned_input.len() as isize,
        )
    } < 0
    {
        return Err(crate::PyError::memory_error(""));
    }
    loop {
        let result = unsafe { pypy_cjk_dec_chunk(state.0) };
        if result == 0 || (!final_input && result == MBERR_TOOFEW) {
            break;
        }
        let (reason, error_size) = if result > 0 {
            ("illegal multibyte sequence", result as usize)
        } else if result == MBERR_TOOFEW {
            (
                "incomplete multibyte sequence",
                unsafe { pypy_cjk_dec_inbuf_remaining(state.0) }.max(0) as usize,
            )
        } else if result == MBERR_NOMEMORY {
            return Err(crate::PyError::memory_error(""));
        } else {
            debug_assert_eq!(result, MBERR_INTERNAL);
            return Err(crate::PyError::runtime_error("internal codec error"));
        };
        let start = unsafe { pypy_cjk_dec_inbuf_consumed(state.0) }.max(0) as usize;
        let end = start.saturating_add(error_size).min(input.len());
        let replacement: &[CjkWchar] = match errors {
            "strict" => {
                return Err(crate::typedef::unicode_decode_error(
                    name, input, start, end, reason,
                ));
            }
            "ignore" => &[],
            "replace" => &[0xfffd],
            _ => {
                crate::module::_codecs::validate_error_handler(errors)?;
                return Err(crate::PyError::not_implemented(format!(
                    "{name} codec does not yet support the '{errors}' error handler"
                )));
            }
        };
        let result = unsafe {
            pypy_cjk_dec_replace_on_error(
                state.0,
                replacement.as_ptr(),
                replacement.len() as isize,
                end as isize,
            )
        };
        if result == MBERR_NOMEMORY {
            return Err(crate::PyError::memory_error(""));
        }
    }
    let consumed = unsafe { pypy_cjk_dec_inbuf_consumed(state.0) }.max(0) as usize;
    let output_len = unsafe { pypy_cjk_dec_outlen(state.0) }.max(0) as usize;
    let output = unsafe { std::slice::from_raw_parts(pypy_cjk_dec_outbuf(state.0), output_len) };
    Ok((wchars_to_text(output), consumed.min(input.len())))
}

fn encode_impl(
    name: &str,
    w_input: PyObjectRef,
    errors: &str,
    final_input: bool,
) -> Result<(Vec<u8>, usize), crate::PyError> {
    if unsafe { !pyre_object::is_str(w_input) } {
        return Err(crate::PyError::type_error("encoder argument must be str"));
    }
    let input = unsafe { pyre_object::w_str_get_wtf8(w_input) };
    let (mut units, boundaries) = text_to_wchars(input);
    let codec = codec_ptr(name).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::LookupError,
            "no such codec is supported.",
        )
    })?;
    // Same ownership order as `decode_impl`: `pypy_cjk_enc_free`
    // (multibytecodec.c:174) dereferences its argument.
    let raw = unsafe { pypy_cjk_enc_new(codec) };
    if raw.is_null() {
        return Err(crate::PyError::memory_error(""));
    }
    let state = EncState(raw);
    if unsafe { pypy_cjk_enc_init(state.0, units.as_mut_ptr(), units.len() as isize) } < 0 {
        return Err(crate::PyError::memory_error(""));
    }
    let flags = if final_input {
        MBENC_FLUSH | MBENC_RESET
    } else {
        0
    };
    loop {
        let result = unsafe { pypy_cjk_enc_chunk(state.0, flags) };
        if result == 0 || (!final_input && result == MBERR_TOOFEW) {
            break;
        }
        let (reason, error_units) = if result > 0 {
            ("illegal multibyte sequence", result as usize)
        } else if result == MBERR_TOOFEW {
            (
                "incomplete multibyte sequence",
                unsafe { pypy_cjk_enc_inbuf_remaining(state.0) }.max(0) as usize,
            )
        } else if result == MBERR_NOMEMORY {
            return Err(crate::PyError::memory_error(""));
        } else {
            debug_assert_eq!(result, MBERR_INTERNAL);
            return Err(crate::PyError::runtime_error("internal codec error"));
        };
        let start_units = unsafe { pypy_cjk_enc_inbuf_consumed(state.0) }.max(0) as usize;
        let end_units = start_units.saturating_add(error_units).min(units.len());
        let start = wchar_consumed_to_codepoints(&boundaries, start_units);
        let end = wchar_consumed_to_codepoints(&boundaries, end_units);
        let replacement: &[u8] = match errors {
            "strict" => {
                return Err(crate::typedef::unicode_encode_error(
                    name, w_input, start, end, reason,
                ));
            }
            "ignore" => b"",
            "replace" => b"?",
            _ => {
                crate::module::_codecs::validate_error_handler(errors)?;
                return Err(crate::PyError::not_implemented(format!(
                    "{name} codec does not yet support the '{errors}' error handler"
                )));
            }
        };
        let result = unsafe {
            pypy_cjk_enc_replace_on_error(
                state.0,
                replacement.as_ptr().cast::<i8>(),
                replacement.len() as isize,
                end_units as isize,
            )
        };
        if result == MBERR_NOMEMORY {
            return Err(crate::PyError::memory_error(""));
        }
    }
    if final_input {
        loop {
            let result = unsafe { pypy_cjk_enc_reset(state.0) };
            if result == 0 {
                break;
            }
            if result == MBERR_NOMEMORY {
                return Err(crate::PyError::memory_error(""));
            }
            return Err(crate::PyError::runtime_error("internal codec error"));
        }
    }
    let consumed_units = unsafe { pypy_cjk_enc_inbuf_consumed(state.0) }.max(0) as usize;
    let consumed = wchar_consumed_to_codepoints(&boundaries, consumed_units.min(units.len()));
    let output_len = unsafe { pypy_cjk_enc_outlen(state.0) }.max(0) as usize;
    let output = unsafe {
        std::slice::from_raw_parts(pypy_cjk_enc_outbuf(state.0).cast::<u8>(), output_len).to_vec()
    };
    Ok((output, consumed))
}

pub(crate) fn getcodec(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(w_name) = args.first().copied() else {
        return Err(crate::PyError::type_error("getcodec() missing codec name"));
    };
    let name = crate::baseobjspace::text_w(w_name)?;
    if codec_ptr(name).is_none() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::LookupError,
            "no such codec is supported.",
        ));
    }
    // `_codecs_jp` reaches this through `from _multibytecodec import
    // __getcodec`, i.e. an import — so resolve the module the same way rather
    // than requiring some earlier importer to have populated `sys.modules`.
    let module = match crate::importing::get_sys_module("_multibytecodec") {
        Some(module) => module,
        None => crate::importing::importhook(
            "_multibytecodec",
            pyre_object::PY_NULL,
            pyre_object::w_tuple_new(vec![pyre_object::w_str_new("MultibyteCodec")]),
            0,
            crate::call::getexecutioncontext(),
        )?,
    };
    let cls = crate::baseobjspace::getattr_str(module, "MultibyteCodec")?;
    crate::call::call_function_impl_result(cls, &[w_name])
}

fn raw_encode(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, _) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() < 4 {
        return Err(crate::PyError::type_error("_encode() requires 4 arguments"));
    }
    let name = crate::baseobjspace::text_w(positional[0])?;
    let errors = crate::baseobjspace::text_w(positional[2])?;
    let final_input = crate::baseobjspace::is_true(positional[3])?;
    let (output, consumed) = encode_impl(name, positional[1], errors, final_input)?;
    Ok(w_tuple_new(vec![
        pyre_object::bytesobject::w_bytes_from_bytes(&output),
        w_int_new(consumed as i64),
    ]))
}

fn raw_decode(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, _) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() < 4 {
        return Err(crate::PyError::type_error("_decode() requires 4 arguments"));
    }
    let name = crate::baseobjspace::text_w(positional[0])?;
    let input = crate::baseobjspace::simple_buffer_bytes(positional[1])?.ok_or_else(|| {
        crate::PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            crate::type_methods::arg_type_name(positional[1])
        ))
    })?;
    let errors = crate::baseobjspace::text_w(positional[2])?;
    let final_input = crate::baseobjspace::is_true(positional[3])?;
    let (output, consumed) = decode_impl(name, input.as_bytes(), errors, final_input)?;
    Ok(w_tuple_new(vec![
        pyre_object::unicodeobject::w_str_from_wtf8_managed(output),
        w_int_new(consumed as i64),
    ]))
}

crate::py_module! {
    "_multibytecodec",
    functions: {
        "__getcodec" / 1 = getcodec,
        "_encode" / 4 = raw_encode,
        "_decode" / 4 = raw_decode,
    },
    extra_init: |ns| {
        // The app classes close over the two engine entry points. Install
        // them only after the macro has populated those functions in the
        // module, as PyPy's MixedModule does before evaluating app-level defs.
        let encode = unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_encode") }
            .expect("_multibytecodec._encode is installed");
        let decode = unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_decode") }
            .expect("_multibytecodec._decode is installed");
        crate::importing::appleveldef_install_seeded(
            ns,
            include_str!("app_multibytecodec.py"),
            "app_multibytecodec.py",
            &[
                "MultibyteCodec",
                "MultibyteIncrementalDecoder",
                "MultibyteIncrementalEncoder",
                "MultibyteStreamReader",
                "MultibyteStreamWriter",
            ],
            &[("_encode", encode), ("_decode", decode)],
        );
    },
}
