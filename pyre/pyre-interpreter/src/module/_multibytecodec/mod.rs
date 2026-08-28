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
    fn pypy_cjkcodec_gb2312() -> *const c_void;
    fn pypy_cjkcodec_gbk() -> *const c_void;
    fn pypy_cjkcodec_gb18030() -> *const c_void;
    fn pypy_cjkcodec_hz() -> *const c_void;
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
    fn pypy_cjk_dec_getstate(state: *mut c_void, output: *mut u8);
    fn pypy_cjk_dec_setstate(state: *mut c_void, input: *const u8);

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
    fn pypy_cjk_enc_getstate(state: *mut c_void, output: *mut u8);
    fn pypy_cjk_enc_setstate(state: *mut c_void, input: *const u8);
}

fn codec_ptr(name: &str) -> Option<*const c_void> {
    // `c_codecs.py`'s complete `_codecs_cn` and `_codecs_jp` getter tables.
    let ptr = unsafe {
        match name {
            "gb2312" => pypy_cjkcodec_gb2312(),
            "gbk" => pypy_cjkcodec_gbk(),
            "gb18030" => pypy_cjkcodec_gb18030(),
            "hz" => pypy_cjkcodec_hz(),
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

/// The app-level port recreates the C buffer for each call, while PyPy keeps
/// one `decodebuf` on `MultibyteIncrementalDecoder`.  Export the native state
/// back to the object-owned bytearray on every exit, including an exception.
struct DecStateExport {
    state: *mut c_void,
    sink_slot: Option<usize>,
}

impl Drop for DecStateExport {
    fn drop(&mut self) {
        let Some(sink_slot) = self.sink_slot else {
            return;
        };
        let sink = pyre_object::gc_roots::shadow_stack_get(sink_slot);
        let output = unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(sink) };
        if output.len() == 8 {
            unsafe { pypy_cjk_dec_getstate(self.state, output.as_mut_ptr()) };
        }
    }
}

/// Encoder counterpart of [`DecStateExport`], preserving PyPy encodebuf state
/// even when `encodeex` raises after changing a shift mode.
struct EncStateExport {
    state: *mut c_void,
    sink_slot: Option<usize>,
}

impl Drop for EncStateExport {
    fn drop(&mut self) {
        let Some(sink_slot) = self.sink_slot else {
            return;
        };
        let sink = pyre_object::gc_roots::shadow_stack_get(sink_slot);
        let output = unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(sink) };
        if output.len() == 8 {
            unsafe { pypy_cjk_enc_getstate(self.state, output.as_mut_ptr()) };
        }
    }
}

fn encoder_state(state: &EncState) -> [u8; 8] {
    let mut output = [0; 8];
    unsafe { pypy_cjk_enc_getstate(state.0, output.as_mut_ptr()) };
    output
}

fn set_encoder_state(state: &EncState, input: &[u8; 8]) {
    unsafe { pypy_cjk_enc_setstate(state.0, input.as_ptr()) };
}

/// PyPy `c_codecs.encode` copies a replacement sub-encode's state back from
/// its `finally` arm even when that sub-encode raises.  The sub-engine guard
/// has exported that state to `sink`; install it on the outer engine before
/// its own guard runs.
fn restore_exported_encoder_state(state: &EncState, sink_slot: Option<usize>) {
    let Some(sink_slot) = sink_slot else {
        return;
    };
    let sink = pyre_object::gc_roots::shadow_stack_get(sink_slot);
    let exported = unsafe { pyre_object::bytearrayobject::w_bytearray_data(sink) };
    if let Ok(exported) = <&[u8; 8]>::try_from(exported) {
        set_encoder_state(state, exported);
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

/// [`wchar_consumed_to_codepoints`] rounded the other way: the first code point
/// that starts at or after `consumed`.
fn wchar_codepoints_covering(boundaries: &[usize], consumed: usize) -> usize {
    match boundaries.binary_search(&consumed) {
        Ok(index) | Err(index) => index,
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
        // Every unit here came from a mapping table or from a replacement the
        // error handler already produced as text, so it is a code point.
        // Skipping a unit that is not would drop a character silently.
        out.push(CodePoint::from_u32(unit).expect("cjkcodecs unit is a code point"));
    }
    out
}

fn decode_impl_with_state(
    name: &str,
    input: &[u8],
    errors: &str,
    final_input: bool,
    initial_state: Option<&[u8; 8]>,
    state_sink: Option<PyObjectRef>,
) -> Result<(Wtf8Buf, usize, [u8; 8]), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let state_sink_slot = state_sink.map(|sink| {
        let slot = roots.base();
        let _ = roots.pin_root(sink);
        slot
    });
    let codec = codec_ptr(name).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::LookupError,
            "no such codec is supported.",
        )
    })?;
    // PyPy `c_codecs.decode` raises before binding the state, and
    // `pypy_cjk_dec_free` dereferences its argument, so
    // the null check has to happen before the RAII wrapper owns the pointer.
    let raw = unsafe { pypy_cjk_dec_new(codec) };
    if raw.is_null() {
        return Err(crate::PyError::memory_error(""));
    }
    let state = DecState(raw);
    if let Some(initial_state) = initial_state {
        unsafe { pypy_cjk_dec_setstate(state.0, initial_state.as_ptr()) };
    }
    let _state_export = DecStateExport {
        state: state.0,
        sink_slot: state_sink_slot,
    };
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
        // `c_codecs.py multibytecodec_decerror`: the three built-in
        // modes answer inline, anything else goes to the registered handler,
        // which also names the position decoding resumes at.
        let (replacement, resume_end): (Vec<CjkWchar>, usize) = match errors {
            "strict" => {
                return Err(crate::typedef::unicode_decode_error(
                    name, input, start, end, reason,
                ));
            }
            "ignore" => (Vec::new(), end),
            "replace" => (vec![0xfffd], end),
            _ => {
                let mut text = Wtf8Buf::new();
                // `multibytecodec_decerror` folds the returned position against
                // the buffer decoding started on and goes on decoding that one,
                // so a handler that put different bytes on `exc.object` neither
                // redirects it nor widens the range a position may name.
                let newpos = crate::type_methods::call_registered_multibyte_decode_error_handler(
                    errors, name, input, start, end, reason, &mut text,
                )?;
                (text_to_wchars(&text).0, newpos)
            }
        };
        let result = unsafe {
            pypy_cjk_dec_replace_on_error(
                state.0,
                replacement.as_ptr(),
                replacement.len() as isize,
                resume_end as isize,
            )
        };
        if result == MBERR_NOMEMORY {
            return Err(crate::PyError::memory_error(""));
        }
    }
    let consumed = unsafe { pypy_cjk_dec_inbuf_consumed(state.0) }.max(0) as usize;
    let output_len = unsafe { pypy_cjk_dec_outlen(state.0) }.max(0) as usize;
    let output = unsafe { std::slice::from_raw_parts(pypy_cjk_dec_outbuf(state.0), output_len) };
    let mut next_state = [0; 8];
    unsafe { pypy_cjk_dec_getstate(state.0, next_state.as_mut_ptr()) };
    Ok((
        wchars_to_text(output),
        consumed.min(input.len()),
        next_state,
    ))
}

fn decode_impl(
    name: &str,
    input: &[u8],
    errors: &str,
    final_input: bool,
) -> Result<(Wtf8Buf, usize), crate::PyError> {
    let (output, consumed, _) =
        decode_impl_with_state(name, input, errors, final_input, None, None)?;
    Ok((output, consumed))
}

/// PyPy `c_codecs.multibytecodec_encerror`: a rettype 'u' replacement is not
/// bytes yet, it goes
/// back through the same codec.  "strict" so a replacement the codec cannot
/// encode raises there instead of re-entering the error handler.
///
/// As PyPy's `c_codecs.encode(copystate=encodebuf)` does, the sub-encode starts
/// from the outer engine state and copies its resulting state back.
fn encode_replacement_text(
    name: &str,
    w_text: PyObjectRef,
    state: &[u8; 8],
    state_sink: Option<PyObjectRef>,
) -> Result<(Vec<u8>, [u8; 8]), crate::PyError> {
    let (output, _, next_state) =
        encode_impl_with_state(name, w_text, "strict", true, Some(state), state_sink)?;
    Ok((output, next_state))
}

fn encode_impl_with_state(
    name: &str,
    w_input: PyObjectRef,
    errors: &str,
    final_input: bool,
    initial_state: Option<&[u8; 8]>,
    state_sink: Option<PyObjectRef>,
) -> Result<(Vec<u8>, usize, [u8; 8]), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let input_slot = roots.base();
    let _ = roots.pin_root(w_input);
    let state_sink_slot = state_sink.map(|sink| {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(sink);
        slot
    });
    let w_input = roots.get(input_slot);
    if unsafe { !pyre_object::is_str(w_input) } {
        return Err(crate::PyError::type_error("encoder argument must be str"));
    }
    // A registered error handler runs Python between iterations of the loop
    // below, which still names this string in the error it reports and hands
    // it to the handler again, so it is held on the shadow stack and reread
    // from there rather than kept as a Rust local across the call.
    let roots = pyre_object::gc_roots::push_roots();
    let input_slot = roots.base();
    let _ = roots.pin_root(w_input);
    let input = unsafe { pyre_object::w_str_get_wtf8(roots.get(input_slot)) };
    let (mut units, boundaries) = text_to_wchars(input);
    let codec = codec_ptr(name).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::LookupError,
            "no such codec is supported.",
        )
    })?;
    // Same ownership order as `decode_impl`: `pypy_cjk_enc_free` dereferences
    // its argument.
    let raw = unsafe { pypy_cjk_enc_new(codec) };
    if raw.is_null() {
        return Err(crate::PyError::memory_error(""));
    }
    let state = EncState(raw);
    if let Some(initial_state) = initial_state {
        set_encoder_state(&state, initial_state);
    }
    let _state_export = EncStateExport {
        state: state.0,
        sink_slot: state_sink_slot,
    };
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
        let error_units_end = start_units.saturating_add(error_units).min(units.len());
        // The engine measures the error in wchar units, and where `wchar_t` is
        // 16 bits an astral code point is two of them.  The span an error
        // handler works in is code points, so a unit that is only half of one
        // widens to the whole code point, and the engine resumes there.
        let start = wchar_consumed_to_codepoints(&boundaries, start_units);
        let end = wchar_codepoints_covering(&boundaries, error_units_end);
        let end_units = boundaries[end];
        // `c_codecs.py multibytecodec_encerror`: `ignore` splices in
        // nothing (rettype 'b'), `replace` and a str from a registered handler
        // are re-encoded through the same codec first (rettype 'u'), and bytes
        // from a handler are spliced in verbatim.  A handler also names the
        // position to resume at as a code-point index, so it goes back through
        // `boundaries` to reach the wchar unit the engine restarts on.
        let (replacement, resume_units): (Vec<u8>, usize) = match errors {
            "strict" => {
                return Err(crate::typedef::unicode_encode_error(
                    name,
                    roots.get(input_slot),
                    start as i64,
                    end as i64,
                    reason,
                ));
            }
            "ignore" => (Vec::new(), end_units),
            "replace" => {
                let replacement = encode_replacement_text(
                    name,
                    w_str_new("?"),
                    &encoder_state(&state),
                    state_sink_slot.map(pyre_object::gc_roots::shadow_stack_get),
                );
                let (bytes, next_state) = match replacement {
                    Ok(result) => result,
                    Err(error) => {
                        restore_exported_encoder_state(&state, state_sink_slot);
                        return Err(error);
                    }
                };
                set_encoder_state(&state, &next_state);
                (bytes, end_units)
            }
            _ => {
                let (replacement, newpos) =
                    crate::type_methods::call_registered_encode_error_handler(
                        errors,
                        name,
                        roots.get(input_slot),
                        boundaries.len() - 1,
                        start,
                        end,
                        reason,
                    )?;
                let bytes = match replacement {
                    crate::type_methods::EncodeReplacement::Bytes(bytes) => bytes,
                    crate::type_methods::EncodeReplacement::Str(points) => {
                        let mut text = Wtf8Buf::new();
                        for point in points {
                            text.push(CodePoint::from_u32(point).unwrap());
                        }
                        let replacement = encode_replacement_text(
                            name,
                            w_str_from_wtf8_managed(text),
                            &encoder_state(&state),
                            state_sink_slot.map(pyre_object::gc_roots::shadow_stack_get),
                        );
                        let (bytes, next_state) = match replacement {
                            Ok(result) => result,
                            Err(error) => {
                                restore_exported_encoder_state(&state, state_sink_slot);
                                return Err(error);
                            }
                        };
                        set_encoder_state(&state, &next_state);
                        bytes
                    }
                };
                (bytes, boundaries[newpos])
            }
        };
        let result = unsafe {
            pypy_cjk_enc_replace_on_error(
                state.0,
                replacement.as_ptr().cast::<i8>(),
                replacement.len() as isize,
                resume_units as isize,
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
    Ok((output, consumed, encoder_state(&state)))
}

fn encode_impl(
    name: &str,
    w_input: PyObjectRef,
    errors: &str,
    final_input: bool,
) -> Result<(Vec<u8>, usize), crate::PyError> {
    let (output, consumed, _) =
        encode_impl_with_state(name, w_input, errors, final_input, None, None)?;
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

fn codec_call_control(
    w_control: PyObjectRef,
) -> Result<(bool, [u8; 8], PyObjectRef), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let control_slot = roots.base();
    let w_control = roots.pin_root(w_control);
    if unsafe { !pyre_object::is_tuple(w_control) || pyre_object::w_tuple_len(w_control) != 2 } {
        return Err(crate::PyError::type_error(
            "codec control must be a (final, state) tuple",
        ));
    }
    let w_final = unsafe { pyre_object::w_tuple_getitem(w_control, 0) }
        .expect("a two-item codec control has a first item");
    let final_input = crate::baseobjspace::is_true(w_final)?;
    // `is_true` may execute Python.  Read the state back from the rooted,
    // possibly forwarded control tuple after that call.
    let w_state = unsafe { pyre_object::w_tuple_getitem(roots.get(control_slot), 1) }
        .expect("a two-item codec control has a second item");
    if unsafe { !pyre_object::is_bytearray(w_state) } {
        return Err(crate::PyError::type_error(
            "codec state must be an exact bytearray",
        ));
    }
    let state: [u8; 8] = unsafe { pyre_object::bytearrayobject::w_bytearray_data(w_state) }
        .try_into()
        .map_err(|_| crate::PyError::value_error("codec state must contain exactly 8 bytes"))?;
    Ok((final_input, state, w_state))
}

fn raw_encode_stateful(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, _) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() < 4 {
        return Err(crate::PyError::type_error(
            "_encode_stateful() requires 4 arguments",
        ));
    }
    let name = crate::baseobjspace::text_w(positional[0])?;
    let errors = crate::baseobjspace::text_w(positional[2])?;
    let (final_input, state, w_state) = codec_call_control(positional[3])?;
    let (output, consumed, _) = encode_impl_with_state(
        name,
        positional[1],
        errors,
        final_input,
        Some(&state),
        Some(w_state),
    )?;
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

fn raw_decode_stateful(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, _) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() < 4 {
        return Err(crate::PyError::type_error(
            "_decode_stateful() requires 4 arguments",
        ));
    }
    let name = crate::baseobjspace::text_w(positional[0])?;
    let input = crate::baseobjspace::simple_buffer_bytes(positional[1])?.ok_or_else(|| {
        crate::PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            crate::type_methods::arg_type_name(positional[1])
        ))
    })?;
    let errors = crate::baseobjspace::text_w(positional[2])?;
    let (final_input, state, w_state) = codec_call_control(positional[3])?;
    let (output, consumed, _) = decode_impl_with_state(
        name,
        input.as_bytes(),
        errors,
        final_input,
        Some(&state),
        Some(w_state),
    )?;
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
        "_encode_stateful" / 4 = raw_encode_stateful,
        "_decode" / 4 = raw_decode,
        "_decode_stateful" / 4 = raw_decode_stateful,
    },
    extra_init: |ns| {
        // The app classes close over the two engine entry points. Install
        // them only after the macro has populated those functions in the
        // module, as PyPy's MixedModule does before evaluating app-level defs.
        let encode = unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_encode") }
            .expect("_multibytecodec._encode is installed");
        let decode = unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_decode") }
            .expect("_multibytecodec._decode is installed");
        let encode_stateful = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_encode_stateful")
        }
        .expect("_multibytecodec._encode_stateful is installed");
        let decode_stateful = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(ns, "_decode_stateful")
        }
        .expect("_multibytecodec._decode_stateful is installed");
        crate::importing::appleveldef_install_seeded(
            ns,
            include_str!("app_multibytecodec.py"),
            "app_multibytecodec.py",
            "_multibytecodec",
            &[
                "MultibyteCodec",
                "MultibyteIncrementalDecoder",
                "MultibyteIncrementalEncoder",
                "MultibyteStreamReader",
                "MultibyteStreamWriter",
            ],
            &[
                ("_encode", encode),
                ("_decode", decode),
                ("_encode_stateful", encode_stateful),
                ("_decode_stateful", decode_stateful),
            ],
        )?;
    },
}
