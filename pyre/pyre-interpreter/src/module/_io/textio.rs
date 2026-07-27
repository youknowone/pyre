//! Text streams — PyPy `pypy/module/_io/interp_textio.py`.
//!
//! Keep the stream state on the typed object, matching
//! `W_TextIOWrapper`.  In particular, the buffer and the
//! ZERO/OK/DETACHED state are not instance-dict side data.

use pyre_object::*;
use rustpython_wtf8::{Wtf8, Wtf8Buf};

const STATE_ZERO: i64 = 0;
const STATE_OK: i64 = 1;
const STATE_DETACHED: i64 = 2;

/// PyPy `PositionCookie`.  Each field occupies one native unsigned word in
/// the opaque integer returned by `tell()`.
#[derive(Default)]
struct PositionCookie {
    start_pos: u64,
    dec_flags: u64,
    bytes_to_feed: u64,
    chars_to_skip: u64,
    need_eof: bool,
}

impl PositionCookie {
    const BITS: usize = u64::BITS as usize;

    fn unpack(mut value: RBigInt) -> Result<Self, crate::PyError> {
        if value.int_lt(0) {
            return Err(crate::PyError::value_error("negative seek position"));
        }
        // interp_textio.py:253-269 `PositionCookie.__init__`: extract each
        // native word with rbigint's mask conversion, then shift the source.
        // In particular, do not call `to_u64()` on the whole cookie — valid
        // packed cookies intentionally occupy several machine words.
        let start_pos = value.ulonglongmask();
        value = value
            .rshift(Self::BITS as i64, false)
            .expect("native-word shift");
        let dec_flags = value.uintmask();
        value = value
            .rshift(Self::BITS as i64, false)
            .expect("native-word shift");
        let bytes_to_feed = value.uintmask();
        value = value
            .rshift(Self::BITS as i64, false)
            .expect("native-word shift");
        let chars_to_skip = value.uintmask();
        value = value
            .rshift(Self::BITS as i64, false)
            .expect("native-word shift");
        let need_eof = value.tobool();
        Ok(Self {
            start_pos,
            dec_flags,
            bytes_to_feed,
            chars_to_skip,
            need_eof,
        })
    }

    fn pack(&self) -> RBigInt {
        let mut result = RBigInt::from(self.start_pos);
        result = result.or_(
            &RBigInt::from(self.dec_flags)
                .lshift(Self::BITS as i64)
                .expect("native-word shift"),
        );
        result = result.or_(
            &RBigInt::from(self.bytes_to_feed)
                .lshift((Self::BITS * 2) as i64)
                .expect("native-word shift"),
        );
        result = result.or_(
            &RBigInt::from(self.chars_to_skip)
                .lshift((Self::BITS * 3) as i64)
                .expect("native-word shift"),
        );
        if self.need_eof {
            result = result.or_(
                &RBigInt::one()
                    .lshift((Self::BITS * 4) as i64)
                    .expect("native-word shift"),
            );
        }
        result
    }

    fn to_object(&self) -> PyObjectRef {
        crate::objspace::descroperation::bigint_result(self.pack())
    }
}

/// PyPy `PositionSnapshot`.
struct PositionSnapshot {
    flags: u64,
    input: Vec<u8>,
}

/// PyPy `interp_textio.DecodeBuffer`.
#[derive(Default)]
struct DecodeBuffer {
    text: Option<Wtf8Buf>,
    pos: usize,
    upos: usize,
    ulen: usize,
}

impl DecodeBuffer {
    fn set(&mut self, decoded: PyObjectRef) -> Result<(), crate::PyError> {
        if unsafe { !pyre_object::is_str(decoded) } {
            return Err(crate::PyError::type_error(format!(
                "decoder should return a string result, not '{}'",
                crate::type_methods::arg_type_name(decoded)
            )));
        }
        let text = unsafe { pyre_object::w_str_get_wtf8(decoded) }.to_wtf8_buf();
        self.ulen = text.code_points().count();
        self.text = Some(text);
        self.pos = 0;
        self.upos = 0;
        Ok(())
    }

    fn reset(&mut self) {
        self.text = None;
        self.pos = 0;
        self.upos = 0;
        self.ulen = 0;
    }

    fn has_data(&self) -> bool {
        self.text.is_some() && !self.exhausted()
    }

    fn exhausted(&self) -> bool {
        self.text.as_ref().is_none_or(|text| self.pos >= text.len())
    }

    fn available(&self) -> usize {
        self.ulen.saturating_sub(self.upos)
    }

    fn advance(&mut self) {
        let text = self.text.as_ref().expect("DecodeBuffer text");
        let cp = text[self.pos..]
            .code_points()
            .next()
            .expect("advance requires available data");
        self.pos += cp.len_wtf8();
        self.upos += 1;
    }

    fn get_chars(&mut self, size: Option<usize>) -> Wtf8Buf {
        let Some(text) = self.text.as_ref() else {
            return Wtf8Buf::new();
        };
        let count = size.unwrap_or(self.available()).min(self.available());
        let start = self.pos;
        let end = text[start..]
            .code_point_indices()
            .nth(count)
            .map_or(text.len(), |(offset, _)| start + offset);
        let result = text[start..end].to_wtf8_buf();
        self.pos = end;
        self.upos += count;
        result
    }

    fn find_char(&mut self, marker: u8, limit: Option<usize>) -> bool {
        let mut scanned = 0;
        while limit.is_none_or(|limit| scanned < limit) && !self.exhausted() {
            let found = self
                .text
                .as_ref()
                .expect("DecodeBuffer text")
                .ascii_byte_at(self.pos)
                == marker;
            self.advance();
            scanned += 1;
            if found {
                return true;
            }
        }
        false
    }

    fn find_newline_universal(&mut self, limit: Option<usize>) -> bool {
        let mut scanned = 0;
        while limit.is_none_or(|limit| scanned < limit) && !self.exhausted() {
            let byte = self
                .text
                .as_ref()
                .expect("DecodeBuffer text")
                .ascii_byte_at(self.pos);
            self.advance();
            scanned += 1;
            if byte == b'\n' {
                return true;
            }
            if byte == b'\r' {
                if limit.is_some_and(|limit| scanned >= limit) || self.exhausted() {
                    return true;
                }
                if self
                    .text
                    .as_ref()
                    .expect("DecodeBuffer text")
                    .ascii_byte_at(self.pos)
                    == b'\n'
                {
                    self.advance();
                }
                return true;
            }
        }
        false
    }

    fn find_crlf(&mut self, limit: Option<usize>) -> bool {
        let mut scanned = 0;
        while limit.is_none_or(|limit| scanned < limit) && !self.exhausted() {
            let byte = self
                .text
                .as_ref()
                .expect("DecodeBuffer text")
                .ascii_byte_at(self.pos);
            if byte != b'\r' {
                self.advance();
                scanned += 1;
                continue;
            }
            let saved_pos = self.pos;
            let saved_upos = self.upos;
            self.advance();
            scanned += 1;
            if limit.is_some_and(|limit| scanned >= limit) {
                return false;
            }
            if self.exhausted() {
                self.pos = saved_pos;
                self.upos = saved_upos;
                return false;
            }
            if self
                .text
                .as_ref()
                .expect("DecodeBuffer text")
                .ascii_byte_at(self.pos)
                == b'\n'
            {
                self.advance();
                return true;
            }
        }
        false
    }

    fn consumed_from(&self, start: usize) -> Wtf8Buf {
        self.text
            .as_ref()
            .map_or_else(Wtf8Buf::new, |text| text[start..self.pos].to_wtf8_buf())
    }

    fn starts_with_lf(&self) -> bool {
        self.text
            .as_ref()
            .is_some_and(|text| self.pos < text.len() && text.ascii_byte_at(self.pos) == b'\n')
    }
}

#[crate::pyre_class("_io.TextIOWrapper")]
pub struct W_TextIOWrapper {
    state: i64,
    w_buffer: PyObjectRef,
    w_encoding: PyObjectRef,
    w_errors: PyObjectRef,
    w_newline: PyObjectRef,
    w_stdio_name: PyObjectRef,
    w_encoder: PyObjectRef,
    w_decoder: PyObjectRef,
    line_buffering: bool,
    write_through: bool,
    decoded: DecodeBuffer,
    snapshot: Option<PositionSnapshot>,
    pending_bytes: Option<Vec<Vec<u8>>>,
    pending_bytes_count: usize,
    chunk_size: i64,
    b2cratio: f64,
    has_read1: bool,
    readuniversal: bool,
    readtranslate: bool,
    seekable_flag: bool,
    telling: bool,
    encoding_start_of_stream: bool,
}

impl Default for W_TextIOWrapper {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            state: STATE_ZERO,
            w_buffer: PY_NULL,
            w_encoding: PY_NULL,
            w_errors: PY_NULL,
            w_newline: PY_NULL,
            w_stdio_name: PY_NULL,
            w_encoder: PY_NULL,
            w_decoder: PY_NULL,
            line_buffering: false,
            write_through: false,
            decoded: DecodeBuffer::default(),
            snapshot: None,
            pending_bytes: None,
            pending_bytes_count: 0,
            chunk_size: 8192,
            b2cratio: 0.0,
            has_read1: false,
            readuniversal: false,
            readtranslate: false,
            seekable_flag: false,
            telling: false,
            encoding_start_of_stream: false,
        }
    }
}

impl W_TextIOWrapper {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    fn check_init(&self) -> Result<(), crate::PyError> {
        if self.state == STATE_ZERO {
            Err(crate::PyError::value_error(
                "I/O operation on uninitialized object",
            ))
        } else {
            Ok(())
        }
    }

    fn check_attached(&self) -> Result<(), crate::PyError> {
        if self.state == STATE_DETACHED {
            return Err(crate::PyError::value_error(
                "underlying buffer has been detached",
            ));
        }
        self.check_init()
    }

    fn buffer_closed(&self) -> Result<bool, crate::PyError> {
        self.check_attached()?;
        let closed = crate::baseobjspace::getattr_str(self.w_buffer, "closed")?;
        crate::baseobjspace::is_true(closed)
    }

    fn check_closed(&self) -> Result<(), crate::PyError> {
        if self.buffer_closed()? {
            Err(crate::PyError::value_error("I/O operation on closed file"))
        } else {
            Ok(())
        }
    }

    fn call_buffer(&self, name: &str, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        super::call_method_result(self.w_buffer, name, args)
    }

    /// PyPy `text0_or_none` / `io_check_errors` first route text through a
    /// real Unicode encoding operation.  Do the same instead of borrowing a
    /// Rust `&str`: a Python string may contain a lone surrogate, which must
    /// raise `UnicodeEncodeError` rather than panic in `w_str_get_value`.
    fn checked_text(
        obj: PyObjectRef,
        default: &str,
        argument: &str,
    ) -> Result<String, crate::PyError> {
        unsafe {
            if pyre_object::is_none(obj) {
                return Ok(default.to_string());
            }
            if !pyre_object::is_str(obj) {
                return Err(crate::PyError::type_error(format!(
                    "{argument} must be a str"
                )));
            }
        }
        let bytes = crate::type_methods::encode_object(obj, "utf-8", "strict")?;
        Ok(String::from_utf8(bytes)
            .expect("the utf-8 codec must only return valid UTF-8 for valid Unicode text"))
    }

    fn checked_text0(
        obj: PyObjectRef,
        default: &str,
        argument: &str,
    ) -> Result<String, crate::PyError> {
        let value = Self::checked_text(obj, default, argument)?;
        if value.contains('\0') {
            return Err(crate::PyError::value_error("embedded null character"));
        }
        Ok(value)
    }

    /// `interp_textio.io_check_errors` tail: `checked_text0` already applied
    /// the text0 + utf-8 strict-encode checks, so the only remaining step is
    /// the dev-mode error-handler name lookup.
    fn io_check_errors(errors: &str) -> Result<(), crate::PyError> {
        if crate::importing::dev_mode_flag() {
            crate::module::_codecs::validate_error_handler(errors)?;
        }
        Ok(())
    }

    /// `encoding="locale"` selects the current locale's encoding; the sandbox
    /// and default environment resolve that to UTF-8.
    fn resolve_locale_encoding(encoding: String) -> String {
        if encoding == "locale" {
            "utf-8".to_string()
        } else {
            encoding
        }
    }

    /// PyPy `_io.interp_iobase.unwrap_newline`.
    fn unwrap_newline(newline: PyObjectRef) -> Result<Option<String>, crate::PyError> {
        unsafe {
            if pyre_object::is_none(newline) {
                return Ok(None);
            }
            if !pyre_object::is_str(newline) {
                return Err(crate::PyError::type_error("illegal newline type"));
            }
            let value = pyre_object::w_str_get_wtf8(newline)
                .as_str()
                .map_err(|_| crate::PyError::value_error("illegal newline value"))?;
            if !matches!(value, "" | "\n" | "\r" | "\r\n") {
                return Err(crate::PyError::value_error(format!(
                    "illegal newline value: {value}"
                )));
            }
            Ok(Some(value.to_string()))
        }
    }

    fn configured_newline(&self) -> Option<&str> {
        unsafe {
            if pyre_object::is_none(self.w_newline) {
                None
            } else {
                pyre_object::w_str_get_value_opt(self.w_newline)
            }
        }
    }

    /// PyPy `W_TextIOWrapper._set_newline`.
    fn set_newline(&mut self, newline: Option<&str>) {
        self.readuniversal = newline.is_none_or(str::is_empty);
        self.readtranslate = newline.is_none();
    }

    /// The string `write` substitutes for `'\n'`, or `None` for no
    /// translation.  PyPy `W_TextIOWrapper` `writetranslate`/`writenl`: an
    /// explicit `'\r'` or `'\r\n'` is honored on every platform; `newline=None`
    /// writes the platform line separator (`'\r\n'` on Windows, untranslated
    /// elsewhere); `''` and `'\n'` are written verbatim.
    fn write_newline(&self) -> Option<&'static str> {
        match self.configured_newline() {
            None => {
                #[cfg(windows)]
                {
                    Some("\r\n")
                }
                #[cfg(not(windows))]
                {
                    None
                }
            }
            Some("\r") => Some("\r"),
            Some("\r\n") => Some("\r\n"),
            _ => None,
        }
    }

    fn size_limit(w_size: PyObjectRef) -> Result<Option<usize>, crate::PyError> {
        if unsafe { pyre_object::is_none(w_size) } {
            return Ok(None);
        }
        let size = crate::builtins::space_index_w(w_size)?;
        if size < 0 {
            Ok(None)
        } else {
            Ok(Some(size as usize))
        }
    }

    /// PyPy `W_TextIOWrapper._set_encoder_decoder`.
    fn set_encoder_decoder(&mut self, codec: PyObjectRef) -> Result<(), crate::PyError> {
        self.w_encoder = PY_NULL;
        self.w_decoder = PY_NULL;

        if codec.is_null() {
            return Ok(());
        }

        if crate::baseobjspace::is_true(super::call_method_result(self.w_buffer, "readable", &[])?)?
        {
            let mut decoder =
                super::call_method_result(codec, "incrementaldecoder", &[self.w_errors])?;
            if self.readuniversal {
                let io = crate::importing::get_sys_module("_io").ok_or_else(|| {
                    crate::PyError::runtime_error("_io module is not initialized")
                })?;
                let decoder_type =
                    crate::baseobjspace::getattr_str(io, "IncrementalNewlineDecoder")?;
                decoder = crate::call::call_function_impl_result(
                    decoder_type,
                    &[decoder, w_bool_from(self.readtranslate)],
                )?;
            }
            self.w_decoder = decoder;
        }

        if crate::baseobjspace::is_true(super::call_method_result(self.w_buffer, "writable", &[])?)?
        {
            self.w_encoder =
                super::call_method_result(codec, "incrementalencoder", &[self.w_errors])?;
        }
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(())
    }

    /// PyPy `_read_chunk` decoder-state validation.
    fn decoder_getstate(&self) -> Result<(Vec<u8>, u64), crate::PyError> {
        let state = super::call_method_result(self.w_decoder, "getstate", &[])?;
        if unsafe { !pyre_object::is_tuple(state) || pyre_object::w_tuple_len(state) != 2 } {
            return Err(crate::PyError::type_error("illegal decoder state"));
        }
        let items = unsafe { pyre_object::w_tuple_items_copy_as_vec(state) };
        if unsafe { !pyre_object::bytesobject::is_bytes(items[0]) } {
            return Err(crate::PyError::type_error(format!(
                "illegal decoder state: the first value should be a bytes object not '{}'",
                crate::type_methods::arg_type_name(items[0])
            )));
        }
        let flags = crate::builtins::space_index_w(items[1])?;
        if flags < 0 {
            return Err(crate::PyError::type_error("illegal decoder state"));
        }
        Ok((
            unsafe { pyre_object::bytesobject::w_bytes_data(items[0]) }.to_vec(),
            flags as u64,
        ))
    }

    /// PyPy `_decoder_setstate`.
    fn decoder_setstate(&self, cookie: &PositionCookie) -> Result<(), crate::PyError> {
        if cookie.start_pos == 0 && cookie.dec_flags == 0 {
            super::call_method_result(self.w_decoder, "reset", &[])?;
        } else {
            let state = w_tuple_new(vec![
                pyre_object::bytesobject::w_bytes_empty(),
                w_int_new(cookie.dec_flags as i64),
            ]);
            super::call_method_result(self.w_decoder, "setstate", &[state])?;
        }
        Ok(())
    }

    /// PyPy `_encoder_reset`.
    fn encoder_reset(&mut self, start_of_stream: bool) -> Result<(), crate::PyError> {
        if start_of_stream {
            super::call_method_result(self.w_encoder, "reset", &[])?;
            self.encoding_start_of_stream = true;
        } else {
            super::call_method_result(self.w_encoder, "setstate", &[w_int_new(0)])?;
            self.encoding_start_of_stream = false;
        }
        Ok(())
    }

    /// PyPy `W_TextIOWrapper._read_chunk`.
    fn read_chunk(&mut self, size_hint: usize) -> Result<bool, crate::PyError> {
        if self.w_decoder.is_null() {
            return Err(super::unsupported("not readable"));
        }
        let (dec_buffer, dec_flags) = if self.telling {
            let (buffer, flags) = self.decoder_getstate()?;
            (Some(buffer), flags)
        } else {
            (None, 0)
        };
        let scaled_hint = if size_hint == 0 {
            0
        } else {
            (self.b2cratio.max(1.0) * size_hint as f64) as usize
        };
        let chunk_size = (self.chunk_size as usize).max(scaled_hint);
        let method = if self.has_read1 { "read1" } else { "read" };
        let input = self.call_buffer(method, &[w_int_new(chunk_size as i64)])?;
        if unsafe { pyre_object::is_none(input) } {
            return Err(super::buffered::make_blocking_error());
        }
        let input_bytes =
            unsafe { crate::builtins::file_write_buffer_bytes(input) }.map_err(|_| {
                crate::PyError::type_error(format!(
                    "underlying {method}() should have returned a bytes-like object, not '{}'",
                    crate::type_methods::arg_type_name(input)
                ))
            })?;
        let nbytes = input_bytes.len();
        let eof = nbytes == 0;
        let bytes = pyre_object::bytesobject::w_bytes_from_bytes(&input_bytes);
        let decoded =
            super::call_method_result(self.w_decoder, "decode", &[bytes, w_bool_from(eof)])?;
        self.decoded.set(decoded)?;
        let nchars = self.decoded.ulen;
        if nchars > 0 {
            self.b2cratio = nbytes as f64 / nchars as f64;
            if let Some(mut next_input) = dec_buffer {
                next_input.extend_from_slice(&input_bytes);
                self.snapshot = Some(PositionSnapshot {
                    flags: dec_flags,
                    input: next_input,
                });
            }
            Ok(true)
        } else {
            self.b2cratio = 0.0;
            if let Some(mut next_input) = dec_buffer {
                next_input.extend_from_slice(&input_bytes);
                self.snapshot = Some(PositionSnapshot {
                    flags: dec_flags,
                    input: next_input,
                });
            }
            Ok(!eof)
        }
    }

    /// PyPy `W_TextIOWrapper._ensure_data`.
    fn ensure_data(&mut self, size_hint: usize) -> Result<bool, crate::PyError> {
        while !self.decoded.has_data() {
            if !self.read_chunk(size_hint)? {
                self.decoded.reset();
                self.snapshot = None;
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn scan_line_ending(&mut self, limit: Option<usize>) -> bool {
        if self.readtranslate {
            return self.decoded.find_char(b'\n', limit);
        }
        if self.readuniversal {
            return self.decoded.find_newline_universal(limit);
        }
        match self.configured_newline().unwrap_or("\n") {
            "\r\n" => self.decoded.find_crlf(limit),
            "\r" => self.decoded.find_char(b'\r', limit),
            _ => self.decoded.find_char(b'\n', limit),
        }
    }

    fn read_all(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let mut result = self.decoded.get_chars(None);
        let input = self.call_buffer("read", &[])?;
        if unsafe { pyre_object::is_none(input) } {
            return Err(super::buffered::make_blocking_error());
        }
        let input_bytes =
            unsafe { crate::builtins::file_write_buffer_bytes(input) }.map_err(|_| {
                crate::PyError::type_error(format!(
                    "underlying read() should have returned a bytes-like object, not '{}'",
                    crate::type_methods::arg_type_name(input)
                ))
            })?;
        let bytes = pyre_object::bytesobject::w_bytes_from_bytes(&input_bytes);
        let decoded =
            super::call_method_result(self.w_decoder, "decode", &[bytes, w_bool_from(true)])?;
        if unsafe { !pyre_object::is_str(decoded) } {
            return Err(crate::PyError::type_error(format!(
                "decoder should return a string result, not '{}'",
                crate::type_methods::arg_type_name(decoded)
            )));
        }
        result.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(decoded) });
        if self.snapshot.is_some() {
            self.decoded.reset();
            self.snapshot = None;
        }
        Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(result))
    }

    fn read_n(&mut self, size: usize) -> Result<PyObjectRef, crate::PyError> {
        let mut remaining = size;
        let mut result = Wtf8Buf::new();
        while remaining > 0 {
            if !self.ensure_data(remaining)? {
                break;
            }
            let chars = self.decoded.get_chars(Some(remaining));
            remaining -= chars.code_points().count();
            result.push_wtf8(&chars);
        }
        Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(result))
    }

    fn readline_impl(&mut self, limit: Option<usize>) -> Result<PyObjectRef, crate::PyError> {
        if limit == Some(0) {
            return Ok(w_str_new(""));
        }
        let mut result = Wtf8Buf::new();
        let mut pending_cr = false;
        loop {
            if !self.ensure_data(0)? {
                if pending_cr {
                    result.push_char('\r');
                }
                break;
            }

            if pending_cr {
                result.push_char('\r');
                pending_cr = false;
                if self.decoded.starts_with_lf() {
                    result.push_char('\n');
                    self.decoded.advance();
                    break;
                }
            }

            let used = result.code_points().count();
            let remaining = limit.map(|limit| limit.saturating_sub(used));
            let start = self.decoded.pos;
            let found = self.scan_line_ending(remaining);
            result.push_wtf8(&self.decoded.consumed_from(start));

            if found || limit.is_some_and(|limit| result.code_points().count() >= limit) {
                break;
            }

            if !self.decoded.exhausted() {
                let remnant = self.decoded.get_chars(None);
                if self.configured_newline() == Some("\r\n")
                    && remnant.code_points().count() == 1
                    && remnant.ascii_byte_at(0) == b'\r'
                {
                    pending_cr = true;
                } else {
                    result.push_wtf8(&remnant);
                }
            }
            self.decoded.reset();
        }
        Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(result))
    }

    /// PyPy `W_TextIOWrapper._fix_encoder_state`.
    fn reset_encoder_state(&mut self) {
        self.encoding_start_of_stream = false;
        if let Ok(w_seekable) = super::call_method_result(self.w_buffer, "seekable", &[]) {
            if crate::baseobjspace::is_true(w_seekable).unwrap_or(false) {
                self.encoding_start_of_stream = true;
                if let Ok(w_position) = super::call_method_result(self.w_buffer, "tell", &[]) {
                    if crate::builtins::space_index_w(w_position).unwrap_or(0) != 0 {
                        self.encoding_start_of_stream = false;
                        if !self.w_encoder.is_null() {
                            let _ = super::call_method_result(
                                self.w_encoder,
                                "setstate",
                                &[w_int_new(0)],
                            );
                        }
                    }
                }
            }
        }
    }

    fn size_args(w_size: PyObjectRef) -> Vec<PyObjectRef> {
        if unsafe { pyre_object::is_none(w_size) } {
            Vec::new()
        } else {
            vec![w_size]
        }
    }

    fn lookup_text_codec(encoding: &str) -> Result<PyObjectRef, crate::PyError> {
        // Normal interpreter startup installs an ExecutionContext before
        // Python-visible I/O can run.  A handful of Rust-level `open()` unit
        // tests deliberately exercise the builtin without booting an
        // interpreter; codec lookup cannot import `encodings` in that host
        // seam because no module globals owner exists yet.
        if crate::call::getexecutioncontext().is_null() {
            return Ok(PY_NULL);
        }
        crate::module::_codecs::lookup_text_codec("open", encoding)
    }

    /// Allocate the typed payload used by the interpreter-created standard
    /// streams.  Their methods and metadata remain instance overrides until
    /// sys stream construction is ported to a real FileIO-backed pipeline.
    pub(crate) fn allocate_stdio(name: &str, encoding: &str, errors: &str) -> PyObjectRef {
        // Establish the Python type and its mapdict layout before allocating
        // a payload whose stdio methods are installed in the instance dict.
        let _ = type_object();
        let obj = Self::allocate_stable(Self {
            state: STATE_OK,
            w_buffer: w_none(),
            w_encoding: w_str_new(encoding),
            w_errors: w_str_new(errors),
            w_newline: w_none(),
            w_stdio_name: w_str_new(name),
            w_encoder: PY_NULL,
            w_decoder: PY_NULL,
            line_buffering: false,
            write_through: false,
            decoded: DecodeBuffer::default(),
            snapshot: None,
            pending_bytes: None,
            pending_bytes_count: 0,
            chunk_size: 8192,
            b2cratio: 0.0,
            has_read1: false,
            readuniversal: true,
            readtranslate: true,
            seekable_flag: false,
            telling: false,
            encoding_start_of_stream: false,
            ..Self::default()
        });
        crate::baseobjspace::setdictvalue_native(obj, "name", w_str_new(name));
        obj
    }
}

#[crate::pyre_methods(
    base = super::text_iobase_type(),
    weakrefable,
    doc = "TextIOWrapper(buffer, encoding=None, errors=None, newline=None, line_buffering=False, write_through=False)"
)]
impl W_TextIOWrapper {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> PyObjectRef {
        let obj = Self::allocate_stable(Self::default());
        // A subclass still uses this concrete storage layout, while its
        // Python-visible class remains `cls`.
        super::tag_io_instance(obj, cls)
    }

    fn __init__(
        &mut self,
        buffer: PyObjectRef,
        #[default(pyre_object::w_none())] encoding: PyObjectRef,
        #[default(pyre_object::w_none())] errors: PyObjectRef,
        #[default(pyre_object::w_none())] newline: PyObjectRef,
        #[default(pyre_object::w_bool_from(false))] line_buffering: PyObjectRef,
        #[default(pyre_object::w_bool_from(false))] write_through: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        // PyPy starts every initialization attempt in STATE_ZERO.  A failed
        // reinitialization must leave all I/O operations uninitialized.
        self.state = STATE_ZERO;
        self.w_buffer = PY_NULL;
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);

        let encoding =
            Self::resolve_locale_encoding(Self::checked_text0(encoding, "utf-8", "encoding")?);
        let errors = Self::checked_text0(errors, "strict", "errors")?;
        Self::io_check_errors(&errors)?;
        let newline_value = Self::unwrap_newline(newline)?;
        let codec = Self::lookup_text_codec(&encoding)?;

        self.w_buffer = buffer;
        self.w_encoding = w_str_new(&encoding);
        self.w_errors = w_str_new(&errors);
        self.w_newline = newline;
        // 3.14's constructor uses the `bool` Argument Clinic converter (truth
        // testing), unlike `reconfigure`'s `int` converter — an object with
        // `__bool__` but no `__index__` is accepted here.
        self.line_buffering = crate::baseobjspace::is_true(line_buffering)?;
        self.write_through = crate::baseobjspace::is_true(write_through)?;
        self.decoded.reset();
        self.snapshot = None;
        self.pending_bytes = None;
        self.pending_bytes_count = 0;
        self.chunk_size = 8192;
        self.b2cratio = 0.0;
        self.set_newline(newline_value.as_deref());
        self.has_read1 = crate::baseobjspace::getattr_str(buffer, "read1").is_ok();
        self.set_encoder_decoder(codec)?;
        self.seekable_flag =
            crate::baseobjspace::is_true(super::call_method_result(buffer, "seekable", &[])?)?;
        self.telling = self.seekable_flag;
        self.state = STATE_OK;
        self.reset_encoder_state();
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(())
    }

    fn read(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        if self.w_decoder.is_null() {
            return Err(super::unsupported("not readable"));
        }
        self.write_flush()?;
        match Self::size_limit(w_size)? {
            None => self.read_all(),
            Some(size) => self.read_n(size),
        }
    }

    fn readline(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        if self.w_decoder.is_null() {
            return Err(super::unsupported("not readable"));
        }
        self.write_flush()?;
        self.readline_impl(Self::size_limit(w_size)?)
    }

    fn readlines(
        &mut self,
        #[default(pyre_object::w_none())] w_hint: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        super::iobase_readlines(&[self.self_obj(), w_hint])
    }

    /// PyPy `_writeflush_loop`; kept separate so the generated JIT can still
    /// trace through the ordinary `write()` fast path.
    fn write_flush_loop(&mut self) -> Result<(), crate::PyError> {
        while self.pending_bytes.is_some() {
            self.write_flush()?;
        }
        Ok(())
    }

    /// PyPy `_writeflush` inlinable fast path.
    fn write_flush(&mut self) -> Result<(), crate::PyError> {
        if self.pending_bytes.is_none() {
            return Ok(());
        }
        self.really_flush()
    }

    /// PyPy `_really_flush`: clear the pending list before invoking
    /// `buffer.write`, so a reentrant `TextIOWrapper.write` starts a new list
    /// which the outer `_writeflush_loop` can subsequently drain.
    fn really_flush(&mut self) -> Result<(), crate::PyError> {
        let chunks = self
            .pending_bytes
            .take()
            .expect("write_flush only calls really_flush with pending bytes");
        self.pending_bytes_count = 0;
        let total = chunks.iter().map(Vec::len).sum();
        let mut pending = Vec::with_capacity(total);
        for chunk in chunks {
            pending.extend_from_slice(&chunk);
        }
        let bytes = pyre_object::bytesobject::w_bytes_from_bytes(&pending);
        self.call_buffer("write", &[bytes])?;
        Ok(())
    }

    fn write(&mut self, text: PyObjectRef) -> Result<i64, crate::PyError> {
        self.check_closed()?;
        if self.w_encoder.is_null() {
            return Err(super::unsupported("not writable"));
        }
        if unsafe { !pyre_object::is_str(text) } {
            return Err(crate::PyError::type_error(format!(
                "unicode argument expected, got '{}'",
                crate::type_methods::arg_type_name(text)
            )));
        }
        let nchars = unsafe { pyre_object::w_str_len(text) };
        let text_wtf8 = unsafe { pyre_object::w_str_get_wtf8(text) };
        let has_lf = text_wtf8
            .code_points()
            .any(|cp| cp.to_u32() == b'\n' as u32);
        let has_cr = text_wtf8
            .code_points()
            .any(|cp| cp.to_u32() == b'\r' as u32);

        let mut to_encode = text;
        if has_lf {
            if let Some(writenl) = self.write_newline() {
                to_encode = super::call_method_result(
                    text,
                    "replace",
                    &[w_str_new("\n"), w_str_new(writenl)],
                )?;
            }
        }

        let need_flush = self.line_buffering && (has_lf || has_cr);
        let text_need_flush = self.write_through;
        let encoded = super::call_method_result(self.w_encoder, "encode", &[to_encode])?;
        if unsafe { !pyre_object::bytesobject::is_bytes(encoded) } {
            return Err(crate::PyError::type_error(format!(
                "encoder should return a bytes object, not '{}'",
                crate::type_methods::arg_type_name(encoded)
            )));
        }
        self.encoding_start_of_stream = false;
        let bytes = unsafe { pyre_object::bytesobject::w_bytes_data(encoded) }.to_vec();

        if bytes.len() >= self.chunk_size as usize && self.pending_bytes.is_some() {
            self.write_flush_loop()?;
        }
        match self.pending_bytes.as_mut() {
            Some(pending) => pending.push(bytes),
            None => self.pending_bytes = Some(vec![bytes]),
        }
        self.pending_bytes_count += self
            .pending_bytes
            .as_ref()
            .and_then(|pending| pending.last())
            .map_or(0, Vec::len);

        if self.pending_bytes_count >= self.chunk_size as usize || need_flush || text_need_flush {
            self.write_flush()?;
        }
        if need_flush {
            self.call_buffer("flush", &[])?;
        }
        if self.snapshot.is_some() {
            self.decoded.reset();
            self.snapshot = None;
        }
        if !self.w_decoder.is_null() {
            super::call_method_result(self.w_decoder, "reset", &[])?;
        }
        Ok(nchars as i64)
    }

    fn writelines(&self, lines: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        super::iobase_writelines(&[self.self_obj(), lines])
    }

    fn flush(&mut self) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        self.telling = self.seekable_flag;
        self.write_flush()?;
        self.call_buffer("flush", &[])
    }

    fn close(&mut self) -> Result<(), crate::PyError> {
        self.check_attached()?;
        // Keep the original buffer alive across the user-overridable
        // `closed` property.  CPython GH-142594: that property may reenter
        // and detach this wrapper.
        let buffer = self.w_buffer;
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(buffer);
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let closed = crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            "closed",
        )?;
        if crate::baseobjspace::is_true(closed)? {
            return Ok(());
        }
        if self.state == STATE_DETACHED {
            super::call_method_result(
                pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                "close",
                &[],
            )?;
            return Ok(());
        }

        // PyPy: `try: self.flush() finally: self.buffer.close()`.  The flush
        // is virtual, and a close failure replaces it while retaining the
        // flush exception as `__context__`.
        let flush_error = super::call_method_result(self.self_obj(), "flush", &[]).err();
        let close_result = super::call_method_result(
            pyre_object::gc_roots::shadow_stack_get(buffer_slot),
            "close",
            &[],
        );
        if let Err(mut close_error) = close_result {
            if let Some(mut flush_error) = flush_error {
                let _roots = pyre_object::gc_roots::push_roots();
                let flush_obj = flush_error.to_exc_object();
                pyre_object::gc_roots::pin_root(flush_obj);
                let flush_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let close_obj = close_error.to_exc_object();
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_context(
                        close_obj,
                        pyre_object::gc_roots::shadow_stack_get(flush_slot),
                    )
                };
                close_error.exc_object = close_obj;
            }
            return Err(close_error);
        }
        if let Some(error) = flush_error {
            return Err(error);
        }
        Ok(())
    }

    fn detach(&mut self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        super::call_method_result(self.self_obj(), "flush", &[])?;
        let buffer = self.w_buffer;
        self.w_buffer = PY_NULL;
        self.state = STATE_DETACHED;
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(buffer)
    }

    fn tell(&mut self) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        if !self.seekable_flag {
            return Err(super::unsupported("underlying stream is not seekable"));
        }
        if !self.telling {
            return Err(crate::PyError::os_error(
                "telling position disabled by next() call",
            ));
        }

        self.write_flush()?;
        super::call_method_result(self.self_obj(), "flush", &[])?;
        let w_pos = self.call_buffer("tell", &[])?;
        if self.w_decoder.is_null() || self.snapshot.is_none() {
            return Ok(w_pos);
        }

        let w_index = crate::baseobjspace::space_index(w_pos)?;
        let raw_pos = unsafe { crate::builtins::obj_to_bigint(w_index) };
        let mut cookie = PositionCookie::unpack(raw_pos)?;
        let snapshot = self.snapshot.as_ref().expect("checked above");
        let input = snapshot.input.clone();
        cookie.dec_flags = snapshot.flags;
        cookie.start_pos = cookie.start_pos.wrapping_sub(input.len() as u64);
        if self.decoded.pos == 0 {
            return Ok(cookie.to_object());
        }

        let mut chars_to_skip = self.decoded.upos as u64;
        let saved_state = super::call_method_result(self.w_decoder, "getstate", &[])?;
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(saved_state);
        let saved_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        let result = (|| -> Result<PyObjectRef, crate::PyError> {
            // PyPy's b2cratio heuristic searches backward for a decoder state
            // with no buffered bytes.
            let mut skip_bytes = ((self.b2cratio * chars_to_skip as f64) as usize).min(input.len());
            let mut skip_back = 1usize;
            while skip_bytes > 0 {
                self.decoder_setstate(&cookie)?;
                let decoded = super::call_method_result(
                    self.w_decoder,
                    "decode",
                    &[pyre_object::bytesobject::w_bytes_from_bytes(
                        &input[..skip_bytes],
                    )],
                )?;
                if unsafe { !pyre_object::is_str(decoded) } {
                    return Err(crate::PyError::type_error(format!(
                        "decoder should return a string result, not '{}'",
                        crate::type_methods::arg_type_name(decoded)
                    )));
                }
                let chars_decoded = unsafe { pyre_object::w_str_len(decoded) } as u64;
                if chars_decoded <= chars_to_skip {
                    let (dec_buffer, flags) = self.decoder_getstate()?;
                    if dec_buffer.is_empty() {
                        cookie.dec_flags = flags;
                        chars_to_skip -= chars_decoded;
                        break;
                    }
                    skip_bytes = skip_bytes.saturating_sub(dec_buffer.len());
                    skip_back = 1;
                } else {
                    skip_bytes = skip_bytes.saturating_sub(skip_back);
                    skip_back = skip_back.saturating_mul(2);
                }
            }
            if skip_bytes == 0 {
                self.decoder_setstate(&cookie)?;
            }
            cookie.start_pos += skip_bytes as u64;
            cookie.chars_to_skip = chars_to_skip;
            if chars_to_skip == 0 {
                return Ok(cookie.to_object());
            }

            self.decoder_setstate(&cookie)?;
            let mut chars_decoded = 0u64;
            let mut i = skip_bytes;
            while i < input.len() {
                let decoded = super::call_method_result(
                    self.w_decoder,
                    "decode",
                    &[pyre_object::bytesobject::w_bytes_from_bytes(
                        &input[i..i + 1],
                    )],
                )?;
                if unsafe { !pyre_object::is_str(decoded) } {
                    return Err(crate::PyError::type_error(format!(
                        "decoder should return a string result, not '{}'",
                        crate::type_methods::arg_type_name(decoded)
                    )));
                }
                chars_decoded += unsafe { pyre_object::w_str_len(decoded) } as u64;
                cookie.bytes_to_feed += 1;

                let (dec_buffer, flags) = self.decoder_getstate()?;
                if dec_buffer.is_empty() && chars_decoded <= chars_to_skip {
                    cookie.start_pos += cookie.bytes_to_feed;
                    chars_to_skip -= chars_decoded;
                    cookie.dec_flags = flags;
                    cookie.bytes_to_feed = 0;
                    chars_decoded = 0;
                }
                if chars_decoded >= chars_to_skip {
                    break;
                }
                i += 1;
            }
            if chars_decoded < chars_to_skip {
                let decoded = super::call_method_result(
                    self.w_decoder,
                    "decode",
                    &[pyre_object::bytesobject::w_bytes_empty(), w_bool_from(true)],
                )?;
                if unsafe { !pyre_object::is_str(decoded) } {
                    return Err(crate::PyError::type_error(format!(
                        "decoder should return a string result, not '{}'",
                        crate::type_methods::arg_type_name(decoded)
                    )));
                }
                chars_decoded += unsafe { pyre_object::w_str_len(decoded) } as u64;
                cookie.need_eof = true;
                if chars_decoded < chars_to_skip {
                    return Err(crate::PyError::os_error(
                        "can't reconstruct logical file position",
                    ));
                }
            }
            cookie.chars_to_skip = chars_to_skip;
            Ok(cookie.to_object())
        })();

        let restore = super::call_method_result(
            self.w_decoder,
            "setstate",
            &[pyre_object::gc_roots::shadow_stack_get(saved_slot)],
        );
        match (result, restore) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(value), Ok(_)) => Ok(value),
        }
    }

    fn seek(
        &mut self,
        cookie: PyObjectRef,
        #[default(0i64)] whence: i64,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        if !self.seekable_flag {
            return Err(super::unsupported("underlying stream is not seekable"));
        }
        let mut w_position = crate::baseobjspace::space_index(cookie)?;
        let mut position = unsafe { crate::builtins::obj_to_bigint(w_position) };

        if whence == 1 {
            if position.tobool() {
                return Err(super::unsupported("can't do nonzero cur-relative seeks"));
            }
            w_position = super::call_method_result(self.self_obj(), "tell", &[])?;
            let indexed = crate::baseobjspace::space_index(w_position)?;
            position = unsafe { crate::builtins::obj_to_bigint(indexed) };
        } else if whence == 2 {
            if position.tobool() {
                return Err(super::unsupported("can't do nonzero end-relative seeks"));
            }
            super::call_method_result(self.self_obj(), "flush", &[])?;
            self.decoded.reset();
            self.snapshot = None;
            if !self.w_decoder.is_null() {
                super::call_method_result(self.w_decoder, "reset", &[])?;
            }
            let result = self.call_buffer("seek", &[w_position, w_int_new(whence)])?;
            if !self.w_encoder.is_null() {
                let at_start = crate::builtins::space_index_w(result)? == 0;
                self.encoder_reset(at_start)?;
            }
            return Ok(result);
        } else if whence != 0 {
            return Err(crate::PyError::value_error(format!(
                "invalid whence ({whence}, should be 0, 1 or 2)"
            )));
        }

        let position_cookie = PositionCookie::unpack(position)?;
        super::call_method_result(self.self_obj(), "flush", &[])?;
        let start = crate::objspace::descroperation::bigint_result(RBigInt::from(
            position_cookie.start_pos,
        ));
        self.call_buffer("seek", &[start])?;
        self.decoded.reset();
        self.snapshot = None;
        if !self.w_decoder.is_null() {
            self.decoder_setstate(&position_cookie)?;
        }

        if position_cookie.chars_to_skip != 0 {
            let chunk =
                self.call_buffer("read", &[w_int_new(position_cookie.bytes_to_feed as i64)])?;
            if unsafe { !pyre_object::bytesobject::is_bytes(chunk) } {
                return Err(crate::PyError::type_error(format!(
                    "underlying read() should have returned a bytes object, not '{}'",
                    crate::type_methods::arg_type_name(chunk)
                )));
            }
            let input = unsafe { pyre_object::bytesobject::w_bytes_data(chunk) }.to_vec();
            self.snapshot = Some(PositionSnapshot {
                flags: position_cookie.dec_flags,
                input,
            });
            let decoded = super::call_method_result(
                self.w_decoder,
                "decode",
                &[chunk, w_bool_from(position_cookie.need_eof)],
            )?;
            if unsafe { !pyre_object::is_str(decoded) } {
                return Err(crate::PyError::type_error(format!(
                    "decoder should return a string result, not '{}'",
                    crate::type_methods::arg_type_name(decoded)
                )));
            }
            if unsafe { pyre_object::w_str_len(decoded) } < position_cookie.chars_to_skip as usize {
                return Err(crate::PyError::os_error(
                    "can't restore logical file position",
                ));
            }
            self.decoded.set(decoded)?;
            self.decoded
                .get_chars(Some(position_cookie.chars_to_skip as usize));
        } else {
            self.snapshot = Some(PositionSnapshot {
                flags: position_cookie.dec_flags,
                input: Vec::new(),
            });
        }

        if !self.w_encoder.is_null() {
            self.encoder_reset(position_cookie.start_pos == 0 && position_cookie.dec_flags == 0)?;
        }
        Ok(w_position)
    }

    fn truncate(
        &self,
        #[default(pyre_object::w_none())] size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        super::call_method_result(self.self_obj(), "flush", &[])?;
        if unsafe { pyre_object::is_none(size) } {
            self.call_buffer("truncate", &[])
        } else {
            self.call_buffer("truncate", &[size])
        }
    }

    fn reconfigure(
        &mut self,
        #[default(pyre_object::w_none())] encoding: PyObjectRef,
        #[default(pyre_object::w_none())] errors: PyObjectRef,
        #[default(pyre_object::PY_NULL)] newline: PyObjectRef,
        #[default(pyre_object::w_none())] line_buffering: PyObjectRef,
        #[default(pyre_object::w_none())] write_through: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        self.check_attached()?;
        if self.decoded.text.is_some()
            && (!unsafe { pyre_object::is_none(encoding) }
                || !unsafe { pyre_object::is_none(errors) }
                || !newline.is_null())
        {
            return Err(super::unsupported(
                "It is not possible to set the encoding or newline of stream after the first read",
            ));
        }

        let new_encoding = if unsafe { pyre_object::is_none(encoding) } {
            None
        } else {
            let value = Self::checked_text(encoding, "", "encoding")?;
            Some(Self::resolve_locale_encoding(value))
        };
        let new_errors = if unsafe { pyre_object::is_none(errors) } {
            None
        } else {
            let value = Self::checked_text0(errors, "", "errors")?;
            Self::io_check_errors(&value)?;
            Some(value)
        };
        let new_newline = if newline.is_null() {
            None
        } else {
            Some(Self::unwrap_newline(newline)?)
        };
        // CPython 3.14's clinic converter still uses the integer/index
        // protocol for these two flags (the CPython tests deliberately
        // distinguish it from truth testing).
        let new_line_buffering = if unsafe { pyre_object::is_none(line_buffering) } {
            None
        } else {
            Some(crate::builtins::space_index_w(line_buffering)? != 0)
        };
        let new_write_through = if unsafe { pyre_object::is_none(write_through) } {
            None
        } else {
            Some(crate::builtins::space_index_w(write_through)? != 0)
        };
        // The incremental newline decoder captures its translation at build
        // time, so a newline change must rebuild the codec whenever it flips
        // `readuniversal` or `readtranslate`; otherwise a universal-to-fixed
        // reconfigure would keep translating `\r\n` with the stale decoder.
        let newline_forces_reset = new_newline.as_ref().is_some_and(|value| {
            let new_readuniversal = value.as_deref().is_none_or(str::is_empty);
            let new_readtranslate = value.is_none();
            self.readuniversal != new_readuniversal || self.readtranslate != new_readtranslate
        });
        let reset_codec = new_encoding.is_some() || new_errors.is_some() || newline_forces_reset;
        // PyPy/CPython prepare the replacement codec before mutating the
        // wrapper.  In particular, a failing codec lookup must leave
        // `encoding`, `errors`, and the incremental encoder/decoder intact.
        let new_codec = if reset_codec {
            let encoding = new_encoding
                .as_deref()
                .unwrap_or_else(|| unsafe { pyre_object::w_str_get_value(self.w_encoding) });
            Some(Self::lookup_text_codec(encoding)?)
        } else {
            None
        };

        // CPython 3.14 `_textiowrapper_writeflush`: every reconfiguration
        // first commits pending output, even when every option is omitted.
        super::call_method_result(self.self_obj(), "flush", &[])?;
        if let Some(value) = new_newline.as_ref() {
            self.w_newline = newline;
            self.set_newline(value.as_deref());
        }
        if let Some(value) = new_encoding.as_ref() {
            self.w_encoding = w_str_new(&value);
            self.w_errors = w_str_new(new_errors.as_deref().unwrap_or("strict"));
        } else if let Some(value) = new_errors.as_ref() {
            self.w_errors = w_str_new(&value);
        }
        if let Some(codec) = new_codec {
            self.set_encoder_decoder(codec)?;
        }
        if let Some(value) = new_line_buffering {
            self.line_buffering = value;
        }
        if let Some(value) = new_write_through {
            self.write_through = value;
        }
        self.reset_encoder_state();
        self.b2cratio = 0.0;
        Ok(())
    }

    fn readable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("readable", &[])
    }

    fn writable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("writable", &[])
    }

    fn seekable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("seekable", &[])
    }

    fn isatty(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("isatty", &[])
    }

    fn fileno(&self) -> Result<PyObjectRef, crate::PyError> {
        self.call_buffer("fileno", &[])
    }

    #[getter]
    fn buffer(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        Ok(self.w_buffer)
    }

    #[getter]
    fn closed(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        if unsafe { pyre_object::is_none(self.w_buffer) } {
            return Ok(w_bool_from(false));
        }
        crate::baseobjspace::getattr_str(self.w_buffer, "closed")
    }

    #[getter]
    fn name(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        if !self.w_stdio_name.is_null() {
            return Ok(self.w_stdio_name);
        }
        crate::baseobjspace::getattr_str(self.w_buffer, "name")
    }

    #[getter]
    fn encoding(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        Ok(self.w_encoding)
    }

    #[getter]
    fn errors(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        Ok(self.w_errors)
    }

    #[getter]
    fn line_buffering(&self) -> Result<bool, crate::PyError> {
        self.check_init()?;
        Ok(self.line_buffering)
    }

    #[getter]
    fn write_through(&self) -> Result<bool, crate::PyError> {
        self.check_init()?;
        Ok(self.write_through)
    }

    #[getter]
    fn newlines(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        if self.w_decoder.is_null() {
            return Ok(w_none());
        }
        match crate::baseobjspace::getattr_str(self.w_decoder, "newlines") {
            Ok(value) => Ok(value),
            Err(error) if error.kind == crate::PyErrorKind::AttributeError => Ok(w_none()),
            Err(error) => Err(error),
        }
    }

    #[getter]
    #[allow(non_snake_case)]
    fn _CHUNK_SIZE(&self) -> Result<i64, crate::PyError> {
        self.check_attached()?;
        Ok(self.chunk_size)
    }

    #[setter]
    #[allow(non_snake_case)]
    fn set__CHUNK_SIZE(&mut self, size: PyObjectRef) -> Result<(), crate::PyError> {
        self.check_attached()?;
        let size = crate::baseobjspace::int_w(size)?;
        if size <= 0 {
            return Err(crate::PyError::value_error(
                "a strictly positive integer is required",
            ));
        }
        self.chunk_size = size;
        Ok(())
    }

    fn __enter__(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        Ok(self.self_obj())
    }

    fn __exit__(
        &mut self,
        _exc_type: PyObjectRef,
        _exc: PyObjectRef,
        _tb: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        self.close()
    }

    fn __iter__(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        Ok(self.self_obj())
    }

    fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
        self.check_attached()?;
        self.telling = false;
        let line = self.readline(w_none())?;
        if unsafe { pyre_object::w_str_len(line) == 0 } {
            self.telling = self.seekable_flag;
            Err(crate::PyError::stop_iteration())
        } else {
            Ok(line)
        }
    }

    fn __repr__(&self) -> Result<String, crate::PyError> {
        self.check_init()?;
        let self_obj = self.self_obj();
        let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
            return Err(crate::PyError::runtime_error(
                "reentrant call inside TextIOWrapper.__repr__",
            ));
        };

        let typename = crate::type_methods::arg_type_name(self_obj);
        let mut fields = String::new();
        if self.state != STATE_DETACHED {
            if let Ok(name) = crate::baseobjspace::getattr_str(self.w_buffer, "name") {
                fields.push_str(" name=");
                fields.push_str(&unsafe { crate::display::py_repr(name)? });
            }
        }
        if let Ok(mode) = crate::baseobjspace::getattr_str(self_obj, "mode") {
            fields.push_str(" mode=");
            fields.push_str(&unsafe { crate::display::py_repr(mode)? });
        }
        fields.push_str(" encoding=");
        fields.push_str(&unsafe { crate::display::py_repr(self.w_encoding)? });
        Ok(format!("<{typename}{fields}>"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_cookie_round_trips_all_native_words() {
        let original = PositionCookie {
            start_pos: 0xfedc_ba98_7654_3210,
            dec_flags: 0x0123_4567_89ab_cdef,
            bytes_to_feed: 0x8877_6655_4433_2211,
            chars_to_skip: 0x1020_3040_5060_7080,
            need_eof: true,
        };
        let packed = original.pack();
        assert!(
            packed.bits() > u64::BITS as u64,
            "a packed cookie must not be narrowed through to_u64"
        );

        let decoded = PositionCookie::unpack(packed).expect("packed cookie");
        assert_eq!(decoded.start_pos, original.start_pos);
        assert_eq!(decoded.dec_flags, original.dec_flags);
        assert_eq!(decoded.bytes_to_feed, original.bytes_to_feed);
        assert_eq!(decoded.chars_to_skip, original.chars_to_skip);
        assert_eq!(decoded.need_eof, original.need_eof);
    }
}
