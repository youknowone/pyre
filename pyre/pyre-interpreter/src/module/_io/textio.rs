//! Text streams — PyPy `pypy/module/_io/interp_textio.py`.
//!
//! Keep the stream state on the typed object, matching
//! `W_TextIOWrapper`.  In particular, the buffer and the
//! ZERO/OK/DETACHED state are not instance-dict side data.

use pyre_object::*;

const STATE_ZERO: i64 = 0;
const STATE_OK: i64 = 1;
const STATE_DETACHED: i64 = 2;

#[crate::pyre_class("_io.TextIOWrapper")]
pub struct W_TextIOWrapper {
    state: i64,
    w_buffer: PyObjectRef,
    w_encoding: PyObjectRef,
    w_errors: PyObjectRef,
    w_newline: PyObjectRef,
    w_stdio_name: PyObjectRef,
    line_buffering: bool,
    write_through: bool,
    has_read: bool,
    decoded: String,
    decoded_pos: usize,
    decoded_loaded: bool,
    encoder_fresh: bool,
    suppress_bom: bool,
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
            line_buffering: false,
            write_through: false,
            has_read: false,
            decoded: String::new(),
            decoded_pos: 0,
            decoded_loaded: false,
            encoder_fresh: true,
            suppress_bom: false,
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

    fn encoding_errors(&self) -> (String, String) {
        let read = |obj: PyObjectRef, default: &str| unsafe {
            if !obj.is_null() && pyre_object::is_str(obj) {
                pyre_object::w_str_get_value(obj).to_string()
            } else {
                default.to_string()
            }
        };
        (
            read(self.w_encoding, "utf-8"),
            read(self.w_errors, "strict"),
        )
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

    fn decode(&self, obj: PyObjectRef) -> Result<String, crate::PyError> {
        let (encoding, errors) = self.encoding_errors();
        let text = unsafe {
            if pyre_object::bytesobject::is_bytes_like(obj) {
                let decoded = crate::typedef::bytes_method_decode(&[
                    obj,
                    w_str_new(&encoding),
                    w_str_new(&errors),
                ])?;
                pyre_object::w_str_get_value(decoded).to_string()
            } else if pyre_object::is_str(obj) {
                pyre_object::w_str_get_value(obj).to_string()
            } else {
                String::new()
            }
        };
        Ok(text)
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

    /// PyPy `DecodeBuffer` + `_read_chunk`.  The current backend fills the
    /// decoded buffer in one chunk; keeping the decoded text and cursor on
    /// the stream still preserves character-sized reads and newline
    /// boundaries independently of the byte buffer's `readline` policy.
    fn ensure_decoded(&mut self) -> Result<(), crate::PyError> {
        if self.decoded_loaded {
            return Ok(());
        }
        let raw = self.call_buffer("read", &[])?;
        let mut text = self.decode(raw)?;
        if self.configured_newline().is_none() {
            text = text.replace("\r\n", "\n").replace('\r', "\n");
        }
        self.decoded = text;
        self.decoded_pos = 0;
        self.decoded_loaded = true;
        Ok(())
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

    fn char_limit(text: &str, count: usize) -> usize {
        text.char_indices()
            .nth(count)
            .map(|(index, _)| index)
            .unwrap_or(text.len())
    }

    fn take_decoded(&mut self, byte_count: usize) -> PyObjectRef {
        let end = self.decoded_pos + byte_count;
        let value = w_str_new(&self.decoded[self.decoded_pos..end]);
        self.decoded_pos = end;
        value
    }

    fn line_end(text: &str, newline: Option<&str>) -> usize {
        match newline {
            // Universal-newline translation has already changed every line
            // ending to LF.
            None | Some("\n") => text.find('\n').map_or(text.len(), |i| i + 1),
            Some("\r") => text.find('\r').map_or(text.len(), |i| i + 1),
            Some("\r\n") => text.find("\r\n").map_or(text.len(), |i| i + 2),
            Some("") => {
                let bytes = text.as_bytes();
                for (i, byte) in bytes.iter().enumerate() {
                    if *byte == b'\n' {
                        return i + 1;
                    }
                    if *byte == b'\r' {
                        return if bytes.get(i + 1) == Some(&b'\n') {
                            i + 2
                        } else {
                            i + 1
                        };
                    }
                }
                text.len()
            }
            Some(_) => text.len(),
        }
    }

    fn reset_encoder_state(&mut self) {
        self.encoder_fresh = true;
        self.suppress_bom = false;
        if let Ok(w_seekable) = super::call_method_result(self.w_buffer, "seekable", &[]) {
            if crate::baseobjspace::is_true(w_seekable).unwrap_or(false) {
                if let Ok(w_position) = super::call_method_result(self.w_buffer, "tell", &[]) {
                    if crate::builtins::space_index_w(w_position).unwrap_or(0) != 0 {
                        self.suppress_bom = true;
                    }
                }
            }
        }
    }

    fn strip_bom<'a>(&mut self, encoded: &'a [u8]) -> &'a [u8] {
        let bom_len = if encoded.starts_with(&[0xef, 0xbb, 0xbf]) {
            3
        } else if encoded.starts_with(&[0xff, 0xfe, 0x00, 0x00])
            || encoded.starts_with(&[0x00, 0x00, 0xfe, 0xff])
        {
            4
        } else if encoded.starts_with(&[0xff, 0xfe]) || encoded.starts_with(&[0xfe, 0xff]) {
            2
        } else {
            0
        };
        let strip = bom_len != 0 && (self.suppress_bom || !self.encoder_fresh);
        self.encoder_fresh = false;
        if strip { &encoded[bom_len..] } else { encoded }
    }

    fn size_args(w_size: PyObjectRef) -> Vec<PyObjectRef> {
        if unsafe { pyre_object::is_none(w_size) } {
            Vec::new()
        } else {
            vec![w_size]
        }
    }

    fn validate_text_codec(encoding: &str) -> Result<(), crate::PyError> {
        // Normal interpreter startup installs an ExecutionContext before
        // Python-visible I/O can run.  A handful of Rust-level `open()` unit
        // tests deliberately exercise the builtin without booting an
        // interpreter; codec lookup cannot import `encodings` in that host
        // seam because no module globals owner exists yet.
        if crate::call::getexecutioncontext().is_null() {
            return Ok(());
        }
        crate::module::_codecs::lookup_text_codec("open", encoding)?;
        Ok(())
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
            line_buffering: false,
            write_through: false,
            has_read: false,
            decoded: String::new(),
            decoded_pos: 0,
            decoded_loaded: false,
            encoder_fresh: true,
            suppress_bom: false,
            ..Self::default()
        });
        crate::baseobjspace::setdictvalue(obj, "name", w_str_new(name));
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
        unsafe { (*obj).w_class = cls };
        obj
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

        let encoding = Self::checked_text0(encoding, "utf-8", "encoding")?;
        let errors = Self::checked_text0(errors, "strict", "errors")?;
        let _newline_value = Self::unwrap_newline(newline)?;
        Self::validate_text_codec(&encoding)?;

        self.w_buffer = buffer;
        self.w_encoding = w_str_new(&encoding);
        self.w_errors = w_str_new(&errors);
        self.w_newline = newline;
        self.line_buffering = crate::baseobjspace::is_true(line_buffering)?;
        self.write_through = crate::baseobjspace::is_true(write_through)?;
        self.has_read = false;
        self.decoded.clear();
        self.decoded_pos = 0;
        self.decoded_loaded = false;
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
        self.ensure_decoded()?;
        self.has_read = true;
        let remaining = &self.decoded[self.decoded_pos..];
        let count = match Self::size_limit(w_size)? {
            None => remaining.len(),
            Some(limit) => Self::char_limit(remaining, limit),
        };
        Ok(self.take_decoded(count))
    }

    fn readline(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        self.ensure_decoded()?;
        self.has_read = true;
        let remaining = &self.decoded[self.decoded_pos..];
        let mut count = Self::line_end(remaining, self.configured_newline());
        if let Some(limit) = Self::size_limit(w_size)? {
            count = count.min(Self::char_limit(remaining, limit));
        }
        Ok(self.take_decoded(count))
    }

    fn readlines(
        &mut self,
        #[default(pyre_object::w_none())] _w_hint: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let mut lines = Vec::new();
        loop {
            let line = self.readline(w_none())?;
            if unsafe { pyre_object::w_str_get_value(line).is_empty() } {
                break;
            }
            lines.push(line);
        }
        Ok(w_list_new(lines))
    }

    fn write(&mut self, text: PyObjectRef) -> Result<i64, crate::PyError> {
        self.check_closed()?;
        if unsafe { !pyre_object::is_str(text) } {
            return Err(crate::PyError::type_error("write() argument must be str"));
        }
        let (encoding, errors) = self.encoding_errors();
        let nchars = unsafe { pyre_object::w_str_len(text) };
        let configured_newline = unsafe {
            if pyre_object::is_none(self.w_newline) {
                None
            } else {
                pyre_object::w_str_get_value_opt(self.w_newline)
            }
        };
        let translated;
        let to_encode = if matches!(configured_newline, Some("\r") | Some("\r\n")) {
            if let Some(value) = unsafe { pyre_object::w_str_get_value_opt(text) } {
                translated = w_str_new(&value.replace('\n', configured_newline.unwrap()));
                translated
            } else {
                text
            }
        } else {
            text
        };
        let encoded = crate::type_methods::encode_object(to_encode, &encoding, &errors)?;
        let encoded = self.strip_bom(&encoded);
        let bytes = pyre_object::bytesobject::w_bytes_from_bytes(encoded);
        self.call_buffer("write", &[bytes])?;

        if self.line_buffering
            && unsafe {
                pyre_object::w_str_get_wtf8(text)
                    .code_points()
                    .any(|cp| matches!(cp.to_u32(), 0x0a | 0x0d))
            }
        {
            super::call_method_result(self.self_obj(), "flush", &[])?;
        }
        Ok(nchars as i64)
    }

    fn writelines(&self, lines: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        super::iobase_writelines(&[self.self_obj(), lines])
    }

    fn flush(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        self.call_buffer("flush", &[])
    }

    fn close(&self) -> Result<(), crate::PyError> {
        self.check_attached()?;
        if self.buffer_closed()? {
            return Ok(());
        }

        // PyPy: `try: self.flush() finally: self.buffer.close()`.  The flush
        // is virtual, and a close failure replaces it while retaining the
        // flush exception as `__context__`.
        let flush_error = super::call_method_result(self.self_obj(), "flush", &[]).err();
        let close_result = super::call_method_result(self.w_buffer, "close", &[]);
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

    fn tell(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        self.call_buffer("tell", &[])
    }

    fn seek(
        &mut self,
        cookie: PyObjectRef,
        #[default(0i64)] whence: i64,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        let position = crate::builtins::space_index_w(cookie)?;
        if position != 0 && whence == 1 {
            return Err(super::unsupported("can't do nonzero cur-relative seeks"));
        }
        if position != 0 && whence == 2 {
            return Err(super::unsupported("can't do nonzero end-relative seeks"));
        }
        let result = self.call_buffer("seek", &[cookie, w_int_new(whence)])?;
        self.decoded.clear();
        self.decoded_pos = 0;
        self.decoded_loaded = false;
        self.has_read = false;
        Ok(result)
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
        if self.has_read
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
            let value = if value == "locale" {
                "utf-8".to_string()
            } else {
                value
            };
            Self::validate_text_codec(&value)?;
            Some(value)
        };
        let new_errors = if unsafe { pyre_object::is_none(errors) } {
            None
        } else {
            Some(Self::checked_text0(errors, "", "errors")?)
        };
        if !newline.is_null() {
            Self::unwrap_newline(newline)?;
        }
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

        // CPython 3.14 `_textiowrapper_writeflush`: every reconfiguration
        // first commits pending output, even when every option is omitted.
        super::call_method_result(self.self_obj(), "flush", &[])?;
        if !newline.is_null() {
            self.w_newline = newline;
        }
        if let Some(value) = new_encoding {
            self.w_encoding = w_str_new(&value);
            self.w_errors = w_str_new(new_errors.as_deref().unwrap_or("strict"));
            self.reset_encoder_state();
        } else if let Some(value) = new_errors {
            self.w_errors = w_str_new(&value);
        }
        if let Some(value) = new_line_buffering {
            self.line_buffering = value;
        }
        if let Some(value) = new_write_through {
            self.write_through = value;
        }
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
        self.check_init()?;
        Ok(w_none())
    }

    fn __enter__(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        Ok(self.self_obj())
    }

    fn __exit__(
        &self,
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
        let line = self.readline(w_none())?;
        if unsafe { pyre_object::w_str_get_value(line).is_empty() } {
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
