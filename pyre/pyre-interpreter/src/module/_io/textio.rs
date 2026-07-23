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
        Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
    }

    fn size_args(w_size: PyObjectRef) -> Vec<PyObjectRef> {
        if unsafe { pyre_object::is_none(w_size) } {
            Vec::new()
        } else {
            vec![w_size]
        }
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

        let encoding = unsafe {
            if pyre_object::is_none(encoding) {
                "utf-8".to_string()
            } else if pyre_object::is_str(encoding) {
                pyre_object::w_str_get_value(encoding).to_string()
            } else {
                return Err(crate::PyError::type_error("encoding must be a str"));
            }
        };
        let errors = unsafe {
            if pyre_object::is_none(errors) {
                "strict".to_string()
            } else if pyre_object::is_str(errors) {
                pyre_object::w_str_get_value(errors).to_string()
            } else {
                return Err(crate::PyError::type_error("errors must be a str"));
            }
        };
        let newline_value = unsafe {
            if pyre_object::is_none(newline) {
                None
            } else if pyre_object::is_str(newline) {
                Some(pyre_object::w_str_get_value(newline))
            } else {
                return Err(crate::PyError::type_error("illegal newline type"));
            }
        };
        if !matches!(
            newline_value,
            None | Some("") | Some("\n") | Some("\r") | Some("\r\n")
        ) {
            return Err(crate::PyError::value_error(format!(
                "illegal newline value: {}",
                newline_value.unwrap_or_default()
            )));
        }

        self.w_buffer = buffer;
        self.w_encoding = w_str_new(&encoding);
        self.w_errors = w_str_new(&errors);
        self.w_newline = newline;
        self.line_buffering = crate::baseobjspace::is_true(line_buffering)?;
        self.write_through = crate::baseobjspace::is_true(write_through)?;
        self.state = STATE_OK;
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(())
    }

    fn read(
        &self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        let args = Self::size_args(w_size);
        let raw = self.call_buffer("read", &args)?;
        Ok(w_str_new(&self.decode(raw)?))
    }

    fn readline(
        &self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed()?;
        let args = Self::size_args(w_size);
        let raw = self.call_buffer("readline", &args)?;
        Ok(w_str_new(&self.decode(raw)?))
    }

    fn readlines(
        &self,
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

    fn write(&self, text: PyObjectRef) -> Result<i64, crate::PyError> {
        self.check_closed()?;
        if unsafe { !pyre_object::is_str(text) } {
            return Err(crate::PyError::type_error("write() argument must be str"));
        }
        let (encoding, errors) = self.encoding_errors();
        let encoded = crate::type_methods::encode_object(text, &encoding, &errors)?;
        let nchars = unsafe { pyre_object::w_str_len(text) };
        let bytes = pyre_object::bytesobject::w_bytes_from_bytes(&encoded);
        self.call_buffer("write", &[bytes])?;

        if self.write_through
            || (self.line_buffering
                && unsafe {
                    let value = pyre_object::w_str_get_value(text);
                    value.contains('\n') || value.contains('\r')
                })
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
        &self,
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
        self.call_buffer("seek", &[cookie, w_int_new(whence)])
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
        #[default(pyre_object::w_none())] _encoding: PyObjectRef,
        #[default(pyre_object::w_none())] _errors: PyObjectRef,
        #[default(pyre_object::w_none())] _newline: PyObjectRef,
        #[default(pyre_object::w_none())] line_buffering: PyObjectRef,
        #[default(pyre_object::w_none())] write_through: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        self.check_attached()?;
        // CPython 3.14 `_textiowrapper_writeflush`: every reconfiguration
        // first commits pending output, even when every option is omitted.
        super::call_method_result(self.self_obj(), "flush", &[])?;
        if unsafe { !pyre_object::is_none(line_buffering) } {
            self.line_buffering = crate::baseobjspace::is_true(line_buffering)?;
        }
        if unsafe { !pyre_object::is_none(write_through) } {
            self.write_through = crate::baseobjspace::is_true(write_through)?;
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

    fn __next__(&self) -> Result<PyObjectRef, crate::PyError> {
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
