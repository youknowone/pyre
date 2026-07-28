//! Buffered binary writer — PyPy `pypy/module/_io/interp_bufferedio.py`.

use pyre_object::*;

use super::DEFAULT_BUFFER_SIZE;
const STATE_ZERO: i64 = 0;
const STATE_OK: i64 = 1;
const STATE_DETACHED: i64 = 2;

pub(super) fn is_blocking_error(error: &crate::PyError) -> bool {
    let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") else {
        return false;
    };
    !error.exc_object.is_null()
        && unsafe { crate::baseobjspace::isinstance_w(error.exc_object, blocking) }
}

pub(super) fn make_write_blocking_error(written: usize) -> crate::PyError {
    let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") else {
        return crate::PyError::os_error("write could not complete without blocking");
    };
    match crate::call::call_function_impl_result(
        blocking,
        &[
            w_int_new(0),
            w_str_new("write could not complete without blocking"),
            w_int_new(written as i64),
        ],
    ) {
        Ok(value) => unsafe { crate::PyError::from_exc_object(value) },
        Err(error) => error,
    }
}

pub(super) fn input_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    if unsafe { pyre_object::bytesobject::is_bytes_like(obj) } {
        return Ok(unsafe { pyre_object::bytesobject::bytes_like_data(obj) }.to_vec());
    }
    let view = crate::builtins::w_memoryview_new(obj)?;
    unsafe {
        crate::builtins::memoryview_check_released(view)?;
        Ok(crate::builtins::memoryview_gather_bytes(view))
    }
}

#[crate::pyre_class("_io.BufferedWriter")]
pub struct W_BufferedWriter {
    state: i64,
    w_raw: PyObjectRef,
    buffer: PyObjectRef,
    buffer_size: i64,
    abs_pos: i64,
    pos: i64,
    raw_pos: i64,
    read_end: i64,
    write_pos: i64,
    write_end: i64,
    readable: bool,
    writable: bool,
    locked: bool,
}

impl Default for W_BufferedWriter {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            state: STATE_ZERO,
            w_raw: PY_NULL,
            buffer: PY_NULL,
            buffer_size: 0,
            abs_pos: 0,
            pos: 0,
            raw_pos: 0,
            read_end: -1,
            write_pos: 0,
            write_end: -1,
            readable: false,
            writable: false,
            locked: false,
        }
    }
}

impl W_BufferedWriter {
    fn self_obj(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    fn check_init(&self) -> Result<(), crate::PyError> {
        match self.state {
            STATE_ZERO => Err(crate::PyError::value_error(
                "I/O operation on uninitialized object",
            )),
            STATE_DETACHED => Err(crate::PyError::value_error("raw stream has been detached")),
            _ => Ok(()),
        }
    }

    fn raw_closed(&self) -> Result<bool, crate::PyError> {
        self.check_init()?;
        let closed = crate::baseobjspace::getattr_str(self.w_raw, "closed")?;
        crate::baseobjspace::is_true(closed)
    }

    fn check_closed(&self, message: &str) -> Result<(), crate::PyError> {
        if self.raw_closed()? {
            Err(crate::PyError::value_error(message))
        } else {
            Ok(())
        }
    }

    fn with_lock<T>(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<T, crate::PyError>,
    ) -> Result<T, crate::PyError> {
        if self.locked {
            return Err(crate::PyError::runtime_error("reentrant call"));
        }
        self.locked = true;
        let result = body(self);
        self.locked = false;
        result
    }

    fn writer_reset_buf(&mut self) {
        self.write_pos = 0;
        self.write_end = -1;
    }

    fn raw_offset(&self) -> i64 {
        if self.raw_pos >= 0
            && ((self.readable && self.read_end != -1) || (self.writable && self.write_end != -1))
        {
            self.raw_pos - self.pos
        } else {
            0
        }
    }

    fn raw_tell(&mut self) -> Result<i64, crate::PyError> {
        let result = super::call_method_result(self.w_raw, "tell", &[])?;
        let pos = crate::baseobjspace::int_w(result)?;
        if pos < 0 {
            return Err(crate::PyError::os_error(
                "raw stream returned invalid position",
            ));
        }
        self.abs_pos = pos;
        Ok(pos)
    }

    fn raw_seek(&mut self, pos: i64, whence: i64) -> Result<i64, crate::PyError> {
        let result =
            super::call_method_result(self.w_raw, "seek", &[w_int_new(pos), w_int_new(whence)])?;
        let pos = crate::baseobjspace::int_w(result)?;
        if pos < 0 {
            return Err(crate::PyError::os_error(
                "Raw stream returned invalid position",
            ));
        }
        self.abs_pos = pos;
        Ok(pos)
    }

    fn raw_write(&mut self, data: &[u8]) -> Result<usize, crate::PyError> {
        let bytes = pyre_object::bytesobject::w_bytes_from_bytes(data);
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(self.self_obj());
        pyre_object::gc_roots::pin_root(bytes);
        let sp = pyre_object::gc_roots::shadow_stack_len() - 2;
        let result = super::call_method_result(
            self.w_raw,
            "write",
            &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
        )?;
        if unsafe { pyre_object::is_none(result) } {
            return Err(make_write_blocking_error(0));
        }
        let written = crate::builtins::space_index_w(result)?;
        if written < 0 || written as usize > data.len() {
            return Err(crate::PyError::os_error(
                "raw write() returned invalid length",
            ));
        }
        if self.abs_pos != -1 {
            self.abs_pos += written;
        }
        Ok(written as usize)
    }

    fn writer_flush_unlocked(&mut self) -> Result<(), crate::PyError> {
        if self.write_end == -1 || self.write_pos == self.write_end {
            return Ok(());
        }
        let rewind = self.raw_offset() + (self.pos - self.write_pos);
        if rewind != 0 {
            self.raw_seek(-rewind, 1)?;
            self.raw_pos -= rewind;
        }
        while self.write_pos < self.write_end {
            let start = self.write_pos as usize;
            let end = self.write_end as usize;
            let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(self.buffer) }
                [start..end]
                .to_vec();
            let written = match self.raw_write(&data) {
                Ok(written) => written,
                Err(error) if is_blocking_error(&error) => {
                    return Err(make_write_blocking_error(0));
                }
                Err(error) => return Err(error),
            };
            self.write_pos += written as i64;
            self.raw_pos = self.write_pos;
        }
        self.writer_reset_buf();
        Ok(())
    }

    fn flush_unlocked(&mut self) -> Result<(), crate::PyError> {
        self.writer_flush_unlocked()
    }

    fn write_bytes(&mut self, data: &[u8]) -> Result<i64, crate::PyError> {
        self.check_closed("write to closed file")?;
        self.with_lock(|this| {
            if !((this.readable && this.read_end != -1) || (this.writable && this.write_end != -1))
            {
                this.pos = 0;
                this.raw_pos = 0;
            }
            let available = this.buffer_size as usize - this.pos as usize;
            if data.len() <= available {
                let start = this.pos as usize;
                let buffer =
                    unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer) };
                buffer[start..start + data.len()].copy_from_slice(data);
                if this.write_end == -1 || this.write_pos > this.pos {
                    this.write_pos = this.pos;
                }
                this.pos += data.len() as i64;
                this.write_end = this.write_end.max(this.pos);
                return Ok(data.len() as i64);
            }

            if let Err(error) = this.writer_flush_unlocked() {
                if !is_blocking_error(&error) {
                    return Err(error);
                }
                let old_start = this.write_pos as usize;
                let old_end = this.write_end as usize;
                let pending = unsafe {
                    pyre_object::bytearrayobject::w_bytearray_data(this.buffer)[old_start..old_end]
                        .to_vec()
                };
                let buffer =
                    unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer) };
                buffer[..pending.len()].copy_from_slice(&pending);
                this.write_end -= this.write_pos;
                this.raw_pos -= this.write_pos;
                this.pos -= this.write_pos;
                this.write_pos = 0;
                let available = this.buffer_size as usize - this.write_end as usize;
                if data.len() <= available {
                    let start = this.write_end as usize;
                    buffer[start..start + data.len()].copy_from_slice(data);
                    this.write_end += data.len() as i64;
                    this.pos += data.len() as i64;
                    return Ok(data.len() as i64);
                }
                buffer[this.write_end as usize..this.write_end as usize + available]
                    .copy_from_slice(&data[..available]);
                this.write_end += available as i64;
                this.pos += available as i64;
                return Err(make_write_blocking_error(available));
            }

            let mut written = 0usize;
            while data.len() - written > this.buffer_size as usize {
                match this.raw_write(&data[written..]) {
                    Ok(0) => return Err(make_write_blocking_error(written)),
                    Ok(size) => written += size,
                    Err(error) if is_blocking_error(&error) => {
                        let remaining = data.len() - written;
                        let take = remaining.min(this.buffer_size as usize);
                        let buffer = unsafe {
                            pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer)
                        };
                        buffer[..take].copy_from_slice(&data[written..written + take]);
                        this.raw_pos = 0;
                        this.pos = take as i64;
                        this.write_pos = 0;
                        this.write_end = take as i64;
                        written += take;
                        return Err(make_write_blocking_error(written));
                    }
                    Err(error) => return Err(error),
                }
            }

            let remaining = data.len() - written;
            if remaining > 0 {
                let buffer =
                    unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer) };
                buffer[..remaining].copy_from_slice(&data[written..]);
                written += remaining;
            }
            this.write_pos = 0;
            this.write_end = remaining as i64;
            this.pos = remaining as i64;
            this.raw_pos = 0;
            Ok(written as i64)
        })
    }
}

#[crate::pyre_methods(
    base = super::buffered_iobase_type(),
    weakrefable,
    doc = "BufferedWriter(raw, buffer_size=DEFAULT_BUFFER_SIZE)"
)]
impl W_BufferedWriter {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> PyObjectRef {
        let obj = W_BufferedWriter::allocate_stable(W_BufferedWriter::default());
        super::tag_io_instance(obj, cls)
    }

    fn __init__(
        &mut self,
        w_raw: PyObjectRef,
        #[default(DEFAULT_BUFFER_SIZE)] buffer_size: i64,
    ) -> Result<(), crate::PyError> {
        self.state = STATE_ZERO;
        let writable = super::call_method_result(w_raw, "writable", &[])?;
        if !crate::baseobjspace::is_true(writable)? {
            return Err(super::unsupported("File or stream is not writable."));
        }
        if buffer_size <= 0 {
            return Err(crate::PyError::value_error(
                "buffer size must be strictly positive",
            ));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_raw);
        let raw_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let buffer = pyre_object::bytearrayobject::w_bytearray_new(buffer_size as usize);
        self.w_raw = pyre_object::gc_roots::shadow_stack_get(raw_slot);
        self.buffer = buffer;
        self.buffer_size = buffer_size;
        self.readable = false;
        self.writable = true;
        self.pos = 0;
        self.raw_pos = 0;
        self.read_end = -1;
        self.writer_reset_buf();
        self.locked = false;
        self.abs_pos = 0;
        let _ = self.raw_tell();
        self.state = STATE_OK;
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(())
    }

    fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
        Err(crate::PyError::type_error(format!(
            "cannot serialize '{}' object",
            crate::type_methods::arg_type_name(self.self_obj())
        )))
    }

    fn write(&mut self, w_data: PyObjectRef) -> Result<i64, crate::PyError> {
        let data = input_bytes(w_data)?;
        self.write_bytes(&data)
    }

    fn flush(&mut self) -> Result<(), crate::PyError> {
        self.check_closed("flush of closed file")?;
        self.with_lock(Self::flush_unlocked)
    }

    #[getter]
    fn raw(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        Ok(self.w_raw)
    }

    #[getter]
    fn closed(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        crate::baseobjspace::getattr_str(self.w_raw, "closed")
    }

    #[getter]
    fn name(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        crate::baseobjspace::getattr_str(self.w_raw, "name")
    }

    #[getter]
    fn mode(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        crate::baseobjspace::getattr_str(self.w_raw, "mode")
    }

    fn readable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        // A writer never reports readable, even over a bidirectional raw.
        Ok(pyre_object::w_bool_from(false))
    }

    fn writable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "writable", &[])
    }

    fn seekable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "seekable", &[])
    }

    fn isatty(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "isatty", &[])
    }

    fn fileno(&self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "fileno", &[])
    }

    fn tell(&mut self) -> Result<i64, crate::PyError> {
        self.check_init()?;
        Ok((self.raw_tell()? - self.raw_offset()).max(0))
    }

    fn seek(
        &mut self,
        w_pos: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] w_whence: PyObjectRef,
    ) -> Result<i64, crate::PyError> {
        self.check_closed("seek of closed file")?;
        let pos = crate::builtins::space_index_w(w_pos)?;
        let whence = crate::builtins::space_index_w(w_whence)?;
        if !(0..=2).contains(&whence) {
            return Err(crate::PyError::value_error(format!(
                "whence must be between 0 and 2, not {whence}"
            )));
        }
        let seekable = super::call_method_result(self.w_raw, "seekable", &[])?;
        if !crate::baseobjspace::is_true(seekable)? {
            return Err(super::unsupported("File or stream is not seekable."));
        }
        self.with_lock(|this| {
            this.writer_flush_unlocked()?;
            this.writer_reset_buf();
            let adjusted = if whence == 1 {
                pos - this.raw_offset()
            } else {
                pos
            };
            let result = this.raw_seek(adjusted, whence)?;
            this.raw_pos = -1;
            Ok(result)
        })
    }

    fn close(&mut self) -> Result<(), crate::PyError> {
        self.check_init()?;
        if self.raw_closed()? {
            return Ok(());
        }
        let self_obj = self.self_obj();
        let flush_error = super::call_method_result(self_obj, "flush", &[]).err();
        let close_result = super::call_method_result(self.w_raw, "close", &[]);
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
        self.buffer = PY_NULL;
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        if let Some(error) = flush_error {
            return Err(error);
        }
        Ok(())
    }

    fn detach(&mut self) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.self_obj(), "flush", &[])?;
        let raw = self.w_raw;
        self.w_raw = PY_NULL;
        self.buffer = PY_NULL;
        self.state = STATE_DETACHED;
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(raw)
    }

    fn truncate(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed("truncate of closed file")?;
        self.with_lock(|this| {
            this.flush_unlocked()?;
            this.abs_pos = -1;
            super::call_method_result(this.w_raw, "truncate", &[w_size])
        })
    }

    fn _dealloc_warn(&self, w_source: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "_dealloc_warn", &[w_source])
    }

    fn __repr__(&self) -> Result<String, crate::PyError> {
        let self_obj = self.self_obj();
        let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
            return Err(crate::PyError::runtime_error(
                "reentrant call inside BufferedWriter.__repr__",
            ));
        };
        let typename = crate::type_methods::arg_type_name(self_obj);
        match crate::baseobjspace::getattr_str(self_obj, "name") {
            Ok(name) => Ok(format!("<{typename} name={}>", unsafe {
                crate::display::py_repr(name)?
            })),
            Err(_) => Ok(format!("<{typename}>")),
        }
    }
}
