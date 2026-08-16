//! Buffered binary streams — PyPy `pypy/module/_io/interp_bufferedio.py`.
//!
//! Keep the buffering state on the typed object, matching `BufferedMixin`.
//! In particular, `raw`, the byte buffer, the per-stream `TryLock`, and the
//! logical/raw positions are not instance-dict side data.

use pyre_object::*;

use super::DEFAULT_BUFFER_SIZE;
const STATE_ZERO: i64 = 0;
const STATE_OK: i64 = 1;
const STATE_DETACHED: i64 = 2;

pub(super) fn is_blocking_error(error: &crate::PyError) -> bool {
    let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") else {
        return false;
    };
    let value = error.exc_object;
    !value.is_null() && unsafe { crate::baseobjspace::isinstance_w(value, blocking) }
}

pub(super) fn make_blocking_error() -> crate::PyError {
    let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") else {
        return crate::PyError::os_error("read could not complete without blocking");
    };
    match crate::call::call_function_impl_result(
        blocking,
        &[
            w_int_new(0),
            w_str_new("read could not complete without blocking"),
        ],
    ) {
        Ok(value) => unsafe { crate::PyError::from_exc_object(value) },
        Err(error) => error,
    }
}

pub(super) fn raw_readinto_size(
    result: PyObjectRef,
    length: usize,
) -> Result<usize, crate::PyError> {
    let size = match crate::baseobjspace::int_w(result) {
        Ok(size) => size,
        Err(mut cause) => {
            // CPython 3.14's buffered C implementation translates a
            // non-index `readinto` result to OSError and preserves the
            // conversion TypeError as the explicit cause.
            let _roots = pyre_object::gc_roots::push_roots();
            let cause_obj = cause.to_exc_object();
            pyre_object::gc_roots::pin_root(cause_obj);
            let cause_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let mut outer = crate::PyError::os_error("raw readinto() returned invalid length");
            let outer_obj = outer.to_exc_object();
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_cause(
                    outer_obj,
                    pyre_object::gc_roots::shadow_stack_get(cause_slot),
                )
            };
            outer.exc_object = outer_obj;
            return Err(outer);
        }
    };
    if size < 0 || size as usize > length {
        return Err(crate::PyError::os_error(format!(
            "raw readinto() returned invalid length {size} (should have been between 0 and {length})"
        )));
    }
    Ok(size as usize)
}

#[crate::pyre_class("_io.BufferedReader")]
pub struct W_BufferedReader {
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
    lock: usize,
}

impl Default for W_BufferedReader {
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
            lock: 0,
        }
    }
}

impl W_BufferedReader {
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
        self.check_init()?;
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
        // Bind the handle once, the way `with self.lock:` does: `body` may run
        // `__init__` again on the same object and install a different lock, and
        // releasing that one trips `Lock.release`'s not-acquired check.
        let lock = self.lock;
        if !super::acquire_buffered_lock(lock) {
            return Err(crate::PyError::runtime_error("reentrant call"));
        }
        let result = body(self);
        super::release_buffered_lock(lock);
        result
    }

    fn readahead(&self) -> usize {
        if self.readable && self.read_end != -1 {
            debug_assert!(self.read_end >= self.pos);
            (self.read_end - self.pos) as usize
        } else {
            0
        }
    }

    fn reader_reset_buf(&mut self) {
        self.read_end = -1;
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

    fn buffer_bytes(&self, start: usize, end: usize) -> Vec<u8> {
        unsafe { pyre_object::bytearrayobject::w_bytearray_data(self.buffer)[start..end].to_vec() }
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

    /// `BufferedMixin._raw_read`: use the raw stream's `readinto` protocol,
    /// validate its result, then copy the exact returned window into the
    /// object's ByteBuffer-equivalent storage.
    fn raw_read(&mut self, start: usize, length: usize) -> Result<usize, crate::PyError> {
        let temp = pyre_object::bytearrayobject::w_bytearray_new(length);
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(self.self_obj());
        pyre_object::gc_roots::pin_root(temp);
        let sp = pyre_object::gc_roots::shadow_stack_len() - 2;
        let result = super::call_method_result(
            self.w_raw,
            "readinto",
            &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
        )?;
        if unsafe { pyre_object::is_none(result) } {
            return Err(make_blocking_error());
        }
        let size = raw_readinto_size(result, length)?;
        let temp = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(temp) };
        let buffer = unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(self.buffer) };
        buffer[start..start + size].copy_from_slice(&data[..size]);
        if self.abs_pos != -1 {
            self.abs_pos += size as i64;
        }
        Ok(size)
    }

    fn fill_buffer(&mut self) -> Result<usize, crate::PyError> {
        self.check_closed("I/O operation on closed file")?;
        let start = if self.read_end == -1 {
            0
        } else {
            self.read_end as usize
        };
        let length = self.buffer_size as usize - start;
        let size = self.raw_read(start, length)?;
        if size > 0 {
            self.read_end = (start + size) as i64;
            self.raw_pos = self.read_end;
        }
        Ok(size)
    }

    fn read_fast(&mut self, n: usize) -> Option<Vec<u8>> {
        if n <= self.readahead() {
            let start = self.pos as usize;
            let end = start + n;
            let result = self.buffer_bytes(start, end);
            self.pos = end as i64;
            Some(result)
        } else {
            None
        }
    }

    fn read_all_unlocked(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let current_size = self.readahead();
        let mut output = Vec::new();
        if current_size > 0 {
            let start = self.pos as usize;
            output.extend_from_slice(&self.buffer_bytes(start, start + current_size));
            self.pos += current_size as i64;
        }
        self.reader_reset_buf();

        // CPython 3.14 `_bufferedreader_read_all` prefers the raw stream's
        // `readall` protocol and invokes it once.  A raw implementation that
        // does not provide `readall` retains the chunked `read()` fallback.
        if let Some(readall) = crate::baseobjspace::findattr_result(self.w_raw, "readall")? {
            let data = crate::call::call_function_impl_result(readall, &[])?;
            if unsafe { pyre_object::is_none(data) } {
                return if output.is_empty() {
                    Ok(data)
                } else {
                    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
                };
            }
            if !unsafe { crate::baseobjspace::isinstance_bytes_w(data) } {
                return Err(crate::PyError::type_error(format!(
                    "expected bytes, got {} object",
                    crate::type_methods::arg_type_name(data)
                )));
            }
            let chunk = unsafe { pyre_object::bytesobject::bytes_like_data(data) };
            output.extend_from_slice(chunk);
            if self.abs_pos != -1 {
                self.abs_pos += chunk.len() as i64;
            }
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output));
        }

        loop {
            let data = super::call_method_result(self.w_raw, "read", &[])?;
            if unsafe { pyre_object::is_none(data) } {
                if output.is_empty() {
                    return Ok(data);
                }
                break;
            }
            if !unsafe { crate::baseobjspace::isinstance_bytes_w(data) } {
                return Err(crate::PyError::type_error(format!(
                    "expected bytes, got {} object",
                    crate::type_methods::arg_type_name(data)
                )));
            }
            let chunk = unsafe { pyre_object::bytesobject::bytes_like_data(data) };
            if chunk.is_empty() {
                break;
            }
            output.extend_from_slice(chunk);
            if self.abs_pos != -1 {
                self.abs_pos += chunk.len() as i64;
            }
        }
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
    }

    fn read_generic_unlocked(&mut self, n: usize) -> Result<Option<Vec<u8>>, crate::PyError> {
        let current_size = self.readahead();
        if n <= current_size {
            return Ok(self.read_fast(n));
        }
        let mut output = Vec::with_capacity(n);
        if current_size > 0 {
            let start = self.pos as usize;
            output.extend_from_slice(&self.buffer_bytes(start, start + current_size));
            self.pos += current_size as i64;
        }
        self.reader_reset_buf();

        let mut remaining = n - output.len();
        while remaining >= self.buffer_size as usize {
            let block = self.buffer_size as usize * (remaining / self.buffer_size as usize);
            let temp = pyre_object::bytearrayobject::w_bytearray_new(block);
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(self.self_obj());
            pyre_object::gc_roots::pin_root(temp);
            let sp = pyre_object::gc_roots::shadow_stack_len() - 2;
            let result = super::call_method_result(
                self.w_raw,
                "readinto",
                &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
            )?;
            if unsafe { pyre_object::is_none(result) } {
                return if output.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(output))
                };
            }
            let size = raw_readinto_size(result, block)?;
            if size == 0 {
                return Ok(Some(output));
            }
            let temp = pyre_object::gc_roots::shadow_stack_get(sp + 1);
            let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(temp) };
            output.extend_from_slice(&data[..size]);
            remaining -= size;
            if self.abs_pos != -1 {
                self.abs_pos += size as i64;
            }
        }

        self.pos = 0;
        self.raw_pos = 0;
        self.read_end = 0;
        while remaining > 0 && self.read_end < self.buffer_size {
            let size = match self.fill_buffer() {
                Ok(size) => size,
                Err(error) if is_blocking_error(&error) => {
                    if output.is_empty() {
                        return Ok(None);
                    }
                    0
                }
                Err(error) => return Err(error),
            };
            if size == 0 {
                break;
            }
            let take = size.min(remaining);
            output
                .extend_from_slice(&self.buffer_bytes(self.pos as usize, self.pos as usize + take));
            self.pos += take as i64;
            remaining -= take;
        }
        Ok(Some(output))
    }

    fn read_size(&mut self, size: i64) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed("read of closed file")?;
        if size == -1 {
            return self.with_lock(Self::read_all_unlocked);
        }
        if size < -1 {
            return Err(crate::PyError::value_error(
                "read length must be positive or -1",
            ));
        }
        let size = size as usize;
        if let Some(result) = self.read_fast(size) {
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&result));
        }
        let result = self.with_lock(|this| this.read_generic_unlocked(size))?;
        Ok(match result {
            Some(data) => pyre_object::bytesobject::w_bytes_from_bytes(&data),
            None => w_none(),
        })
    }

    fn read1_size(&mut self, mut size: i64) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed("read of closed file")?;
        if size < 0 {
            size = self.buffer_size;
        }
        if size == 0 {
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
        }
        self.with_lock(|this| {
            let mut have = this.readahead();
            if have == 0 {
                // CPython 3.14 `_bufferedreader_read1`: an empty buffer and a
                // request larger than the buffer performs one direct raw
                // read of the requested size.  The older PyPy source fills
                // its fixed buffer here (and calls that behavior probably
                // wrong); 3.14 semantics win for pyre.
                if size > this.buffer_size {
                    this.reader_reset_buf();
                    let requested = size as usize;
                    let temp = pyre_object::bytearrayobject::w_bytearray_new(requested);
                    let _roots = pyre_object::gc_roots::push_roots();
                    pyre_object::gc_roots::pin_root(this.self_obj());
                    pyre_object::gc_roots::pin_root(temp);
                    let temp_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                    let result = super::call_method_result(
                        this.w_raw,
                        "readinto",
                        &[pyre_object::gc_roots::shadow_stack_get(temp_slot)],
                    )?;
                    if unsafe { pyre_object::is_none(result) } {
                        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
                    }
                    let read = raw_readinto_size(result, requested)?;
                    if this.abs_pos != -1 {
                        this.abs_pos += read as i64;
                    }
                    let temp = pyre_object::gc_roots::shadow_stack_get(temp_slot);
                    let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(temp) };
                    return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data[..read]));
                }
                this.reader_reset_buf();
                this.pos = 0;
                have = match this.fill_buffer() {
                    Ok(size) => size,
                    Err(error) if is_blocking_error(&error) => 0,
                    Err(error) => return Err(error),
                };
            }
            let take = (size as usize).min(have);
            let start = this.pos as usize;
            let data = this.buffer_bytes(start, start + take);
            this.pos += take as i64;
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
        })
    }

    fn readinto_impl(
        &mut self,
        w_buffer: PyObjectRef,
        read_once: bool,
    ) -> Result<i64, crate::PyError> {
        self.check_init()?;
        self.check_closed("readinto of closed file")?;
        let mut target = unsafe { crate::builtins::WritableBuffer::acquire(w_buffer) }?;
        let requested = unsafe { target.as_mut_slice() }.len();
        if requested == 0 {
            return Ok(0);
        }

        // CPython/PyPy's buffered readinto state machine first drains all
        // readahead.  It may then perform direct raw reads; readinto1 stops
        // after the first such read, rather than behaving like read1().
        self.with_lock(|this| {
            let mut written = 0usize;
            let have = this.readahead();
            if have > 0 {
                let take = have.min(requested);
                let start = this.pos as usize;
                let data = this.buffer_bytes(start, start + take);
                (unsafe { target.as_mut_slice() })[..take].copy_from_slice(&data);
                this.pos += take as i64;
                written = take;
                if written == requested {
                    return Ok(written as i64);
                }
            }

            this.reader_reset_buf();
            while written < requested {
                let remaining = requested - written;
                if remaining > this.buffer_size as usize {
                    let temp = pyre_object::bytearrayobject::w_bytearray_new(remaining);
                    let _roots = pyre_object::gc_roots::push_roots();
                    pyre_object::gc_roots::pin_root(this.self_obj());
                    pyre_object::gc_roots::pin_root(temp);
                    let sp = pyre_object::gc_roots::shadow_stack_len() - 2;
                    let result = super::call_method_result(
                        this.w_raw,
                        "readinto",
                        &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
                    )?;
                    if unsafe { pyre_object::is_none(result) } {
                        break;
                    }
                    let size = raw_readinto_size(result, remaining)?;
                    if size == 0 {
                        break;
                    }
                    let temp = pyre_object::gc_roots::shadow_stack_get(sp + 1);
                    let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(temp) };
                    (unsafe { target.as_mut_slice() })[written..written + size]
                        .copy_from_slice(&data[..size]);
                    written += size;
                    if this.abs_pos != -1 {
                        this.abs_pos += size as i64;
                    }
                    if read_once {
                        break;
                    }
                    continue;
                }

                if read_once && written > 0 {
                    break;
                }

                this.pos = 0;
                this.raw_pos = 0;
                this.read_end = 0;
                let size = match this.fill_buffer() {
                    Ok(size) => size,
                    Err(error) if is_blocking_error(&error) => 0,
                    Err(error) => return Err(error),
                };
                if size == 0 {
                    break;
                }
                let take = size.min(remaining);
                let data = this.buffer_bytes(0, take);
                (unsafe { target.as_mut_slice() })[written..written + take].copy_from_slice(&data);
                this.pos = take as i64;
                written += take;
                if read_once {
                    break;
                }
            }
            Ok(written as i64)
        })
    }
}

#[crate::pyre_methods(
    base = super::buffered_iobase_type(),
    weakrefable,
    doc = "BufferedReader(raw, buffer_size=DEFAULT_BUFFER_SIZE)",
    _text_signature_ = "(raw, buffer_size=DEFAULT_BUFFER_SIZE)"
)]
impl W_BufferedReader {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> PyObjectRef {
        let obj = W_BufferedReader::allocate_stable(W_BufferedReader::default());
        super::tag_io_instance(obj, cls)
    }

    fn __init__(
        &mut self,
        w_raw: PyObjectRef,
        #[default(DEFAULT_BUFFER_SIZE)] buffer_size: i64,
    ) -> Result<(), crate::PyError> {
        self.state = STATE_ZERO;
        let readable = super::call_method_result(w_raw, "readable", &[])?;
        if !crate::baseobjspace::is_true(readable)? {
            return Err(super::unsupported("File or stream is not readable."));
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
        self.readable = true;
        self.writable = false;
        self.pos = 0;
        self.raw_pos = 0;
        self.read_end = -1;
        self.write_pos = 0;
        self.write_end = -1;
        self.lock = super::allocate_buffered_lock();
        // Where the raw stream sits is left unknown rather than asked for:
        // `seek` refreshes it before its fast path reads it and `tell` always
        // asks, which is every use, so construction owes no `lseek`.
        self.abs_pos = -1;
        self.state = STATE_OK;
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(())
    }

    fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
        Err(crate::PyError::type_error(format!(
            "cannot pickle '{}' object",
            crate::type_methods::arg_type_name(self.self_obj())
        )))
    }

    fn read(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let size = super::iobase_convert_size(w_size)?;
        self.read_size(size)
    }

    fn read1(&mut self, #[default(-1)] size: i64) -> Result<PyObjectRef, crate::PyError> {
        self.read1_size(size)
    }

    fn peek(&mut self, #[default(0)] _size: i64) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed("peek of closed file")?;
        self.with_lock(|this| {
            let mut have = this.readahead();
            if have == 0 {
                this.reader_reset_buf();
                have = match this.fill_buffer() {
                    Ok(size) => size,
                    Err(error) if is_blocking_error(&error) => 0,
                    Err(error) => return Err(error),
                };
                this.pos = 0;
            }
            let data = this.buffer_bytes(this.pos as usize, this.pos as usize + have);
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
        })
    }

    fn readline(
        &mut self,
        #[default(pyre_object::w_none())] w_limit: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed("readline of closed file")?;
        let mut limit = super::iobase_convert_size(w_limit)?;
        let mut have = self.readahead();
        if limit >= 0 {
            have = have.min(limit as usize);
        }
        let start = self.pos as usize;
        let buffered = self.buffer_bytes(start, start + have);
        if let Some(index) = buffered.iter().position(|byte| *byte == b'\n') {
            let end = index + 1;
            self.pos += end as i64;
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(
                &buffered[..end],
            ));
        }
        if limit >= 0 && have == limit as usize {
            self.pos += have as i64;
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buffered));
        }

        self.with_lock(|this| {
            let mut output = buffered;
            this.pos += have as i64;
            if limit >= 0 {
                limit -= have as i64;
            }
            loop {
                this.reader_reset_buf();
                let filled = match this.fill_buffer() {
                    Ok(size) => size,
                    Err(error) if is_blocking_error(&error) => 0,
                    Err(error) => return Err(error),
                };
                if filled == 0 {
                    break;
                }
                let take = if limit >= 0 {
                    filled.min(limit as usize)
                } else {
                    filled
                };
                let chunk = this.buffer_bytes(0, take);
                let newline = chunk.iter().position(|byte| *byte == b'\n');
                let end = newline.map_or(take, |index| index + 1);
                output.extend_from_slice(&chunk[..end]);
                this.pos = end as i64;
                if newline.is_some() || (limit >= 0 && end == limit as usize) {
                    break;
                }
                if limit >= 0 {
                    limit -= end as i64;
                }
            }
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
        })
    }

    fn readinto(&mut self, w_buffer: PyObjectRef) -> Result<i64, crate::PyError> {
        self.readinto_impl(w_buffer, false)
    }

    fn readinto1(&mut self, w_buffer: PyObjectRef) -> Result<i64, crate::PyError> {
        self.readinto_impl(w_buffer, true)
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
        super::call_method_result(self.w_raw, "readable", &[])
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
        let raw_tell = self.raw_tell()?;
        Ok((raw_tell - self.raw_offset()).max(0))
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
        if whence != 2 {
            if self.abs_pos == -1 {
                let _ = self.raw_tell()?;
            }
            let current = self.abs_pos;
            let available = self.readahead() as i64;
            if available > 0 {
                let offset = if whence == 0 {
                    pos - (current - self.raw_offset())
                } else {
                    pos
                };
                if -self.pos <= offset && offset <= available {
                    self.pos += offset;
                    return Ok((current - available + offset).max(0));
                }
            }
        }
        self.with_lock(|this| {
            let adjusted = if whence == 1 {
                pos - this.raw_offset()
            } else {
                pos
            };
            let result = this.raw_seek(adjusted, whence)?;
            this.raw_pos = -1;
            this.reader_reset_buf();
            Ok(result)
        })
    }

    fn flush(&self) -> Result<PyObjectRef, crate::PyError> {
        // CPython 3.14 test_io.BufferedReaderTest.test_read_on_closed requires
        // the derived raw `closed` state here. PyPy's `simple_flush_w` checks
        // only initialization and therefore lets BufferedReader(BytesIO)
        // flush after close; that observable result is the spec exception.
        self.check_closed("flush of closed file")?;
        super::call_method_result(self.w_raw, "flush", &[])
    }

    fn close(&mut self) -> Result<(), crate::PyError> {
        self.check_init()?;
        if self.with_lock(|this| this.raw_closed())? {
            return Ok(());
        }
        let self_obj = self.self_obj();
        let flush_error = super::call_method_result(self_obj, "flush", &[]).err();
        let close_result =
            self.with_lock(|this| super::call_method_result(this.w_raw, "close", &[]).map(|_| ()));
        if let Err(mut close_error) = close_result {
            if let Some(mut flush_error) = flush_error {
                // PyPy's `try: flush() finally: raw.close()` exposes the
                // earlier flush exception as the close exception's context.
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
        &self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed("truncate of closed file")?;
        let _ = w_size;
        Err(super::unsupported("truncate"))
    }

    fn _dealloc_warn(&self, w_source: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "_dealloc_warn", &[w_source])
    }

    fn __repr__(&self) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
        let self_obj = self.self_obj();
        let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
            return Err(crate::PyError::runtime_error(
                "reentrant call inside BufferedReader.__repr__",
            ));
        };
        let typename = crate::type_methods::arg_type_name(self_obj);
        match crate::baseobjspace::getattr_str(self_obj, "name") {
            Ok(name) => Ok(crate::display::wtf8_format!(
                format!("<{typename} name="),
                unsafe { crate::display::py_repr_wtf8(name)? },
                ">"
            )),
            Err(_) => Ok(rustpython_wtf8::Wtf8Buf::from_string(format!(
                "<{typename}>"
            ))),
        }
    }
}
