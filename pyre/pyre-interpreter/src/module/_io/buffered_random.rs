//! Seekable buffered binary streams — PyPy `W_BufferedRandom`.

use pyre_object::*;

use super::DEFAULT_BUFFER_SIZE;

const STATE_ZERO: i64 = 0;
const STATE_OK: i64 = 1;
const STATE_DETACHED: i64 = 2;

#[crate::pyre_class("_io.BufferedRandom")]
pub struct W_BufferedRandom {
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

impl Default for W_BufferedRandom {
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

impl W_BufferedRandom {
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
        if self.locked {
            return Err(crate::PyError::runtime_error("reentrant call"));
        }
        self.locked = true;
        let result = body(self);
        self.locked = false;
        result
    }

    fn reader_reset_buf(&mut self) {
        self.read_end = -1;
    }

    fn writer_reset_buf(&mut self) {
        self.write_pos = 0;
        self.write_end = -1;
    }

    fn readahead(&self) -> usize {
        if self.readable && self.read_end != -1 {
            debug_assert!(self.read_end >= self.pos);
            (self.read_end - self.pos) as usize
        } else {
            0
        }
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

    fn adjust_position(&mut self, new_pos: i64) {
        debug_assert!(new_pos >= 0);
        self.pos = new_pos;
        if self.readable && self.read_end != -1 && self.read_end < new_pos {
            self.read_end = new_pos;
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
            return Err(super::buffered::make_blocking_error());
        }
        let size = super::buffered::raw_readinto_size(result, length)?;
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
            return Err(super::buffered_writer::make_write_blocking_error(0));
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
            let data = self.buffer_bytes(self.write_pos as usize, self.write_end as usize);
            let written = match self.raw_write(&data) {
                Ok(written) => written,
                Err(error) if super::buffered_writer::is_blocking_error(&error) => {
                    return Err(super::buffered_writer::make_write_blocking_error(0));
                }
                Err(error) => return Err(error),
            };
            self.write_pos += written as i64;
            self.raw_pos = self.write_pos;
        }
        self.writer_reset_buf();
        Ok(())
    }

    fn flush_and_rewind_unlocked(&mut self) -> Result<(), crate::PyError> {
        self.writer_flush_unlocked()?;
        if self.readable {
            let offset = self.raw_offset();
            let result = self.raw_seek(-offset, 1).map(|_| ());
            self.reader_reset_buf();
            result?;
        }
        Ok(())
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
                this.adjust_position(this.pos + data.len() as i64);
                this.write_end = this.write_end.max(this.pos);
                return Ok(data.len() as i64);
            }

            if let Err(error) = this.writer_flush_unlocked() {
                if !super::buffered_writer::is_blocking_error(&error) {
                    return Err(error);
                }
                if this.readable {
                    this.reader_reset_buf();
                }
                let pending = this.buffer_bytes(this.write_pos as usize, this.write_end as usize);
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
                return Err(super::buffered_writer::make_write_blocking_error(available));
            }

            let offset = this.raw_offset();
            if offset != 0 {
                this.raw_seek(-offset, 1)?;
                this.raw_pos -= offset;
            }

            let mut written = 0usize;
            while data.len() - written > this.buffer_size as usize {
                match this.raw_write(&data[written..]) {
                    Ok(0) => {
                        return Err(super::buffered_writer::make_write_blocking_error(written));
                    }
                    Ok(size) => written += size,
                    Err(error) if super::buffered_writer::is_blocking_error(&error) => {
                        let take = (data.len() - written).min(this.buffer_size as usize);
                        let buffer = unsafe {
                            pyre_object::bytearrayobject::w_bytearray_data_mut(this.buffer)
                        };
                        buffer[..take].copy_from_slice(&data[written..written + take]);
                        this.raw_pos = 0;
                        this.adjust_position(take as i64);
                        this.write_pos = 0;
                        this.write_end = take as i64;
                        written += take;
                        return Err(super::buffered_writer::make_write_blocking_error(written));
                    }
                    Err(error) => return Err(error),
                }
            }

            if this.readable {
                this.reader_reset_buf();
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
            this.adjust_position(remaining as i64);
            this.raw_pos = 0;
            Ok(written as i64)
        })
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
        if self.writable {
            self.flush_and_rewind_unlocked()?;
        }
        self.reader_reset_buf();
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
        if self.writable {
            self.flush_and_rewind_unlocked()?;
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
            let size = super::buffered::raw_readinto_size(result, block)?;
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
                Err(error) if super::buffered::is_blocking_error(&error) => {
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
                if this.writable {
                    this.flush_and_rewind_unlocked()?;
                }
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
                    let read = super::buffered::raw_readinto_size(result, requested)?;
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
                    Err(error) if super::buffered::is_blocking_error(&error) => 0,
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
            if this.writable {
                this.flush_and_rewind_unlocked()?;
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
                    let size = super::buffered::raw_readinto_size(result, remaining)?;
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
                    Err(error) if super::buffered::is_blocking_error(&error) => 0,
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
    doc = "BufferedRandom(raw, buffer_size=DEFAULT_BUFFER_SIZE)"
)]
impl W_BufferedRandom {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (params, kwargs) = crate::builtins::split_builtin_kwargs(args);
        let given = params.len().saturating_sub(1) + crate::builtins::real_kwarg_count(kwargs);
        if given > 2 {
            return Err(crate::PyError::type_error(format!(
                "BufferedRandom() takes at most 2 arguments ({} given)",
                given
            )));
        }
        let obj = W_BufferedRandom::allocate_stable(W_BufferedRandom::default());
        Ok(super::tag_io_instance(obj, cls))
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
        let writable = super::call_method_result(w_raw, "writable", &[])?;
        if !crate::baseobjspace::is_true(writable)? {
            return Err(super::unsupported("File or stream is not writable."));
        }
        let seekable = super::call_method_result(w_raw, "seekable", &[])?;
        if !crate::baseobjspace::is_true(seekable)? {
            return Err(super::unsupported("File or stream is not seekable."));
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
        self.writable = true;
        self.pos = 0;
        self.raw_pos = 0;
        self.reader_reset_buf();
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

    fn read(
        &mut self,
        #[default(pyre_object::w_none())] w_size: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        let size = super::iobase_convert_size(Some(w_size))?;
        self.read_size(size)
    }

    fn read1(&mut self, #[default(-1)] size: i64) -> Result<PyObjectRef, crate::PyError> {
        self.read1_size(size)
    }

    fn peek(&mut self, #[default(0)] _size: i64) -> Result<PyObjectRef, crate::PyError> {
        self.check_closed("peek of closed file")?;
        self.with_lock(|this| {
            if this.writable {
                this.flush_and_rewind_unlocked()?;
            }
            let mut have = this.readahead();
            if have == 0 {
                this.reader_reset_buf();
                have = match this.fill_buffer() {
                    Ok(size) => size,
                    Err(error) if super::buffered::is_blocking_error(&error) => 0,
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
        let mut limit = super::iobase_convert_size(Some(w_limit))?;
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
            if this.writable {
                this.flush_and_rewind_unlocked()?;
            }
            loop {
                this.reader_reset_buf();
                let filled = match this.fill_buffer() {
                    Ok(size) => size,
                    Err(error) if super::buffered::is_blocking_error(&error) => 0,
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

    fn write(&mut self, w_data: PyObjectRef) -> Result<i64, crate::PyError> {
        let data = super::buffered_writer::input_bytes(w_data)?;
        self.write_bytes(&data)
    }

    fn flush(&mut self) -> Result<(), crate::PyError> {
        self.check_closed("flush of closed file")?;
        self.with_lock(Self::flush_and_rewind_unlocked)
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
        if whence != 2 && self.readable {
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
            if this.writable {
                this.writer_flush_unlocked()?;
                this.writer_reset_buf();
            }
            let adjusted = if whence == 1 {
                pos - this.raw_offset()
            } else {
                pos
            };
            let result = this.raw_seek(adjusted, whence)?;
            this.raw_pos = -1;
            if this.readable {
                this.reader_reset_buf();
            }
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
            if this.writable {
                this.flush_and_rewind_unlocked()?;
            }
            this.abs_pos = -1;
            super::call_method_result(this.w_raw, "truncate", &[w_size])
        })
    }

    fn _dealloc_warn(&self, w_source: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        self.check_init()?;
        super::call_method_result(self.w_raw, "_dealloc_warn", &[w_source])
    }

    fn __repr__(&self) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
        let self_obj = self.self_obj();
        let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
            return Err(crate::PyError::runtime_error(
                "reentrant call inside BufferedRandom.__repr__",
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
