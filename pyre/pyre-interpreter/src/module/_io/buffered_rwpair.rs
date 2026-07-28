//! Paired buffered streams — PyPy `W_BufferedRWPair`.

use pyre_object::*;

use super::DEFAULT_BUFFER_SIZE;

#[crate::pyre_class("_io.BufferedRWPair")]
pub struct W_BufferedRWPair {
    w_reader: PyObjectRef,
    w_writer: PyObjectRef,
}

impl Default for W_BufferedRWPair {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            w_reader: PY_NULL,
            w_writer: PY_NULL,
        }
    }
}

impl W_BufferedRWPair {
    fn check_reader(&self) -> Result<PyObjectRef, crate::PyError> {
        if self.w_reader.is_null() {
            Err(crate::PyError::value_error(
                "I/O operation on uninitialized object",
            ))
        } else {
            Ok(self.w_reader)
        }
    }

    fn check_writer(&self) -> Result<PyObjectRef, crate::PyError> {
        if self.w_writer.is_null() {
            Err(crate::PyError::value_error(
                "I/O operation on uninitialized object",
            ))
        } else {
            Ok(self.w_writer)
        }
    }

    fn reader_call(&self, name: &str, args: &[PyObjectRef]) -> crate::PyResult {
        super::call_method_result(self.check_reader()?, name, args)
    }

    fn writer_call(&self, name: &str, args: &[PyObjectRef]) -> crate::PyResult {
        super::call_method_result(self.check_writer()?, name, args)
    }
}

#[crate::pyre_methods(
    base = super::buffered_iobase_type(),
    weakrefable,
    doc = "BufferedRWPair(reader, writer, buffer_size=DEFAULT_BUFFER_SIZE)"
)]
impl W_BufferedRWPair {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> PyObjectRef {
        let obj = W_BufferedRWPair::allocate_stable(W_BufferedRWPair::default());
        // interp_bufferedio.py:1175-1177 `needs_finalizer`: `self.w_writer` and
        // `self.w_reader` have their own finalizer, so the pair itself needs
        // none. A subclass keeps one — its `close` may do anything.
        let needs_finalizer = !cls.is_null() && !std::ptr::eq(cls, type_object());
        super::tag_io_instance_with_finalizer(obj, cls, needs_finalizer)
    }

    fn __init__(
        &mut self,
        w_reader: PyObjectRef,
        w_writer: PyObjectRef,
        #[default(DEFAULT_BUFFER_SIZE)] buffer_size: i64,
    ) -> Result<(), crate::PyError> {
        self.w_reader = PY_NULL;
        self.w_writer = PY_NULL;

        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_reader);
        pyre_object::gc_roots::pin_root(w_writer);
        let input_sp = pyre_object::gc_roots::shadow_stack_len() - 2;
        let reader_size = w_int_new(buffer_size);
        let reader = crate::call::call_function_impl_result(
            super::buffered::type_object(),
            &[
                pyre_object::gc_roots::shadow_stack_get(input_sp),
                reader_size,
            ],
        )?;
        pyre_object::gc_roots::pin_root(reader);
        let reader_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let writer_size = w_int_new(buffer_size);
        let writer = crate::call::call_function_impl_result(
            super::buffered_writer::type_object(),
            &[
                pyre_object::gc_roots::shadow_stack_get(input_sp + 1),
                writer_size,
            ],
        )?;
        self.w_reader = pyre_object::gc_roots::shadow_stack_get(reader_slot);
        self.w_writer = writer;
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(())
    }

    fn read(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("read", &args[1..])
    }

    fn peek(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("peek", &args[1..])
    }

    fn read1(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("read1", &args[1..])
    }

    fn readinto(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("readinto", &args[1..])
    }

    fn readable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.reader_call("readable", &[])
    }

    fn write(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        self.writer_call("write", &args[1..])
    }

    fn flush(&self) -> Result<PyObjectRef, crate::PyError> {
        self.writer_call("flush", &[])
    }

    fn writable(&self) -> Result<PyObjectRef, crate::PyError> {
        self.writer_call("writable", &[])
    }

    fn close(&self) -> Result<(), crate::PyError> {
        let writer = self.check_writer()?;
        let mut writer_error = super::call_method_result(writer, "close", &[]).err();
        let reader = self.check_reader()?;
        if let Err(mut reader_error) = super::call_method_result(reader, "close", &[]) {
            if let Some(mut context) = writer_error {
                let _roots = pyre_object::gc_roots::push_roots();
                let context_obj = context.to_exc_object();
                pyre_object::gc_roots::pin_root(context_obj);
                let context_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let reader_obj = reader_error.to_exc_object();
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_context(
                        reader_obj,
                        pyre_object::gc_roots::shadow_stack_get(context_slot),
                    )
                };
                reader_error.exc_object = reader_obj;
            }
            return Err(reader_error);
        }
        if let Some(error) = writer_error.take() {
            return Err(error);
        }
        Ok(())
    }

    fn isatty(&self) -> Result<PyObjectRef, crate::PyError> {
        let writer = self.writer_call("isatty", &[])?;
        if crate::baseobjspace::is_true(writer)? {
            Ok(w_bool_from(true))
        } else {
            self.reader_call("isatty", &[])
        }
    }

    #[getter]
    fn closed(&self) -> Result<PyObjectRef, crate::PyError> {
        crate::baseobjspace::getattr_str(self.check_writer()?, "closed")
    }
}
