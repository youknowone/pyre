//! `__pypy__.PickleBuffer` — `pypy/module/__pypy__/interp_buffer.py
//! W_PickleBuffer`. Wraps a bytes-like object so the `_pickle` accelerator
//! can serialize it either in-band or out-of-band (protocol 5). The
//! `_pickle` save path recognizes the wrapper via `from_obj` and reads its
//! contents through `buffer_view`.

use pyre_object::PyObjectRef;

use crate::PyError;

#[crate::pyre_class("__pypy__.PickleBuffer")]
pub struct W_PickleBuffer {
    /// The wrapped buffer-supporting object, or `None` after `release()`.
    w_obj: PyObjectRef,
}

#[crate::pyre_methods(
    doc = "PickleBuffer(buffer) -> wrapper for potentially out-of-band serialization."
)]
impl W_PickleBuffer {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> PyObjectRef {
        W_PickleBuffer::allocate(W_PickleBuffer {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_obj: pyre_object::w_none(),
        })
    }

    fn __init__(&mut self, buffer: PyObjectRef) -> Result<(), PyError> {
        if !is_buffer_like(buffer) {
            let name = type_name(buffer);
            return Err(PyError::type_error(format!(
                "a bytes-like object is required, not '{name}'"
            )));
        }
        self.w_obj = buffer;
        Ok(())
    }

    /// `raw()` — a memoryview onto the wrapped buffer.
    fn raw(&self) -> Result<PyObjectRef, PyError> {
        let w_obj = self.w_obj;
        if unsafe { pyre_object::is_none(w_obj) } {
            return Err(released_error());
        }
        let mv_type = memoryview_type()
            .ok_or_else(|| PyError::runtime_error("memoryview type unavailable"))?;
        crate::module::_pickle::call_fn(mv_type, &[w_obj])
    }

    /// `release()` — drop the reference to the underlying buffer.
    fn release(&mut self) {
        self.w_obj = pyre_object::w_none();
    }
}

impl W_PickleBuffer {
    /// The wrapped buffer object (`None` after `release()`), read by the
    /// `_pickle` save path.
    pub(crate) fn wrapped(&self) -> PyObjectRef {
        self.w_obj
    }
}

fn released_error() -> PyError {
    PyError::value_error("operation forbidden on released PickleBuffer object")
}

fn type_name(obj: PyObjectRef) -> String {
    match crate::typedef::r#type(obj) {
        Some(t) => unsafe { pyre_object::w_type_get_name(t) }.to_string(),
        None => "object".to_string(),
    }
}

/// `bytes` / `bytearray` / `memoryview` are the buffer-supporting objects the
/// wrapper accepts.
fn is_buffer_like(obj: PyObjectRef) -> bool {
    unsafe { pyre_object::is_bytes(obj) || pyre_object::is_bytearray(obj) || is_memoryview(obj) }
}

fn is_memoryview(obj: PyObjectRef) -> bool {
    match crate::typedef::r#type(obj) {
        Some(t) => unsafe { pyre_object::w_type_get_name(t) == "memoryview" },
        None => false,
    }
}

/// Extract `(contents, readonly)` from a buffer-supporting object: `bytes`
/// is read-only, `bytearray` is mutable, and a `memoryview` reports both
/// through its own contents and `readonly` flag.
pub(crate) fn buffer_view(obj: PyObjectRef) -> Result<(Vec<u8>, bool), PyError> {
    unsafe {
        if pyre_object::is_bytes(obj) {
            return Ok((pyre_object::bytesobject::w_bytes_data(obj).to_vec(), true));
        }
        if pyre_object::is_bytearray(obj) {
            return Ok((
                pyre_object::bytearrayobject::w_bytearray_data(obj).to_vec(),
                false,
            ));
        }
    }
    if is_memoryview(obj) {
        let w_data = crate::module::_pickle::call_meth(obj, "tobytes", &[])?;
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_data) }.to_vec();
        let w_ro = crate::baseobjspace::getattr_str(obj, "readonly")?;
        return Ok((data, crate::baseobjspace::is_true(w_ro)?));
    }
    Err(PyError::type_error(format!(
        "a bytes-like object is required, not '{}'",
        type_name(obj)
    )))
}

/// The `memoryview` builtin type via the live execution context.
fn memoryview_type() -> Option<PyObjectRef> {
    let frame = crate::eval::CURRENT_FRAME.with(|f| f.get());
    if frame.is_null() {
        return None;
    }
    let ec = unsafe { (*frame).execution_context };
    if ec.is_null() {
        return None;
    }
    unsafe { (*ec).lookup_builtin("memoryview") }
}
