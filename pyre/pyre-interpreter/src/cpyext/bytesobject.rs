//! `bytes` -- PyPy `cpyext/bytesobject.py`.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char, c_int};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_FromString(text: *const c_char) -> *mut CPyObject {
    if text.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    pyobject::make_ref(pyre_object::bytesobject::w_bytes_from_bytes(bytes))
}

/// A NULL `text` asks CPython for an uninitialized buffer the caller then
/// fills through `PyBytes_AS_STRING`.  Pyre's `bytes` is immutable from the
/// moment it is built and its storage is not the address `PyBytes_AsString`
/// hands out, so that pattern cannot work here and is rejected instead of
/// silently producing a `bytes` the caller's writes never reach.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_FromStringAndSize(
    text: *const c_char,
    size: isize,
) -> *mut CPyObject {
    if size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if text.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyBytes_FromStringAndSize(NULL, size) is not implemented yet",
        ));
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, size as usize) };
    pyobject::make_ref(pyre_object::bytesobject::w_bytes_from_bytes(bytes))
}

fn bytes_argument(object: *mut CPyObject, function: &str) -> Option<pyre_object::PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::bytesobject::is_bytes(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): bytes expected"
        )));
        return None;
    }
    Some(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_AsString(object: *mut CPyObject) -> *mut c_char {
    let Some(value) = bytes_argument(object, "PyBytes_AsString") else {
        return std::ptr::null_mut();
    };
    let (pointer, _) = unsafe {
        pyobject::cached_bytes(object, || {
            pyre_object::bytesobject::w_bytes_data(value).to_vec()
        })
    };
    pointer as *mut c_char
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_AsStringAndSize(
    object: *mut CPyObject,
    buffer: *mut *mut c_char,
    size: *mut isize,
) -> c_int {
    let Some(value) = bytes_argument(object, "PyBytes_AsStringAndSize") else {
        return -1;
    };
    if buffer.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let (pointer, length) = unsafe {
        pyobject::cached_bytes(object, || {
            pyre_object::bytesobject::w_bytes_data(value).to_vec()
        })
    };
    unsafe {
        *buffer = pointer as *mut c_char;
        if !size.is_null() {
            *size = length as isize;
        } else if pyre_object::bytesobject::w_bytes_data(value).contains(&0) {
            super::pyerrors::set_pending_error(crate::PyError::value_error("embedded null byte"));
            return -1;
        }
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_Size(object: *mut CPyObject) -> isize {
    let Some(value) = bytes_argument(object, "PyBytes_Size") else {
        return -1;
    };
    unsafe { pyre_object::bytesobject::w_bytes_len(value) as isize }
}

/// `bytesobject.py:261-268 PyBytes_FromObject` — the `bytes` an object's buffer
/// or its elements make, without the `bytes(n)` count spelling and without a
/// codec.
///
/// A `__bytes__` override is *not* consulted here.  That is
/// `invoke_bytes_method`, and the only entry point that runs it is
/// [`super::object::PyObject_Bytes`] (`object.py:199-201`).
pub(super) fn bytes_of(object: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::bytesobject::is_bytes(object) } {
        // An exact `bytes` is its own answer; a subclass instance is copied
        // down to one.
        if unsafe {
            pyre_object::pyobject::is_exact_type(object, &pyre_object::bytesobject::BYTES_TYPE)
        } {
            return Ok(object);
        }
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(object) }.to_vec();
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data));
    }
    if unsafe { pyre_object::is_str(object) } {
        return Err(crate::PyError::type_error(
            "cannot convert 'str' object to bytes",
        ));
    }
    // `_convert_from_buffer_or_iterable`: a read-only buffer if the object
    // exports one, otherwise its elements as bytes.
    if let Some(buffer) = crate::baseobjspace::full_ro_buffer_bytes(object)? {
        let data = buffer.as_bytes().to_vec();
        buffer.release();
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let iterator = match crate::baseobjspace::iter(object) {
        Ok(iterator) => iterator,
        Err(error) => {
            if unsafe { crate::baseobjspace::lookup(object, "__iter__") }.is_none() {
                return Err(crate::PyError::type_error(format!(
                    "cannot convert '{}' object to bytes",
                    crate::type_methods::arg_type_name(object)
                )));
            }
            return Err(error);
        }
    };
    let iterator_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(iterator);
    let mut data = Vec::new();
    loop {
        let item =
            crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iterator_slot));
        match item {
            Ok(item) => data.push(unsafe { crate::baseobjspace::byte_w(item, "bytes") }?),
            Err(error) if error.kind == crate::PyErrorKind::StopIteration => break,
            Err(error) => return Err(error),
        }
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
}

/// `PyBytes_FromObject(object)` — [`bytes_of`] as an entry point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_FromObject(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    super::object::result(bytes_of(object))
}

/// `PyBytes_AS_STRING(object)` — the unchecked spelling of
/// [`PyBytes_AsString`], which takes `void *` so a caller may hand it a
/// `PyBytesObject *` without a cast.
///
/// # Safety
/// `object` must be a live `bytes` mirror.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_AS_STRING(object: *mut std::ffi::c_void) -> *mut c_char {
    unsafe { PyBytes_AsString(object as *mut CPyObject) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::bytesobject::is_bytes(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::bytesobject::BYTES_TYPE))
        as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyBytes_FromString as *const ());
    std::hint::black_box(PyBytes_FromStringAndSize as *const ());
    std::hint::black_box(PyBytes_AsString as *const ());
    std::hint::black_box(PyBytes_AsStringAndSize as *const ());
    std::hint::black_box(PyBytes_Size as *const ());
    std::hint::black_box(PyBytes_FromObject as *const ());
    std::hint::black_box(PyBytes_AS_STRING as *const ());
    std::hint::black_box(PyBytes_Check as *const ());
    std::hint::black_box(PyBytes_CheckExact as *const ());
}
