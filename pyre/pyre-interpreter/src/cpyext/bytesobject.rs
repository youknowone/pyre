//! `bytes` -- PyPy `cpyext/bytesobject.py`.

use super::object::argument;
use super::pyobject::{self, CPyObject};
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::bytesobject::is_bytes(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_CheckExact(object: *mut CPyObject) -> c_int {
    unsafe { PyBytes_Check(object) }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyBytes_FromString as *const ());
    std::hint::black_box(PyBytes_FromStringAndSize as *const ());
    std::hint::black_box(PyBytes_AsString as *const ());
    std::hint::black_box(PyBytes_AsStringAndSize as *const ());
    std::hint::black_box(PyBytes_Size as *const ());
    std::hint::black_box(PyBytes_Check as *const ());
    std::hint::black_box(PyBytes_CheckExact as *const ());
}
