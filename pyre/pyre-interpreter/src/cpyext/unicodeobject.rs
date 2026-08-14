//! `str` -- PyPy `cpyext/unicodeobject.py`.
//!
//! `PyUnicode_AsUTF8` has to hand out a stable, NUL-terminated address.  The
//! interpreter's own storage is WTF-8, unterminated and movable, so the bytes
//! are copied into the mirror's cache — the counterpart of the `c_utf8` field
//! upstream fills on the `PyUnicodeObject` mirror.  The address is therefore
//! valid for exactly as long as the caller's reference to the object.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use std::ffi::{CStr, c_char, c_int};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FromString(text: *const c_char) -> *mut CPyObject {
    if text.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    from_utf8_bytes(bytes)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FromStringAndSize(
    text: *const c_char,
    size: isize,
) -> *mut CPyObject {
    if text.is_null() || size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, size as usize) };
    from_utf8_bytes(bytes)
}

fn from_utf8_bytes(bytes: &[u8]) -> *mut CPyObject {
    match std::str::from_utf8(bytes) {
        Ok(text) => pyobject::make_ref(pyre_object::w_str_new(text)),
        Err(error) => {
            super::pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::UnicodeDecodeError,
                format!(
                    "'utf-8' codec can't decode byte at position {}",
                    error.valid_up_to()
                ),
            ));
            std::ptr::null_mut()
        }
    }
}

fn text_argument(object: *mut CPyObject, function: &str) -> Option<pyre_object::PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::unicodeobject::is_str(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): str expected"
        )));
        return None;
    }
    Some(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsUTF8(object: *mut CPyObject) -> *const c_char {
    unsafe { PyUnicode_AsUTF8AndSize(object, std::ptr::null_mut()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsUTF8AndSize(
    object: *mut CPyObject,
    size: *mut isize,
) -> *const c_char {
    let Some(value) = text_argument(object, "PyUnicode_AsUTF8") else {
        return std::ptr::null();
    };
    let (pointer, length) = unsafe {
        pyobject::cached_bytes(object, || {
            pyre_object::w_str_get_wtf8(value).as_bytes().to_vec()
        })
    };
    if !size.is_null() {
        unsafe { *size = length as isize };
    }
    pointer
}

/// The number of code points, which is what `len()` reports.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_GetLength(object: *mut CPyObject) -> isize {
    let Some(value) = text_argument(object, "PyUnicode_GetLength") else {
        return -1;
    };
    unsafe { pyre_object::unicodeobject::w_str_len(value) as isize }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::unicodeobject::is_str(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_CheckExact(object: *mut CPyObject) -> c_int {
    unsafe { PyUnicode_Check(object) }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyUnicode_FromString as *const ());
    std::hint::black_box(PyUnicode_FromStringAndSize as *const ());
    std::hint::black_box(PyUnicode_AsUTF8 as *const ());
    std::hint::black_box(PyUnicode_AsUTF8AndSize as *const ());
    std::hint::black_box(PyUnicode_GetLength as *const ());
    std::hint::black_box(PyUnicode_Check as *const ());
    std::hint::black_box(PyUnicode_CheckExact as *const ());
}
