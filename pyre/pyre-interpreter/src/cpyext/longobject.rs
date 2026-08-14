//! `int` -- PyPy `cpyext/longobject.py` (and `intobject.py`).

use super::object::{argument, result};
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{c_double, c_int, c_long, c_longlong, c_ulong, c_ulonglong};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromLong(value: c_long) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value as i64))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromLongLong(value: c_longlong) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value as i64))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromSsize_t(value: isize) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_int_new(value as i64))
}

/// Values above `i64::MAX` need the big-integer path, which the `int` slice
/// has not reached; report the overflow rather than truncate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromUnsignedLong(value: c_ulong) -> *mut CPyObject {
    from_unsigned(value as u64)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromUnsignedLongLong(value: c_ulonglong) -> *mut CPyObject {
    from_unsigned(value as u64)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromSize_t(value: usize) -> *mut CPyObject {
    from_unsigned(value as u64)
}

fn from_unsigned(value: u64) -> *mut CPyObject {
    match i64::try_from(value) {
        Ok(value) => pyobject::make_ref(pyre_object::w_int_new(value)),
        Err(_) => {
            super::pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::OverflowError,
                "unsigned value does not fit in a pyre int yet",
            ));
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromDouble(value: c_double) -> *mut CPyObject {
    if !value.is_finite() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::OverflowError,
            "cannot convert float infinity or NaN to integer",
        ));
        return std::ptr::null_mut();
    }
    pyobject::make_ref(pyre_object::w_int_new(value.trunc() as i64))
}

/// `PyLong_FromString`, base 0 or 2..=36.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_FromString(
    text: *const std::ffi::c_char,
    end: *mut *mut std::ffi::c_char,
    base: c_int,
) -> *mut CPyObject {
    if text.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if base != 0 && !(2..=36).contains(&base) {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "int() base must be >= 2 and <= 36, or 0",
        ));
        return std::ptr::null_mut();
    }
    let raw = unsafe { std::ffi::CStr::from_ptr(text) };
    if !end.is_null() {
        unsafe { *end = text.add(raw.to_bytes().len()) as *mut std::ffi::c_char };
    }
    let text = raw.to_string_lossy().trim().to_string();
    let (digits, radix) = split_digits(&text, base);
    match i64::from_str_radix(&digits, radix) {
        Ok(value) => pyobject::make_ref(pyre_object::w_int_new(value)),
        Err(_) => {
            super::pyerrors::set_pending_error(crate::PyError::value_error(format!(
                "invalid literal for int() with base {base}: '{text}'"
            )));
            std::ptr::null_mut()
        }
    }
}

/// The digits `from_str_radix` parses, sign included, and the base they are in.
///
/// Base 0 reads the base from a `0x` / `0o` / `0b` prefix; an explicit 16, 8 or
/// 2 also accepts the prefix that names it, which is what `PyLong_FromString`
/// documents.
fn split_digits(text: &str, base: c_int) -> (String, u32) {
    let (sign, rest) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text.strip_prefix('+').unwrap_or(text)),
    };
    let prefixed = match rest.get(..2).map(str::to_ascii_lowercase).as_deref() {
        Some("0x") => Some(16),
        Some("0o") => Some(8),
        Some("0b") => Some(2),
        _ => None,
    };
    match prefixed {
        Some(detected) if base == 0 || base == detected => {
            (format!("{sign}{}", &rest[2..]), detected as u32)
        }
        _ if base == 0 => (format!("{sign}{rest}"), 10),
        _ => (format!("{sign}{rest}"), base as u32),
    }
}

fn as_i64(object: *mut CPyObject) -> Option<i64> {
    let object = argument(object)?;
    trap(crate::baseobjspace::gateway_int_w(object))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsLong(object: *mut CPyObject) -> c_long {
    as_i64(object).unwrap_or(-1) as c_long
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsLongLong(object: *mut CPyObject) -> c_longlong {
    as_i64(object).unwrap_or(-1) as c_longlong
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsSsize_t(object: *mut CPyObject) -> isize {
    as_i64(object).unwrap_or(-1) as isize
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsUnsignedLong(object: *mut CPyObject) -> c_ulong {
    as_unsigned(object).unwrap_or(u64::MAX) as c_ulong
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsUnsignedLongLong(object: *mut CPyObject) -> c_ulonglong {
    as_unsigned(object).unwrap_or(u64::MAX) as c_ulonglong
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsSize_t(object: *mut CPyObject) -> usize {
    as_unsigned(object).unwrap_or(u64::MAX) as usize
}

fn as_unsigned(object: *mut CPyObject) -> Option<u64> {
    let value = as_i64(object)?;
    match u64::try_from(value) {
        Ok(value) => Some(value),
        Err(_) => {
            super::pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::OverflowError,
                "can't convert negative value to unsigned int",
            ));
            None
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_AsDouble(object: *mut CPyObject) -> c_double {
    let Some(object) = argument(object) else {
        return -1.0;
    };
    trap(crate::baseobjspace::float_w(object)).unwrap_or(-1.0) as c_double
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_int(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyLong_CheckExact(object: *mut CPyObject) -> c_int {
    unsafe { PyLong_Check(object) }
}

/// `PyNumber_Long` — `int(object)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Long(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::gateway_int_w(object).map(pyre_object::w_int_new))
}

fn bool_of(value: bool) -> PyObjectRef {
    pyre_object::boolobject::w_bool_from(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBool_FromLong(value: c_long) -> *mut CPyObject {
    pyobject::make_ref(bool_of(value != 0))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyLong_FromLong as *const ());
    std::hint::black_box(PyLong_FromLongLong as *const ());
    std::hint::black_box(PyLong_FromSsize_t as *const ());
    std::hint::black_box(PyLong_FromUnsignedLong as *const ());
    std::hint::black_box(PyLong_FromUnsignedLongLong as *const ());
    std::hint::black_box(PyLong_FromSize_t as *const ());
    std::hint::black_box(PyLong_FromDouble as *const ());
    std::hint::black_box(PyLong_FromString as *const ());
    std::hint::black_box(PyLong_AsLong as *const ());
    std::hint::black_box(PyLong_AsLongLong as *const ());
    std::hint::black_box(PyLong_AsSsize_t as *const ());
    std::hint::black_box(PyLong_AsUnsignedLong as *const ());
    std::hint::black_box(PyLong_AsUnsignedLongLong as *const ());
    std::hint::black_box(PyLong_AsSize_t as *const ());
    std::hint::black_box(PyLong_AsDouble as *const ());
    std::hint::black_box(PyLong_Check as *const ());
    std::hint::black_box(PyLong_CheckExact as *const ());
    std::hint::black_box(PyNumber_Long as *const ());
    std::hint::black_box(PyBool_FromLong as *const ());
}
