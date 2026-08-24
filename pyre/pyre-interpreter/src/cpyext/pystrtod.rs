//! String to double -- PyPy `cpyext/pystrtod.py`.

use super::pyobject::CPyObject;
use std::ffi::{CStr, CString, c_char, c_double};

/// How many leading bytes of `text` read as a floating-point number.
///
/// This is not the `float()` grammar: the conversion takes a prefix rather
/// than the whole string, and accepts neither leading whitespace nor the
/// underscore separators a Python literal may carry.
fn float_prefix(text: &[u8]) -> usize {
    let signed = usize::from(matches!(text.first(), Some(b'+' | b'-')));
    let body = &text[signed..];
    for word in [&b"infinity"[..], &b"inf"[..], &b"nan"[..]] {
        if body.len() >= word.len() && body[..word.len()].eq_ignore_ascii_case(word) {
            return signed + word.len();
        }
    }
    let run = |from: usize| {
        let mut end = from;
        while body.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        end
    };
    let whole = run(0);
    let mantissa = match body.get(whole) {
        Some(b'.') => run(whole + 1),
        _ => whole,
    };
    // The point stands in for either run of digits but not for both: `.5` and
    // `5.` are numbers, and `.` alone is not.
    if whole == 0 && mantissa <= 1 {
        return 0;
    }
    // An `e` that no exponent follows belongs to whatever comes after the
    // number rather than to the number.
    let exponent = mantissa + 1 + usize::from(matches!(body.get(mantissa + 1), Some(b'+' | b'-')));
    let end = match body.get(mantissa) {
        Some(b'e' | b'E') => match run(exponent) {
            end if end > exponent => end,
            _ => mantissa,
        },
        _ => mantissa,
    };
    signed + end
}

/// How the failures below quote the string back.
fn quoted(text: &[u8]) -> String {
    String::from_utf8_lossy(&text[..text.len().min(200)]).into_owned()
}

/// `pystrtod.py PyOS_string_to_double`.
///
/// `endptr` decides how much of `s` has to be a number: a caller that passes
/// one means to carry on where the number ended and owns whatever follows it,
/// and a caller that passes none is reading the whole string.
///
/// # Safety
/// `s` must be a NUL-terminated string, and `endptr` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyOS_string_to_double(
    s: *const c_char,
    endptr: *mut *mut c_char,
    overflow_exception: *mut CPyObject,
) -> c_double {
    if s.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1.0;
    }
    let text = unsafe { CStr::from_ptr(s) }.to_bytes();
    let read = float_prefix(text);
    if !endptr.is_null() {
        unsafe { *endptr = s.add(read) as *mut c_char };
    }
    let source = &text[..read];
    // A prefix of nothing is not a number; a remainder is only the caller's
    // to deal with when they asked where the number ended.
    let complete = read != 0 && (!endptr.is_null() || read == text.len());
    let parsed = std::str::from_utf8(source)
        .ok()
        .and_then(|source| source.parse::<f64>().ok());
    let value = match parsed {
        Some(value) if complete => value,
        _ => {
            super::pyerrors::set_pending_error(crate::PyError::value_error(format!(
                "could not convert string to float: '{}'",
                quoted(text)
            )));
            return -1.0;
        }
    };
    // `errno == ERANGE && fabs(x) >= 1.0`: a magnitude too large for the type
    // reads back as infinity, which only a source that spelled it can be.
    let spelled = source
        .iter()
        .any(|byte| byte.is_ascii_alphabetic() && !matches!(byte, b'e' | b'E'));
    if value.is_infinite() && !spelled && !overflow_exception.is_null() {
        let message = CString::new(format!(
            "value too large to convert to float: '{}'",
            quoted(text)
        ))
        .unwrap_or_default();
        unsafe { super::pyerrors::PyErr_SetString(overflow_exception, message.as_ptr()) };
        return -1.0;
    }
    value
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyOS_string_to_double as *const ());
}
