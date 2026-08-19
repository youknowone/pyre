//! `PyUnicodeWriter` -- the buffer an extension builds a `str` in, piece by
//! piece.  Upstream has no counterpart; the shape follows the reference
//! header, whose handle is opaque.
//!
//! Being opaque, what the writer holds is this layer's own text rather than a
//! partly built `str`: nothing is an object until [`PyUnicodeWriter_Finish`]
//! is asked for one, and a writer that is discarded made none.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use rustpython_wtf8::{CodePoint, Wtf8Buf};
use std::ffi::{CStr, c_char, c_int};

/// The text written so far.  Opaque to C, which only ever holds the pointer.
pub struct CPyUnicodeWriter {
    text: Wtf8Buf,
}

/// The writer a call names, or `None` with the failure already recorded.
///
/// # Safety
/// `writer` must be null or a writer [`PyUnicodeWriter_Create`] answered with.
unsafe fn writer<'a>(writer: *mut CPyUnicodeWriter) -> Option<&'a mut CPyUnicodeWriter> {
    if writer.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(unsafe { &mut *writer })
}

/// `PyUnicodeWriter_Create(length)` — `length` is what the caller expects to
/// write, which is a reservation rather than a limit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_Create(length: isize) -> *mut CPyUnicodeWriter {
    if length < 0 {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "length must be positive".to_owned(),
        ));
        return std::ptr::null_mut();
    }
    let mut text = Wtf8Buf::new();
    if text.try_reserve(length as usize).is_err() {
        unsafe { super::pyerrors::PyErr_NoMemory() };
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(CPyUnicodeWriter { text }))
}

/// `PyUnicodeWriter_Discard(writer)` — give the writer up without asking for
/// what it holds.
///
/// # Safety
/// `writer` must be null or a writer this call takes over.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_Discard(writer: *mut CPyUnicodeWriter) {
    if writer.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(writer) });
}

/// `PyUnicodeWriter_Finish(writer)` — the `str` the writer holds.
///
/// The writer is given up either way, so a caller that got NULL must not
/// discard it as well.
///
/// # Safety
/// `writer` must be a writer this call takes over.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_Finish(writer: *mut CPyUnicodeWriter) -> *mut CPyObject {
    if writer.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let written = unsafe { Box::from_raw(writer) };
    pyobject::make_ref(pyre_object::w_str_from_wtf8_managed(written.text))
}

/// `PyUnicodeWriter_WriteChar(writer, ch)` — one code point, a surrogate
/// among them: a `str` holds those, and only a value outside the range has
/// nothing to stand for.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_WriteChar(
    handle: *mut CPyUnicodeWriter,
    ch: u32,
) -> c_int {
    let Some(handle) = (unsafe { writer(handle) }) else {
        return -1;
    };
    let Some(point) = CodePoint::from_u32(ch) else {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "character must be in range(0x110000)".to_owned(),
        ));
        return -1;
    };
    handle.text.push(point);
    0
}

/// The bytes a `const char *` and a size name, `-1` meaning NUL-terminated.
///
/// # Safety
/// `string` must address `size` readable bytes, or be NUL-terminated.
unsafe fn sized_bytes<'a>(string: *const c_char, size: isize) -> Option<&'a [u8]> {
    if size == 0 {
        // Nothing to read, so the caller need not have named a buffer at all.
        return Some(&[]);
    }
    if string.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(match size {
        ..0 => unsafe { CStr::from_ptr(string) }.to_bytes(),
        size => unsafe { std::slice::from_raw_parts(string as *const u8, size as usize) },
    })
}

/// `PyUnicodeWriter_WriteUTF8(writer, str, size)`.
///
/// Strict: bytes that are not UTF-8 leave the writer holding exactly what it
/// held before, so a caller that goes on to discard it loses nothing else.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_WriteUTF8(
    handle: *mut CPyUnicodeWriter,
    string: *const c_char,
    size: isize,
) -> c_int {
    let (Some(handle), Some(bytes)) = (unsafe { writer(handle) }, unsafe {
        sized_bytes(string, size)
    }) else {
        return -1;
    };
    match std::str::from_utf8(bytes) {
        Ok(text) => {
            handle.text.push_str(text);
            0
        }
        Err(error) => {
            let position = error.valid_up_to();
            super::pyerrors::set_pending_error(crate::typedef::unicode_decode_error(
                "utf-8",
                bytes,
                position,
                position + 1,
                utf8_reason(&error),
            ));
            -1
        }
    }
}

/// What a strict UTF-8 decode refused the byte for.
fn utf8_reason(error: &std::str::Utf8Error) -> &'static str {
    match error.error_len() {
        None => "unexpected end of data",
        Some(_) => "invalid start byte",
    }
}

/// `PyUnicodeWriter_WriteASCII(writer, str, size)` — the caller promises the
/// bytes are ASCII, so each one is the code point it spells.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_WriteASCII(
    handle: *mut CPyUnicodeWriter,
    string: *const c_char,
    size: isize,
) -> c_int {
    let (Some(handle), Some(bytes)) = (unsafe { writer(handle) }, unsafe {
        sized_bytes(string, size)
    }) else {
        return -1;
    };
    for &byte in bytes {
        handle.text.push(CodePoint::from(byte as char));
    }
    0
}

/// `PyUnicodeWriter_WriteWideChar(writer, str, size)`, whose unit is one code
/// point where a `wchar_t` is four bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_WriteWideChar(
    handle: *mut CPyUnicodeWriter,
    string: *const super::unicodeobject::wchar_t,
    size: isize,
) -> c_int {
    let Some(handle) = (unsafe { writer(handle) }) else {
        return -1;
    };
    if size == 0 {
        // Nothing to read, so the caller need not have named a buffer at all.
        return 0;
    }
    if string.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let mut index = 0isize;
    loop {
        if size >= 0 && index >= size {
            break;
        }
        let unit = unsafe { *string.offset(index) } as u32;
        if size < 0 && unit == 0 {
            break;
        }
        match CodePoint::from_u32(unit) {
            Some(point) => handle.text.push(point),
            None => {
                super::pyerrors::set_pending_error(crate::PyError::value_error(
                    "character must be in range(0x110000)".to_owned(),
                ));
                return -1;
            }
        }
        index += 1;
    }
    0
}

/// `PyUnicodeWriter_WriteUCS4(writer, str, size)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_WriteUCS4(
    handle: *mut CPyUnicodeWriter,
    string: *mut u32,
    size: isize,
) -> c_int {
    let Some(handle) = (unsafe { writer(handle) }) else {
        return -1;
    };
    if size < 0 {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "size must be positive".to_owned(),
        ));
        return -1;
    }
    if size == 0 {
        return 0;
    }
    if string.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    for &unit in unsafe { std::slice::from_raw_parts(string as *const u32, size as usize) } {
        match CodePoint::from_u32(unit) {
            Some(point) => handle.text.push(point),
            None => {
                super::pyerrors::set_pending_error(crate::PyError::value_error(
                    "character must be in range(0x110000)".to_owned(),
                ));
                return -1;
            }
        }
    }
    0
}

/// Append what `convert` makes of `object`, which is where a writer runs
/// Python code and so where it can fail for reasons of its own.
fn write_converted(
    handle: *mut CPyUnicodeWriter,
    object: *mut CPyObject,
    convert: unsafe fn(PyObjectRef) -> Result<Wtf8Buf, crate::PyError>,
) -> c_int {
    super::object::realize_all([object]);
    let (Some(_), Some(value)) = (unsafe { writer(handle) }, argument(object)) else {
        return -1;
    };
    let text = match unsafe { convert(value) } {
        Ok(text) => text,
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            return -1;
        }
    };
    // The conversion runs Python code, so the writer is reached only once it
    // is over and nothing of the heap is held across it.
    let Some(handle) = (unsafe { writer(handle) }) else {
        return -1;
    };
    handle.text.push_wtf8(&text);
    0
}

/// `PyUnicodeWriter_WriteStr(writer, obj)` -- `str(obj)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_WriteStr(
    handle: *mut CPyUnicodeWriter,
    object: *mut CPyObject,
) -> c_int {
    write_converted(handle, object, crate::display::py_str_wtf8)
}

/// `PyUnicodeWriter_WriteRepr(writer, obj)` -- `repr(obj)`.
///
/// A NULL object is spelled `<NULL>` rather than refused: the caller reaching
/// here is often reporting a failure, and there is nothing left to report with
/// if describing what failed can fail in turn.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_WriteRepr(
    handle: *mut CPyUnicodeWriter,
    object: *mut CPyObject,
) -> c_int {
    if object.is_null() {
        let Some(handle) = (unsafe { writer(handle) }) else {
            return -1;
        };
        handle.text.push_str("<NULL>");
        return 0;
    }
    write_converted(handle, object, crate::display::py_repr_wtf8)
}

/// `PyUnicodeWriter_WriteSubstring(writer, str, start, end)` — `str[start:end]`
/// in code points.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicodeWriter_WriteSubstring(
    handle: *mut CPyUnicodeWriter,
    object: *mut CPyObject,
    start: isize,
    end: isize,
) -> c_int {
    super::object::realize_all([object]);
    let (Some(_), Some(value)) = (unsafe { writer(handle) }, argument(object)) else {
        return -1;
    };
    if !unsafe { pyre_object::unicodeobject::is_str(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "expect str, not {}",
            crate::type_methods::arg_type_name(value)
        )));
        return -1;
    }
    let points: Vec<CodePoint> = unsafe { pyre_object::w_str_get_wtf8(value) }
        .code_points()
        .collect();
    if start < 0 || start > end {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "invalid start argument".to_owned(),
        ));
        return -1;
    }
    if end > points.len() as isize {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "invalid end argument".to_owned(),
        ));
        return -1;
    }
    let Some(handle) = (unsafe { writer(handle) }) else {
        return -1;
    };
    for &point in &points[start as usize..end as usize] {
        handle.text.push(point);
    }
    0
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyUnicodeWriter_Create as *const ());
    std::hint::black_box(PyUnicodeWriter_Discard as *const ());
    std::hint::black_box(PyUnicodeWriter_Finish as *const ());
    std::hint::black_box(PyUnicodeWriter_WriteChar as *const ());
    std::hint::black_box(PyUnicodeWriter_WriteUTF8 as *const ());
    std::hint::black_box(PyUnicodeWriter_WriteASCII as *const ());
    std::hint::black_box(PyUnicodeWriter_WriteWideChar as *const ());
    std::hint::black_box(PyUnicodeWriter_WriteUCS4 as *const ());
    std::hint::black_box(PyUnicodeWriter_WriteStr as *const ());
    std::hint::black_box(PyUnicodeWriter_WriteRepr as *const ());
    std::hint::black_box(PyUnicodeWriter_WriteSubstring as *const ());
}
