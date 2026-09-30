//! File-object entry points -- PyPy `cpyext/pyfile.py`.
//!
//! `PyFile_FromString` is absent: 3.14 no longer declares it. `n < 0` on
//! `PyFile_GetLine` strips one trailing newline and raises `EOFError` on an
//! empty read, which `pyfile.py` leaves as an `XXX`.

use super::object::{argument, call_method, realize_all, result};
use super::pyerrors::set_pending_error;
use super::pyobject::{self, CPyObject};
use super::unicodeobject::utf8_text;
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char, c_int};

const PY_PRINT_RAW: c_int = 1;

fn status(value: Result<(), crate::PyError>) -> c_int {
    match value {
        Ok(()) => 0,
        Err(error) => {
            set_pending_error(error);
            -1
        }
    }
}

fn c_text(pointer: *const c_char) -> Result<Option<PyObjectRef>, crate::PyError> {
    if pointer.is_null() {
        return Ok(None);
    }
    let bytes = unsafe { CStr::from_ptr(pointer) }.to_bytes();
    Ok(Some(pyre_object::w_str_new(utf8_text(bytes)?)))
}

fn text_or_none(pointer: *const c_char) -> Result<PyObjectRef, crate::PyError> {
    Ok(c_text(pointer)?.unwrap_or_else(pyre_object::w_none))
}

/// `pyfile.py PyFile_GetLine`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFile_GetLine(file: *mut CPyObject, n: c_int) -> *mut CPyObject {
    let Some(file) = argument(file) else {
        return std::ptr::null_mut();
    };
    let read = if n > 0 {
        call_method(file, "readline", &[pyre_object::w_int_new(n as i64)])
    } else {
        call_method(file, "readline", &[])
    };
    result(read.and_then(|line| finish_line(line, n)))
}

fn finish_line(line: PyObjectRef, n: c_int) -> Result<PyObjectRef, crate::PyError> {
    let bytes = unsafe { pyre_object::is_bytes(line) };
    let text = unsafe { pyre_object::is_str(line) };
    if !bytes && !text {
        return Err(crate::PyError::type_error(
            "object.readline() returned non-string",
        ));
    }
    if n >= 0 {
        return Ok(line);
    }
    let length = if text {
        unsafe { pyre_object::w_str_len(line) }
    } else {
        unsafe { pyre_object::w_bytes_len(line) }
    };
    if length == 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::EOFError,
            "EOF when reading a line",
        ));
    }
    let last_is_newline = if text {
        unsafe { pyre_object::w_str_codepoint_at(line, length - 1) }
            .is_some_and(|point| point.to_u32() == u32::from(b'\n'))
    } else {
        unsafe { pyre_object::w_bytes_getitem(line, length - 1) == b'\n' }
    };
    if !last_is_newline {
        return Ok(line);
    }
    if text {
        Ok(unsafe { pyre_object::w_str_slice_codepoints(line, 0, 1, (length - 1) as i64) })
    } else {
        let data = unsafe { pyre_object::w_bytes_data(line) };
        Ok(pyre_object::w_bytes_from_bytes(&data[..length - 1]))
    }
}

/// `pyfile.py PyFile_FromFd` — `_io.open` on an already-open descriptor.
/// `name` is ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFile_FromFd(
    fd: c_int,
    _name: *const c_char,
    mode: *const c_char,
    buffering: c_int,
    encoding: *const c_char,
    errors: *const c_char,
    newline: *const c_char,
    closefd: c_int,
) -> *mut CPyObject {
    let opened = (|| {
        let Some(mode) = c_text(mode)? else {
            return Err(crate::PyError::value_error("mode is required"));
        };
        crate::builtins::builtin_open(&[
            pyre_object::w_int_new(fd as i64),
            mode,
            pyre_object::w_int_new(buffering as i64),
            text_or_none(encoding)?,
            text_or_none(errors)?,
            text_or_none(newline)?,
            pyre_object::w_bool_from(closefd != 0),
        ])
    })();
    result(opened)
}

/// `pyfile.py PyFile_WriteString`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFile_WriteString(text: *const c_char, file: *mut CPyObject) -> c_int {
    if argument(file).is_none() {
        return -1;
    }
    let text = unsafe { super::unicodeobject::PyUnicode_FromString(text) };
    if text.is_null() {
        return -1;
    }
    let file = unsafe { pyobject::from_ref(file) };
    let decoded = unsafe { pyobject::from_ref(text) };
    let wrote = status(call_method(file, "write", &[decoded]).map(|_| ()));
    unsafe { pyobject::decref(text) };
    wrote
}

/// `pyfile.py PyFile_WriteObject`. A NULL object is written as `<NULL>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFile_WriteObject(
    object: *mut CPyObject,
    file: *mut CPyObject,
    flags: c_int,
) -> c_int {
    realize_all([object, file]);
    let Some(file) = argument(file) else {
        return -1;
    };
    let rendered = if object.is_null() {
        Ok(pyre_object::w_str_new("<NULL>"))
    } else if let Some(object) = argument(object) {
        if flags & PY_PRINT_RAW != 0 {
            unsafe { crate::display::py_str_wtf8(object) }.map(pyre_object::w_str_from_wtf8)
        } else {
            unsafe { crate::display::py_repr_wtf8(object) }.map(pyre_object::w_str_from_wtf8)
        }
    } else {
        return -1;
    };
    status(rendered.and_then(|text| call_method(file, "write", &[text]).map(|_| ())))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyFile_GetLine as *const ());
    std::hint::black_box(PyFile_FromFd as *const ());
    std::hint::black_box(PyFile_WriteString as *const ());
    std::hint::black_box(PyFile_WriteObject as *const ());
}
