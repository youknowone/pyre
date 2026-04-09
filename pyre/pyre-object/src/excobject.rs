//! W_ExceptionObject — Python exception instance.
//!
//! Each exception carries a `kind` tag (mapping to PyErrorKind) and
//! a message string. The `ob_type` pointer is `EXCEPTION_TYPE` for all
//! exception instances; the `kind` field distinguishes the actual type.

use crate::pyobject::*;

pub static EXCEPTION_TYPE: PyType = crate::pyobject::new_pytype("exception");

/// Numeric tags for exception kinds — must stay in sync with PyErrorKind.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcKind {
    BaseException = 0,
    Exception = 1,
    TypeError = 2,
    ValueError = 3,
    ZeroDivisionError = 4,
    NameError = 5,
    IndexError = 6,
    KeyError = 7,
    AttributeError = 8,
    RuntimeError = 9,
    StopIteration = 10,
    OverflowError = 11,
    ArithmeticError = 12,
    ImportError = 13,
    NotImplementedError = 14,
    AssertionError = 15,
    /// Raised by `_weakref` when a proxy is dereferenced after the
    /// referent has been collected — pypy/module/_weakref/interp__weakref.py:347
    /// `oefmt(space.w_ReferenceError, "weakly referenced object no longer exists")`.
    ReferenceError = 16,
}

/// Layout: `[ob_type: *const PyType | kind: ExcKind | message: *mut String]`
#[repr(C)]
pub struct W_ExceptionObject {
    pub ob_header: PyObject,
    pub kind: ExcKind,
    pub message: *mut String,
}

pub const EXC_KIND_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, kind);
pub const EXC_MESSAGE_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, message);

/// Allocate a new exception object on the heap.
pub fn w_exception_new(kind: ExcKind, message: &str) -> PyObjectRef {
    let msg = Box::into_raw(Box::new(message.to_string()));
    let obj = Box::new(W_ExceptionObject {
        ob_header: PyObject {
            ob_type: &EXCEPTION_TYPE as *const PyType,
            w_class: get_instantiate(&EXCEPTION_TYPE),
        },
        kind,
        message: msg,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Check if an object is an exception instance.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_exception(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &EXCEPTION_TYPE) }
}

/// Get the exception kind tag.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_kind(obj: PyObjectRef) -> ExcKind {
    unsafe { (*(obj as *const W_ExceptionObject)).kind }
}

/// Get the exception message.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_message(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const W_ExceptionObject)).message }
}

/// Get the Python type name string for an ExcKind.
pub fn exc_kind_name(kind: ExcKind) -> &'static str {
    match kind {
        ExcKind::BaseException => "BaseException",
        ExcKind::Exception => "Exception",
        ExcKind::TypeError => "TypeError",
        ExcKind::ValueError => "ValueError",
        ExcKind::ZeroDivisionError => "ZeroDivisionError",
        ExcKind::NameError => "NameError",
        ExcKind::IndexError => "IndexError",
        ExcKind::KeyError => "KeyError",
        ExcKind::AttributeError => "AttributeError",
        ExcKind::RuntimeError => "RuntimeError",
        ExcKind::StopIteration => "StopIteration",
        ExcKind::OverflowError => "OverflowError",
        ExcKind::ArithmeticError => "ArithmeticError",
        ExcKind::ImportError => "ImportError",
        ExcKind::NotImplementedError => "NotImplementedError",
        ExcKind::AssertionError => "AssertionError",
        ExcKind::ReferenceError => "ReferenceError",
    }
}

/// Check if `exc_kind` matches `type_name`, considering Python's
/// exception hierarchy (e.g. ZeroDivisionError is-a ArithmeticError
/// is-a Exception is-a BaseException).
pub fn exc_kind_matches(kind: ExcKind, type_name: &str) -> bool {
    if type_name == "BaseException" {
        return true;
    }
    if type_name == "Exception" {
        return kind != ExcKind::BaseException;
    }
    if type_name == "ArithmeticError" {
        return matches!(
            kind,
            ExcKind::ArithmeticError | ExcKind::ZeroDivisionError | ExcKind::OverflowError
        );
    }
    exc_kind_name(kind) == type_name
}

/// Convert a Python exception type name to an ExcKind.
pub fn exc_kind_from_name(name: &str) -> Option<ExcKind> {
    match name {
        "BaseException" => Some(ExcKind::BaseException),
        "Exception" => Some(ExcKind::Exception),
        "TypeError" => Some(ExcKind::TypeError),
        "ValueError" => Some(ExcKind::ValueError),
        "ZeroDivisionError" => Some(ExcKind::ZeroDivisionError),
        "NameError" => Some(ExcKind::NameError),
        "IndexError" => Some(ExcKind::IndexError),
        "KeyError" => Some(ExcKind::KeyError),
        "AttributeError" => Some(ExcKind::AttributeError),
        "RuntimeError" => Some(ExcKind::RuntimeError),
        "StopIteration" => Some(ExcKind::StopIteration),
        "OverflowError" => Some(ExcKind::OverflowError),
        "ArithmeticError" => Some(ExcKind::ArithmeticError),
        "ImportError" => Some(ExcKind::ImportError),
        "NotImplementedError" => Some(ExcKind::NotImplementedError),
        "AssertionError" => Some(ExcKind::AssertionError),
        "ReferenceError" => Some(ExcKind::ReferenceError),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exception_create_and_read() {
        let obj = w_exception_new(ExcKind::ValueError, "bad value");
        unsafe {
            assert!(is_exception(obj));
            assert_eq!(w_exception_get_kind(obj), ExcKind::ValueError);
            assert_eq!(w_exception_get_message(obj), "bad value");
        }
    }

    #[test]
    fn test_exc_kind_matches_hierarchy() {
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "ZeroDivisionError"
        ));
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "ArithmeticError"
        ));
        assert!(exc_kind_matches(ExcKind::ZeroDivisionError, "Exception"));
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "BaseException"
        ));
        assert!(!exc_kind_matches(ExcKind::ZeroDivisionError, "ValueError"));
    }

    #[test]
    fn test_exc_kind_from_name_roundtrip() {
        for kind in [
            ExcKind::TypeError,
            ExcKind::ValueError,
            ExcKind::ZeroDivisionError,
        ] {
            let name = exc_kind_name(kind);
            assert_eq!(exc_kind_from_name(name), Some(kind));
        }
    }
}
