//! W_StrObject -- Python `str` type backed by a heap-allocated String.
//!
//! Strings are opaque to the JIT: all operations go through residual
//! `extern "C"` helper calls rather than inline IR arithmetic.

use crate::pyobject::*;

/// Python string object.
///
/// Layout: `[ob_type: *const PyType | value: *mut String]`
/// The `value` pointer owns a heap-allocated `String` (via `Box::into_raw`).
#[repr(C)]
pub struct W_StrObject {
    pub ob_header: PyObject,
    pub value: *mut String,
}

/// Field offset of `value` within `W_StrObject`, for JIT field access.
pub const STR_VALUE_OFFSET: usize = std::mem::offset_of!(W_StrObject, value);

/// Allocate a new W_StrObject on the heap.
///
/// Phase 1: uses `Box::leak` for simplicity (objects are never freed).
/// The inner `String` is also `Box::into_raw`'d so it can be recovered.
pub fn w_str_new(s: &str) -> PyObjectRef {
    let inner = Box::into_raw(Box::new(s.to_string()));
    let obj = Box::new(W_StrObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
        },
        value: inner,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Extract the &str value from a known W_StrObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_StrObject`.
#[inline]
pub unsafe fn w_str_get_value(obj: PyObjectRef) -> &'static str {
    unsafe {
        let str_obj = obj as *const W_StrObject;
        &*(*str_obj).value
    }
}

/// Check if an object is a str.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_str(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &STR_TYPE) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_str_create_and_read() {
        let obj = w_str_new("hello");
        unsafe {
            assert!(is_str(obj));
            assert!(!is_int(obj));
            assert_eq!(w_str_get_value(obj), "hello");
        }
    }

    #[test]
    fn test_str_empty() {
        let obj = w_str_new("");
        unsafe {
            assert!(is_str(obj));
            assert_eq!(w_str_get_value(obj), "");
        }
    }

    #[test]
    fn test_str_field_offset() {
        assert_eq!(STR_VALUE_OFFSET, 8); // after *const PyType (8 bytes on 64-bit)
    }
}
