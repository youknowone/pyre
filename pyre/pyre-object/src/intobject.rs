//! W_IntObject — Python `int` type backed by i64.
//!
//! Phase 1 uses a fixed i64 representation. BigInt support (like PyPy's
//! `W_LongObject`) will be added in Phase 4.

use crate::pyobject::*;

/// Python integer object.
///
/// Layout: `[ob_type: *const PyType | intval: i64]`
/// The JIT reads `intval` via `GetfieldGcI` at offset 8 (after the type pointer).
#[repr(C)]
pub struct W_IntObject {
    pub ob_header: PyObject,
    pub intval: i64,
}

/// Field offset of `intval` within `W_IntObject`, for JIT field access.
pub const INT_INTVAL_OFFSET: usize = std::mem::offset_of!(W_IntObject, intval);

/// Allocate a new W_IntObject on the heap.
///
/// Phase 1: uses `Box::leak` for simplicity (objects are never freed).
/// A proper GC will replace this allocation strategy.
pub fn w_int_new(value: i64) -> PyObjectRef {
    let obj = Box::new(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
        },
        intval: value,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Extract the i64 value from a known W_IntObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_IntObject`.
#[inline]
pub unsafe fn w_int_get_value(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_IntObject)).intval }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_create_and_read() {
        let obj = w_int_new(42);
        unsafe {
            assert!(is_int(obj));
            assert!(!is_bool(obj));
            assert_eq!(w_int_get_value(obj), 42);
            // Clean up
            drop(Box::from_raw(obj as *mut W_IntObject));
        }
    }

    #[test]
    fn test_int_negative() {
        let obj = w_int_new(-7);
        unsafe {
            assert_eq!(w_int_get_value(obj), -7);
            drop(Box::from_raw(obj as *mut W_IntObject));
        }
    }

    #[test]
    fn test_int_field_offset() {
        assert_eq!(INT_INTVAL_OFFSET, 8); // after *const PyType (8 bytes on 64-bit)
    }
}
