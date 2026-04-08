//! W_LongObject -- arbitrary-precision integer backed by `BigInt`.
//!
//! Used when i64 overflow is detected in `W_IntObject` arithmetic.
//! The JIT never inlines bigint operations; `GuardClass(INT_TYPE)` rejects
//! `W_LongObject` and deoptimizes back to the interpreter.

use malachite_bigint::BigInt;

use crate::pyobject::*;

/// Arbitrary-precision integer object.
///
/// Layout: `[ob_type: *const PyType | value: *mut BigInt]`
/// The `value` pointer owns a heap-allocated `BigInt` (via `Box::into_raw`).
#[repr(C)]
pub struct W_LongObject {
    pub ob_header: PyObject,
    pub value: *mut BigInt,
}

// Safety: BigInt is Send+Sync and W_LongObject only stores a raw pointer
// that is effectively owned.
unsafe impl Send for W_LongObject {}
unsafe impl Sync for W_LongObject {}

/// Field offset of `value` within `W_LongObject`, for potential JIT field access.
pub const LONG_VALUE_OFFSET: usize = std::mem::offset_of!(W_LongObject, value);

/// Allocate a new W_LongObject on the heap.
///
/// Phase 1: uses `Box::leak` (objects are never freed).
pub fn w_long_new(value: BigInt) -> PyObjectRef {
    let inner = Box::into_raw(Box::new(value));
    let obj = Box::new(W_LongObject {
        ob_header: PyObject {
            ob_type: &LONG_TYPE as *const PyType,
            w_class: get_instantiate(&LONG_TYPE),
        },
        value: inner,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Create a W_LongObject from an i64 value.
pub fn w_long_from_i64(v: i64) -> PyObjectRef {
    w_long_new(BigInt::from(v))
}

/// Box a bigint constant into a heap Python int object.
pub fn box_bigint_constant(value: &BigInt) -> PyObjectRef {
    w_long_new(value.clone())
}

/// Extract a reference to the BigInt value from a known W_LongObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_LongObject`.
#[inline]
pub unsafe fn w_long_get_value(obj: PyObjectRef) -> &'static BigInt {
    unsafe {
        let long_obj = obj as *const W_LongObject;
        &*(*long_obj).value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_create_and_read() {
        let obj = w_long_new(BigInt::from(42));
        unsafe {
            assert!(is_long(obj));
            assert!(!is_int(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(42));
        }
    }

    #[test]
    fn test_long_from_i64() {
        let obj = w_long_from_i64(i64::MAX);
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(i64::MAX));
        }
    }

    #[test]
    fn test_long_large_value() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big.clone());
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), big);
        }
    }

    #[test]
    fn test_long_field_offset() {
        assert_eq!(LONG_VALUE_OFFSET, 16);
    }

    #[test]
    fn test_long_type_name_is_int() {
        // Python users see "int" for both W_IntObject and W_LongObject
        assert_eq!(LONG_TYPE.name, "int");
    }
}
