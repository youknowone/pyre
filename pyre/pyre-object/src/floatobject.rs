//! W_FloatObject — Python `float` type backed by f64.

use crate::pyobject::*;

/// Python float object.
///
/// Layout: `[ob_type: *const PyType | floatval: f64]`
/// The JIT reads `floatval` via `GetfieldGcF` at offset 8 (after the type pointer).
#[repr(C)]
pub struct W_FloatObject {
    pub ob_header: PyObject,
    pub floatval: f64,
}

/// Field offset of `floatval` within `W_FloatObject`, for JIT field access.
pub const FLOAT_FLOATVAL_OFFSET: usize = std::mem::offset_of!(W_FloatObject, floatval);

/// Allocate a new W_FloatObject on the heap.
///
/// Phase 1: uses `Box::leak` for simplicity (objects are never freed).
/// A proper GC will replace this allocation strategy.
pub fn w_float_new(value: f64) -> PyObjectRef {
    let obj = Box::new(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        floatval: value,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Box a float constant into a heap Python float object.
pub fn box_float_constant(value: f64) -> PyObjectRef {
    w_float_new(value)
}

/// Extract the f64 value from a known W_FloatObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_FloatObject`.
#[inline]
pub unsafe fn w_float_get_value(obj: PyObjectRef) -> f64 {
    unsafe { (*(obj as *const W_FloatObject)).floatval }
}

pub extern "C" fn jit_w_float_new(value_bits: i64) -> i64 {
    let value = f64::from_bits(value_bits as u64);
    w_float_new(value) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_create_and_read() {
        let obj = w_float_new(3.14);
        unsafe {
            assert!(is_float(obj));
            assert!(!is_int(obj));
            assert_eq!(w_float_get_value(obj), 3.14);
            drop(Box::from_raw(obj as *mut W_FloatObject));
        }
    }

    #[test]
    fn test_float_negative() {
        let obj = w_float_new(-2.5);
        unsafe {
            assert_eq!(w_float_get_value(obj), -2.5);
            drop(Box::from_raw(obj as *mut W_FloatObject));
        }
    }

    #[test]
    fn test_box_float_constant_reads_back() {
        let obj = box_float_constant(6.25);
        unsafe {
            assert_eq!(w_float_get_value(obj), 6.25);
            drop(Box::from_raw(obj as *mut W_FloatObject));
        }
    }

    #[test]
    fn test_float_field_offset() {
        assert_eq!(FLOAT_FLOATVAL_OFFSET, 16); // after *const PyType (8 bytes on 64-bit)
    }
}
