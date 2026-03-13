//! W_BoolObject — Python `bool` type.
//!
//! Booleans are a subtype of int in Python. In pyre, they use a separate
//! struct for JIT type specialization via `GuardClass`.

use crate::pyobject::*;

/// Python boolean object.
#[repr(C)]
pub struct W_BoolObject {
    pub ob_header: PyObject,
    pub boolval: bool,
}

/// Allocate a new W_BoolObject.
pub fn w_bool_new(value: bool) -> PyObjectRef {
    let obj = Box::new(W_BoolObject {
        ob_header: PyObject {
            ob_type: &BOOL_TYPE as *const PyType,
        },
        boolval: value,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Extract the bool value from a known W_BoolObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_BoolObject`.
#[inline]
pub unsafe fn w_bool_get_value(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_BoolObject)).boolval }
}

/// Get a boolean PyObjectRef from a bool value.
///
/// Phase 1: allocates new objects each time. In production, True/False
/// would be pre-allocated singletons.
pub fn w_bool_from(value: bool) -> PyObjectRef {
    w_bool_new(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_true() {
        let obj = w_bool_new(true);
        unsafe {
            assert!(is_bool(obj));
            assert!(!is_int(obj));
            assert!(w_bool_get_value(obj));
            drop(Box::from_raw(obj as *mut W_BoolObject));
        }
    }

    #[test]
    fn test_bool_false() {
        let obj = w_bool_new(false);
        unsafe {
            assert!(!w_bool_get_value(obj));
            drop(Box::from_raw(obj as *mut W_BoolObject));
        }
    }
}
