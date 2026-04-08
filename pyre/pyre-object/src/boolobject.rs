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

/// Field offset of `boolval` within `W_BoolObject`, for JIT field access.
pub const BOOL_BOOLVAL_OFFSET: usize = std::mem::offset_of!(W_BoolObject, boolval);

/// Allocate a new W_BoolObject.
pub fn w_bool_new(value: bool) -> PyObjectRef {
    let obj = Box::new(W_BoolObject {
        ob_header: PyObject {
            ob_type: &BOOL_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
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

// ── Bool singletons ──────────────────────────────────────────────────

static TRUE_SINGLETON: W_BoolObject = W_BoolObject {
    ob_header: PyObject {
        ob_type: &BOOL_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
    boolval: true,
};

static FALSE_SINGLETON: W_BoolObject = W_BoolObject {
    ob_header: PyObject {
        ob_type: &BOOL_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
    boolval: false,
};

/// Get a boolean PyObjectRef from a bool value.
///
/// Returns a pointer to a pre-allocated static singleton,
/// avoiding heap allocation on every comparison/branch.
#[inline]
pub fn w_bool_from(value: bool) -> PyObjectRef {
    if value {
        (&TRUE_SINGLETON as *const W_BoolObject).cast_mut() as PyObjectRef
    } else {
        (&FALSE_SINGLETON as *const W_BoolObject).cast_mut() as PyObjectRef
    }
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
