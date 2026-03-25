//! W_InstanceObject — instance of a user-defined class.
//!
//! PyPy equivalent: pypy/objspace/std/objectobject.py → W_ObjectObject
//!
//! An instance holds a pointer to its W_TypeObject (class).
//! Per-instance attributes are stored in the thread-local ATTR_TABLE
//! side table (pyre-objspace), matching PyPy's instance __dict__.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python instance object.
///
/// Layout: `[ob_type | w_type]`
///
/// - `ob_type`: always &INSTANCE_TYPE (for is_instance() checks)
/// - `w_type`: pointer to the W_TypeObject this is an instance of
///
/// PyPy stores the type in ob_type directly, but pyre uses a fixed
/// INSTANCE_TYPE for dispatch and stores the actual class separately.
#[repr(C)]
pub struct W_InstanceObject {
    pub ob_header: PyObject,
    /// The class (W_TypeObject) this object is an instance of.
    pub w_type: PyObjectRef,
}

/// Allocate a new instance of a user-defined class.
///
/// PyPy equivalent: object.__new__(space, w_type) → allocate_instance
pub fn w_instance_new(w_type: PyObjectRef) -> PyObjectRef {
    let obj = Box::new(W_InstanceObject {
        ob_header: PyObject {
            ob_type: &INSTANCE_TYPE as *const PyType,
        },
        w_type,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Get the class (W_TypeObject) of an instance.
pub unsafe fn w_instance_get_type(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_InstanceObject)).w_type
}

/// Check if an object is an instance of a user-defined class.
#[inline]
pub unsafe fn is_instance(obj: PyObjectRef) -> bool {
    py_type_check(obj, &INSTANCE_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_create_and_check() {
        // Use a sentinel as the "type"
        let fake_type = PY_NULL;
        let obj = w_instance_new(fake_type);
        unsafe {
            assert!(is_instance(obj));
            assert!(!is_int(obj));
            assert!(!is_type(obj));
            assert_eq!(w_instance_get_type(obj), fake_type);
        }
    }
}
