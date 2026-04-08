//! W_NoneObject — Python `None` singleton.

use crate::pyobject::*;

/// Python None object (singleton).
#[repr(C)]
pub struct W_NoneObject {
    pub ob_header: PyObject,
}

/// Global None singleton.
static NONE_SINGLETON: W_NoneObject = W_NoneObject {
    ob_header: PyObject {
        ob_type: &NONE_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
};

/// Get the None singleton as a PyObjectRef.
pub fn w_none() -> PyObjectRef {
    &NONE_SINGLETON as *const W_NoneObject as *mut PyObject
}

/// Python NotImplemented singleton.
/// PyPy: space.w_NotImplemented
#[repr(C)]
pub struct W_NotImplementedObject {
    pub ob_header: PyObject,
}

static NOT_IMPLEMENTED_SINGLETON: W_NotImplementedObject = W_NotImplementedObject {
    ob_header: PyObject {
        ob_type: &NOTIMPLEMENTED_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
};

/// Get the NotImplemented singleton.
pub fn w_not_implemented() -> PyObjectRef {
    &NOT_IMPLEMENTED_SINGLETON as *const W_NotImplementedObject as *mut PyObject
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_none_is_singleton() {
        let a = w_none();
        let b = w_none();
        assert_eq!(a, b);
        unsafe {
            assert!(is_none(a));
            assert!(!is_int(a));
        }
    }
}
