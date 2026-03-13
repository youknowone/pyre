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
    },
};

/// Get the None singleton as a PyObjectRef.
pub fn w_none() -> PyObjectRef {
    &NONE_SINGLETON as *const W_NoneObject as *mut PyObject
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
