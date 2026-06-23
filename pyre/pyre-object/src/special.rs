//! `pypy/interpreter/special.py` singleton payloads.

use crate::pyobject::*;

/// Python NotImplemented singleton.
/// PyPy: pypy/interpreter/special.py NotImplemented
#[repr(C)]
pub struct NotImplemented {
    pub ob_header: PyObject,
}

static NOT_IMPLEMENTED_SINGLETON: NotImplemented = NotImplemented {
    ob_header: PyObject {
        ob_type: &NOTIMPLEMENTED_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
};

/// Get the NotImplemented singleton.
pub fn w_not_implemented() -> PyObjectRef {
    &NOT_IMPLEMENTED_SINGLETON as *const NotImplemented as *mut PyObject
}

/// Python Ellipsis singleton (`...`).
/// PyPy: pypy/interpreter/special.py Ellipsis
#[repr(C)]
pub struct Ellipsis {
    pub ob_header: PyObject,
}

static ELLIPSIS_SINGLETON: Ellipsis = Ellipsis {
    ob_header: PyObject {
        ob_type: &ELLIPSIS_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
};

/// Get the Ellipsis singleton.
pub fn w_ellipsis() -> PyObjectRef {
    &ELLIPSIS_SINGLETON as *const Ellipsis as *mut PyObject
}
