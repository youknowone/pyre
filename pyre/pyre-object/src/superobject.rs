//! W_Super — Python `super` proxy object.
//!
//! PyPy equivalent: pypy/objspace/descroperation.py + superobject.py
//!
//! Stores (super_type, obj) and resolves attribute lookups
//! starting from the next class after super_type in obj's MRO.

use crate::pyobject::*;

pub static SUPER_TYPE: PyType = crate::pyobject::new_pytype("super");

/// super proxy: [ob_type | super_type (cls) | obj (self)]
#[repr(C)]
pub struct W_SuperObject {
    pub ob: PyObject,
    /// The class passed to super() — lookup starts after this in MRO.
    pub super_type: PyObjectRef,
    /// The instance (self) or class for classmethod.
    pub obj: PyObjectRef,
}

/// Create a new super proxy.
pub fn w_super_new(super_type: PyObjectRef, obj: PyObjectRef) -> PyObjectRef {
    let s = Box::new(W_SuperObject {
        ob: PyObject {
            ob_type: &SUPER_TYPE as *const PyType,
            w_class: get_instantiate(&SUPER_TYPE),
        },
        super_type,
        obj,
    });
    Box::into_raw(s) as PyObjectRef
}

#[inline]
pub unsafe fn is_super(obj: PyObjectRef) -> bool {
    py_type_check(obj, &SUPER_TYPE)
}

/// Get the super_type (cls) from a super proxy.
#[inline]
pub unsafe fn w_super_get_type(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SuperObject)).super_type
}

/// Get the bound object (self) from a super proxy.
#[inline]
pub unsafe fn w_super_get_obj(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SuperObject)).obj
}
