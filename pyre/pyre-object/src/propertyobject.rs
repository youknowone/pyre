//! W_PropertyObject — Python `property` descriptor.
//!
//! PyPy equivalent: pypy/module/__builtin__/descriptor.py → W_Property
//!
//! A property holds fget, fset, fdel function references.
//! Used by the descriptor protocol in py_getattr/py_setattr.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python property descriptor object.
///
/// Layout: `[ob_type | fget | fset | fdel]`
#[repr(C)]
pub struct W_PropertyObject {
    pub ob_header: PyObject,
    pub fget: PyObjectRef,
    pub fset: PyObjectRef,
    pub fdel: PyObjectRef,
}

pub static PROPERTY_TYPE: PyType = PyType {
    tp_name: "property",
};

/// Allocate a new property object.
///
/// PyPy: W_Property.__init__(space, w_fget, w_fset, w_fdel, w_doc)
pub fn w_property_new(fget: PyObjectRef, fset: PyObjectRef, fdel: PyObjectRef) -> PyObjectRef {
    let obj = Box::new(W_PropertyObject {
        ob_header: PyObject {
            ob_type: &PROPERTY_TYPE as *const PyType,
        },
        fget,
        fset,
        fdel,
    });
    Box::into_raw(obj) as PyObjectRef
}

pub unsafe fn w_property_get_fget(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_PropertyObject)).fget
}

pub unsafe fn w_property_get_fset(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_PropertyObject)).fset
}

pub unsafe fn w_property_get_fdel(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_PropertyObject)).fdel
}

#[inline]
pub unsafe fn is_property(obj: PyObjectRef) -> bool {
    py_type_check(obj, &PROPERTY_TYPE)
}

// ── StaticMethod ─────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py StaticMethod
//
// __get__ returns the wrapped function unchanged (no self binding).

/// Python staticmethod descriptor.
#[repr(C)]
pub struct W_StaticMethodObject {
    pub ob_header: PyObject,
    pub w_function: PyObjectRef,
}

pub static STATICMETHOD_TYPE: PyType = PyType {
    tp_name: "staticmethod",
};

pub fn w_staticmethod_new(func: PyObjectRef) -> PyObjectRef {
    let obj = Box::new(W_StaticMethodObject {
        ob_header: PyObject {
            ob_type: &STATICMETHOD_TYPE as *const PyType,
        },
        w_function: func,
    });
    Box::into_raw(obj) as PyObjectRef
}

pub unsafe fn w_staticmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_StaticMethodObject)).w_function
}

#[inline]
pub unsafe fn is_staticmethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &STATICMETHOD_TYPE)
}

// ── ClassMethod ──────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py ClassMethod
//
// __get__ returns a bound method with the class as first arg.

/// Python classmethod descriptor.
#[repr(C)]
pub struct W_ClassMethodObject {
    pub ob_header: PyObject,
    pub w_function: PyObjectRef,
}

pub static CLASSMETHOD_TYPE: PyType = PyType {
    tp_name: "classmethod",
};

pub fn w_classmethod_new(func: PyObjectRef) -> PyObjectRef {
    let obj = Box::new(W_ClassMethodObject {
        ob_header: PyObject {
            ob_type: &CLASSMETHOD_TYPE as *const PyType,
        },
        w_function: func,
    });
    Box::into_raw(obj) as PyObjectRef
}

pub unsafe fn w_classmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_ClassMethodObject)).w_function
}

#[inline]
pub unsafe fn is_classmethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &CLASSMETHOD_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_create() {
        let obj = w_property_new(PY_NULL, PY_NULL, PY_NULL);
        unsafe {
            assert!(is_property(obj));
            assert!(!is_int(obj));
        }
    }
}
