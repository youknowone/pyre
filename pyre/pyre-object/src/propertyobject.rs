//! W_PropertyObject — Python `property` descriptor.
//!
//! PyPy equivalent: pypy/module/__builtin__/descriptor.py → W_Property
//!
//! A property holds fget, fset, fdel function references.
//! Used by the descriptor protocol in getattr/setattr.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// Python property descriptor object.
///
/// Layout: `[ob_type | fget | fset | fdel]`
#[pyre_class("property", type_id = 19, static_name = "PROPERTY")]
pub struct W_PropertyObject {
    pub fget: PyObjectRef,
    pub fset: PyObjectRef,
    pub fdel: PyObjectRef,
}

/// Allocate a new property object.
///
/// PyPy: W_Property.__init__(space, w_fget, w_fset, w_fdel, w_doc)
pub fn w_property_new(fget: PyObjectRef, fset: PyObjectRef, fdel: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(fget);
    crate::gc_roots::pin_root(fset);
    crate::gc_roots::pin_root(fdel);
    W_PropertyObject::allocate(W_PropertyObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        fget,
        fset,
        fdel,
    })
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
#[pyre_class("staticmethod", type_id = 20, static_name = "STATICMETHOD")]
pub struct W_StaticMethodObject {
    pub w_function: PyObjectRef,
}

pub fn w_staticmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(func);
    W_StaticMethodObject::allocate(W_StaticMethodObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_function: func,
    })
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
#[pyre_class("classmethod", type_id = 21, static_name = "CLASSMETHOD")]
pub struct W_ClassMethodObject {
    pub w_function: PyObjectRef,
}

pub fn w_classmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(func);
    W_ClassMethodObject::allocate(W_ClassMethodObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_function: func,
    })
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

    #[test]
    fn w_property_gc_type_id_matches_descr() {
        assert_eq!(W_PROPERTY_GC_TYPE_ID, 19);
        assert_eq!(
            <W_PropertyObject as crate::lltype::GcType>::type_id(),
            W_PROPERTY_GC_TYPE_ID
        );
        assert_eq!(
            <W_PropertyObject as crate::lltype::GcType>::SIZE,
            W_PROPERTY_OBJECT_SIZE
        );
    }

    #[test]
    fn w_staticmethod_gc_type_id_matches_descr() {
        assert_eq!(W_STATICMETHOD_GC_TYPE_ID, 20);
        assert_eq!(
            <W_StaticMethodObject as crate::lltype::GcType>::type_id(),
            W_STATICMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <W_StaticMethodObject as crate::lltype::GcType>::SIZE,
            W_STATICMETHOD_OBJECT_SIZE
        );
    }

    #[test]
    fn w_classmethod_gc_type_id_matches_descr() {
        assert_eq!(W_CLASSMETHOD_GC_TYPE_ID, 21);
        assert_eq!(
            <W_ClassMethodObject as crate::lltype::GcType>::type_id(),
            W_CLASSMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <W_ClassMethodObject as crate::lltype::GcType>::SIZE,
            W_CLASSMETHOD_OBJECT_SIZE
        );
    }
}
