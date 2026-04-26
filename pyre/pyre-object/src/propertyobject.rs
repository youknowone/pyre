//! W_PropertyObject — Python `property` descriptor.
//!
//! PyPy equivalent: pypy/module/__builtin__/descriptor.py → W_Property
//!
//! A property holds fget, fset, fdel function references.
//! Used by the descriptor protocol in getattr/setattr.

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

pub static PROPERTY_TYPE: PyType = crate::pyobject::new_pytype("property");

/// Field offsets of inline `PyObjectRef` slots within `W_PropertyObject`.
pub const PROPERTY_FGET_OFFSET: usize = std::mem::offset_of!(W_PropertyObject, fget);
pub const PROPERTY_FSET_OFFSET: usize = std::mem::offset_of!(W_PropertyObject, fset);
pub const PROPERTY_FDEL_OFFSET: usize = std::mem::offset_of!(W_PropertyObject, fdel);

/// GC type id assigned to `W_PropertyObject` at JitDriver init time.
pub const W_PROPERTY_GC_TYPE_ID: u32 = 19;

/// Fixed payload size (`framework.py:811`).
pub const W_PROPERTY_OBJECT_SIZE: usize = std::mem::size_of::<W_PropertyObject>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_PROPERTY_GC_PTR_OFFSETS: [usize; 3] = [
    PROPERTY_FGET_OFFSET,
    PROPERTY_FSET_OFFSET,
    PROPERTY_FDEL_OFFSET,
];

impl crate::lltype::GcType for W_PropertyObject {
    const TYPE_ID: u32 = W_PROPERTY_GC_TYPE_ID;
    const SIZE: usize = W_PROPERTY_OBJECT_SIZE;
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

    crate::lltype::malloc_typed(W_PropertyObject {
        ob_header: PyObject {
            ob_type: &PROPERTY_TYPE as *const PyType,
            w_class: get_instantiate(&PROPERTY_TYPE),
        },
        fget,
        fset,
        fdel,
    }) as PyObjectRef
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

pub static STATICMETHOD_TYPE: PyType = crate::pyobject::new_pytype("staticmethod");

/// Field offset of `w_function` within `W_StaticMethodObject`.
pub const STATICMETHOD_W_FUNCTION_OFFSET: usize =
    std::mem::offset_of!(W_StaticMethodObject, w_function);

/// GC type id assigned to `W_StaticMethodObject` at JitDriver init time.
pub const W_STATICMETHOD_GC_TYPE_ID: u32 = 20;

/// Fixed payload size (`framework.py:811`).
pub const W_STATICMETHOD_OBJECT_SIZE: usize = std::mem::size_of::<W_StaticMethodObject>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_STATICMETHOD_GC_PTR_OFFSETS: [usize; 1] = [STATICMETHOD_W_FUNCTION_OFFSET];

impl crate::lltype::GcType for W_StaticMethodObject {
    const TYPE_ID: u32 = W_STATICMETHOD_GC_TYPE_ID;
    const SIZE: usize = W_STATICMETHOD_OBJECT_SIZE;
}

pub fn w_staticmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(func);

    crate::lltype::malloc_typed(W_StaticMethodObject {
        ob_header: PyObject {
            ob_type: &STATICMETHOD_TYPE as *const PyType,
            w_class: get_instantiate(&STATICMETHOD_TYPE),
        },
        w_function: func,
    }) as PyObjectRef
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

pub static CLASSMETHOD_TYPE: PyType = crate::pyobject::new_pytype("classmethod");

/// Field offset of `w_function` within `W_ClassMethodObject`.
pub const CLASSMETHOD_W_FUNCTION_OFFSET: usize =
    std::mem::offset_of!(W_ClassMethodObject, w_function);

/// GC type id assigned to `W_ClassMethodObject` at JitDriver init time.
pub const W_CLASSMETHOD_GC_TYPE_ID: u32 = 21;

/// Fixed payload size (`framework.py:811`).
pub const W_CLASSMETHOD_OBJECT_SIZE: usize = std::mem::size_of::<W_ClassMethodObject>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_CLASSMETHOD_GC_PTR_OFFSETS: [usize; 1] = [CLASSMETHOD_W_FUNCTION_OFFSET];

impl crate::lltype::GcType for W_ClassMethodObject {
    const TYPE_ID: u32 = W_CLASSMETHOD_GC_TYPE_ID;
    const SIZE: usize = W_CLASSMETHOD_OBJECT_SIZE;
}

pub fn w_classmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(func);

    crate::lltype::malloc_typed(W_ClassMethodObject {
        ob_header: PyObject {
            ob_type: &CLASSMETHOD_TYPE as *const PyType,
            w_class: get_instantiate(&CLASSMETHOD_TYPE),
        },
        w_function: func,
    }) as PyObjectRef
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
            <W_PropertyObject as crate::lltype::GcType>::TYPE_ID,
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
            <W_StaticMethodObject as crate::lltype::GcType>::TYPE_ID,
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
            <W_ClassMethodObject as crate::lltype::GcType>::TYPE_ID,
            W_CLASSMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <W_ClassMethodObject as crate::lltype::GcType>::SIZE,
            W_CLASSMETHOD_OBJECT_SIZE
        );
    }
}
