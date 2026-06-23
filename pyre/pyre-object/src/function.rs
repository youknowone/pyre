//! `pypy/interpreter/function.py` method descriptor ports.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

// ── StaticMethod ─────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py StaticMethod
//
// __get__ returns the wrapped function unchanged (no self binding).

/// Python staticmethod descriptor.
#[pyre_class("staticmethod", type_id = 20, static_name = "STATICMETHOD")]
pub struct StaticMethod {
    pub w_function: PyObjectRef,
}

pub fn w_staticmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // wrapped function across the GC malloc and read its relocated address.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(func);

    let header = PyObject {
        ob_type: &STATICMETHOD_TYPE as *const PyType,
        w_class: get_instantiate(&STATICMETHOD_TYPE),
    };
    let raw =
        crate::gc_hook::try_gc_alloc_stable(W_STATICMETHOD_GC_TYPE_ID, W_STATICMETHOD_OBJECT_SIZE)
            .filter(|p| !p.is_null());
    let func = crate::gc_roots::shadow_stack_get(save_point);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut StaticMethod,
                StaticMethod {
                    ob: header,
                    w_function: func,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    StaticMethod::allocate(StaticMethod {
        ob: header,
        w_function: func,
    })
}

pub unsafe fn w_staticmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const StaticMethod)).w_function
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
pub struct ClassMethod {
    pub w_function: PyObjectRef,
}

pub fn w_classmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // wrapped function across the GC malloc and read its relocated address.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(func);

    let header = PyObject {
        ob_type: &CLASSMETHOD_TYPE as *const PyType,
        w_class: get_instantiate(&CLASSMETHOD_TYPE),
    };
    let raw =
        crate::gc_hook::try_gc_alloc_stable(W_CLASSMETHOD_GC_TYPE_ID, W_CLASSMETHOD_OBJECT_SIZE)
            .filter(|p| !p.is_null());
    let func = crate::gc_roots::shadow_stack_get(save_point);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut ClassMethod,
                ClassMethod {
                    ob: header,
                    w_function: func,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    ClassMethod::allocate(ClassMethod {
        ob: header,
        w_function: func,
    })
}

pub unsafe fn w_classmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const ClassMethod)).w_function
}

#[inline]
pub unsafe fn is_classmethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &CLASSMETHOD_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_staticmethod_gc_type_id_matches_descr() {
        assert_eq!(W_STATICMETHOD_GC_TYPE_ID, 20);
        assert_eq!(
            <StaticMethod as crate::lltype::GcType>::type_id(),
            W_STATICMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <StaticMethod as crate::lltype::GcType>::SIZE,
            W_STATICMETHOD_OBJECT_SIZE
        );
    }

    #[test]
    fn w_classmethod_gc_type_id_matches_descr() {
        assert_eq!(W_CLASSMETHOD_GC_TYPE_ID, 21);
        assert_eq!(
            <ClassMethod as crate::lltype::GcType>::type_id(),
            W_CLASSMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <ClassMethod as crate::lltype::GcType>::SIZE,
            W_CLASSMETHOD_OBJECT_SIZE
        );
    }
}
