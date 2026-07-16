//! `pypy/interpreter/function.py` method descriptor ports.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

// ── Method ───────────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py Method

/// Python bound method wrapper.
#[pyre_class("method", type_id = 16, static_name = "METHOD")]
pub struct Method {
    pub w_function: PyObjectRef,
    pub w_self: PyObjectRef,
    pub w_class: PyObjectRef,
}

/// Field offsets of inline `PyObjectRef` slots within `Method`.
/// Consumed by `pyre-jit-trace/src/descr.rs` to emit field-access IR;
/// the macro's own `W_METHOD_GC_PTR_OFFSETS` aggregate is independent
/// and does not depend on these per-field consts.
pub const METHOD_W_FUNCTION_OFFSET: usize = std::mem::offset_of!(Method, w_function);
pub const METHOD_W_SELF_OFFSET: usize = std::mem::offset_of!(Method, w_self);
pub const METHOD_W_CLASS_OFFSET: usize = std::mem::offset_of!(Method, w_class);

pub fn w_method_new(
    w_function: PyObjectRef,
    w_self: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // three members across the GC malloc and re-read their relocated
    // addresses afterwards (a minor collection inside the malloc may move
    // them). A bound method whose `w_function`/`w_self`/`w_class` is
    // reachable only through it must be GC-traced; a `malloc_typed` method
    // is invisible to mark-sweep, whereas `register_pyre_class` registers
    // this layout's `ptr_offsets`, so mark-sweep follows the members. The
    // write barrier below keeps the old-gen method in the remembered set so
    // young members survive a later minor collection.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(w_function);
    crate::gc_roots::pin_root(w_self);
    crate::gc_roots::pin_root(w_class);
    let header = PyObject {
        ob_type: &METHOD_TYPE as *const PyType,
        w_class: get_instantiate(&METHOD_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_METHOD_GC_TYPE_ID, W_METHOD_OBJECT_SIZE);
    // Re-read the pinned roots after the allocation; a minor collection
    // inside the GC malloc may have relocated them.
    let w_function = crate::gc_roots::shadow_stack_get(save_point);
    let w_self = crate::gc_roots::shadow_stack_get(save_point + 1);
    let w_class = crate::gc_roots::shadow_stack_get(save_point + 2);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut Method,
                Method {
                    ob: header,
                    w_function,
                    w_self,
                    w_class,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    Method::allocate(Method {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_function,
        w_self,
        w_class,
    })
}

#[inline]
pub unsafe fn is_method(obj: PyObjectRef) -> bool {
    py_type_check(obj, &METHOD_TYPE)
}

#[inline]
pub unsafe fn w_method_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const Method)).w_function
}

#[inline]
pub unsafe fn w_method_get_self(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const Method)).w_self
}

#[inline]
pub unsafe fn w_method_get_class(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const Method)).w_class
}

// ── StaticMethod ─────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py StaticMethod
//
// __get__ returns the wrapped function unchanged (no self binding).

/// Python staticmethod descriptor.
#[pyre_class("staticmethod", type_id = 20, static_name = "STATICMETHOD")]
pub struct StaticMethod {
    pub w_function: PyObjectRef,
    /// function.py:676 `self.w_dict = None` — lazily allocated by
    /// `StaticMethod.getdict`, then populated by `descr_init` with the
    /// wrapped function's presentation attributes.
    pub w_dict: PyObjectRef,
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
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(
        W_STATICMETHOD_GC_TYPE_ID,
        W_STATICMETHOD_OBJECT_SIZE,
    );
    let func = crate::gc_roots::shadow_stack_get(save_point);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut StaticMethod,
                StaticMethod {
                    ob: header,
                    w_function: func,
                    w_dict: PY_NULL,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    StaticMethod::allocate(StaticMethod {
        ob: header,
        w_function: func,
        w_dict: PY_NULL,
    })
}

pub unsafe fn w_staticmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const StaticMethod)).w_function
}

/// function.py:697 `self.w_function = w_function`.
#[inline]
pub unsafe fn w_staticmethod_set_func(obj: PyObjectRef, func: PyObjectRef) {
    unsafe {
        (*(obj as *mut StaticMethod)).w_function = func;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

/// function.py:678-681 `StaticMethod.getdict` — allocate the instance
/// dictionary on first access and retain its identity.
#[inline]
pub unsafe fn w_staticmethod_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let sm = obj as *mut StaticMethod;
        if (*sm).w_dict.is_null() {
            (*sm).w_dict = crate::w_dict_new();
            crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
        }
        (*sm).w_dict
    }
}

/// function.py:683-688 `StaticMethod.setdict`; the caller performs the
/// dict type check before replacing this field.
#[inline]
pub unsafe fn w_staticmethod_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe {
        (*(obj as *mut StaticMethod)).w_dict = w_dict;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
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
    /// function.py:724 `self.w_dict = None` — a real per-wrapper field,
    /// allocated lazily by `ClassMethod.getdict`.
    pub w_dict: PyObjectRef,
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
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(
        W_CLASSMETHOD_GC_TYPE_ID,
        W_CLASSMETHOD_OBJECT_SIZE,
    );
    let func = crate::gc_roots::shadow_stack_get(save_point);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut ClassMethod,
                ClassMethod {
                    ob: header,
                    w_function: func,
                    w_dict: PY_NULL,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    ClassMethod::allocate(ClassMethod {
        ob: header,
        w_function: func,
        w_dict: PY_NULL,
    })
}

pub unsafe fn w_classmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const ClassMethod)).w_function
}

/// function.py:752 `self.w_function = w_function`.
#[inline]
pub unsafe fn w_classmethod_set_func(obj: PyObjectRef, func: PyObjectRef) {
    unsafe {
        (*(obj as *mut ClassMethod)).w_function = func;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

/// function.py:726-729 `ClassMethod.getdict`.
#[inline]
pub unsafe fn w_classmethod_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let cm = obj as *mut ClassMethod;
        if (*cm).w_dict.is_null() {
            (*cm).w_dict = crate::w_dict_new();
            crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
        }
        (*cm).w_dict
    }
}

/// function.py:731-736 `ClassMethod.setdict`; the object-space layer checks
/// for dict or a dict subclass before replacing the field.
#[inline]
pub unsafe fn w_classmethod_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe {
        (*(obj as *mut ClassMethod)).w_dict = w_dict;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

#[inline]
pub unsafe fn is_classmethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &CLASSMETHOD_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard against drift between the constant colocated with
    /// `Method` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_CELL/FUNCTION trip-wire tests.
    #[test]
    fn w_method_gc_type_id_matches_descr() {
        assert_eq!(W_METHOD_GC_TYPE_ID, 16);
        assert_eq!(
            <Method as crate::lltype::GcType>::type_id(),
            W_METHOD_GC_TYPE_ID
        );
        assert_eq!(
            <Method as crate::lltype::GcType>::SIZE,
            W_METHOD_OBJECT_SIZE
        );
    }

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
