//! W_MethodObject - bound method wrapper.
//!
//! PyPy equivalent: pypy/interpreter/function.py Method

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("method", type_id = 16, static_name = "METHOD")]
pub struct W_MethodObject {
    pub w_function: PyObjectRef,
    pub w_self: PyObjectRef,
    pub w_class: PyObjectRef,
}

/// Field offsets of inline `PyObjectRef` slots within `W_MethodObject`.
/// Consumed by `pyre-jit-trace/src/descr.rs` to emit field-access IR;
/// the macro's own `W_METHOD_GC_PTR_OFFSETS` aggregate is independent
/// and does not depend on these per-field consts.
pub const METHOD_W_FUNCTION_OFFSET: usize = std::mem::offset_of!(W_MethodObject, w_function);
pub const METHOD_W_SELF_OFFSET: usize = std::mem::offset_of!(W_MethodObject, w_self);
pub const METHOD_W_CLASS_OFFSET: usize = std::mem::offset_of!(W_MethodObject, w_class);

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
    let raw = crate::gc_hook::try_gc_alloc_stable(W_METHOD_GC_TYPE_ID, W_METHOD_OBJECT_SIZE)
        .filter(|p| !p.is_null());
    // Re-read the pinned roots after the allocation; a minor collection
    // inside the GC malloc may have relocated them.
    let w_function = crate::gc_roots::shadow_stack_get(save_point);
    let w_self = crate::gc_roots::shadow_stack_get(save_point + 1);
    let w_class = crate::gc_roots::shadow_stack_get(save_point + 2);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut W_MethodObject,
                W_MethodObject {
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
    W_MethodObject::allocate(W_MethodObject {
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
    (*(obj as *const W_MethodObject)).w_function
}

#[inline]
pub unsafe fn w_method_get_self(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_MethodObject)).w_self
}

#[inline]
pub unsafe fn w_method_get_class(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_MethodObject)).w_class
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard against drift between the constant colocated with
    /// `W_MethodObject` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_CELL/FUNCTION trip-wire tests.
    #[test]
    fn w_method_gc_type_id_matches_descr() {
        assert_eq!(W_METHOD_GC_TYPE_ID, 16);
        assert_eq!(
            <W_MethodObject as crate::lltype::GcType>::type_id(),
            W_METHOD_GC_TYPE_ID
        );
        assert_eq!(
            <W_MethodObject as crate::lltype::GcType>::SIZE,
            W_METHOD_OBJECT_SIZE
        );
    }
}
