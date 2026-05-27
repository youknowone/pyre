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
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the `lltype::malloc_typed` call below. All three inputs are live
    // PyObjectRef roots that must survive a potential collection inside
    // the allocation point once the malloc body swaps to a
    // managed allocator.
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_function);
    crate::gc_roots::pin_root(w_self);
    crate::gc_roots::pin_root(w_class);
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
