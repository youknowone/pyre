//! W_MethodObject - bound method wrapper.
//!
//! PyPy equivalent: pypy/interpreter/function.py Method

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

#[repr(C)]
pub struct W_MethodObject {
    pub ob_header: PyObject,
    pub w_function: PyObjectRef,
    pub w_self: PyObjectRef,
    pub w_class: PyObjectRef,
}

pub static METHOD_TYPE: PyType = crate::pyobject::new_pytype("method");

/// Field offsets of inline `PyObjectRef` slots within `W_MethodObject`.
pub const METHOD_W_FUNCTION_OFFSET: usize = std::mem::offset_of!(W_MethodObject, w_function);
pub const METHOD_W_SELF_OFFSET: usize = std::mem::offset_of!(W_MethodObject, w_self);
pub const METHOD_W_CLASS_OFFSET: usize = std::mem::offset_of!(W_MethodObject, w_class);

/// GC type id assigned to `W_MethodObject` at JitDriver init time. Held
/// as a constant alongside the struct (rather than runtime-queried) so
/// the allocation hook can reach it without a back-channel, mirroring
/// `W_CELL_GC_TYPE_ID` / `FUNCTION_GC_TYPE_ID`.
pub const W_METHOD_GC_TYPE_ID: u32 = 16;

/// Fixed payload size used by `gct_fv_gc_malloc`'s `c_size`
/// (`framework.py:811`).
pub const W_METHOD_OBJECT_SIZE: usize = std::mem::size_of::<W_MethodObject>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace
/// during minor collection. `ob.w_class` is intentionally absent — see
/// `cellobject.rs` for the rationale.
pub const W_METHOD_GC_PTR_OFFSETS: [usize; 3] = [
    METHOD_W_FUNCTION_OFFSET,
    METHOD_W_SELF_OFFSET,
    METHOD_W_CLASS_OFFSET,
];

impl crate::lltype::GcType for W_MethodObject {
    const TYPE_ID: u32 = W_METHOD_GC_TYPE_ID;
    const SIZE: usize = W_METHOD_OBJECT_SIZE;
}

pub fn w_method_new(
    w_function: PyObjectRef,
    w_self: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the `lltype::malloc_typed` call below. All three inputs are live
    // PyObjectRef roots that must survive a potential collection inside
    // the allocation point once Phase 2 swaps the malloc body to a
    // managed allocator.
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_function);
    crate::gc_roots::pin_root(w_self);
    crate::gc_roots::pin_root(w_class);

    crate::lltype::malloc_typed(W_MethodObject {
        ob_header: PyObject {
            ob_type: &METHOD_TYPE as *const PyType,
            w_class: get_instantiate(&METHOD_TYPE),
        },
        w_function,
        w_self,
        w_class,
    }) as PyObjectRef
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
            <W_MethodObject as crate::lltype::GcType>::TYPE_ID,
            W_METHOD_GC_TYPE_ID
        );
        assert_eq!(
            <W_MethodObject as crate::lltype::GcType>::SIZE,
            W_METHOD_OBJECT_SIZE
        );
    }
}
