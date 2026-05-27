//! W_SliceObject — Python `slice` type.

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("slice", type_id = 17, static_name = "SLICE")]
pub struct W_SliceObject {
    pub start: PyObjectRef,
    pub stop: PyObjectRef,
    pub step: PyObjectRef,
}

/// Field offsets of the inline `PyObjectRef` slots within
/// `W_SliceObject`.  Consumed by `pyre-jit-trace/src/descr.rs` to
/// emit field-access IR; the macro emits its own
/// `W_SLICE_GC_PTR_OFFSETS` aggregate that does NOT depend on these
/// per-field consts, so they live here only for IR-construction use.
pub const SLICE_START_OFFSET: usize = std::mem::offset_of!(W_SliceObject, start);
pub const SLICE_STOP_OFFSET: usize = std::mem::offset_of!(W_SliceObject, stop);
pub const SLICE_STEP_OFFSET: usize = std::mem::offset_of!(W_SliceObject, step);

pub fn w_slice_new(start: PyObjectRef, stop: PyObjectRef, step: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(start);
    crate::gc_roots::pin_root(stop);
    crate::gc_roots::pin_root(step);
    W_SliceObject::allocate(W_SliceObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        start,
        stop,
        step,
    })
}

pub unsafe fn is_slice(obj: PyObjectRef) -> bool {
    unsafe { !obj.is_null() && (*obj).ob_type == &SLICE_TYPE as *const PyType }
}

pub unsafe fn w_slice_get_start(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SliceObject)).start }
}

pub unsafe fn w_slice_get_stop(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SliceObject)).stop }
}

pub unsafe fn w_slice_get_step(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SliceObject)).step }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_slice_gc_type_id_matches_descr() {
        assert_eq!(W_SLICE_GC_TYPE_ID, 17);
        assert_eq!(
            <W_SliceObject as crate::lltype::GcType>::type_id(),
            W_SLICE_GC_TYPE_ID
        );
        assert_eq!(
            <W_SliceObject as crate::lltype::GcType>::SIZE,
            W_SLICE_OBJECT_SIZE
        );
    }
}
