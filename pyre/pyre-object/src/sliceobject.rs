//! W_SliceObject — Python `slice` type.

use crate::pyobject::*;

pub static SLICE_TYPE: PyType = crate::pyobject::new_pytype("slice");

#[repr(C)]
pub struct W_SliceObject {
    pub ob_header: PyObject,
    pub start: PyObjectRef,
    pub stop: PyObjectRef,
    pub step: PyObjectRef,
}

/// Field offsets of the inline `PyObjectRef` slots within
/// `W_SliceObject`.
pub const SLICE_START_OFFSET: usize = std::mem::offset_of!(W_SliceObject, start);
pub const SLICE_STOP_OFFSET: usize = std::mem::offset_of!(W_SliceObject, stop);
pub const SLICE_STEP_OFFSET: usize = std::mem::offset_of!(W_SliceObject, step);

/// GC type id assigned to `W_SliceObject` at JitDriver init time.
pub const W_SLICE_GC_TYPE_ID: u32 = 17;

/// Fixed payload size used by `gct_fv_gc_malloc`'s `c_size`
/// (`framework.py:811`).
pub const W_SLICE_OBJECT_SIZE: usize = std::mem::size_of::<W_SliceObject>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_SLICE_GC_PTR_OFFSETS: [usize; 3] =
    [SLICE_START_OFFSET, SLICE_STOP_OFFSET, SLICE_STEP_OFFSET];

impl crate::lltype::GcType for W_SliceObject {
    const TYPE_ID: u32 = W_SLICE_GC_TYPE_ID;
    const SIZE: usize = W_SLICE_OBJECT_SIZE;
}

pub fn w_slice_new(start: PyObjectRef, stop: PyObjectRef, step: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the `lltype::malloc_typed` call below.
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(start);
    crate::gc_roots::pin_root(stop);
    crate::gc_roots::pin_root(step);

    crate::lltype::malloc_typed(W_SliceObject {
        ob_header: PyObject {
            ob_type: &SLICE_TYPE as *const PyType,
            w_class: get_instantiate(&SLICE_TYPE),
        },
        start,
        stop,
        step,
    }) as PyObjectRef
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
            <W_SliceObject as crate::lltype::GcType>::TYPE_ID,
            W_SLICE_GC_TYPE_ID
        );
        assert_eq!(
            <W_SliceObject as crate::lltype::GcType>::SIZE,
            W_SLICE_OBJECT_SIZE
        );
    }
}
