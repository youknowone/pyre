//! W_SliceObject — Python `slice` type.

use crate::pyobject::*;

pub static SLICE_TYPE: PyType = PyType { tp_name: "slice" };

#[repr(C)]
pub struct W_SliceObject {
    pub ob_header: PyObject,
    pub start: PyObjectRef,
    pub stop: PyObjectRef,
    pub step: PyObjectRef,
}

pub fn w_slice_new(start: PyObjectRef, stop: PyObjectRef, step: PyObjectRef) -> PyObjectRef {
    let obj = Box::new(W_SliceObject {
        ob_header: PyObject {
            ob_type: &SLICE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        start,
        stop,
        step,
    });
    Box::into_raw(obj) as PyObjectRef
}

pub unsafe fn is_slice(obj: PyObjectRef) -> bool {
    !obj.is_null() && (*obj).ob_type == &SLICE_TYPE as *const PyType
}

pub unsafe fn w_slice_get_start(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SliceObject)).start
}

pub unsafe fn w_slice_get_stop(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SliceObject)).stop
}

pub unsafe fn w_slice_get_step(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SliceObject)).step
}
