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

pub static METHOD_TYPE: PyType = PyType { tp_name: "method" };

pub fn w_method_new(
    w_function: PyObjectRef,
    w_self: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    let obj = Box::new(W_MethodObject {
        ob_header: PyObject {
            ob_type: &METHOD_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        w_function,
        w_self,
        w_class,
    });
    Box::into_raw(obj) as PyObjectRef
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
