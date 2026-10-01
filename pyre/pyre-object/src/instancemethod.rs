//! `instancemethod` — PyPy `cpyext/classobject.py` `InstanceMethod`.
//!
//! The typedef there is named `cinstancemethod`. `type(im).__name__` is
//! `instancemethod`.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// A function that becomes a bound method when read from an instance.
#[pyre_class("instancemethod", static_name = "INSTANCEMETHOD")]
pub struct InstanceMethod {
    pub w_function: PyObjectRef,
}

/// `classobject.py InstanceMethod.__init__`: store the callable.
pub fn w_instancemethod_new(func: PyObjectRef) -> PyObjectRef {
    let roots = crate::gc_roots::push_roots();
    let slot = roots.base();
    let _ = roots.pin_root(func);
    InstanceMethod::allocate_stable(InstanceMethod {
        ob: PyObject {
            ob_type: &INSTANCEMETHOD_TYPE as *const PyType,
            w_class: get_instantiate(&INSTANCEMETHOD_TYPE),
        },
        w_function: roots.get(slot),
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_instancemethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &INSTANCEMETHOD_TYPE)
}

/// # Safety
/// `obj` must be an `instancemethod`.
#[inline]
pub unsafe fn w_instancemethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const InstanceMethod)).w_function
}
