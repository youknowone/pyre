//! Instance-method entry points -- PyPy `cpyext/classobject.py`.
//!
//! `PyInstanceMethod_Check` and `PyInstanceMethod_GET_FUNCTION` are macros.
//! The two functions here are the exported entry points.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};

/// `classobject.py PyInstanceMethod_New`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInstanceMethod_New(func: *mut CPyObject) -> *mut CPyObject {
    let Some(func) = argument(func) else {
        return std::ptr::null_mut();
    };
    if !crate::baseobjspace::callable_w(func) {
        let got = crate::typedef::r#type(func)
            .map(|tp| unsafe { pyre_object::w_type_get_name(tp.as_ptr()) }.to_string())
            .unwrap_or_else(|| "?".to_string());
        return result(Err(crate::PyError::type_error(format!(
            "instancemethod expected a callable, got {got}"
        ))));
    }
    result(Ok(pyre_object::instancemethod::w_instancemethod_new(func)))
}

/// `classobject.py PyInstanceMethod_Function` — the wrapped callable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInstanceMethod_Function(im: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(im) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::instancemethod::is_instancemethod(object) } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    pyobject::make_ref(unsafe { pyre_object::instancemethod::w_instancemethod_get_func(object) })
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyInstanceMethod_New as *const ());
    std::hint::black_box(PyInstanceMethod_Function as *const ());
}
