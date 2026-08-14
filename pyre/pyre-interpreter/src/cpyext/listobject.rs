//! `list` -- PyPy `cpyext/listobject.py`.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::c_int;

/// The slots start as `None`, for the reason `PyTuple_New` documents.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_New(size: isize) -> *mut CPyObject {
    if size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let items = vec![pyre_object::w_none(); size as usize];
    pyobject::make_ref(pyre_object::listobject::w_list_new_object(items))
}

fn list_argument(object: *mut CPyObject, function: &str) -> Option<PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::is_list(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): list expected"
        )));
        return None;
    }
    Some(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Size(object: *mut CPyObject) -> isize {
    let Some(value) = list_argument(object, "PyList_Size") else {
        return -1;
    };
    unsafe { pyre_object::listobject::w_list_len(value) as isize }
}

/// Borrowed, owned by the list's mirror.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_GetItem(object: *mut CPyObject, index: isize) -> *mut CPyObject {
    let Some(value) = list_argument(object, "PyList_GetItem") else {
        return std::ptr::null_mut();
    };
    let Some(item) = (unsafe { pyre_object::listobject::w_list_getitem(value, index as i64) })
    else {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "list index out of range",
        ));
        return std::ptr::null_mut();
    };
    pyobject::borrow_from(object, item)
}

/// Steals a reference to `item`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_SetItem(
    object: *mut CPyObject,
    index: isize,
    item: *mut CPyObject,
) -> c_int {
    let Some(value) = list_argument(object, "PyList_SetItem") else {
        unsafe { pyobject::decref(item) };
        return -1;
    };
    let w_item = unsafe { pyobject::from_ref(item) };
    let stored = unsafe { pyre_object::listobject::w_list_setitem(value, index as i64, w_item) };
    unsafe { pyobject::decref(item) };
    if !stored {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "list assignment index out of range",
        ));
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Append(object: *mut CPyObject, item: *mut CPyObject) -> c_int {
    let Some(value) = list_argument(object, "PyList_Append") else {
        return -1;
    };
    let Some(item) = argument(item) else {
        return -1;
    };
    unsafe { pyre_object::listobject::w_list_append(value, item) };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_list(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_CheckExact(object: *mut CPyObject) -> c_int {
    unsafe { PyList_Check(object) }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyList_New as *const ());
    std::hint::black_box(PyList_Size as *const ());
    std::hint::black_box(PyList_GetItem as *const ());
    std::hint::black_box(PyList_SetItem as *const ());
    std::hint::black_box(PyList_Append as *const ());
    std::hint::black_box(PyList_Check as *const ());
    std::hint::black_box(PyList_CheckExact as *const ());
}
