//! `dict` -- PyPy `cpyext/dictobject.py`.

use super::object::argument;
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char, c_int};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_New() -> *mut CPyObject {
    pyobject::make_ref(pyre_object::dictmultiobject::w_dict_new())
}

fn dict_argument(object: *mut CPyObject, function: &str) -> Option<PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::is_dict(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): dict expected"
        )));
        return None;
    }
    Some(value)
}

fn key_name(name: *const c_char) -> Option<String> {
    if name.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_SetItem(
    object: *mut CPyObject,
    key: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    let Some(dict) = dict_argument(object, "PyDict_SetItem") else {
        return -1;
    };
    let (Some(key), Some(value)) = (argument(key), argument(value)) else {
        return -1;
    };
    if trap(crate::baseobjspace::setitem(dict, key, value)).is_none() {
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_SetItemString(
    object: *mut CPyObject,
    key: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    let Some(dict) = dict_argument(object, "PyDict_SetItemString") else {
        return -1;
    };
    let (Some(key), Some(value)) = (key_name(key), argument(value)) else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(dict);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(value);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            &key,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        )
    };
    0
}

/// Borrowed, and — as upstream and CPython both specify — it does not set an
/// exception when the key is missing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItem(
    object: *mut CPyObject,
    key: *mut CPyObject,
) -> *mut CPyObject {
    let Some(dict) = dict_argument(object, "PyDict_GetItem") else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return std::ptr::null_mut();
    };
    let Some(key) = argument(key) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return std::ptr::null_mut();
    };
    match crate::baseobjspace::getitem(dict, key) {
        Ok(value) => pyobject::borrow_from(object, value),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItemString(
    object: *mut CPyObject,
    key: *const c_char,
) -> *mut CPyObject {
    let Some(dict) = dict_argument(object, "PyDict_GetItemString") else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return std::ptr::null_mut();
    };
    let Some(key) = key_name(key) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return std::ptr::null_mut();
    };
    match unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, &key) } {
        Some(value) => pyobject::borrow_from(object, value),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_DelItem(object: *mut CPyObject, key: *mut CPyObject) -> c_int {
    let Some(dict) = dict_argument(object, "PyDict_DelItem") else {
        return -1;
    };
    let Some(key) = argument(key) else {
        return -1;
    };
    if trap(crate::baseobjspace::delitem(dict, key)).is_none() {
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Size(object: *mut CPyObject) -> isize {
    let Some(dict) = dict_argument(object, "PyDict_Size") else {
        return -1;
    };
    unsafe { pyre_object::dictmultiobject::w_dict_len(dict) as isize }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Contains(object: *mut CPyObject, key: *mut CPyObject) -> c_int {
    let Some(dict) = dict_argument(object, "PyDict_Contains") else {
        return -1;
    };
    let Some(key) = argument(key) else {
        return -1;
    };
    match trap(crate::baseobjspace::contains(dict, key)) {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_dict(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_CheckExact(object: *mut CPyObject) -> c_int {
    unsafe { PyDict_Check(object) }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyDict_New as *const ());
    std::hint::black_box(PyDict_SetItem as *const ());
    std::hint::black_box(PyDict_SetItemString as *const ());
    std::hint::black_box(PyDict_GetItem as *const ());
    std::hint::black_box(PyDict_GetItemString as *const ());
    std::hint::black_box(PyDict_DelItem as *const ());
    std::hint::black_box(PyDict_Size as *const ());
    std::hint::black_box(PyDict_Contains as *const ());
    std::hint::black_box(PyDict_Check as *const ());
    std::hint::black_box(PyDict_CheckExact as *const ());
}
