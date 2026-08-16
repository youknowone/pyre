//! The mapping protocol -- PyPy `cpyext/mapping.py`.

use super::object::{argument, arguments, result};
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char, c_int};

/// A mapping is anything subscriptable that is not a list or a tuple.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return 0;
    }
    let mapping = unsafe {
        !pyre_object::is_list(object)
            && !pyre_object::is_tuple(object)
            && !pyre_object::unicodeobject::is_str(object)
            && crate::baseobjspace::lookup(object, "__getitem__").is_some()
    };
    mapping as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_Size(object: *mut CPyObject) -> isize {
    unsafe { super::object::PyObject_Size(object) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_Length(object: *mut CPyObject) -> isize {
    unsafe { super::object::PyObject_Size(object) }
}

/// The `str` key a `*String` entry point names.
///
/// Minting it allocates and reading a mirror can allocate too, so both ends
/// need an order: a caller realizes every mirror it was handed first, then
/// mints the key, then reads the receiver out. Once the mirrors are realized
/// no conversion can allocate, so the key cannot be moved out from under the
/// call that follows.
fn key_of(pointer: *const c_char) -> Option<PyObjectRef> {
    if pointer.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(pyre_object::w_str_new(
        &unsafe { CStr::from_ptr(pointer) }.to_string_lossy(),
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_GetItemString(
    object: *mut CPyObject,
    key: *const c_char,
) -> *mut CPyObject {
    super::object::realize_all([object]);
    let (Some(key), Some(object)) = (key_of(key), argument(object)) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getitem(object, key))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_SetItemString(
    object: *mut CPyObject,
    key: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, value]);
    let (Some(key), Some(object)) = (key_of(key), argument(object)) else {
        return -1;
    };
    let Some(value) = argument(value) else {
        return -1;
    };
    if trap(crate::baseobjspace::setitem(object, key, value)).is_none() {
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_DelItemString(
    object: *mut CPyObject,
    key: *const c_char,
) -> c_int {
    super::object::realize_all([object]);
    let (Some(key), Some(object)) = (key_of(key), argument(object)) else {
        return -1;
    };
    if trap(crate::baseobjspace::delitem(object, key)).is_none() {
        return -1;
    }
    0
}

/// `PyMapping_HasKey` swallows the lookup's exception, as CPython's does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_HasKey(object: *mut CPyObject, key: *mut CPyObject) -> c_int {
    let Some([object, key]) = arguments([object, key]) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return 0;
    };
    match crate::baseobjspace::getitem(object, key) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_HasKeyString(
    object: *mut CPyObject,
    key: *const c_char,
) -> c_int {
    super::object::realize_all([object]);
    let (Some(key), Some(object)) = (key_of(key), argument(object)) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return 0;
    };
    match crate::baseobjspace::getitem(object, key) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

/// `keys()`, `values()` and `items()` as lists, which is what the C API
/// promises whatever the mapping returns.
fn as_list(object: PyObjectRef, method: &str) -> Result<PyObjectRef, crate::PyError> {
    let view = crate::baseobjspace::getattr_str(object, method)
        .and_then(|method| crate::call::call_function_impl_result(method, &[]))?;
    Ok(pyre_object::w_list_new(
        crate::baseobjspace::unpackiterable(view, -1)?,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_Keys(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(as_list(object, "keys"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_Values(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(as_list(object, "values"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMapping_Items(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(as_list(object, "items"))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyMapping_Keys as *const ());
    std::hint::black_box(PyMapping_Values as *const ());
    std::hint::black_box(PyMapping_Items as *const ());
    std::hint::black_box(PyMapping_Check as *const ());
    std::hint::black_box(PyMapping_Size as *const ());
    std::hint::black_box(PyMapping_Length as *const ());
    std::hint::black_box(PyMapping_GetItemString as *const ());
    std::hint::black_box(PyMapping_SetItemString as *const ());
    std::hint::black_box(PyMapping_DelItemString as *const ());
    std::hint::black_box(PyMapping_HasKey as *const ());
    std::hint::black_box(PyMapping_HasKeyString as *const ());
}
