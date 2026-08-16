//! `tuple` -- PyPy `cpyext/tupleobject.py`.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::c_int;

/// A tuple of NULL slots for the caller to fill through `PyTuple_SetItem`.
///
/// This is the layout the interpreter's own two-step tuple construction uses
/// (`module/marshal`: `w_tuple_new_array_backed(vec![PY_NULL; len])` followed
/// by `w_tuple_setitem_initializing`), and the only one that has a setter:
/// `w_tuple_new` would pick the arity-2 specialisation, which stores its items
/// inline and cannot be filled afterwards.  Reading a slot before writing it
/// therefore yields NULL, exactly as it does in CPython.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTuple_New(size: isize) -> *mut CPyObject {
    if size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let Some(items) = super::object::item_slots(size, pyre_object::PY_NULL) else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(pyre_object::tupleobject::w_tuple_new_array_backed(items))
}

fn tuple_argument(object: *mut CPyObject, function: &str) -> Option<PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::is_tuple(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): tuple expected"
        )));
        return None;
    }
    Some(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTuple_Size(object: *mut CPyObject) -> isize {
    let Some(value) = tuple_argument(object, "PyTuple_Size") else {
        return -1;
    };
    unsafe { pyre_object::tupleobject::w_tuple_len(value) as isize }
}

/// Borrowed, owned by the tuple's mirror for as long as it lives.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTuple_GetItem(object: *mut CPyObject, index: isize) -> *mut CPyObject {
    let Some(value) = tuple_argument(object, "PyTuple_GetItem") else {
        return std::ptr::null_mut();
    };
    let Some(item) = (unsafe { pyre_object::tupleobject::w_tuple_getitem(value, index as i64) })
    else {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "tuple index out of range",
        ));
        return std::ptr::null_mut();
    };
    pyobject::borrow_from(object, item)
}

/// Steals a reference to `item`, and is only defined on a tuple no other code
/// has seen yet — the contract `PyTuple_SetItem` documents.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTuple_SetItem(
    object: *mut CPyObject,
    index: isize,
    item: *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, item]);
    let Some(value) = tuple_argument(object, "PyTuple_SetItem") else {
        unsafe { pyobject::decref(item) };
        return -1;
    };
    let length = unsafe { pyre_object::tupleobject::w_tuple_len(value) } as isize;
    if index < 0 || index >= length {
        unsafe { pyobject::decref(item) };
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "tuple assignment index out of range",
        ));
        return -1;
    }
    let w_item = unsafe { pyobject::from_ref(item) };
    let stored = unsafe {
        pyre_object::tupleobject::w_tuple_setitem_initializing(value, index as usize, w_item)
    };
    unsafe { pyobject::decref(item) };
    if !stored {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyTuple_SetItem(): the tuple is not one PyTuple_New built",
        ));
        return -1;
    }
    0
}

/// `PyTuple_GetSlice(tuple, low, high)` — `tuple[low:high]`
/// (`tupleobject.py:216-221`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTuple_GetSlice(
    object: *mut CPyObject,
    low: isize,
    high: isize,
) -> *mut CPyObject {
    let slice = super::sliceobject::range_slice(low, high);
    let Some(value) = tuple_argument(object, "PyTuple_GetSlice") else {
        return std::ptr::null_mut();
    };
    super::object::result(crate::baseobjspace::getitem(value, slice))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTuple_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_tuple(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTuple_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::TUPLE_TYPE)) as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyTuple_New as *const ());
    std::hint::black_box(PyTuple_Size as *const ());
    std::hint::black_box(PyTuple_GetItem as *const ());
    std::hint::black_box(PyTuple_SetItem as *const ());
    std::hint::black_box(PyTuple_GetSlice as *const ());
    std::hint::black_box(PyTuple_Check as *const ());
    std::hint::black_box(PyTuple_CheckExact as *const ());
}
