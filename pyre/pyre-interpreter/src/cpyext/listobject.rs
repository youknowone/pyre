//! `list` -- PyPy `cpyext/listobject.py`.

use super::object::{argument, arguments, result};
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
    let Some(items) = super::object::item_slots(size, pyre_object::w_none()) else {
        return std::ptr::null_mut();
    };
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
    super::object::realize_all([object, item]);
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
    super::object::realize_all([object, item]);
    let Some(value) = list_argument(object, "PyList_Append") else {
        return -1;
    };
    let Some(item) = argument(item) else {
        return -1;
    };
    unsafe { pyre_object::listobject::w_list_append(value, item) };
    0
}

/// `PyList_GetItemRef(list, index)` — `PyList_GetItem`'s strong-reference
/// spelling, the one that stays defined while another thread shrinks the list.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_GetItemRef(object: *mut CPyObject, index: isize) -> *mut CPyObject {
    let Some(value) = list_argument(object, "PyList_GetItemRef") else {
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
    pyobject::make_ref(item)
}

/// A list, or the `SystemError` `PyErr_BadInternalCall` records.
///
/// The rejection `PyList_Sort` and `PyList_Reverse` make, which is not the
/// `TypeError` the reading entry points make (`listobject.py`, `:139`).
fn internal_list(object: *mut CPyObject) -> Option<PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::is_list(value) } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(value)
}

/// The `list` type's own method, which is what `space.call_method(space.w_list,
/// name, ...)` reaches: an override on a subclass is deliberately not consulted
/// (`listobject.py:96-142`).
fn list_method(
    method: fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
    arguments: &[PyObjectRef],
) -> c_int {
    match super::pyerrors::trap(method(arguments)) {
        Some(_) => 0,
        None => -1,
    }
}

/// `PyList_Insert(list, index, item)` (`listobject.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Insert(
    object: *mut CPyObject,
    index: isize,
    item: *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, item]);
    let index = pyre_object::w_int_new(index as i64);
    let Some([value, item]) = arguments([object, item]) else {
        return -1;
    };
    list_method(
        crate::type_methods::list_method_insert,
        &[value, index, item],
    )
}

/// `PyList_Sort(list)` (`listobject.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Sort(object: *mut CPyObject) -> c_int {
    let Some(value) = internal_list(object) else {
        return -1;
    };
    list_method(crate::type_methods::list_method_sort, &[value])
}

/// `PyList_Reverse(list)` (`listobject.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Reverse(object: *mut CPyObject) -> c_int {
    let Some(value) = internal_list(object) else {
        return -1;
    };
    list_method(crate::type_methods::list_method_reverse, &[value])
}

/// `PyList_Clear(list)` — drop every item.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Clear(object: *mut CPyObject) -> c_int {
    let Some(value) = internal_list(object) else {
        return -1;
    };
    list_method(crate::type_methods::list_method_clear, &[value])
}

/// `PyList_Extend(list, iterable)` — append everything `iterable` yields.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Extend(object: *mut CPyObject, iterable: *mut CPyObject) -> c_int {
    super::object::realize_all([object, iterable]);
    let (Some(value), Some(iterable)) = (internal_list(object), argument(iterable)) else {
        return -1;
    };
    list_method(crate::type_methods::list_method_extend, &[value, iterable])
}

/// `PyList_AsTuple(list)` — `tuple(list)` (`listobject.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_AsTuple(object: *mut CPyObject) -> *mut CPyObject {
    let Some(value) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(
        crate::baseobjspace::unpackiterable(value, -1).map(pyre_object::tupleobject::w_tuple_new),
    )
}

/// `PyList_GetSlice(list, low, high)` — `list[low:high]`, with no negative
/// index handling (`listobject.py:144-152`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_GetSlice(
    object: *mut CPyObject,
    low: isize,
    high: isize,
) -> *mut CPyObject {
    super::object::realize_all([object]);
    let slice = super::sliceobject::range_slice(low, high);
    let Some(value) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getitem(value, slice))
}

/// `PyList_SetSlice(list, low, high, sequence)` — a NULL `sequence` deletes the
/// slice (`listobject.py:154-166`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_SetSlice(
    object: *mut CPyObject,
    low: isize,
    high: isize,
    sequence: *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, sequence]);
    let slice = super::sliceobject::range_slice(low, high);
    let Some(value) = argument(object) else {
        return -1;
    };
    let items = unsafe { pyobject::from_ref(sequence) };
    let assigned = if items.is_null() {
        crate::baseobjspace::delitem(value, slice)
    } else {
        crate::baseobjspace::setitem(value, slice, items).map(drop)
    };
    if super::pyerrors::trap(assigned).is_none() {
        -1
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_list(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyList_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::LIST_TYPE)) as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyList_New as *const ());
    std::hint::black_box(PyList_Size as *const ());
    std::hint::black_box(PyList_GetItem as *const ());
    std::hint::black_box(PyList_SetItem as *const ());
    std::hint::black_box(PyList_Append as *const ());
    std::hint::black_box(PyList_GetItemRef as *const ());
    std::hint::black_box(PyList_Insert as *const ());
    std::hint::black_box(PyList_Clear as *const ());
    std::hint::black_box(PyList_Extend as *const ());
    std::hint::black_box(PyList_Sort as *const ());
    std::hint::black_box(PyList_Reverse as *const ());
    std::hint::black_box(PyList_AsTuple as *const ());
    std::hint::black_box(PyList_GetSlice as *const ());
    std::hint::black_box(PyList_SetSlice as *const ());
    std::hint::black_box(PyList_Check as *const ());
    std::hint::black_box(PyList_CheckExact as *const ());
}
