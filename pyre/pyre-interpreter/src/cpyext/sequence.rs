//! The sequence protocol -- PyPy `cpyext/sequence.py`.

use super::object::{argument, arguments, result};
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::c_int;

/// A sequence is anything with `sq_item`, which at interpreter level is
/// `__getitem__` on something that is not a mapping.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return 0;
    }
    let sequence = unsafe {
        !crate::baseobjspace::isinstance_dict_w(object)
            && crate::baseobjspace::lookup(object, "__getitem__").is_some()
    };
    sequence as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Size(object: *mut CPyObject) -> isize {
    unsafe { super::object::PyObject_Size(object) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Length(object: *mut CPyObject) -> isize {
    unsafe { super::object::PyObject_Size(object) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Concat(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::add(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_InPlaceConcat(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        crate::bytecode::BinaryOperator::InplaceAdd,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Repeat(object: *mut CPyObject, count: isize) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::mul(object, pyre_object::w_int_new(count as i64)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_InPlaceRepeat(
    object: *mut CPyObject,
    count: isize,
) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        object,
        pyre_object::w_int_new(count as i64),
        crate::bytecode::BinaryOperator::InplaceMultiply,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_GetItem(
    object: *mut CPyObject,
    index: isize,
) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getitem(
        object,
        pyre_object::w_int_new(index as i64),
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_SetItem(
    object: *mut CPyObject,
    index: isize,
    value: *mut CPyObject,
) -> c_int {
    let Some([object, value]) = arguments([object, value]) else {
        return -1;
    };
    let assigned =
        crate::baseobjspace::setitem(object, pyre_object::w_int_new(index as i64), value);
    if trap(assigned).is_none() { -1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_DelItem(object: *mut CPyObject, index: isize) -> c_int {
    let Some(object) = argument(object) else {
        return -1;
    };
    let deleted = crate::baseobjspace::delitem(object, pyre_object::w_int_new(index as i64));
    if trap(deleted).is_none() { -1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Contains(
    object: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    let Some([object, value]) = arguments([object, value]) else {
        return -1;
    };
    match trap(crate::baseobjspace::contains(object, value)) {
        Some(found) => found as c_int,
        None => -1,
    }
}

/// The index of the first item equal to `value`, or -1 with `ValueError` set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Index(object: *mut CPyObject, value: *mut CPyObject) -> isize {
    let Some([object, value]) = arguments([object, value]) else {
        return -1;
    };
    trap(index_of(object, value)).unwrap_or(-1)
}

fn index_of(object: PyObjectRef, value: PyObjectRef) -> Result<isize, crate::PyError> {
    // `eq_w` runs the element's own `__eq__`, so every operand crosses a
    // collection point on each turn.  A `Vec` is not somewhere the collector
    // looks: the elements go on the shadow stack and are read back per turn.
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(value);
    let object_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(object);
    let items = crate::baseobjspace::unpackiterable(
        pyre_object::gc_roots::shadow_stack_get(object_slot),
        -1,
    )?;
    let elements = pyre_object::gc_roots::pin_roots(&items);
    for index in 0..items.len() {
        if crate::baseobjspace::eq_w(
            pyre_object::gc_roots::shadow_stack_get(elements + index),
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        )? {
            return Ok(index as isize);
        }
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::ValueError,
        "sequence.index(x): x not in sequence",
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_List(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::unpackiterable(object, -1).map(pyre_object::w_list_new))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Tuple(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(
        crate::baseobjspace::unpackiterable(object, -1).map(pyre_object::tupleobject::w_tuple_new),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_GetSlice(
    object: *mut CPyObject,
    start: isize,
    stop: isize,
) -> *mut CPyObject {
    super::object::realize_all([object]);
    let slice = super::sliceobject::range_slice(start, stop);
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getitem(object, slice))
}

/// `PySequence_SetSlice(o, low, high, value)` (`sequence.py:127-132`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_SetSlice(
    object: *mut CPyObject,
    low: isize,
    high: isize,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, value]);
    let slice = super::sliceobject::range_slice(low, high);
    let Some([object, value]) = arguments([object, value]) else {
        return -1;
    };
    let assigned = crate::baseobjspace::setitem(object, slice, value);
    if trap(assigned).is_none() { -1 } else { 0 }
}

/// `PySequence_DelSlice(o, low, high)` (`sequence.py:134-139`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_DelSlice(
    object: *mut CPyObject,
    low: isize,
    high: isize,
) -> c_int {
    super::object::realize_all([object]);
    let slice = super::sliceobject::range_slice(low, high);
    let Some(object) = argument(object) else {
        return -1;
    };
    let deleted = crate::baseobjspace::delitem(object, slice);
    if trap(deleted).is_none() { -1 } else { 0 }
}

/// `PySequence_In(o, value)` — the older spelling of `PySequence_Contains`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_In(object: *mut CPyObject, value: *mut CPyObject) -> c_int {
    unsafe { PySequence_Contains(object, value) }
}

/// `PySequence_Count(o, value)` — how many items compare equal to `value`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Count(object: *mut CPyObject, value: *mut CPyObject) -> isize {
    let Some([object, value]) = arguments([object, value]) else {
        return -1;
    };
    trap(count_of(object, value)).unwrap_or(-1)
}

fn count_of(object: PyObjectRef, value: PyObjectRef) -> Result<isize, crate::PyError> {
    // The same shadow-stack discipline as [`index_of`], for the same reason.
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(value);
    let object_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(object);
    let items = crate::baseobjspace::unpackiterable(
        pyre_object::gc_roots::shadow_stack_get(object_slot),
        -1,
    )?;
    let elements = pyre_object::gc_roots::pin_roots(&items);
    let mut seen = 0;
    for index in 0..items.len() {
        if crate::baseobjspace::eq_w(
            pyre_object::gc_roots::shadow_stack_get(elements + index),
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        )? {
            seen += 1;
        }
    }
    Ok(seen)
}

/// `PySequence_Fast(o, message)` — `o` itself when it is already a list or a
/// tuple, and a list of its items otherwise, with `message` replacing the
/// `TypeError` a non-iterable raises (`sequence.py:44-66`).
///
/// The fallback builds a list rather than upstream's tuple, which is what 3.14
/// documents; `PySequence_Fast_GET_ITEM` reads either.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Fast(
    object: *mut CPyObject,
    message: *const std::ffi::c_char,
) -> *mut CPyObject {
    let Some(value) = argument(object) else {
        return std::ptr::null_mut();
    };
    if unsafe { pyre_object::is_list(value) || pyre_object::is_tuple(value) } {
        unsafe { pyobject::incref(object) };
        return object;
    }
    match crate::baseobjspace::unpackiterable(value, -1) {
        Ok(items) => pyobject::make_ref(pyre_object::w_list_new(items)),
        Err(error) => {
            let error = if error.kind == crate::PyErrorKind::TypeError && !message.is_null() {
                crate::PyError::type_error(
                    unsafe { std::ffi::CStr::from_ptr(message) }
                        .to_string_lossy()
                        .into_owned(),
                )
            } else {
                error
            };
            super::pyerrors::set_pending_error(error);
            std::ptr::null_mut()
        }
    }
}

/// `PySequence_Fast_GET_SIZE(o)` (`sequence.py:83-98`).
///
/// A function rather than the macro the reference header spells: a mirror has
/// no item array of its own, so the length comes from the interpreter object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Fast_GET_SIZE(object: *mut CPyObject) -> isize {
    let value = unsafe { pyobject::from_ref(object) };
    if unsafe { !value.is_null() && pyre_object::is_tuple(value) } {
        return unsafe { super::tupleobject::PyTuple_Size(object) };
    }
    unsafe { super::listobject::PyList_Size(object) }
}

/// `PySequence_Fast_GET_ITEM(o, index)` — borrowed (`sequence.py:68-81`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_Fast_GET_ITEM(
    object: *mut CPyObject,
    index: isize,
) -> *mut CPyObject {
    let value = unsafe { pyobject::from_ref(object) };
    if unsafe { !value.is_null() && pyre_object::is_tuple(value) } {
        return unsafe { super::tupleobject::PyTuple_GetItem(object, index) };
    }
    unsafe { super::listobject::PyList_GetItem(object, index) }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PySequence_Check as *const ());
    std::hint::black_box(PySequence_Size as *const ());
    std::hint::black_box(PySequence_Length as *const ());
    std::hint::black_box(PySequence_Concat as *const ());
    std::hint::black_box(PySequence_InPlaceConcat as *const ());
    std::hint::black_box(PySequence_Repeat as *const ());
    std::hint::black_box(PySequence_InPlaceRepeat as *const ());
    std::hint::black_box(PySequence_GetItem as *const ());
    std::hint::black_box(PySequence_SetItem as *const ());
    std::hint::black_box(PySequence_DelItem as *const ());
    std::hint::black_box(PySequence_Contains as *const ());
    std::hint::black_box(PySequence_Index as *const ());
    std::hint::black_box(PySequence_List as *const ());
    std::hint::black_box(PySequence_Tuple as *const ());
    std::hint::black_box(PySequence_GetSlice as *const ());
    std::hint::black_box(PySequence_SetSlice as *const ());
    std::hint::black_box(PySequence_DelSlice as *const ());
    std::hint::black_box(PySequence_In as *const ());
    std::hint::black_box(PySequence_Count as *const ());
    std::hint::black_box(PySequence_Fast as *const ());
    std::hint::black_box(PySequence_Fast_GET_SIZE as *const ());
    std::hint::black_box(PySequence_Fast_GET_ITEM as *const ());
}
