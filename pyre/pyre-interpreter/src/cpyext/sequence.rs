//! The sequence protocol -- PyPy `cpyext/sequence.py`.

use super::object::{argument, result};
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
        !pyre_object::is_dict(object)
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
    let (Some(left), Some(right)) = (argument(left), argument(right)) else {
        return std::ptr::null_mut();
    };
    result(crate::add(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySequence_InPlaceConcat(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let (Some(left), Some(right)) = (argument(left), argument(right)) else {
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
    let (Some(object), Some(value)) = (argument(object), argument(value)) else {
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
    let (Some(object), Some(value)) = (argument(object), argument(value)) else {
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
    let (Some(object), Some(value)) = (argument(object), argument(value)) else {
        return -1;
    };
    trap(index_of(object, value)).unwrap_or(-1)
}

fn index_of(object: PyObjectRef, value: PyObjectRef) -> Result<isize, crate::PyError> {
    let items = crate::baseobjspace::unpackiterable(object, -1)?;
    for (index, &item) in items.iter().enumerate() {
        if crate::baseobjspace::eq_w(item, value)? {
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
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let slice = pyre_object::sliceobject::w_slice_new(
        pyre_object::w_int_new(start as i64),
        pyre_object::w_int_new(stop as i64),
        pyre_object::w_none(),
    );
    result(crate::baseobjspace::getitem(object, slice))
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
}
