//! `tuple` -- PyPy `cpyext/tupleobject.py`.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use super::typeobject::{CPyTypeObject, CPyVarObject};
use pyre_object::PyObjectRef;
use std::ffi::c_int;

/// `PyTupleObject` -- the header, and the items that follow it.
///
/// `tupleobject.py:31-35 PyTupleObjectFields` gives the mirror an array of
/// `ob_size` `PyObject *`, filled when the mirror is built, so that
/// `PyTuple_GET_ITEM` reads a slot the way CPython's macro does.  The array is
/// the mirror's own rather than a view on the tuple, and has to be: the
/// arity-2 specialisations of `specialisedtupleobject.py` build a fresh item
/// on every read, so a borrowed reference into one would name something
/// nothing keeps alive.
///
/// `ob_item` is declared at one slot, which is what `tupleobject.h` spells;
/// [`item_bytes`] is what decides how many the block really has.
#[repr(C)]
pub struct CPyTupleObject {
    pub ob_base: CPyVarObject,
    pub ob_item: [*mut CPyObject; 1],
}

/// The array follows the header with nothing between, which is what lets a
/// block hold it and a reader find it.
const _: () = assert!(
    std::mem::offset_of!(CPyTupleObject, ob_item) == size_of::<CPyVarObject>(),
    "a tuple mirror's items start where its header ends"
);

/// Whether a mirror of `ob_type` carries its items in its own block.
///
/// The array starts where the header ends, so only a tuple whose type declared
/// nothing past it has the room; one that added fields of its own keeps the
/// array beside the block instead, which is what
/// [`pyobject::items_or_build`] answers with.  The flag is what makes the
/// layout a tuple's rather than that of any other type sized like one.
fn carries_items(ob_type: *mut CPyTypeObject) -> bool {
    if ob_type.is_null() {
        return false;
    }
    let flags = unsafe { (*ob_type).tp_flags };
    flags & super::typeobject::PY_TPFLAGS_TUPLE_SUBCLASS != 0
        && unsafe { (*ob_type).tp_basicsize } == size_of::<CPyVarObject>() as isize
}

/// `pyobject.py:96-100 allocate`'s `size += itemcount * itemsize`, for the one
/// type whose items live in its block.
pub(super) fn item_bytes(w_obj: PyObjectRef, ob_type: *mut CPyTypeObject) -> usize {
    if !carries_items(ob_type) || unsafe { !pyre_object::is_tuple(w_obj) } {
        return 0;
    }
    unsafe { pyre_object::tupleobject::w_tuple_len(w_obj) }
        .saturating_mul(size_of::<*mut CPyObject>())
}

/// Where `raw`'s items sit, for a block that carries them.
///
/// The count is `ob_size`, which [`super::typeobject::stamp_ob_size`] filled
/// from the same length [`item_bytes`] sized the block by.
fn items_in_block(raw: *mut CPyObject) -> Option<(*mut *mut CPyObject, usize)> {
    if raw.is_null() || !carries_items(unsafe { (*raw).ob_type }) {
        return None;
    }
    let length = unsafe { (*(raw as *mut CPyVarObject)).ob_size }.max(0) as usize;
    // Reached from the block's base rather than through the declared slot: an
    // empty tuple's block ends where its array starts, and that address is
    // still the one a reader is handed.
    let items = unsafe { raw.byte_add(size_of::<CPyVarObject>()) } as *mut *mut CPyObject;
    Some((items, length))
}

/// Give the block the items it does not yet hold --
/// `tupleobject.py:70-96 tuple_attach`, run on the first read rather than when
/// the mirror is built.
///
/// Upstream fills the array before `track_reference` publishes the mirror, so
/// nothing can reach a half-filled one.  Here the link is created first, and
/// `make_ref` below reaches `ensure_mirror` again -- for a type, all the way
/// into `finish_interpreter_type`, which walks the very MRO tuple being filled.
/// Filling on demand keeps that recursion out of mirror creation, and a NUL
/// slot is what marks one still owed: a slot holding a genuine NUL resolves to
/// NUL again, so a repeated pass is the same answer rather than a second
/// reference.
fn fill_missing(raw: *mut CPyObject, items: *mut *mut CPyObject, length: usize) {
    for index in 0..length {
        if !unsafe { items.add(index).read() }.is_null() {
            continue;
        }
        // Read the tuple back through the mirror before each `make_ref`: that
        // call allocates, and the tuple moves under a collection where the
        // block does not.
        let w_tuple = unsafe { pyobject::from_ref(raw) };
        if w_tuple.is_null() || unsafe { !pyre_object::is_tuple(w_tuple) } {
            return;
        }
        let item = unsafe { pyre_object::tupleobject::w_tuple_getitem(w_tuple, index as i64) };
        let Some(item) = item else { continue };
        let entry = pyobject::make_ref(item);
        // Written only if the slot is still owed: the `make_ref` above can
        // reach a reader that filled it, and two references would be one more
        // than the block gives back.
        match unsafe { items.add(index).read() }.is_null() {
            true => unsafe { items.add(index).write(entry) },
            false => unsafe { pyobject::decref(entry) },
        }
    }
}

/// Give back the references a dying tuple mirror's items owned --
/// `tupleobject.py tuple_dealloc`.
pub(super) fn forget_block(raw: *mut CPyObject) {
    let Some((items, length)) = items_in_block(raw) else {
        return;
    };
    for index in 0..length {
        // Cleared before the release: a deallocator this runs can reach the
        // same block, and has to find a slot it will not release twice.
        let entry = unsafe { items.add(index).replace(std::ptr::null_mut()) };
        unsafe { pyobject::decref(entry) };
    }
}

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

/// The tuple `object` stands for, or nothing -- with nothing said about why.
///
/// For what stands in for a macro, which reports no error and is not written
/// to be checked for one: setting one would leave it pending for whatever
/// entry point the extension reaches next.
fn tuple_value(object: *mut CPyObject) -> Option<PyObjectRef> {
    let value = unsafe { pyobject::from_ref(object) };
    (!value.is_null() && unsafe { pyre_object::is_tuple(value) }).then_some(value)
}

/// Whether a write can reach `value`'s items.
///
/// The arity-2 specialisations of `specialisedtupleobject.py` keep their two
/// in fields of their own rather than in a block; a general tuple carries the
/// block at the one offset, a subclass included.
fn is_array_backed(value: PyObjectRef) -> bool {
    use pyre_object::specialisedtupleobject::{
        is_specialised_tuple_ff, is_specialised_tuple_ii, is_specialised_tuple_oo,
    };
    unsafe {
        !is_specialised_tuple_ii(value)
            && !is_specialised_tuple_ff(value)
            && !is_specialised_tuple_oo(value)
    }
}

/// `PyTuple_FromArray(array, size)` — the tuple a C array of references
/// makes, as a new reference.  The array's own references are left alone.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTuple_FromArray(
    array: *const *mut CPyObject,
    size: isize,
) -> *mut CPyObject {
    if size < 0 || (array.is_null() && size != 0) {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    unsafe { super::object::tuple_from_vector(array, size as usize) }
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

/// The item array a tuple mirror hands out — `tupleobject.py` gives its mirror
/// an `ob_item` of `ob_size` `PyObject *`, and `PyTuple_GET_ITEM` reads it.
///
/// Built on first ask and kept: what a tuple holds does not change, save
/// through [`PyTuple_SetItem`] and [`_PyTuple_SET_ITEM`], which write the slot
/// they changed.  The entries are references the array owns, so a slot read
/// through it is good for as long as the mirror is, and they go back when the
/// mirror dies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyTuple_ITEMS(object: *mut CPyObject) -> *mut *mut CPyObject {
    let Some(value) = tuple_value(object) else {
        return std::ptr::null_mut();
    };
    if let Some((items, length)) = items_in_block(object) {
        fill_missing(object, items, length);
        return items;
    }
    pyobject::items_or_build(object, || {
        let length = unsafe { pyre_object::tupleobject::w_tuple_len(value) };
        (0..length)
            .map(|index| {
                // Read immediately before `make_ref`, which allocates: the
                // tuple is reached through the mirror's link, which the
                // collector keeps current.
                let w_tuple = unsafe { pyobject::from_ref(object) };
                let item =
                    unsafe { pyre_object::tupleobject::w_tuple_getitem(w_tuple, index as i64) };
                match item {
                    Some(item) => pyobject::make_ref(item) as usize,
                    None => 0,
                }
            })
            .collect()
    })
}

/// Put `item` in slot `index` of whichever array `object` hands out, answering
/// with what it displaced.
///
/// A caller holding the address `PyTuple_GET_ITEM` gave it reads the new value
/// through it, so the slot is written wherever the array lives.
fn replace_item(
    object: *mut CPyObject,
    index: usize,
    item: *mut CPyObject,
) -> Option<*mut CPyObject> {
    let Some((items, length)) = items_in_block(object) else {
        return pyobject::replace_cached_item(object, index, item);
    };
    if index >= length {
        return None;
    }
    Some(unsafe { items.add(index).replace(item) })
}

/// `PyTuple_SET_ITEM(op, i, v)` -- slot `i` is given `v`, whatever it held
/// before, and `v`'s reference goes with it.
///
/// The macro's assignment, which is not [`PyTuple_SetItem`]: it does not give
/// back what the slot held, and an extension counts on that.  cffi's
/// `ffi_obj.c _ffi_callback_decorator` reads a slot, puts a borrowed function
/// in its place for the length of one call, and puts the old value back --
/// neither write owns what it stores, and the ledger balances only because
/// neither release happens.
///
/// Silent about a receiver it cannot serve, as the macro is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyTuple_SET_ITEM(
    object: *mut CPyObject,
    index: isize,
    item: *mut CPyObject,
) {
    super::object::realize_all([object, item]);
    if unsafe { _PyTuple_ITEMS(object) }.is_null() || index < 0 {
        return;
    }
    let Some(value) = tuple_value(object) else {
        return;
    };
    if !is_array_backed(value) {
        return;
    }
    let w_item = unsafe { pyobject::from_ref(item) };
    let written = unsafe {
        pyre_object::tupleobject::w_tuple_setitem_unchecked(value, index as usize, w_item)
    };
    if written.is_some() {
        // What comes back is dropped rather than released: see above.
        replace_item(object, index as usize, item);
    }
}

/// Steals a reference to `item`, gives back the one the slot held, and is only
/// defined on a tuple no other code has seen yet — the contract
/// `PyTuple_SetItem` documents.
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
    let stored = is_array_backed(value)
        .then(|| unsafe {
            pyre_object::tupleobject::w_tuple_setitem_unchecked(value, index as usize, w_item)
        })
        .flatten();
    if stored.is_none() {
        unsafe { pyobject::decref(item) };
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyTuple_SetItem(): the tuple is not one PyTuple_New built",
        ));
        return -1;
    }
    // A caller holding the address `PyTuple_GET_ITEM` gave it reads the new
    // value through it, so the slot is written rather than the array rebuilt,
    // and the stolen reference is what the array is given.  Where there is no
    // array to take it, it is released here instead.
    match replace_item(object, index as usize, item) {
        Some(previous) => unsafe { pyobject::decref(previous) },
        None => unsafe { pyobject::decref(item) },
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
    std::hint::black_box(_PyTuple_ITEMS as *const ());
    std::hint::black_box(PyTuple_SetItem as *const ());
    std::hint::black_box(PyTuple_GetSlice as *const ());
    std::hint::black_box(PyTuple_Check as *const ());
    std::hint::black_box(PyTuple_CheckExact as *const ());
}
