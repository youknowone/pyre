//! The set protocol -- PyPy `cpyext/setobject.py`.
//!
//! Each entry point goes through the interpreter's own `set` method of the
//! same name, so a subclass that overrides one is honoured and either operand
//! may be a pyre object.

use super::object::{argument, arguments, result};
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::c_int;

/// Call `name` on `receiver`, as upstream's `space.call_method` does.
fn method(
    receiver: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(receiver);
    for &argument in args {
        roots.pin_root(argument);
    }
    let bound =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(base), name)?;
    let bound_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(bound);
    let args: Vec<PyObjectRef> = (0..args.len())
        .map(|index| pyre_object::gc_roots::shadow_stack_get(base + 1 + index))
        .collect();
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(bound_slot),
        &args,
    )
}

/// Reject a receiver that is not a `set` or `frozenset`, as upstream's
/// `PySet_Check` guard does (`setobject.py:86-89`).
fn any_set(object: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::setobject::is_set_or_frozenset(object) } {
        return Ok(object);
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::SystemError,
        format!("{name}() argument is not a set"),
    ))
}

/// `PySet_New(iterable)` — `set(iterable)`, or an empty set for NULL
/// (`setobject.py:63-72`).
///
/// # Safety
/// `iterable` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySet_New(iterable: *mut CPyObject) -> *mut CPyObject {
    let iterable = unsafe { pyobject::from_ref(iterable) };
    if iterable.is_null() {
        return pyobject::make_ref(pyre_object::setobject::w_set_new());
    }
    let w_set = crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
    if w_set.is_null() {
        return std::ptr::null_mut();
    }
    result(crate::call::call_function_impl_result(w_set, &[iterable]))
}

/// `PyFrozenSet_New(iterable)` (`setobject.py`).
///
/// # Safety
/// `iterable` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFrozenSet_New(iterable: *mut CPyObject) -> *mut CPyObject {
    let iterable = unsafe { pyobject::from_ref(iterable) };
    if iterable.is_null() {
        return pyobject::make_ref(pyre_object::setobject::w_frozenset_new());
    }
    let w_frozenset = crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE);
    if w_frozenset.is_null() {
        return std::ptr::null_mut();
    }
    result(crate::call::call_function_impl_result(
        w_frozenset,
        &[iterable],
    ))
}

/// Store into a frozenset still being built — `setobject.py:647`
/// `W_FrozensetObject.cpyext_add_frozen`.
///
/// The element is hashed while both operands are rooted, because a user
/// `__hash__` is a collection point, and the digest keys the store.
fn add_frozen(set: PyObjectRef, key: PyObjectRef) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(set);
    roots.pin_root(key);
    let hash = crate::builtins::try_hash_value(pyre_object::gc_roots::shadow_stack_get(base + 1))
        .map_err(|error| {
        crate::baseobjspace::wrap_set_element_hash_error(
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            error,
        )
    })?;
    unsafe {
        pyre_object::setobject::w_set_add_hashed_checked(
            pyre_object::gc_roots::shadow_stack_get(base),
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            hash,
        )
    }
    .map_err(crate::baseobjspace::map_set_update_error)
}

/// `PySet_Add(set, key)` (`setobject.py`).
///
/// A frozenset is accepted while it is still being filled — the documented way
/// to build one before it is exposed.  "Still being filled" is "has not
/// answered `__hash__` yet": once the hash is cached, the value has been
/// published and a store would invalidate it.
///
/// # Safety
/// Both arguments must be live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySet_Add(set: *mut CPyObject, key: *mut CPyObject) -> c_int {
    let Some([set, key]) = arguments([set, key]) else {
        return -1;
    };
    let call = if unsafe { pyre_object::setobject::is_set(set) } {
        method(set, "add", &[key])
    } else if unsafe {
        pyre_object::setobject::is_frozenset(set)
            && pyre_object::setobject::w_frozenset_cached_hash(set).is_none()
    } {
        add_frozen(set, key).map(|()| pyre_object::w_none())
    } else {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    };
    match super::pyerrors::trap(call) {
        Some(_) => 0,
        None => -1,
    }
}

/// `PySet_Discard(set, key)` — 1 removed, 0 absent, -1 on error.  A missing
/// key is not an error, so the `KeyError` `remove` raises is turned into 0
/// (`setobject.py:94-110`).
///
/// # Safety
/// Both arguments must be live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySet_Discard(set: *mut CPyObject, key: *mut CPyObject) -> c_int {
    let Some([set, key]) = arguments([set, key]) else {
        return -1;
    };
    let call = any_set(set, "PySet_Discard").and_then(|set| method(set, "remove", &[key]));
    match call {
        Ok(_) => 1,
        Err(error) if error.kind == crate::PyErrorKind::KeyError => 0,
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            -1
        }
    }
}

/// `PySet_Pop(set)` (`setobject.py`).
///
/// # Safety
/// `set` must be a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySet_Pop(set: *mut CPyObject) -> *mut CPyObject {
    let Some(set) = argument(set) else {
        return std::ptr::null_mut();
    };
    result(any_set(set, "PySet_Pop").and_then(|set| method(set, "pop", &[])))
}

/// `PySet_Clear(set)` (`setobject.py`).
///
/// # Safety
/// `set` must be a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySet_Clear(set: *mut CPyObject) -> c_int {
    let Some(set) = argument(set) else {
        return -1;
    };
    let call = any_set(set, "PySet_Clear").and_then(|set| method(set, "clear", &[]));
    match super::pyerrors::trap(call) {
        Some(_) => 0,
        None => -1,
    }
}

/// `PySet_Size(anyset)` (`setobject.py`).
///
/// # Safety
/// `set` must be a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySet_Size(set: *mut CPyObject) -> isize {
    let Some(set) = argument(set) else {
        return -1;
    };
    match super::pyerrors::trap(any_set(set, "PySet_Size")) {
        Some(set) => (unsafe { pyre_object::setobject::w_set_len(set) }) as isize,
        None => -1,
    }
}

/// `PySet_Contains(anyset, key)` — 1 found, 0 not, -1 on error
/// (`setobject.py:140-147`).
///
/// # Safety
/// Both arguments must be live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySet_Contains(set: *mut CPyObject, key: *mut CPyObject) -> c_int {
    let Some([set, key]) = arguments([set, key]) else {
        return -1;
    };
    let call = any_set(set, "PySet_Contains")
        .and_then(|set| crate::baseobjspace::contains(set, key).map_err(Into::into));
    match super::pyerrors::trap(call) {
        Some(found) => found as c_int,
        None => -1,
    }
}

/// `PySet_Check(ob)` — a `set` or a subclass of one.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PySet_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::setobject::is_set(object) }) as c_int
}

/// `PyFrozenSet_Check(ob)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFrozenSet_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::setobject::is_frozenset(object) }) as c_int
}

/// `PyAnySet_Check(ob)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyAnySet_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::setobject::is_set_or_frozenset(object) }) as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PySet_New as *const ());
    std::hint::black_box(PyFrozenSet_New as *const ());
    std::hint::black_box(PySet_Add as *const ());
    std::hint::black_box(PySet_Discard as *const ());
    std::hint::black_box(PySet_Pop as *const ());
    std::hint::black_box(PySet_Clear as *const ());
    std::hint::black_box(PySet_Size as *const ());
    std::hint::black_box(PySet_Contains as *const ());
    std::hint::black_box(PySet_Check as *const ());
    std::hint::black_box(PyFrozenSet_Check as *const ());
    std::hint::black_box(PyAnySet_Check as *const ());
}
