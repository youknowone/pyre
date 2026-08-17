//! `weakref` -- PyPy `cpyext/weakrefobject.py`.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};
use crate::module::_weakref::interp__weakref;
use pyre_object::PyObjectRef;
use std::ffi::c_int;

/// Whether `object` is an instance of one of the weak-reference types.
fn is_a(object: PyObjectRef, kind: PyObjectRef) -> bool {
    !object.is_null() && unsafe { crate::baseobjspace::isinstance_w(object, kind) }
}

/// `weakrefobject.py:65 PyWeakref_CheckRef`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_CheckRef(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    is_a(object, interp__weakref::weakref_type()) as c_int
}

/// `weakrefobject.py:75 PyWeakref_CheckRefExact`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_CheckRefExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return 0;
    }
    let own = crate::typedef::r#type(object).map_or(pyre_object::PY_NULL, |p| p.as_ptr());
    (own == interp__weakref::weakref_type()) as c_int
}

/// `weakrefobject.py:82 PyWeakref_CheckProxy` — either proxy type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_CheckProxy(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (is_a(object, interp__weakref::proxy_type())
        || is_a(object, interp__weakref::callable_proxy_type())) as c_int
}

/// `weakrefobject.py:91 PyWeakref_Check` — a reference or a proxy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_Check(object: *mut CPyObject) -> c_int {
    (unsafe { PyWeakref_CheckRef(object) } != 0 || unsafe { PyWeakref_CheckProxy(object) } != 0)
        as c_int
}

/// The referent of a reference or a proxy, or NULL with a `TypeError` recorded
/// when `object` is neither.
///
/// Both kinds keep the referent in the same `W_WeakrefBase` slot, which is what
/// [`interp__weakref::dereference`] reads, so neither has to be told apart here
/// beyond refusing what is not one of them.
fn referent_of(object: *mut CPyObject, function: &str) -> Option<PyObjectRef> {
    let value = argument(object)?;
    if unsafe { PyWeakref_Check(object) } == 0 {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): expected a weakref"
        )));
        return None;
    }
    Some(interp__weakref::dereference(value))
}

/// `weakrefobject.py:6 PyWeakref_NewRef(ob, callback)`.
///
/// A NULL callback is the "no callback" spelling, which is the one-argument
/// call rather than one passing `None`: `ref(ob, None)` is a reference with a
/// callback that happens to be `None`, and upstream's `space.call_function`
/// with a null `w_callback` passes no second argument at all.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_NewRef(
    object: *mut CPyObject,
    callback: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([object, callback]);
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let callback = unsafe { pyobject::from_ref(callback) };
    let arguments: &[PyObjectRef] = if callback.is_null() {
        &[object]
    } else {
        &[object, callback]
    };
    result(crate::call::call_function_impl_result(
        interp__weakref::weakref_type(),
        arguments,
    ))
}

/// `weakrefobject.py:19 PyWeakref_NewProxy(ob, callback)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_NewProxy(
    object: *mut CPyObject,
    callback: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([object, callback]);
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let callback = unsafe { pyobject::from_ref(callback) };
    let arguments: Vec<PyObjectRef> = if callback.is_null() {
        vec![object]
    } else {
        vec![object, callback]
    };
    result(interp__weakref::proxy(&arguments))
}

/// `weakrefobject.py:33 PyWeakref_GetObject` — borrowed, and `None` rather
/// than NULL once the referent is gone.
///
/// Borrowed here is [`pyobject::borrow_mirror`] and not the reference a
/// container owns on an item's behalf: that one lives until the container's
/// mirror dies, which for a weak reference is exactly the object it must not
/// keep alive.  What is answered instead lives as long as the referent does,
/// and no longer — the hazard this entry point was deprecated for, and the
/// reason [`PyWeakref_GetRef`] exists.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_GetObject(reference: *mut CPyObject) -> *mut CPyObject {
    let Some(referent) = referent_of(reference, "PyWeakref_GetObject") else {
        return std::ptr::null_mut();
    };
    let referent = if referent.is_null() {
        pyre_object::w_none()
    } else {
        referent
    };
    pyobject::borrow_mirror(referent)
}

/// `PyWeakref_GetRef(ref, &result)` — 1 with a new reference in `result` while
/// the referent is alive, 0 with NULL once it is not, -1 on a bad argument.
///
/// The spelling that replaces [`PyWeakref_GetObject`], whose borrowed answer
/// can be freed by the collection its own caller triggers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_GetRef(
    reference: *mut CPyObject,
    into: *mut *mut CPyObject,
) -> c_int {
    if !into.is_null() {
        unsafe { *into = std::ptr::null_mut() };
    }
    let Some(referent) = referent_of(reference, "PyWeakref_GetRef") else {
        return -1;
    };
    if referent.is_null() {
        return 0;
    }
    if !into.is_null() {
        unsafe { *into = pyobject::make_ref(referent) };
    }
    1
}

/// `PyWeakref_IsDead(ref)` — 1 once the referent is gone, 0 while it is alive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyWeakref_IsDead(reference: *mut CPyObject) -> c_int {
    let Some(referent) = referent_of(reference, "PyWeakref_IsDead") else {
        return -1;
    };
    referent.is_null() as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyWeakref_Check as *const ());
    std::hint::black_box(PyWeakref_CheckRef as *const ());
    std::hint::black_box(PyWeakref_CheckRefExact as *const ());
    std::hint::black_box(PyWeakref_CheckProxy as *const ());
    std::hint::black_box(PyWeakref_NewRef as *const ());
    std::hint::black_box(PyWeakref_NewProxy as *const ());
    std::hint::black_box(PyWeakref_GetObject as *const ());
    std::hint::black_box(PyWeakref_GetRef as *const ());
    std::hint::black_box(PyWeakref_IsDead as *const ());
}
