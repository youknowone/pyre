//! `dict` -- PyPy `cpyext/dictobject.py`.

use super::object::{argument, arguments};
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
    super::object::realize_all([object, key, value]);
    let Some(dict) = dict_argument(object, "PyDict_SetItem") else {
        return -1;
    };
    let Some([key, value]) = arguments([key, value]) else {
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
    super::object::realize_all([object, value]);
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
    super::object::realize_all([object, key]);
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
    super::object::realize_all([object, key]);
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
    super::object::realize_all([object, key]);
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
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::DICT_TYPE)) as c_int
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

/// `obj.keys()`.
fn call_keys(object: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let method = crate::baseobjspace::getattr_str(object, "keys")?;
    crate::call::call_function_impl_result(method, &[])
}

/// Every item an iterable yields.
fn unpack_all(object: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    crate::baseobjspace::unpackiterable(object, -1)
}

/// `PyDict_Clear(dict)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Clear(object: *mut CPyObject) {
    let Some(dict) = dict_argument(object, "PyDict_Clear") else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return;
    };
    unsafe { pyre_object::dictmultiobject::w_dict_clear(dict) };
}

/// `PyDict_Copy(dict)` — a new `dict` with the same items.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Copy(object: *mut CPyObject) -> *mut CPyObject {
    let Some(dict) = dict_argument(object, "PyDict_Copy") else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(unsafe { pyre_object::dictmultiobject::w_dict_copy(dict) })
}

/// `PyDict_DelItemString(dict, key)`.
///
/// # Safety
/// `key` must be a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_DelItemString(object: *mut CPyObject, key: *const c_char) -> c_int {
    let (Some(dict), Some(name)) = (dict_argument(object, "PyDict_DelItemString"), key_name(key))
    else {
        return -1;
    };
    let w_key = pyre_object::w_str_new(&name);
    match trap(crate::baseobjspace::delitem(dict, w_key)) {
        Some(()) => 0,
        None => -1,
    }
}

/// `PyDict_GetItemWithError(dict, key)` — a borrowed reference, or NULL with
/// no exception set when the key is simply absent (`dictobject.py:106-118`).
///
/// # Safety
/// Both arguments must be null or live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItemWithError(
    object: *mut CPyObject,
    key: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([object, key]);
    let (Some(dict), Some(key)) = (
        dict_argument(object, "PyDict_GetItemWithError"),
        argument(key),
    ) else {
        return std::ptr::null_mut();
    };
    match crate::baseobjspace::getitem(dict, key) {
        Ok(value) => pyobject::borrow_from(object, value),
        Err(error) if error.kind == crate::PyErrorKind::KeyError => std::ptr::null_mut(),
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            std::ptr::null_mut()
        }
    }
}

/// Fill `*out` with a *strong* reference to `dict[key]`, answering 1 when the
/// key was there, 0 when it was not and -1 on error.
///
/// # Safety
/// `out` must be null or a writable `PyObject *`.
unsafe fn get_item_ref(
    found: Result<Option<PyObjectRef>, crate::PyError>,
    out: *mut *mut CPyObject,
) -> c_int {
    if !out.is_null() {
        unsafe { *out = std::ptr::null_mut() };
    }
    match found {
        Ok(None) => 0,
        Ok(Some(value)) => {
            if !out.is_null() {
                unsafe { *out = pyobject::make_ref(value) };
            }
            1
        }
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            -1
        }
    }
}

/// `PyDict_GetItemRef(dict, key, &result)`.
///
/// # Safety
/// `result` must be null or a writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItemRef(
    object: *mut CPyObject,
    key: *mut CPyObject,
    result: *mut *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, key]);
    let (Some(dict), Some(key)) = (dict_argument(object, "PyDict_GetItemRef"), argument(key))
    else {
        return -1;
    };
    let found = match crate::baseobjspace::getitem(dict, key) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind == crate::PyErrorKind::KeyError => Ok(None),
        Err(error) => Err(error),
    };
    unsafe { get_item_ref(found, result) }
}

/// `PyDict_GetItemStringRef(dict, key, &result)`.
///
/// # Safety
/// `key` must be NUL-terminated and `result` null or a writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItemStringRef(
    object: *mut CPyObject,
    key: *const c_char,
    result: *mut *mut CPyObject,
) -> c_int {
    let (Some(dict), Some(name)) = (
        dict_argument(object, "PyDict_GetItemStringRef"),
        key_name(key),
    ) else {
        return -1;
    };
    let found = Ok(unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, &name) });
    unsafe { get_item_ref(found, result) }
}

/// Build one of the three view lists.  Upstream answers a `list` rather than a
/// view object (`dictobject.py:180-206`), the C caller expecting something it
/// can index.
fn view_list(
    object: *mut CPyObject,
    name: &str,
    part: fn(PyObjectRef) -> Vec<PyObjectRef>,
) -> *mut CPyObject {
    let Some(dict) = dict_argument(object, name) else {
        return std::ptr::null_mut();
    };
    let roots = pyre_object::gc_roots::push_roots();
    roots.pin_root(dict);
    pyobject::make_ref(pyre_object::listobject::w_list_new(part(dict)))
}

/// `PyDict_Keys(dict)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Keys(object: *mut CPyObject) -> *mut CPyObject {
    view_list(object, "PyDict_Keys", |dict| {
        unsafe { pyre_object::dictmultiobject::w_dict_items(dict) }
            .into_iter()
            .map(|(key, _)| key)
            .collect()
    })
}

/// `PyDict_Values(dict)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Values(object: *mut CPyObject) -> *mut CPyObject {
    view_list(object, "PyDict_Values", |dict| {
        unsafe { pyre_object::dictmultiobject::w_dict_items(dict) }
            .into_iter()
            .map(|(_, value)| value)
            .collect()
    })
}

/// `PyDict_Items(dict)` — a list of `(key, value)` pairs.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Items(object: *mut CPyObject) -> *mut CPyObject {
    view_list(object, "PyDict_Items", |dict| {
        unsafe { pyre_object::dictmultiobject::w_dict_items(dict) }
            .into_iter()
            .map(|(key, value)| pyre_object::tupleobject::w_tuple_new(vec![key, value]))
            .collect()
    })
}

/// `PyDict_Merge(a, b, override)` — copy `b`'s items into `a`, replacing an
/// existing key only when `over` is non-zero.
///
/// # Safety
/// Both arguments must be null or live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Merge(
    into: *mut CPyObject,
    from: *mut CPyObject,
    over: c_int,
) -> c_int {
    super::object::realize_all([into, from]);
    let (Some(target), Some(source)) = (dict_argument(into, "PyDict_Merge"), argument(from)) else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(target);
    roots.pin_root(source);
    let target = || pyre_object::gc_roots::shadow_stack_get(base);
    let source_now = pyre_object::gc_roots::shadow_stack_get(base + 1);

    // The pairs go on the shadow stack rather than staying in the vector they
    // arrive in: `contains` and `setitem` below run the target's own
    // `__contains__` / `__setitem__` and re-hash the key, so either can
    // collect, and a `Vec` is not somewhere the collector looks.  The
    // containers are already re-read per iteration for exactly that reason.
    //
    // A mapping that is not a dict is read through `keys()`, as upstream does
    // for the non-dict case (`dictobject.py:214-231`).
    let (pairs, count) = if unsafe { pyre_object::is_dict(source_now) } {
        let items: Vec<(PyObjectRef, PyObjectRef)> =
            unsafe { pyre_object::dictmultiobject::w_dict_items(source_now) };
        let flat: Vec<PyObjectRef> = items
            .iter()
            .flat_map(|&(key, value)| [key, value])
            .collect();
        (pyre_object::gc_roots::pin_roots(&flat), items.len())
    } else {
        let keys = match trap(call_keys(source_now)) {
            Some(keys) => keys,
            None => return -1,
        };
        let keys_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(keys);
        let keys = match trap(unpack_all(pyre_object::gc_roots::shadow_stack_get(
            keys_slot,
        ))) {
            Some(keys) => keys,
            None => return -1,
        };
        // `getitem` runs the mapping's own `__getitem__`, so the keys are read
        // back per iteration and each pair is published as it is produced.
        let unpacked = pyre_object::gc_roots::pin_roots(&keys);
        let pairs = pyre_object::gc_roots::shadow_stack_len();
        for index in 0..keys.len() {
            let value = match trap(crate::baseobjspace::getitem(
                pyre_object::gc_roots::shadow_stack_get(base + 1),
                pyre_object::gc_roots::shadow_stack_get(unpacked + index),
            )) {
                Some(value) => value,
                None => return -1,
            };
            roots.pin_root(pyre_object::gc_roots::shadow_stack_get(unpacked + index));
            roots.pin_root(value);
        }
        (pairs, keys.len())
    };

    for index in 0..count {
        if over == 0 {
            match crate::baseobjspace::contains(
                target(),
                pyre_object::gc_roots::shadow_stack_get(pairs + index * 2),
            ) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(error) => {
                    super::pyerrors::set_pending_error(error);
                    return -1;
                }
            }
        }
        let key = pyre_object::gc_roots::shadow_stack_get(pairs + index * 2);
        let value = pyre_object::gc_roots::shadow_stack_get(pairs + index * 2 + 1);
        if trap(crate::baseobjspace::setitem(target(), key, value)).is_none() {
            return -1;
        }
    }
    0
}

/// `PyDict_Update(a, b)` — `PyDict_Merge(a, b, 1)` (`dictobject.py:233-238`).
///
/// # Safety
/// Both arguments must be null or live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Update(into: *mut CPyObject, from: *mut CPyObject) -> c_int {
    unsafe { PyDict_Merge(into, from, 1) }
}

/// `PyDict_MergeFromSeq2(a, seq2, override)` — `seq2` yields `(key, value)`
/// pairs.
///
/// # Safety
/// Both arguments must be null or live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_MergeFromSeq2(
    into: *mut CPyObject,
    seq2: *mut CPyObject,
    over: c_int,
) -> c_int {
    super::object::realize_all([into, seq2]);
    let (Some(target), Some(sequence)) =
        (dict_argument(into, "PyDict_MergeFromSeq2"), argument(seq2))
    else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(target);
    roots.pin_root(sequence);
    let target = || pyre_object::gc_roots::shadow_stack_get(base);

    let Some(pairs) = trap(unpack_all(pyre_object::gc_roots::shadow_stack_get(
        base + 1,
    ))) else {
        return -1;
    };
    // The unpacked sequence and each pair's two elements are read back out of
    // the shadow stack, not carried in their vectors: `unpack_all`, `contains`
    // and `setitem` all run user code, and a `Vec` is not somewhere the
    // collector looks.  Each iteration opens its own scope so the pushes it
    // makes are popped again rather than growing with the sequence.
    let unpacked = pyre_object::gc_roots::pin_roots(&pairs);
    for index in 0..pairs.len() {
        let pair_roots = pyre_object::gc_roots::push_roots();
        let pair_slot = pyre_object::gc_roots::shadow_stack_len();
        pair_roots.pin_root(pyre_object::gc_roots::shadow_stack_get(unpacked + index));
        let Some(items) = trap(unpack_all(pyre_object::gc_roots::shadow_stack_get(
            pair_slot,
        ))) else {
            return -1;
        };
        if items.len() != 2 {
            super::pyerrors::set_pending_error(crate::PyError::value_error(format!(
                "dictionary update sequence element #{index} has length {}; 2 is required",
                items.len()
            )));
            return -1;
        }
        let element = pyre_object::gc_roots::pin_roots(&items);
        if over == 0 {
            match crate::baseobjspace::contains(
                target(),
                pyre_object::gc_roots::shadow_stack_get(element),
            ) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(error) => {
                    super::pyerrors::set_pending_error(error);
                    return -1;
                }
            }
        }
        let key = pyre_object::gc_roots::shadow_stack_get(element);
        let value = pyre_object::gc_roots::shadow_stack_get(element + 1);
        if trap(crate::baseobjspace::setitem(target(), key, value)).is_none() {
            return -1;
        }
    }
    0
}

/// The key snapshot `PyDict_Next` walks, one per dictionary being iterated.
///
/// Upstream keeps it in the mirror's own `_tmpkeys` field, holding a strong
/// reference so the keys stay alive and the collector keeps forwarding them
/// (`dictobject.py:301-311`).  Pyre's mirror has no such field, so the
/// reference is held here instead, keyed by the mirror's address; it is the
/// reference, not this table, that roots the list.
type SnapshotMap = std::collections::HashMap<
    usize,
    usize,
    std::hash::BuildHasherDefault<std::hash::DefaultHasher>,
>;
static ITERATION_KEYS: super::ForkMutex<SnapshotMap> = super::ForkMutex::new(
    SnapshotMap::with_hasher(std::hash::BuildHasherDefault::new()),
);

/// `PyDict_Next(dict, &pos, &key, &value)` — 1 for each pair, 0 once they are
/// all reported.  `pos` starts at 0 and is only ever handed back
/// (`dictobject.py:255-329`).
///
/// The keys are snapshotted on the first call, so a value reassigned during
/// the walk is seen but a new key is not -- which is the documented contract.
///
/// # Safety
/// `pos` must be a writable `Py_ssize_t` the caller initialised to 0, and
/// `key` / `value` must be null or writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Next(
    object: *mut CPyObject,
    pos: *mut isize,
    key: *mut *mut CPyObject,
    value: *mut *mut CPyObject,
) -> c_int {
    let Some(dict) = dict_argument(object, "PyDict_Next") else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return 0;
    };
    if pos.is_null() {
        return 0;
    }
    let index = unsafe { *pos };

    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(dict);

    let snapshot = if index == 0 {
        let keys: Vec<PyObjectRef> = unsafe {
            pyre_object::dictmultiobject::w_dict_items(pyre_object::gc_roots::shadow_stack_get(
                dict_slot,
            ))
        }
        .into_iter()
        .map(|(key, _)| key)
        .collect();
        let held = pyobject::make_ref(pyre_object::listobject::w_list_new(keys));
        let previous = ITERATION_KEYS.lock().insert(object as usize, held as usize);
        if let Some(previous) = previous {
            unsafe { pyobject::decref(previous as *mut CPyObject) };
        }
        held
    } else {
        match ITERATION_KEYS.lock().get(&(object as usize)) {
            // `pos` should have been 0; there is nothing to walk.
            None => return 0,
            Some(&held) => held as *mut CPyObject,
        }
    };

    let w_keys = unsafe { pyobject::from_ref(snapshot) };
    let length = unsafe { pyre_object::listobject::w_list_len(w_keys) } as isize;
    if index >= length {
        if let Some(held) = ITERATION_KEYS.lock().remove(&(object as usize)) {
            unsafe { pyobject::decref(held as *mut CPyObject) };
        }
        return 0;
    }

    let Some(w_key) = (unsafe { pyre_object::listobject::w_list_getitem(w_keys, index as i64) })
    else {
        return 0;
    };
    // The lookup below can collect, so the key is pinned first and both it and
    // the value are read back from the shadow stack afterwards.
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_key);
    let Some(w_value) = trap(crate::baseobjspace::getitem(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
    )) else {
        return 0;
    };
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_value);
    unsafe { *pos = index + 1 };
    if !key.is_null() {
        unsafe {
            *key = pyobject::borrow_from(object, pyre_object::gc_roots::shadow_stack_get(key_slot))
        };
    }
    if !value.is_null() {
        unsafe {
            *value =
                pyobject::borrow_from(object, pyre_object::gc_roots::shadow_stack_get(value_slot))
        };
    }
    1
}

/// Drop the snapshot a `PyDict_Next` walk left behind.
///
/// A caller that stops walking before the end never reaches the arm that frees
/// it, so the mirror's own deallocation is the other end of that lifetime.
pub(super) fn forget_iteration(mirror: usize) {
    if let Some(held) = ITERATION_KEYS.lock().remove(&mirror) {
        unsafe { pyobject::decref(held as *mut CPyObject) };
    }
}
