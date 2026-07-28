//! Shared `__slots__` storage helpers for native subclasses (float, complex).
//!
//! Each host object stores its `__slots__` values in a list held by a
//! `w_slots: PyObjectRef` field.  Callers pass an accessor that yields that
//! field's address from the object pointer, so a single implementation serves
//! every layout without duplicating the GC-rooting dance.

use crate::pyobject::*;

/// Field accessor: given an object pointer, return the address of its
/// `w_slots` field.
///
/// # Safety
/// The accessor is called only with a pointer to the concrete host struct it
/// was written for.
pub type SlotsField = unsafe fn(PyObjectRef) -> *mut PyObjectRef;

/// Read slot `index`, or `None` when unset (null list or null entry).
///
/// # Safety
/// `obj` must be a valid pointer to the host struct `slots_field` targets.
pub unsafe fn slot_get(
    obj: PyObjectRef,
    index: usize,
    slots_field: SlotsField,
) -> Option<PyObjectRef> {
    let slots = unsafe { *slots_field(obj) };
    if slots.is_null() {
        return None;
    }
    unsafe { crate::listobject::w_list_getitem(slots, index as i64) }.filter(|value| !value.is_null())
}

/// Store `value` into slot `index`, growing (or creating) the backing list as
/// needed.  Creating or growing the list allocates, and `w_list_new` /
/// `w_list_append` are GC safepoints that can relocate `obj`, `value`, and the
/// list itself, so pin them and reload the receiver after every allocation
/// before storing through it.
///
/// # Safety
/// `obj` must be a valid pointer to the host struct `slots_field` targets.
pub unsafe fn slot_set(
    obj: PyObjectRef,
    index: usize,
    value: PyObjectRef,
    slots_field: SlotsField,
) {
    let slots = unsafe { *slots_field(obj) };
    if !slots.is_null() && unsafe { crate::listobject::w_list_len(slots) } > index {
        unsafe { crate::listobject::w_list_setitem(slots, index as i64, value) };
        return;
    }
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(obj);
    crate::gc_roots::pin_root(value);
    if slots.is_null() {
        let new_slots = crate::listobject::w_list_new(vec![PY_NULL; index + 1]);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        unsafe { *slots_field(obj) = new_slots };
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    } else {
        let mut slots = slots;
        while unsafe { crate::listobject::w_list_len(slots) } <= index {
            unsafe { crate::listobject::w_list_append(slots, PY_NULL) };
            let obj = crate::gc_roots::shadow_stack_get(root_base);
            slots = unsafe { *slots_field(obj) };
        }
    }
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let slots = unsafe { *slots_field(obj) };
    let value = crate::gc_roots::shadow_stack_get(root_base + 1);
    unsafe { crate::listobject::w_list_setitem(slots, index as i64, value) };
}

/// Clear slot `index`, returning whether it held a value beforehand.
///
/// # Safety
/// `obj` must be a valid pointer to the host struct `slots_field` targets.
pub unsafe fn slot_del(obj: PyObjectRef, index: usize, slots_field: SlotsField) -> bool {
    let slots = unsafe { *slots_field(obj) };
    if slots.is_null() || unsafe { crate::listobject::w_list_len(slots) } <= index {
        return false;
    }
    let present = unsafe { crate::listobject::w_list_getitem(slots, index as i64) }
        .is_some_and(|value| !value.is_null());
    if present {
        unsafe { crate::listobject::w_list_setitem(slots, index as i64, PY_NULL) };
    }
    present
}
