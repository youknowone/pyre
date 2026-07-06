//! W_SetObject — Python `set` type.
//!
//! PyPy equivalent: pypy/objspace/std/setobject.py
//!
//! Stores arbitrary PyObjectRef elements with element equality
//! reusing dict_keys_equal semantics. PyPy carries multiple set
//! strategies (EmptySet, IntegerSet, etc.); pyre starts with a single
//! Vec to keep parity tractable while bringing the type online.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

pub static SET_TYPE: PyType = crate::pyobject::new_pytype("set");
pub static FROZENSET_TYPE: PyType = crate::pyobject::new_pytype("frozenset");

/// Python set object.
///
/// Layout: `[ob_type | items | len]`. `items` is heap-owned via
/// `Box::into_raw` to keep the struct trivially `Copy`-friendly for the
/// JIT raw-pointer model.
#[repr(C)]
pub struct W_SetObject {
    pub ob_header: PyObject,
    pub items: *mut Vec<PyObjectRef>,
    pub len: usize,
}

/// GC type id assigned to `W_SetObject` at JitDriver init time.
pub const W_SET_GC_TYPE_ID: u32 = 30;

/// Fixed payload size (`framework.py:811`).
pub const W_SET_OBJECT_SIZE: usize = std::mem::size_of::<W_SetObject>();

impl crate::lltype::GcType for W_SetObject {
    fn type_id() -> u32 {
        W_SET_GC_TYPE_ID
    }
    const SIZE: usize = W_SET_OBJECT_SIZE;
}

#[inline]
pub unsafe fn is_set(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SET_TYPE) }
}

#[inline]
pub unsafe fn is_frozenset(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FROZENSET_TYPE) }
}

#[inline]
pub unsafe fn is_set_or_frozenset(obj: PyObjectRef) -> bool {
    unsafe { is_set(obj) || is_frozenset(obj) }
}

/// Element equality. Delegates to dict_keys_equal so that set membership
/// follows the same rules as dict key equality (int / bool / str / tuple
/// / frozenset, with pointer identity as a fallback for everything else).
unsafe fn set_keys_equal(a: PyObjectRef, b: PyObjectRef) -> bool {
    crate::dictmultiobject::dict_keys_equal(a, b)
}

/// Fire the GC write barrier for a set whose element storage just gained
/// a possibly-young element. `set_object_custom_trace` only forwards the
/// `items` slots when the set is reached by a collection; an old-gen set
/// that stored a young element is reached on a minor GC only if it sits in
/// the remembered set, so the barrier must run after every insert. Mirrors
/// `dict_write_barrier`.
#[inline]
fn set_write_barrier(obj: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

fn alloc_set_with_type(tp: &'static PyType) -> PyObjectRef {
    let items = crate::lltype::malloc_raw(Vec::new());
    let header = PyObject {
        ob_type: tp as *const PyType,
        w_class: get_instantiate(tp),
    };
    // Allocate the body in GC old-gen (mark-sweep, non-moving) so it
    // carries TRACK_YOUNG_PTRS, mirroring `w_list_new` / `w_tuple_new`.
    // `w_set_add` stores possibly-young elements into `items`; the write
    // barrier (`set_write_barrier`) only remembers the set on a minor
    // collection when the body is an old-gen object, so a body allocated
    // through the plain `malloc_typed` (no TRACK_YOUNG_PTRS) would leave
    // young elements unforwarded and collected. Falls back to
    // `malloc_typed` when no GC hook is installed (unit tests).
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_SET_GC_TYPE_ID, W_SET_OBJECT_SIZE);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut W_SetObject,
                W_SetObject {
                    ob_header: header,
                    items,
                    len: 0,
                },
            );
        }
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(W_SetObject {
            ob_header: header,
            items,
            len: 0,
        }) as PyObjectRef
    }
}

/// Allocate an empty `set`.
pub fn w_set_new() -> PyObjectRef {
    alloc_set_with_type(&SET_TYPE)
}

/// Allocate an empty `frozenset`.
pub fn w_frozenset_new() -> PyObjectRef {
    alloc_set_with_type(&FROZENSET_TYPE)
}

/// Allocate a populated set from a slice of elements (deduped).
pub fn w_set_from_items(items: &[PyObjectRef]) -> PyObjectRef {
    let s = w_set_new();
    for &item in items {
        unsafe { w_set_add(s, item) };
    }
    s
}

/// Allocate a populated frozenset from a slice of elements (deduped).
pub fn w_frozenset_from_items(items: &[PyObjectRef]) -> PyObjectRef {
    let s = w_frozenset_new();
    for &item in items {
        unsafe { w_set_add(s, item) };
    }
    s
}

/// Insert an element. No-op when already present.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_add(obj: PyObjectRef, item: PyObjectRef) {
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *s.items;
    for &existing in entries.iter() {
        if set_keys_equal(existing, item) {
            return;
        }
    }
    entries.push(item);
    s.len += 1;
    set_write_barrier(obj);
}

/// Membership test.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_contains(obj: PyObjectRef, item: PyObjectRef) -> bool {
    let s = &*(obj as *const W_SetObject);
    let entries = &*s.items;
    entries.iter().any(|&e| set_keys_equal(e, item))
}

/// Remove an element if present. Returns true when removed.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_discard(obj: PyObjectRef, item: PyObjectRef) -> bool {
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *s.items;
    if let Some(idx) = entries.iter().position(|&e| set_keys_equal(e, item)) {
        entries.remove(idx);
        s.len -= 1;
        true
    } else {
        false
    }
}

/// Number of elements in the set.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_len(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetObject)).len
}

/// Snapshot the contained elements as a `Vec`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_items(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let s = &*(obj as *const W_SetObject);
    (*s.items).clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    #[test]
    fn add_dedupes() {
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert_eq!(w_set_len(s), 2);
            assert!(w_set_contains(s, w_int_new(1)));
            assert!(w_set_contains(s, w_int_new(2)));
            assert!(!w_set_contains(s, w_int_new(3)));
        }
    }

    #[test]
    fn discard_removes() {
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert!(w_set_discard(s, w_int_new(1)));
            assert!(!w_set_discard(s, w_int_new(99)));
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, w_int_new(2)));
        }
    }

    #[test]
    fn frozenset_distinct_type() {
        let s = w_set_new();
        let fs = w_frozenset_new();
        unsafe {
            assert!(is_set(s));
            assert!(!is_frozenset(s));
            assert!(is_frozenset(fs));
            assert!(!is_set(fs));
        }
    }

    #[test]
    fn w_set_gc_type_id_matches_descr() {
        assert_eq!(W_SET_GC_TYPE_ID, 30);
        assert_eq!(
            <W_SetObject as crate::lltype::GcType>::type_id(),
            W_SET_GC_TYPE_ID
        );
        assert_eq!(
            <W_SetObject as crate::lltype::GcType>::SIZE,
            W_SET_OBJECT_SIZE
        );
    }
}
