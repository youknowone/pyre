//! `pypy/objspace/std/iterobject.py` sequence iterator port.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

// ── Sequence iterator (list/tuple) ──

#[pyre_class("sequenceiterator", type_id = 23, static_name = "SEQ_ITER")]
pub struct W_SeqIterObject {
    pub seq: PyObjectRef,
    pub index: i64,
    pub length: i64,
}

/// `iterobject.py W_FastListIterObject`.  PyPy shares the abstract
/// `sequenceiterator` typedef, while CPython 3.14 exposes the specialized
/// concrete type as `list_iterator`; keep the PyPy payload/algorithm and the
/// 3.14-visible type identity.
#[pyre_class("list_iterator", static_name = "LIST_ITER")]
pub struct W_ListIterObject {
    pub seq: PyObjectRef,
    pub index: i64,
}

/// `iterobject.py W_ReverseSeqIterObject`, specialized to the list producer
/// required by CPython 3.14's `list_reverseiterator` identity.
#[pyre_class("list_reverseiterator", static_name = "LIST_REVERSE_ITER")]
pub struct W_ListReverseIterObject {
    pub seq: PyObjectRef,
    pub index: i64,
}

/// PyPy's abstract sequence iterator specialized to immutable tuple storage;
/// CPython 3.14 exposes this concrete identity as `tuple_iterator`.
#[pyre_class("tuple_iterator", static_name = "TUPLE_ITER")]
pub struct W_TupleIterObject {
    pub seq: PyObjectRef,
    pub index: i64,
}

pub fn w_seq_iter_new(seq: PyObjectRef, length: usize) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(seq);
    W_SeqIterObject::allocate(W_SeqIterObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        seq,
        index: 0,
        length: length as i64,
    })
}

pub fn w_list_iter_new(seq: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(seq);
    W_ListIterObject::allocate(W_ListIterObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        seq,
        index: 0,
    })
}

pub fn w_list_reverse_iter_new(seq: PyObjectRef, index: i64) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(seq);
    W_ListReverseIterObject::allocate(W_ListReverseIterObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        seq,
        index,
    })
}

pub fn w_tuple_iter_new(seq: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(seq);
    W_TupleIterObject::allocate(W_TupleIterObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        seq,
        index: 0,
    })
}

pub unsafe fn is_seq_iter(obj: PyObjectRef) -> bool {
    // A tagged immediate is an `int`, never a seq-iter; short-circuit before
    // the `ob_type` deref so the GC value-stack walker (`walk_raw_immortal_roots`)
    // never dereferences one. Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && unsafe { (*obj).ob_type == &SEQ_ITER_TYPE as *const PyType }
}

#[inline]
pub unsafe fn is_list_iter(obj: PyObjectRef) -> bool {
    !obj.is_null() && (*obj).ob_type == &LIST_ITER_TYPE as *const PyType
}

#[inline]
pub unsafe fn is_list_reverse_iter(obj: PyObjectRef) -> bool {
    !obj.is_null() && (*obj).ob_type == &LIST_REVERSE_ITER_TYPE as *const PyType
}

#[inline]
pub unsafe fn is_tuple_iter(obj: PyObjectRef) -> bool {
    !obj.is_null() && (*obj).ob_type == &TUPLE_ITER_TYPE as *const PyType
}

#[inline]
pub unsafe fn w_list_iter_seq(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_ListIterObject)).seq
}

#[inline]
pub unsafe fn w_list_iter_index(obj: PyObjectRef) -> i64 {
    (*(obj as *const W_ListIterObject)).index
}

#[inline]
pub unsafe fn w_list_iter_set_seq(obj: PyObjectRef, seq: PyObjectRef) {
    (*(obj as *mut W_ListIterObject)).seq = seq;
}

#[inline]
pub unsafe fn w_list_iter_set_index(obj: PyObjectRef, index: i64) {
    (*(obj as *mut W_ListIterObject)).index = index;
}

#[inline]
pub unsafe fn w_list_reverse_iter_seq(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_ListReverseIterObject)).seq
}

#[inline]
pub unsafe fn w_list_reverse_iter_index(obj: PyObjectRef) -> i64 {
    (*(obj as *const W_ListReverseIterObject)).index
}

#[inline]
pub unsafe fn w_list_reverse_iter_set_seq(obj: PyObjectRef, seq: PyObjectRef) {
    (*(obj as *mut W_ListReverseIterObject)).seq = seq;
}

#[inline]
pub unsafe fn w_list_reverse_iter_set_index(obj: PyObjectRef, index: i64) {
    (*(obj as *mut W_ListReverseIterObject)).index = index;
}

#[inline]
pub unsafe fn w_tuple_iter_seq(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_TupleIterObject)).seq
}

#[inline]
pub unsafe fn w_tuple_iter_index(obj: PyObjectRef) -> i64 {
    (*(obj as *const W_TupleIterObject)).index
}

#[inline]
pub unsafe fn w_tuple_iter_set_seq(obj: PyObjectRef, seq: PyObjectRef) {
    (*(obj as *mut W_TupleIterObject)).seq = seq;
}

#[inline]
pub unsafe fn w_tuple_iter_set_index(obj: PyObjectRef, index: i64) {
    (*(obj as *mut W_TupleIterObject)).index = index;
}

/// The wrapped sequence the iterator walks.
///
/// # Safety
/// `obj` must point to a valid `W_SeqIterObject`.
#[inline]
pub unsafe fn w_seq_iter_seq(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SeqIterObject)).seq }
}

/// The current cursor position.
///
/// # Safety
/// `obj` must point to a valid `W_SeqIterObject`.
#[inline]
pub unsafe fn w_seq_iter_index(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_SeqIterObject)).index }
}

/// The captured sequence length.
///
/// # Safety
/// `obj` must point to a valid `W_SeqIterObject`.
#[inline]
pub unsafe fn w_seq_iter_length(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_SeqIterObject)).length }
}

/// Set the cursor position.
///
/// # Safety
/// `obj` must point to a valid `W_SeqIterObject`.
#[inline]
pub unsafe fn w_seq_iter_set_index(obj: PyObjectRef, value: i64) {
    unsafe {
        (*(obj as *mut W_SeqIterObject)).index = value;
    }
}

#[cfg(test)]
mod seq_iter_tests {
    use super::*;

    #[test]
    fn w_seq_iter_gc_type_id_matches_descr() {
        assert_eq!(W_SEQ_ITER_GC_TYPE_ID, 23);
        assert_eq!(
            <W_SeqIterObject as crate::lltype::GcType>::type_id(),
            W_SEQ_ITER_GC_TYPE_ID
        );
        assert_eq!(
            <W_SeqIterObject as crate::lltype::GcType>::SIZE,
            W_SEQ_ITER_OBJECT_SIZE
        );
    }
}
