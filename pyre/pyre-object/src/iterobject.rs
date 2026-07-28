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
    /// Python 3.14 has producer-specific iterator types whose exhausted
    /// reduce form retains the producer's empty shape.  Pyre shares this
    /// payload for those iterators, so retain that type tag on the iterator
    /// itself: 0 = tuple, 1 = str.
    pub empty_kind: u8,
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

// Python 3.14 gives str / bytes / bytearray / memoryview iteration its own
// concrete type per producer, while PyPy serves all four from the abstract
// `sequenceiterator`.  Keep PyPy's single payload and `descr_next` and give
// each producer the 3.14-visible identity through its own `PyType`, the way
// `dictmultiobject`'s six view iterators share `W_BaseDictMultiIterObject`.
pub static STR_ASCII_ITER_TYPE: PyType = crate::pyobject::new_pytype("str_ascii_iterator");
pub static STR_ITER_TYPE: PyType = crate::pyobject::new_pytype("str_iterator");
pub static BYTES_ITER_TYPE: PyType = crate::pyobject::new_pytype("bytes_iterator");
pub static BYTEARRAY_ITER_TYPE: PyType = crate::pyobject::new_pytype("bytearray_iterator");
pub static MEMORY_ITER_TYPE: PyType = crate::pyobject::new_pytype("memory_iterator");
pub static ARRAY_ITER_TYPE: PyType = crate::pyobject::new_pytype("array.arrayiterator");

/// The Python-visible iterator type for a sequence iterator over `seq`.
///
/// 3.14 splits str iteration by storage: an all-ASCII str yields
/// `str_ascii_iterator`, anything wider yields `str_iterator`.  Producers with
/// no specialized type keep the shared `sequenceiterator`.
fn seq_iter_type_for(seq: PyObjectRef) -> &'static PyType {
    unsafe {
        if crate::is_str(seq) {
            if crate::unicodeobject::w_str_is_ascii(seq) {
                &STR_ASCII_ITER_TYPE
            } else {
                &STR_ITER_TYPE
            }
        } else if crate::bytesobject::is_bytes(seq) {
            &BYTES_ITER_TYPE
        } else if crate::bytearrayobject::is_bytearray(seq) {
            &BYTEARRAY_ITER_TYPE
        } else if crate::memoryview::is_w_memoryview(seq) {
            &MEMORY_ITER_TYPE
        } else if crate::interp_array::is_array(seq) {
            &ARRAY_ITER_TYPE
        } else {
            &SEQ_ITER_TYPE
        }
    }
}

pub fn w_seq_iter_new(seq: PyObjectRef, length: usize) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    let seq_slot = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(seq);
    let seq = crate::gc_roots::shadow_stack_get(seq_slot);
    let tp = seq_iter_type_for(seq);
    let value = W_SeqIterObject {
        ob: PyObject {
            ob_type: tp as *const PyType,
            w_class: crate::pyobject::get_instantiate(tp),
        },
        seq,
        index: 0,
        length: length as i64,
        empty_kind: unsafe { if crate::is_str(seq) { 1 } else { 0 } },
    };
    crate::lltype::malloc_typed_stable(value) as PyObjectRef
}

pub fn w_list_iter_new(seq: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(seq);
    W_ListIterObject::allocate_stable(W_ListIterObject {
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
    W_ListReverseIterObject::allocate_stable(W_ListReverseIterObject {
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
    W_TupleIterObject::allocate_stable(W_TupleIterObject {
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
    if obj.is_null() {
        return false;
    }
    // Every producer-specific identity minted by `seq_iter_type_for` carries
    // the same `W_SeqIterObject` payload, so all of them answer yes here.
    let tp = unsafe { (*obj).ob_type };
    tp == &SEQ_ITER_TYPE as *const PyType
        || tp == &STR_ASCII_ITER_TYPE as *const PyType
        || tp == &STR_ITER_TYPE as *const PyType
        || tp == &BYTES_ITER_TYPE as *const PyType
        || tp == &BYTEARRAY_ITER_TYPE as *const PyType
        || tp == &MEMORY_ITER_TYPE as *const PyType
        || tp == &ARRAY_ITER_TYPE as *const PyType
}

/// `memory_iterator` — the one producer-specific `W_SeqIterObject` identity
/// that exposes no pickle protocol (`__length_hint__` / `__setstate__` absent,
/// `__reduce__` inherited from `object`, which refuses).
#[inline]
pub unsafe fn is_memory_iter(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && (*obj).ob_type == &MEMORY_ITER_TYPE as *const PyType
}

/// `array.arrayiterator` — carries `__reduce__` / `__setstate__` but, unlike
/// the str and bytes flavours, no `__length_hint__`.
#[inline]
pub unsafe fn is_array_iter(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && (*obj).ob_type == &ARRAY_ITER_TYPE as *const PyType
}

#[inline]
pub unsafe fn is_list_iter(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && (*obj).ob_type == &LIST_ITER_TYPE as *const PyType
}

#[inline]
pub unsafe fn is_list_reverse_iter(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && (*obj).ob_type == &LIST_REVERSE_ITER_TYPE as *const PyType
}

#[inline]
pub unsafe fn is_tuple_iter(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
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
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
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
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
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
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
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

/// Empty producer shape used by Python 3.14's exhausted reduce form.
#[inline]
pub unsafe fn w_seq_iter_empty_kind(obj: PyObjectRef) -> u8 {
    unsafe { (*(obj as *const W_SeqIterObject)).empty_kind }
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
