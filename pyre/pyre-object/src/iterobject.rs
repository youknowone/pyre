//! `pypy/objspace/std/iterobject.py` sequence iterator port.

use crate::pyobject::*;
use pyre_macros::pyre_class;

// ── Sequence iterator (list/tuple) ──

#[pyre_class("sequenceiterator", type_id = 23, static_name = "SEQ_ITER")]
pub struct W_SeqIterObject {
    pub seq: PyObjectRef,
    pub index: i64,
    pub length: i64,
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

pub unsafe fn is_seq_iter(obj: PyObjectRef) -> bool {
    !obj.is_null() && unsafe { (*obj).ob_type == &SEQ_ITER_TYPE as *const PyType }
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
