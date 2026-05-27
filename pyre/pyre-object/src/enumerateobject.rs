//! `pypy/module/__builtin__/functional.py:218-310 W_Enumerate` line-by-line port.
//!
//! ```python
//! class W_Enumerate(W_Root):
//!     def __init__(self, w_iter_or_list, start, w_start):
//!         self.w_iter_or_list = w_iter_or_list
//!         self.index = start
//!         self.w_index = w_start
//!     ...
//! ```
//!
//! `w_iter_or_list` is either the source iterator (general case) OR
//! the source list itself (start == 0 + exact-list source, line 268-269).
//! Pyre takes the simpler "always store the iterator" subset for now —
//! the list fast-path is a layered optimisation pyre does not need
//! today (covered by the `is_list` fast path at the call site).
//!
//! `index: i64` is the fast counter; once it overflows i64, `w_index`
//! carries the bigint value (PyPy line 297-303
//! `space.add(w_index, space.newint(1))` after `rarithmetic.ovfcheck`).

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("enumerate", type_id = 41, static_name = "ENUMERATE")]
pub struct W_Enumerate {
    /// `functional.py:225 self.w_iter_or_list` — either the source
    /// iterator (general case) or the source list itself
    /// (start == 0 + exact-list source).  When the iterator is
    /// exhausted, set to `PY_NULL` per `:294-295`.
    pub w_iter_or_list: PyObjectRef,
    /// `functional.py:226 self.index` — i64 fast-path counter.  When
    /// negative, indicates the bigint slot below is active.
    pub index: i64,
    /// `functional.py:227 self.w_index` — bigint counter activated
    /// after i64 overflow per `:298-301`.  `PY_NULL` when the i64
    /// fast path is still active.
    pub w_index: PyObjectRef,
}

/// Allocate a `W_Enumerate`.  Mirrors `functional.py:222-227 __init__`.
/// `w_index` may be `PY_NULL` (i64 fast-path) or a bigint
/// `PyObjectRef` (overflow path).
pub fn w_enumerate_new(
    w_iter_or_list: PyObjectRef,
    start: i64,
    w_index: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_iter_or_list);
    crate::gc_roots::pin_root(w_index);
    W_Enumerate::allocate(W_Enumerate {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iter_or_list,
        index: start,
        w_index,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_enumerate(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ENUMERATE_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_get_iter_or_list(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Enumerate)).w_iter_or_list }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_set_iter_or_list(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Enumerate)).w_iter_or_list = value;
    }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_get_index(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_Enumerate)).index }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_set_index(obj: PyObjectRef, value: i64) {
    unsafe {
        (*(obj as *mut W_Enumerate)).index = value;
    }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_get_w_index(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Enumerate)).w_index }
}

/// # Safety
/// `obj` must point to a valid `W_Enumerate`.
#[inline]
pub unsafe fn w_enumerate_set_w_index(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Enumerate)).w_index = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_enumerate_gc_type_id_matches_descr() {
        assert_eq!(W_ENUMERATE_GC_TYPE_ID, 41);
        assert_eq!(
            <W_Enumerate as crate::lltype::GcType>::type_id(),
            W_ENUMERATE_GC_TYPE_ID
        );
        assert_eq!(
            <W_Enumerate as crate::lltype::GcType>::SIZE,
            W_ENUMERATE_OBJECT_SIZE
        );
    }
}
