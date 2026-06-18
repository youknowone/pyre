//! `pypy/module/__builtin__/functional.py:351-440 W_ReversedIterator`
//! line-by-line port.
//!
//! ```python
//! class W_ReversedIterator(W_Root):
//!     def __init__(self, space, w_sequence):
//!         self.remaining = space.len_w(w_sequence) - 1
//!         if not space.issequence_w(w_sequence):
//!             raise oefmt(space.w_TypeError, ...)
//!         self.w_sequence = w_sequence
//! ```
//!
//! A lazy reverse iterator over a sequence: `descr_next` does
//! `getitem(w_sequence, remaining)` then decrements `remaining`.  When
//! exhausted, `w_sequence` is dropped to `PY_NULL` and `remaining` to
//! `-1` (`:392-393`, `:403-404`).  This replaces the earlier eager
//! materialisation into a `seq_iter`, restoring the lazy CPython 3.14 /
//! PyPy `reversed` object whose `__reduce__` is
//! `(reversed, (sequence,), remaining)`.

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("reversed", static_name = "REVERSED")]
pub struct W_ReversedIterator {
    /// `functional.py:359 self.w_sequence` — the source sequence; set to
    /// `PY_NULL` once the iterator is exhausted (`:393`, `:404`).
    pub w_sequence: PyObjectRef,
    /// `functional.py:355 self.remaining` — index of the next element to
    /// yield, counting down from `len(seq) - 1`; `-1` once exhausted.
    pub remaining: i64,
}

/// Allocate a `W_ReversedIterator`.  Mirrors `functional.py:354-359
/// __init__` with `remaining` already computed as `len(seq) - 1` by the
/// caller.
pub fn w_reversed_new(w_sequence: PyObjectRef, remaining: i64) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_sequence);
    W_ReversedIterator::allocate(W_ReversedIterator {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_sequence,
        remaining,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_reversed(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &REVERSED_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_ReversedIterator`.
#[inline]
pub unsafe fn w_reversed_get_sequence(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ReversedIterator)).w_sequence }
}

/// # Safety
/// `obj` must point to a valid `W_ReversedIterator`.
#[inline]
pub unsafe fn w_reversed_set_sequence(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ReversedIterator)).w_sequence = value;
    }
}

/// # Safety
/// `obj` must point to a valid `W_ReversedIterator`.
#[inline]
pub unsafe fn w_reversed_get_remaining(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_ReversedIterator)).remaining }
}

/// # Safety
/// `obj` must point to a valid `W_ReversedIterator`.
#[inline]
pub unsafe fn w_reversed_set_remaining(obj: PyObjectRef, value: i64) {
    unsafe {
        (*(obj as *mut W_ReversedIterator)).remaining = value;
    }
}
