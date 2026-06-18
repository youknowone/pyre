//! `pypy/module/__builtin__/functional.py:916-1007 W_Filter` line-by-line
//! port.
//!
//! ```python
//! class W_Filter(W_Root):
//!     def __init__(self, space, w_predicate, w_iterable):
//!         if space.is_w(w_predicate, space.w_None):
//!             self.w_predicate = None
//!         else:
//!             self.w_predicate = w_predicate
//!         self.w_iterable = space.iter(w_iterable)
//! ```
//!
//! A lazy filter: `descr_next` pulls from `w_iterable` until the predicate
//! (or truthiness, when the predicate is `None`) passes.  This replaces the
//! earlier eager materialisation into a `seq_iter`, restoring the lazy
//! `filter` object whose `__reduce__` is `(filter, (predicate, iterable))`.

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("filter", static_name = "FILTER")]
pub struct W_Filter {
    /// `functional.py:921-924 self.w_predicate` — the predicate callable, or
    /// `PY_NULL` when the Python-level predicate was `None`.
    pub w_predicate: PyObjectRef,
    /// `functional.py:925 self.w_iterable` — the source iterator
    /// (`space.iter(w_iterable)`).
    pub w_iterable: PyObjectRef,
}

/// Allocate a `W_Filter`.  `w_iterable` must already be an iterator;
/// `w_predicate` is `PY_NULL` for a `None` predicate (`__init__`).
pub fn w_filter_new(w_predicate: PyObjectRef, w_iterable: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    if !w_predicate.is_null() {
        crate::gc_roots::pin_root(w_predicate);
    }
    crate::gc_roots::pin_root(w_iterable);
    W_Filter::allocate(W_Filter {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_predicate,
        w_iterable,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_filter(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FILTER_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_Filter`.
#[inline]
pub unsafe fn w_filter_get_predicate(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Filter)).w_predicate }
}

/// # Safety
/// `obj` must point to a valid `W_Filter`.
#[inline]
pub unsafe fn w_filter_get_iterable(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Filter)).w_iterable }
}
