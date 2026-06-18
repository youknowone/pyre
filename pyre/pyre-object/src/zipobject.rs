//! `pypy/module/__builtin__/functional.py:1010-1123 W_Zip` line-by-line port.
//!
//! ```python
//! class W_Zip(W_Root):
//!     def __init__(self, space, args_w, strict=False):
//!         self.strict = strict
//!         self.iterators_w = build_iterators_from_args(space, args_w)
//! ```
//!
//! A lazy zip: `descr_next` pulls one item from each sub-iterator and
//! returns the tuple, stopping when the shortest is exhausted; in `strict`
//! mode a length mismatch raises `ValueError`.  `descr_setstate` toggles
//! `strict`.  This replaces the earlier eager materialisation into a
//! `seq_iter`.

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("zip", static_name = "ZIP")]
pub struct W_Zip {
    /// `functional.py:1016 self.iterators_w` — a `list` of sub-iterators, one
    /// per input iterable (`build_iterators_from_args`).
    pub w_iterators: PyObjectRef,
    /// `functional.py:1014 self.strict`; `descr_setstate` toggles it.
    pub strict: bool,
}

/// Allocate a `W_Zip`.  `w_iterators` is a `list` of already-built
/// iterators (`build_iterators_from_args`).
pub fn w_zip_new(w_iterators: PyObjectRef, strict: bool) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_iterators);
    W_Zip::allocate(W_Zip {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iterators,
        strict,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_zip(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ZIP_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_Zip`.
#[inline]
pub unsafe fn w_zip_get_iterators(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Zip)).w_iterators }
}

/// # Safety
/// `obj` must point to a valid `W_Zip`.
#[inline]
pub unsafe fn w_zip_get_strict(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Zip)).strict }
}

/// # Safety
/// `obj` must point to a valid `W_Zip`.
#[inline]
pub unsafe fn w_zip_set_strict(obj: PyObjectRef, value: bool) {
    unsafe {
        (*(obj as *mut W_Zip)).strict = value;
    }
}
