//! `pypy/module/__builtin__/functional.py:838-914 W_Map` line-by-line port,
//! extended with the CPython 3.14 `strict` keyword (mirrors `zip`).
//!
//! ```python
//! class W_Map(W_Root):
//!     def __init__(self, space, w_fun, args_w):
//!         self.w_fun = w_fun
//!         self.iterators_w = build_iterators_from_args(space, args_w)
//! ```
//!
//! A lazy map: `descr_next` pulls one item from each sub-iterator, then
//! `call(w_fun, *items)`.  Stops when the shortest sub-iterator is
//! exhausted; in `strict` mode a length mismatch raises `ValueError`.  This
//! replaces the earlier eager materialisation into a `seq_iter`.

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("map", static_name = "MAP")]
pub struct W_Map {
    /// `functional.py:843 self.w_fun` — the mapped callable.
    pub w_fun: PyObjectRef,
    /// `functional.py:844 self.iterators_w` — a `list` of sub-iterators, one
    /// per input iterable (`build_iterators_from_args`).
    pub w_iterators: PyObjectRef,
    /// CPython 3.14 `strict` flag; `descr_setstate` toggles it.
    pub strict: bool,
}

/// Allocate a `W_Map`.  `w_iterators` is a `list` of already-built
/// iterators (`build_iterators_from_args`).
pub fn w_map_new(w_fun: PyObjectRef, w_iterators: PyObjectRef, strict: bool) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_fun);
    crate::gc_roots::pin_root(w_iterators);
    W_Map::allocate(W_Map {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_fun,
        w_iterators,
        strict,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_map(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &MAP_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_Map`.
#[inline]
pub unsafe fn w_map_get_fun(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Map)).w_fun }
}

/// # Safety
/// `obj` must point to a valid `W_Map`.
#[inline]
pub unsafe fn w_map_get_iterators(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Map)).w_iterators }
}

/// # Safety
/// `obj` must point to a valid `W_Map`.
#[inline]
pub unsafe fn w_map_get_strict(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Map)).strict }
}

/// # Safety
/// `obj` must point to a valid `W_Map`.
#[inline]
pub unsafe fn w_map_set_strict(obj: PyObjectRef, value: bool) {
    unsafe {
        (*(obj as *mut W_Map)).strict = value;
    }
}
