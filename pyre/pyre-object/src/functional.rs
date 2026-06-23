//! `pypy/module/__builtin__/functional.py` line-by-line ports for built-in iterator functionals.

use crate::pyobject::*;
use pyre_macros::pyre_class;

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py:218-310 W_Enumerate` line-by-line port.
//
// ```python
// class W_Enumerate(W_Root):
//     def __init__(self, w_iter_or_list, start, w_start):
//         self.w_iter_or_list = w_iter_or_list
//         self.index = start
//         self.w_index = w_start
//     ...
// ```
//
// `w_iter_or_list` is either the source iterator (general case) OR
// the source list itself (start == 0 + exact-list source, line 268-269).
// Pyre takes the simpler "always store the iterator" subset for now —
// the list fast-path is a layered optimisation pyre does not need
// today (covered by the `is_list` fast path at the call site).
//
// `index: i64` is the fast counter; once it overflows i64, `w_index`
// carries the bigint value (PyPy line 297-303
// `space.add(w_index, space.newint(1))` after `rarithmetic.ovfcheck`).

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

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py:351-440 W_ReversedIterator`
// line-by-line port.
//
// ```python
// class W_ReversedIterator(W_Root):
//     def __init__(self, space, w_sequence):
//         self.remaining = space.len_w(w_sequence) - 1
//         if not space.issequence_w(w_sequence):
//             raise oefmt(space.w_TypeError, ...)
//         self.w_sequence = w_sequence
// ```
//
// A lazy reverse iterator over a sequence: `descr_next` does
// `getitem(w_sequence, remaining)` then decrements `remaining`.  When
// exhausted, `w_sequence` is dropped to `PY_NULL` and `remaining` to
// `-1` (`:392-393`, `:403-404`).  This replaces the earlier eager
// materialisation into a `seq_iter`, restoring the lazy CPython 3.14 /
// PyPy `reversed` object whose `__reduce__` is
// `(reversed, (sequence,), remaining)`.

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

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py:838-914 W_Map` line-by-line port,
// extended with the CPython 3.14 `strict` keyword (mirrors `zip`).
//
// ```python
// class W_Map(W_Root):
//     def __init__(self, space, w_fun, args_w):
//         self.w_fun = w_fun
//         self.iterators_w = build_iterators_from_args(space, args_w)
// ```
//
// A lazy map: `descr_next` pulls one item from each sub-iterator, then
// `call(w_fun, *items)`.  Stops when the shortest sub-iterator is
// exhausted; in `strict` mode a length mismatch raises `ValueError`.  This
// replaces the earlier eager materialisation into a `seq_iter`.

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

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py:916-1007 W_Filter` line-by-line
// port.
//
// ```python
// class W_Filter(W_Root):
//     def __init__(self, space, w_predicate, w_iterable):
//         if space.is_w(w_predicate, space.w_None):
//             self.w_predicate = None
//         else:
//             self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
// ```
//
// A lazy filter: `descr_next` pulls from `w_iterable` until the predicate
// (or truthiness, when the predicate is `None`) passes.  This replaces the
// earlier eager materialisation into a `seq_iter`, restoring the lazy
// `filter` object whose `__reduce__` is `(filter, (predicate, iterable))`.

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

// ── functional.rs ─────────────────────────────────────────────

// `pypy/module/__builtin__/functional.py:1010-1123 W_Zip` line-by-line port.
//
// ```python
// class W_Zip(W_Root):
//     def __init__(self, space, args_w, strict=False):
//         self.strict = strict
//         self.iterators_w = build_iterators_from_args(space, args_w)
// ```
//
// A lazy zip: `descr_next` pulls one item from each sub-iterator and
// returns the tuple, stopping when the shortest is exhausted; in `strict`
// mode a length mismatch raises `ValueError`.  `descr_setstate` toggles
// `strict`.  This replaces the earlier eager materialisation into a
// `seq_iter`.

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
