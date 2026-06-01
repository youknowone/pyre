//! Two-argument `iter(callable, sentinel)` product
//! (`pypy/module/__builtin__/operation.py:114-160 iter_sentinel`,
//! `Objects/iterobject.c calliterobject`).  Pyre's `iter` is
//! interpreter-native, so the result is a small native iterator object
//! rather than an app-level generator:
//!
//! ```text
//! __next__():
//!     if it_callable is NULL: raise StopIteration   # exhausted
//!     result = it_callable()
//!     if it_sentinel == result:                     # rich __eq__
//!         it_callable = NULL                         # latch exhausted
//!         raise StopIteration
//!     return result
//! ```
//!
//! `callable` is set to `PY_NULL` once the sentinel has been seen, so a
//! second `next()` keeps raising `StopIteration` without re-invoking the
//! callable (`calliter_iternext` `it->it_callable == NULL`).

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("callable_iterator", type_id = 42, static_name = "CALLABLE_ITERATOR")]
pub struct W_CallableIterator {
    /// The zero-argument callable invoked on each `__next__`.  Set to
    /// `PY_NULL` once the sentinel has been returned, latching the
    /// iterator exhausted.
    pub callable: PyObjectRef,
    /// The sentinel value compared (via `__eq__`) against each call
    /// result; equality ends iteration.
    pub sentinel: PyObjectRef,
}

/// Allocate a `W_CallableIterator` for `iter(callable, sentinel)`.
pub fn w_callable_iterator_new(callable: PyObjectRef, sentinel: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(callable);
    crate::gc_roots::pin_root(sentinel);
    W_CallableIterator::allocate(W_CallableIterator {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        callable,
        sentinel,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_callable_iterator(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CALLABLE_ITERATOR_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_CallableIterator`.
#[inline]
pub unsafe fn w_callable_iterator_get_callable(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_CallableIterator)).callable }
}

/// # Safety
/// `obj` must point to a valid `W_CallableIterator`.
#[inline]
pub unsafe fn w_callable_iterator_set_callable(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_CallableIterator)).callable = value;
    }
}

/// # Safety
/// `obj` must point to a valid `W_CallableIterator`.
#[inline]
pub unsafe fn w_callable_iterator_get_sentinel(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_CallableIterator)).sentinel }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_callable_iterator_gc_type_id_matches_descr() {
        assert_eq!(W_CALLABLE_ITERATOR_GC_TYPE_ID, 42);
        assert_eq!(
            <W_CallableIterator as crate::lltype::GcType>::type_id(),
            W_CALLABLE_ITERATOR_GC_TYPE_ID
        );
        assert_eq!(
            <W_CallableIterator as crate::lltype::GcType>::SIZE,
            W_CALLABLE_ITERATOR_OBJECT_SIZE
        );
    }
}
