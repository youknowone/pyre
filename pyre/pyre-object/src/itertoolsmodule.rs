//! itertools module — iterator objects.
//!
//! PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Line-by-line port of the W_Count / W_Repeat / W_Cycle / W_Chain etc.
//! classes. Each class becomes a `#[repr(C)]` struct with a static PyType.

use crate::pyobject::*;
use pyre_macros::pyre_class;

// ── W_Count — pypy/module/itertools/interp_itertools.py:class W_Count ──
//
// ```python
// class W_Count(W_Root):
//     def __init__(self, space, w_firstval, w_step):
//         self.space = space
//         self.w_c = w_firstval
//         self.w_step = w_step
//
//     def iter_w(self):
//         return self
//
//     def next_w(self):
//         w_c = self.w_c
//         self.w_c = self.space.add(w_c, self.w_step)
//         return w_c
// ```
//
// The receiver stores `w_c` (current value) and `w_step` which are both
// PyObjectRef so that count(1.5, 0.5) works for float too.

#[pyre_class("itertools.count", type_id = 24, static_name = "COUNT")]
pub struct W_Count {
    pub w_c: PyObjectRef,
    pub w_step: PyObjectRef,
}

pub fn w_count_new(w_firstval: PyObjectRef, w_step: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_firstval);
    crate::gc_roots::pin_root(w_step);
    W_Count::allocate(W_Count {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_c: w_firstval,
        w_step,
    })
}

/// Check if an object is a `W_Count`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_count(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COUNT_TYPE) }
}

/// Read the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_c(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Count)).w_c }
}

/// Write the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_set_c(obj: PyObjectRef, v: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Count)).w_c = v;
    }
}

/// Read the `w_step` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_step(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Count)).w_step }
}

// ── W_Repeat — pypy/module/itertools/interp_itertools.py:class W_Repeat ──
//
// ```python
// class W_Repeat(W_Root):
//     def __init__(self, space, w_obj, w_times):
//         self.space = space
//         self.w_obj = w_obj
//         if w_times is None:
//             self.counting = False
//             self.count = 0
//         else:
//             self.counting = True
//             self.count = max(self.space.int_w(w_times), 0)
//
//     def next_w(self):
//         if self.counting:
//             if self.count <= 0:
//                 raise OperationError(self.space.w_StopIteration, self.space.w_None)
//             self.count -= 1
//         return self.w_obj
// ```

#[pyre_class("itertools.repeat", type_id = 25, static_name = "REPEAT")]
pub struct W_Repeat {
    pub w_obj: PyObjectRef,
    pub counting: bool,
    pub count: i64,
}

pub fn w_repeat_new(w_obj: PyObjectRef, w_times: Option<i64>) -> PyObjectRef {
    let (counting, count) = match w_times {
        None => (false, 0),
        Some(n) => (true, n.max(0)),
    };
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_obj);
    W_Repeat::allocate(W_Repeat {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_obj,
        counting,
        count,
    })
}

/// Check if an object is a `W_Repeat`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_repeat(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &REPEAT_TYPE) }
}

/// Read the `w_obj` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Repeat)).w_obj }
}

/// Read the `counting` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_counting(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Repeat)).counting }
}

/// Read the `count` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_count(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_Repeat)).count }
}

/// Decrement the `count` field by 1.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_dec_count(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Repeat)).count -= 1;
    }
}

// ── W_TakeWhile — pypy/module/itertools/interp_itertools.py:class W_TakeWhile ──
//
// ```python
// class W_TakeWhile(W_Root):
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
//         self.stopped = False
// ```
//
// `next_w` lives in the interpreter (`baseobjspace::next`) because it
// calls the predicate.

#[pyre_class("itertools.takewhile", type_id = 54, static_name = "TAKEWHILE")]
pub struct W_TakeWhile {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
    pub stopped: bool,
}

/// `w_iterable` must already be an iterator (`space.iter` applied by the
/// caller, matching `W_TakeWhile.__init__`).
pub fn w_takewhile_new(w_predicate: PyObjectRef, w_iterable: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_predicate);
    crate::gc_roots::pin_root(w_iterable);
    W_TakeWhile::allocate(W_TakeWhile {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_predicate,
        w_iterable,
        stopped: false,
    })
}

/// Check if an object is a `W_TakeWhile`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_takewhile(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &TAKEWHILE_TYPE) }
}

// ── W_DropWhile — pypy/module/itertools/interp_itertools.py:class W_DropWhile ──
//
// ```python
// class W_DropWhile(W_Root):
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
//         self.started = False
// ```

#[pyre_class("itertools.dropwhile", type_id = 55, static_name = "DROPWHILE")]
pub struct W_DropWhile {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
    pub started: bool,
}

/// `w_iterable` must already be an iterator (`space.iter` applied by the
/// caller, matching `W_DropWhile.__init__`).
pub fn w_dropwhile_new(w_predicate: PyObjectRef, w_iterable: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_predicate);
    crate::gc_roots::pin_root(w_iterable);
    W_DropWhile::allocate(W_DropWhile {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_predicate,
        w_iterable,
        started: false,
    })
}

/// Check if an object is a `W_DropWhile`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dropwhile(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DROPWHILE_TYPE) }
}

// ── W_FilterFalse — pypy/module/itertools/interp_itertools.py:class W_FilterFalse ──
//
// Subclass of `W_Filter` (`pypy/module/__builtin__/functional.py:916`)
// with `reverse = True`:
//
// ```python
// class W_Filter(W_Root):
//     reverse = False # set to True in itertools
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         if space.is_w(w_predicate, space.w_None):
//             self.w_predicate = None
//         else:
//             self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
// ```
//
// `w_predicate` is PY_NULL when the Python-level predicate was None.

#[pyre_class("itertools.filterfalse", type_id = 56, static_name = "FILTERFALSE")]
pub struct W_FilterFalse {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
}

/// `w_iterable` must already be an iterator; `w_predicate` is PY_NULL
/// for a None predicate (`W_Filter.__init__`).
pub fn w_filterfalse_new(w_predicate: PyObjectRef, w_iterable: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    if !w_predicate.is_null() {
        crate::gc_roots::pin_root(w_predicate);
    }
    crate::gc_roots::pin_root(w_iterable);
    W_FilterFalse::allocate(W_FilterFalse {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_predicate,
        w_iterable,
    })
}

/// Check if an object is a `W_FilterFalse`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_filterfalse(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FILTERFALSE_TYPE) }
}

// ── W_Pairwise — pypy/module/itertools/interp_itertools.py:class W_Pairwise ──
//
// ```python
// class W_Pairwise(W_Root):
//     def __init__(self, space, w_iterator):
//         self.space = space
//         self.w_iterator = w_iterator
//         self.w_prev = None
// ```
//
// `w_prev` is PY_NULL until the first `next_w`.

#[pyre_class("itertools.pairwise", type_id = 57, static_name = "PAIRWISE")]
pub struct W_Pairwise {
    pub w_iterator: PyObjectRef,
    pub w_prev: PyObjectRef,
}

/// `w_iterator` must already be an iterator (`W_Pairwise__new__` applies
/// `space.iter`).
pub fn w_pairwise_new(w_iterator: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_iterator);
    W_Pairwise::allocate(W_Pairwise {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iterator,
        w_prev: std::ptr::null_mut(),
    })
}

/// Check if an object is a `W_Pairwise`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_pairwise(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &PAIRWISE_TYPE) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_count_gc_type_id_matches_descr() {
        assert_eq!(W_COUNT_GC_TYPE_ID, 24);
        assert_eq!(
            <W_Count as crate::lltype::GcType>::type_id(),
            W_COUNT_GC_TYPE_ID
        );
        assert_eq!(
            <W_Count as crate::lltype::GcType>::SIZE,
            W_COUNT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_repeat_gc_type_id_matches_descr() {
        assert_eq!(W_REPEAT_GC_TYPE_ID, 25);
        assert_eq!(
            <W_Repeat as crate::lltype::GcType>::type_id(),
            W_REPEAT_GC_TYPE_ID
        );
        assert_eq!(
            <W_Repeat as crate::lltype::GcType>::SIZE,
            W_REPEAT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_takewhile_gc_type_id_matches_descr() {
        assert_eq!(W_TAKEWHILE_GC_TYPE_ID, 54);
        assert_eq!(
            <W_TakeWhile as crate::lltype::GcType>::type_id(),
            W_TAKEWHILE_GC_TYPE_ID
        );
        assert_eq!(
            <W_TakeWhile as crate::lltype::GcType>::SIZE,
            W_TAKEWHILE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_dropwhile_gc_type_id_matches_descr() {
        assert_eq!(W_DROPWHILE_GC_TYPE_ID, 55);
        assert_eq!(
            <W_DropWhile as crate::lltype::GcType>::type_id(),
            W_DROPWHILE_GC_TYPE_ID
        );
        assert_eq!(
            <W_DropWhile as crate::lltype::GcType>::SIZE,
            W_DROPWHILE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_filterfalse_gc_type_id_matches_descr() {
        assert_eq!(W_FILTERFALSE_GC_TYPE_ID, 56);
        assert_eq!(
            <W_FilterFalse as crate::lltype::GcType>::type_id(),
            W_FILTERFALSE_GC_TYPE_ID
        );
        assert_eq!(
            <W_FilterFalse as crate::lltype::GcType>::SIZE,
            W_FILTERFALSE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_pairwise_gc_type_id_matches_descr() {
        assert_eq!(W_PAIRWISE_GC_TYPE_ID, 57);
        assert_eq!(
            <W_Pairwise as crate::lltype::GcType>::type_id(),
            W_PAIRWISE_GC_TYPE_ID
        );
        assert_eq!(
            <W_Pairwise as crate::lltype::GcType>::SIZE,
            W_PAIRWISE_OBJECT_SIZE
        );
    }
}
