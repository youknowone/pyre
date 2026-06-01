//! W_RangeIterator -- simplified range iterator.
//!
//! `range()` returns an iterator directly (no separate range object).
//! The JIT specializes `for i in range(N)` to pure integer arithmetic
//! by reading/writing `current`, `stop`, `step` via field descriptors.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// Range iterator object.
///
/// Layout: `[ob_type | current: i64 | stop: i64 | step: i64]`
/// The JIT reads `current` and `stop` via `GetfieldGcI` and writes
/// `current` via `SetfieldGcI` to advance the loop counter in registers.
#[pyre_class("range_iterator", type_id = 6, static_name = "RANGE_ITER")]
pub struct W_RangeIterator {
    pub current: i64,
    pub stop: i64,
    pub step: i64,
}

/// Field offsets of inline scalar slots — consumed by JIT field-access
/// IR (`pyre-jit/src/jit/codewriter.rs` GetfieldGcI / SetfieldGcI).
/// The macro's auto-generated `W_RANGE_ITER_GC_PTR_OFFSETS` is empty
/// here (no PyObjectRef fields) and does not depend on these.
pub const RANGE_ITER_CURRENT_OFFSET: usize = std::mem::offset_of!(W_RangeIterator, current);
pub const RANGE_ITER_STOP_OFFSET: usize = std::mem::offset_of!(W_RangeIterator, stop);
pub const RANGE_ITER_STEP_OFFSET: usize = std::mem::offset_of!(W_RangeIterator, step);

/// Allocate a new `W_RangeIterator` on the heap.
pub fn w_range_iter_new(start: i64, stop: i64, step: i64) -> PyObjectRef {
    W_RangeIterator::allocate(W_RangeIterator {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        current: start,
        stop,
        step,
    })
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_range_iter_new(start: i64, stop: i64, step: i64) -> i64 {
    w_range_iter_new(start, stop, step) as i64
}

/// Advance the range iterator and return the next value, or `None` if exhausted.
///
/// # Safety
/// `obj` must point to a valid `W_RangeIterator`.
pub unsafe fn w_range_iter_next(obj: PyObjectRef) -> Option<PyObjectRef> {
    let iter = obj as *mut W_RangeIterator;
    unsafe {
        if !w_range_iter_has_next(obj) {
            None
        } else {
            let current = (*iter).current;
            let step = (*iter).step;
            (*iter).current = current + step;
            Some(crate::intobject::w_int_new(current))
        }
    }
}

/// Check whether a range iterator has another element without advancing it.
///
/// # Safety
/// `obj` must point to a valid `W_RangeIterator`.
pub unsafe fn w_range_iter_has_next(obj: PyObjectRef) -> bool {
    let iter = obj as *const W_RangeIterator;
    unsafe {
        let current = (*iter).current;
        let stop = (*iter).stop;
        let step = (*iter).step;
        if step > 0 {
            current < stop
        } else {
            current > stop
        }
    }
}

/// Check if an object is a range iterator.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_range_iter(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &RANGE_ITER_TYPE) }
}

/// Read the `(current, stop, step)` triple of a range iterator.
///
/// # Safety
/// `obj` must point to a valid `W_RangeIterator`.
pub unsafe fn w_range_iter_fields(obj: PyObjectRef) -> (i64, i64, i64) {
    let iter = obj as *const W_RangeIterator;
    unsafe { ((*iter).current, (*iter).stop, (*iter).step) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_get_value;

    #[test]
    fn test_range_iter_basic() {
        let iter = w_range_iter_new(0, 3, 1);
        unsafe {
            assert!(is_range_iter(iter));
            assert!(!is_int(iter));

            let v0 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v0), 0);

            let v1 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v1), 1);

            let v2 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v2), 2);

            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_start_stop() {
        let iter = w_range_iter_new(5, 8, 1);
        unsafe {
            let v0 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v0), 5);

            let v1 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v1), 6);

            let v2 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v2), 7);

            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_negative_step() {
        let iter = w_range_iter_new(5, 2, -1);
        unsafe {
            let v0 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v0), 5);

            let v1 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v1), 4);

            let v2 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v2), 3);

            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_empty() {
        let iter = w_range_iter_new(5, 5, 1);
        unsafe {
            assert!(!w_range_iter_has_next(iter));
            assert!(w_range_iter_next(iter).is_none());
        }
    }

    #[test]
    fn test_range_iter_has_next_is_pure_probe() {
        let iter = w_range_iter_new(0, 2, 1);
        unsafe {
            assert!(w_range_iter_has_next(iter));
            assert!(w_range_iter_has_next(iter));
            let v0 = w_range_iter_next(iter).unwrap();
            assert_eq!(w_int_get_value(v0), 0);
        }
    }

    #[test]
    fn test_range_iter_field_offsets() {
        assert_eq!(RANGE_ITER_CURRENT_OFFSET, 16);
        assert_eq!(RANGE_ITER_STOP_OFFSET, 24);
        assert_eq!(RANGE_ITER_STEP_OFFSET, 32);
    }
}

// ── Range sequence object ──
//
// `objspace/std/rangeobject.py W_AbstractRangeObject` / `W_RangeObject`:
// an immutable arithmetic sequence carrying `(start, stop, step)`.
// Distinct from `W_RangeIterator` (the cursor produced by `iter()`), so a
// range is reusable, sized, indexable and comparable.

/// `range(start, stop, step)` sequence object.
#[pyre_class("range", type_id = 7, static_name = "RANGE")]
pub struct W_Range {
    pub start: i64,
    pub stop: i64,
    pub step: i64,
}

/// Allocate a `W_Range`.  `step` must already be non-zero (the caller
/// raises `ValueError` for a zero step before reaching here).
pub fn w_range_new(start: i64, stop: i64, step: i64) -> PyObjectRef {
    W_Range::allocate(W_Range {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        start,
        stop,
        step,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_w_range(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &RANGE_TYPE) }
}

/// Read the `(start, stop, step)` triple of a range object.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
#[inline]
pub unsafe fn w_range_fields(obj: PyObjectRef) -> (i64, i64, i64) {
    let r = obj as *const W_Range;
    unsafe { ((*r).start, (*r).stop, (*r).step) }
}

/// Number of elements in a `(start, stop, step)` range —
/// `rangeobject.py compute_range_length`.
pub fn range_length(start: i64, stop: i64, step: i64) -> i64 {
    if step > 0 {
        if start < stop {
            (stop - start - 1) / step + 1
        } else {
            0
        }
    } else if start > stop {
        (start - stop - 1) / (-step) + 1
    } else {
        0
    }
}

/// Length of a `W_Range`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_len(obj: PyObjectRef) -> i64 {
    let (start, stop, step) = unsafe { w_range_fields(obj) };
    range_length(start, stop, step)
}

/// `start + index * step` for an already-normalised, in-bounds `index`
/// (`0 <= index < len`).  Returns `None` when out of range so the caller
/// can raise `IndexError`; negative indices are folded by adding `len`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_getitem(obj: PyObjectRef, index: i64) -> Option<i64> {
    let (start, _stop, step) = unsafe { w_range_fields(obj) };
    let len = unsafe { w_range_len(obj) };
    let i = if index < 0 { index + len } else { index };
    if i < 0 || i >= len {
        None
    } else {
        Some(start + i * step)
    }
}

/// Whether integer `value` is a member of the range —
/// `rangeobject.py W_RangeObject.descr_contains` integer fast path.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_contains_int(obj: PyObjectRef, value: i64) -> bool {
    let (start, stop, step) = unsafe { w_range_fields(obj) };
    if step > 0 {
        if value < start || value >= stop {
            return false;
        }
    } else if value > start || value <= stop {
        return false;
    }
    (value - start) % step == 0
}

#[cfg(test)]
mod range_obj_tests {
    use super::*;

    #[test]
    fn w_range_gc_type_id_matches_descr() {
        assert_eq!(W_RANGE_GC_TYPE_ID, 7);
        assert_eq!(
            <W_Range as crate::lltype::GcType>::type_id(),
            W_RANGE_GC_TYPE_ID
        );
        assert_eq!(
            <W_Range as crate::lltype::GcType>::SIZE,
            W_RANGE_OBJECT_SIZE
        );
    }

    #[test]
    fn range_length_matches_cpython() {
        assert_eq!(range_length(0, 5, 1), 5);
        assert_eq!(range_length(0, 0, 1), 0);
        assert_eq!(range_length(5, 0, 1), 0);
        assert_eq!(range_length(0, 10, 3), 4);
        assert_eq!(range_length(10, 0, -1), 10);
        assert_eq!(range_length(10, 0, -3), 4);
    }

    #[test]
    fn w_range_getitem_and_contains() {
        let r = w_range_new(0, 10, 2);
        unsafe {
            assert_eq!(w_range_len(r), 5);
            assert_eq!(w_range_getitem(r, 0), Some(0));
            assert_eq!(w_range_getitem(r, 4), Some(8));
            assert_eq!(w_range_getitem(r, -1), Some(8));
            assert_eq!(w_range_getitem(r, 5), None);
            assert!(w_range_contains_int(r, 4));
            assert!(!w_range_contains_int(r, 5));
        }
    }
}

// ── Sequence iterator (list/tuple) ──

#[pyre_class("list_iterator", type_id = 23, static_name = "SEQ_ITER")]
pub struct W_SeqIterator {
    pub seq: PyObjectRef,
    pub index: i64,
    pub length: i64,
}

pub fn w_seq_iter_new(seq: PyObjectRef, length: usize) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(seq);
    W_SeqIterator::allocate(W_SeqIterator {
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

#[cfg(test)]
mod seq_iter_tests {
    use super::*;

    #[test]
    fn w_seq_iter_gc_type_id_matches_descr() {
        assert_eq!(W_SEQ_ITER_GC_TYPE_ID, 23);
        assert_eq!(
            <W_SeqIterator as crate::lltype::GcType>::type_id(),
            W_SEQ_ITER_GC_TYPE_ID
        );
        assert_eq!(
            <W_SeqIterator as crate::lltype::GcType>::SIZE,
            W_SEQ_ITER_OBJECT_SIZE
        );
    }
}
