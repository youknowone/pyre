//! W_RangeIterator -- simplified range iterator.
//!
//! `range()` returns an iterator directly (no separate range object).
//! The JIT specializes `for i in range(N)` to pure integer arithmetic
//! by reading/writing `current`, `stop`, `step` via field descriptors.

use crate::pyobject::*;

/// Type descriptor for range iterators.
pub static RANGE_ITER_TYPE: PyType = PyType {
    tp_name: "range_iterator",
};

/// Range iterator object.
///
/// Layout: `[ob_type | current: i64 | stop: i64 | step: i64]`
/// The JIT reads `current` and `stop` via `GetfieldGcI` and writes
/// `current` via `SetfieldGcI` to advance the loop counter in registers.
#[repr(C)]
pub struct W_RangeIterator {
    pub ob: PyObject,
    pub current: i64,
    pub stop: i64,
    pub step: i64,
}

/// Field offset of `current` within `W_RangeIterator`.
pub const RANGE_ITER_CURRENT_OFFSET: usize = std::mem::offset_of!(W_RangeIterator, current);

/// Field offset of `stop` within `W_RangeIterator`.
pub const RANGE_ITER_STOP_OFFSET: usize = std::mem::offset_of!(W_RangeIterator, stop);

/// Field offset of `step` within `W_RangeIterator`.
pub const RANGE_ITER_STEP_OFFSET: usize = std::mem::offset_of!(W_RangeIterator, step);

/// Allocate a new `W_RangeIterator` on the heap.
pub fn w_range_iter_new(start: i64, stop: i64, step: i64) -> PyObjectRef {
    let obj = Box::new(W_RangeIterator {
        ob: PyObject {
            ob_type: &RANGE_ITER_TYPE as *const PyType,
        },
        current: start,
        stop,
        step,
    });
    Box::into_raw(obj) as PyObjectRef
}

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
        assert_eq!(RANGE_ITER_CURRENT_OFFSET, 8);
        assert_eq!(RANGE_ITER_STOP_OFFSET, 16);
        assert_eq!(RANGE_ITER_STEP_OFFSET, 24);
    }
}
