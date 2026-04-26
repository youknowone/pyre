//! W_RangeIterator -- simplified range iterator.
//!
//! `range()` returns an iterator directly (no separate range object).
//! The JIT specializes `for i in range(N)` to pure integer arithmetic
//! by reading/writing `current`, `stop`, `step` via field descriptors.

use crate::pyobject::*;

/// Type descriptor for range iterators.
pub static RANGE_ITER_TYPE: PyType = crate::pyobject::new_pytype("range_iterator");

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

/// Fixed payload size (`framework.py:811`).
pub const W_RANGE_ITER_OBJECT_SIZE: usize = std::mem::size_of::<W_RangeIterator>();

impl crate::lltype::GcType for W_RangeIterator {
    /// Mirrors `pyre_jit_trace::descr::RANGE_ITER_GC_TYPE_ID`. The JIT
    /// init's `debug_assert_eq!` cross-checks any drift.
    const TYPE_ID: u32 = 6;
    const SIZE: usize = W_RANGE_ITER_OBJECT_SIZE;
}

/// Allocate a new `W_RangeIterator` on the heap.
pub fn w_range_iter_new(start: i64, stop: i64, step: i64) -> PyObjectRef {
    crate::lltype::malloc_typed(W_RangeIterator {
        ob: PyObject {
            ob_type: &RANGE_ITER_TYPE as *const PyType,
            w_class: get_instantiate(&RANGE_ITER_TYPE),
        },
        current: start,
        stop,
        step,
    }) as PyObjectRef
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

// ── Sequence iterator (list/tuple) ──

pub static SEQ_ITER_TYPE: PyType = crate::pyobject::new_pytype("list_iterator");

#[repr(C)]
pub struct W_SeqIterator {
    pub ob: PyObject,
    pub seq: PyObjectRef,
    pub index: i64,
    pub length: i64,
}

/// Field offset of `seq` within `W_SeqIterator`.
pub const SEQ_ITER_SEQ_OFFSET: usize = std::mem::offset_of!(W_SeqIterator, seq);

/// GC type id assigned to `W_SeqIterator` at JitDriver init time.
pub const W_SEQ_ITER_GC_TYPE_ID: u32 = 23;

/// Fixed payload size (`framework.py:811`).
pub const W_SEQ_ITER_OBJECT_SIZE: usize = std::mem::size_of::<W_SeqIterator>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
pub const W_SEQ_ITER_GC_PTR_OFFSETS: [usize; 1] = [SEQ_ITER_SEQ_OFFSET];

impl crate::lltype::GcType for W_SeqIterator {
    const TYPE_ID: u32 = W_SEQ_ITER_GC_TYPE_ID;
    const SIZE: usize = W_SEQ_ITER_OBJECT_SIZE;
}

pub fn w_seq_iter_new(seq: PyObjectRef, length: usize) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(seq);

    crate::lltype::malloc_typed(W_SeqIterator {
        ob: PyObject {
            ob_type: &SEQ_ITER_TYPE as *const PyType,
            w_class: get_instantiate(&SEQ_ITER_TYPE),
        },
        seq,
        index: 0,
        length: length as i64,
    }) as PyObjectRef
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
            <W_SeqIterator as crate::lltype::GcType>::TYPE_ID,
            W_SEQ_ITER_GC_TYPE_ID
        );
        assert_eq!(
            <W_SeqIterator as crate::lltype::GcType>::SIZE,
            W_SEQ_ITER_OBJECT_SIZE
        );
    }
}
