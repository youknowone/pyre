//! Range objects and their iterators.
//!
//! `range()` builds a `W_Range` sequence carrying wrapped `(start, stop,
//! step, length)` bounds; `iter()` produces a `W_RangeIterator` (machine
//! int, JIT-specializable) when every bound fits a word, else a
//! `W_LongRangeIterator` (bignum), mirroring `rangeiterator` /
//! `longrange_iterator`.  The JIT specializes `for i in range(N)` to pure
//! integer arithmetic by reading/writing the iterator's `current`,
//! `stop`, `step` via field descriptors.

use crate::pyobject::*;
use malachite_bigint::BigInt;
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

/// Count of elements not yet produced by a range iterator — the number
/// of `current += step` steps before `current` crosses `stop`.
///
/// # Safety
/// `obj` must point to a valid `W_RangeIterator`.
pub unsafe fn w_range_iter_remaining(obj: PyObjectRef) -> i64 {
    let (current, stop, step) = unsafe { w_range_iter_fields(obj) };
    if step > 0 {
        if current >= stop {
            0
        } else {
            (stop - current + step - 1) / step
        }
    } else if current <= stop {
        0
    } else {
        (current - stop - step - 1) / (-step)
    }
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
// an immutable arithmetic sequence carrying `(start, stop, step)`.  PyPy
// stores the three bounds as wrapped ints, so a range can describe values
// beyond a machine word; pyre keeps the same three wrapped fields.
//
// The hot `for i in range(n)` loop never reads these fields — `iter()`
// produces a `W_RangeIterator` (i64, JIT-specialized) when every bound
// fits a machine word, and a `W_LongRangeIterator` otherwise, mirroring
// PyPy's `rangeiterator` / `longrange_iterator` split.

/// `range(start, stop, step)` sequence object.
///
/// `w_length` is precomputed once at construction (`descr_new` →
/// `compute_range_length`) and read back by `descr_len` / `descr_bool`.
#[pyre_class("range", type_id = 7, static_name = "RANGE")]
pub struct W_Range {
    pub start: PyObjectRef,
    pub stop: PyObjectRef,
    pub step: PyObjectRef,
    pub length: PyObjectRef,
}

/// Allocate a `W_Range` from three wrapped int/long bounds.  `step` must
/// already be non-zero (the caller raises `ValueError` for a zero step
/// before reaching here).  The element count is computed once here and
/// stored, mirroring `descr_new`'s `compute_range_length`.
pub fn w_range_new(start: PyObjectRef, stop: PyObjectRef, step: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(start);
    crate::gc_roots::pin_root(stop);
    crate::gc_roots::pin_root(step);
    let length = unsafe {
        let len_big = range_length_big(
            &range_obj_to_bigint(start),
            &range_obj_to_bigint(stop),
            &range_obj_to_bigint(step),
        );
        range_bigint_to_obj(len_big)
    };
    crate::gc_roots::pin_root(length);
    W_Range::allocate(W_Range {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        start,
        stop,
        step,
        length,
    })
}

/// Convenience constructor wrapping three machine-int bounds.
pub fn w_range_new_i64(start: i64, stop: i64, step: i64) -> PyObjectRef {
    w_range_new(
        crate::intobject::w_int_new(start),
        crate::intobject::w_int_new(stop),
        crate::intobject::w_int_new(step),
    )
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_w_range(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &RANGE_TYPE) }
}

/// Read the wrapped `(start, stop, step)` triple of a range object.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
#[inline]
pub unsafe fn w_range_fields(obj: PyObjectRef) -> (PyObjectRef, PyObjectRef, PyObjectRef) {
    let r = obj as *const W_Range;
    unsafe { ((*r).start, (*r).stop, (*r).step) }
}

/// Read an int/long/bool operand as a `BigInt`.
///
/// # Safety
/// `obj` must be a valid int, long, or bool object.
pub unsafe fn range_obj_to_bigint(obj: PyObjectRef) -> BigInt {
    unsafe {
        // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first;
        // a bool reads through `w_bool_get_value`, not the int accessor.
        if is_bool(obj) {
            BigInt::from(crate::boolobject::w_bool_get_value(obj) as i64)
        } else if is_int(obj) {
            BigInt::from(crate::intobject::w_int_get_value(obj))
        } else {
            crate::longobject::w_long_get_value(obj).clone()
        }
    }
}

/// Wrap a `BigInt` as a machine int when it fits, otherwise a long.
pub fn range_bigint_to_obj(value: BigInt) -> PyObjectRef {
    use num_traits::ToPrimitive;
    match value.to_i64() {
        Some(v) => crate::intobject::w_int_new(v),
        None => crate::longobject::w_long_new(value),
    }
}

/// A single int/long/bool bound as an i64 when it fits, else `None`.
///
/// # Safety
/// `obj` must be a valid int/long/bool object.
pub unsafe fn range_obj_as_i64(obj: PyObjectRef) -> Option<i64> {
    unsafe {
        // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first.
        if is_bool(obj) {
            Some(crate::boolobject::w_bool_get_value(obj) as i64)
        } else if is_int(obj) {
            Some(crate::intobject::w_int_get_value(obj))
        } else {
            use num_traits::ToPrimitive;
            crate::longobject::w_long_get_value(obj).to_i64()
        }
    }
}

/// The `(start, stop, step)` triple as machine ints when all three fit a
/// machine word (the common case); `None` if any bound is a bignum.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_fields_i64(obj: PyObjectRef) -> Option<(i64, i64, i64)> {
    unsafe {
        let (s, e, t) = w_range_fields(obj);
        Some((
            range_obj_as_i64(s)?,
            range_obj_as_i64(e)?,
            range_obj_as_i64(t)?,
        ))
    }
}

/// The precomputed wrapped element count — `descr_len → self.w_length`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
#[inline]
pub unsafe fn w_range_length(obj: PyObjectRef) -> PyObjectRef {
    let r = obj as *const W_Range;
    unsafe { (*r).length }
}

/// The element count as a machine int when it fits, else `None`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
#[inline]
pub unsafe fn w_range_length_i64(obj: PyObjectRef) -> Option<i64> {
    unsafe { range_obj_as_i64(w_range_length(obj)) }
}

/// `descr_bool → space.nonzero(self.w_length)` — a range is truthy iff it
/// has at least one element.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_bool(obj: PyObjectRef) -> bool {
    use num_traits::Zero;
    unsafe { !range_obj_to_bigint(w_range_length(obj)).is_zero() }
}

/// `descr_iter` — a `rangeiterator` (`W_RangeIterator`, machine-int and
/// JIT-specializable) when every bound fits a machine word, otherwise a
/// `longrange_iterator` (`W_LongRangeIterator`).
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_iter(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        // `descr_iter` takes the machine-int iterator only when start, stop,
        // step AND length all fit a machine word.  `W_RangeIterator` stops
        // when `current` crosses `stop`, advancing `current += step` after
        // each element; the post-final `start + length*step` must therefore
        // also fit a word, or the wrapped `current` would never reach `stop`
        // (an infinite loop).  When it would overflow — or any bound/length
        // exceeds a word — the bignum iterator is used instead, which stops
        // on the wrapped length the way `W_IntRangeIterator` counts down its
        // remaining.
        if let (Some((start, stop, step)), Some(length)) =
            (w_range_fields_i64(obj), w_range_length_i64(obj))
        {
            let one_past = start as i128 + length as i128 * step as i128;
            if i64::try_from(one_past).is_ok() {
                return w_range_iter_new(start, stop, step);
            }
        }
        let (start, _stop, step) = w_range_fields(obj);
        let len = w_range_length(obj);
        w_long_range_iter_new(start, step, len)
    }
}

/// `descr_reversed` — walk the span backwards.  The fast path keeps a
/// machine-int `W_RangeIterator` so `for i in reversed(range(n))` stays
/// JIT-specializable; otherwise a `W_LongRangeIterator` from
/// `(start + (length-1)*step, -step, length)`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_reversed(obj: PyObjectRef) -> PyObjectRef {
    use num_traits::One;
    unsafe {
        if let (Some((start, _stop, step)), Some(len)) =
            (w_range_fields_i64(obj), w_range_length_i64(obj))
        {
            if len == 0 {
                return w_range_iter_new(0, 0, 1);
            }
            // The reversed machine-int iterator walks `last` down to `start`
            // by `-step`, stopping past `start - step`; that one-past value,
            // the negated step, and `last` must all fit a word or the
            // wrapped `current` would loop forever — fall back to bignum.
            let last = start as i128 + (len as i128 - 1) * step as i128;
            let one_past = start as i128 - step as i128;
            if let (Ok(last), Ok(stop_rev), Some(neg_step)) = (
                i64::try_from(last),
                i64::try_from(one_past),
                step.checked_neg(),
            ) {
                return w_range_iter_new(last, stop_rev, neg_step);
            }
        }
        let (start, _stop, step) = w_range_fields(obj);
        let len_obj = w_range_length(obj);
        let start_b = range_obj_to_bigint(start);
        let step_b = range_obj_to_bigint(step);
        let len_b = range_obj_to_bigint(len_obj);
        let lastitem = &start_b + (&len_b - BigInt::one()) * &step_b;
        let _roots = crate::gc_roots::push_roots();
        crate::gc_roots::pin_root(len_obj);
        let w_lastitem = range_bigint_to_obj(lastitem);
        crate::gc_roots::pin_root(w_lastitem);
        let w_negstep = range_bigint_to_obj(-step_b);
        crate::gc_roots::pin_root(w_negstep);
        w_long_range_iter_new(w_lastitem, w_negstep, len_obj)
    }
}

/// `_compute_item` — the member at `index` (negative folded against the
/// length), or `None` when out of bounds (`IndexError` at the call site).
/// `index` is the already-`__index__`'d operand as a `BigInt`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_compute_item(obj: PyObjectRef, index: &BigInt) -> Option<PyObjectRef> {
    use num_traits::Zero;
    unsafe {
        let (start, _stop, step) = w_range_fields(obj);
        let len_b = range_obj_to_bigint(w_range_length(obj));
        let mut idx = index.clone();
        if idx < BigInt::zero() {
            idx += &len_b;
        }
        if idx >= len_b || idx < BigInt::zero() {
            return None;
        }
        let value = range_obj_to_bigint(start) + idx * range_obj_to_bigint(step);
        Some(range_bigint_to_obj(value))
    }
}

/// `_contains_long` — O(1) membership test for an integer `item`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_contains_bigint(obj: PyObjectRef, item: &BigInt) -> bool {
    use num_traits::Zero;
    unsafe {
        let (start, stop, step) = w_range_fields(obj);
        let start_b = range_obj_to_bigint(start);
        let stop_b = range_obj_to_bigint(stop);
        let step_b = range_obj_to_bigint(step);
        if step_b > BigInt::zero() {
            // positive steps: start <= ob < stop
            if !(start_b <= *item && *item < stop_b) {
                return false;
            }
        } else {
            // negative steps: stop < ob <= start
            if !(stop_b < *item && *item <= start_b) {
                return false;
            }
        }
        // The stride must not invalidate membership.
        ((item - &start_b) % &step_b).is_zero()
    }
}

/// `descr_index` — the position of `item` (known to be in range), i.e.
/// `(item - start) // step`.
///
/// # Safety
/// `obj` must point to a valid `W_Range`.
pub unsafe fn w_range_index_of(obj: PyObjectRef, item: &BigInt) -> PyObjectRef {
    unsafe {
        let (start, _stop, step) = w_range_fields(obj);
        let value = (item - range_obj_to_bigint(start)) / range_obj_to_bigint(step);
        range_bigint_to_obj(value)
    }
}

/// `descr_eq` — two ranges are equal iff they generate the same sequence:
/// equal lengths, and for a non-empty range equal start and (for length
/// > 1) equal step.  The caller has already established both operands are
/// ranges.
///
/// # Safety
/// `a` and `b` must point to valid `W_Range` objects.
pub unsafe fn w_range_eq(a: PyObjectRef, b: PyObjectRef) -> bool {
    use num_traits::One;
    unsafe {
        let la = range_obj_to_bigint(w_range_length(a));
        let lb = range_obj_to_bigint(w_range_length(b));
        if la != lb {
            return false;
        }
        let (astart, _astop, astep) = w_range_fields(a);
        let (bstart, _bstop, bstep) = w_range_fields(b);
        if la == BigInt::from(0) {
            return true;
        }
        if range_obj_to_bigint(astart) != range_obj_to_bigint(bstart) {
            return false;
        }
        if la == BigInt::one() {
            return true;
        }
        range_obj_to_bigint(astep) == range_obj_to_bigint(bstep)
    }
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

/// Bignum `compute_range_length` — always non-negative.
pub fn range_length_big(start: &BigInt, stop: &BigInt, step: &BigInt) -> BigInt {
    use num_traits::{One, Zero};
    let zero = BigInt::zero();
    if *step > zero {
        if *start < *stop {
            (stop - start - BigInt::one()) / step + BigInt::one()
        } else {
            BigInt::zero()
        }
    } else if *start > *stop {
        (start - stop - BigInt::one()) / (-step) + BigInt::one()
    } else {
        BigInt::zero()
    }
}

// ── Long range iterator ──
//
// `objspace/std/iterobject.py W_LongRangeIterator` — the cursor `iter()`
// produces for a range whose bounds exceed a machine word.  `start`, `step`
// and `len` are set once at construction; `index` is a wrapped integer that
// advances by one each step (`self.w_index`), so the cursor keeps arbitrary
// precision and never overflows for a range longer than a machine word.

#[pyre_class("range_iterator", type_id = 8, static_name = "LONG_RANGE_ITER")]
pub struct W_LongRangeIterator {
    pub start: PyObjectRef,
    pub step: PyObjectRef,
    pub len: PyObjectRef,
    pub index: PyObjectRef,
}

/// Allocate a `W_LongRangeIterator`.
pub fn w_long_range_iter_new(
    start: PyObjectRef,
    step: PyObjectRef,
    len: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(start);
    crate::gc_roots::pin_root(step);
    crate::gc_roots::pin_root(len);
    let index = crate::intobject::w_int_new(0);
    crate::gc_roots::pin_root(index);
    W_LongRangeIterator::allocate(W_LongRangeIterator {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        start,
        step,
        len,
        index,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_long_range_iter(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LONG_RANGE_ITER_TYPE) }
}

/// Number of elements not yet produced — `__length_hint__`.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator`.
pub unsafe fn w_long_range_iter_len(obj: PyObjectRef) -> BigInt {
    unsafe {
        let it = obj as *const W_LongRangeIterator;
        let len = range_obj_to_bigint((*it).len);
        let rem = len - range_obj_to_bigint((*it).index);
        if rem < BigInt::from(0) {
            BigInt::from(0)
        } else {
            rem
        }
    }
}

/// Read the `(start, step, len, index)` fields of a long-range iterator
/// as wrapped int/long objects.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator`.
pub unsafe fn w_long_range_iter_fields(
    obj: PyObjectRef,
) -> (PyObjectRef, PyObjectRef, PyObjectRef, PyObjectRef) {
    let it = obj as *const W_LongRangeIterator;
    unsafe { ((*it).start, (*it).step, (*it).len, (*it).index) }
}

/// Whether the long-range iterator has elements left — peek, non-mutating.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator`.
pub unsafe fn w_long_range_iter_has_next(obj: PyObjectRef) -> bool {
    unsafe {
        let it = obj as *const W_LongRangeIterator;
        range_obj_to_bigint((*it).index) < range_obj_to_bigint((*it).len)
    }
}

/// Advance the long-range iterator and return the next value, or `None`
/// once exhausted — `start + index * step`.
///
/// # Safety
/// `obj` must point to a valid `W_LongRangeIterator`.
pub unsafe fn w_long_range_iter_next(obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        let it = obj as *mut W_LongRangeIterator;
        let index = range_obj_to_bigint((*it).index);
        let len = range_obj_to_bigint((*it).len);
        if index >= len {
            return None;
        }
        let start = range_obj_to_bigint((*it).start);
        let step = range_obj_to_bigint((*it).step);
        // `w_result = self.w_index * self.w_step + self.w_start`, then
        // `self.w_index = self.w_index + 1` (wrapped, arbitrary precision).
        let value = start + index.clone() * step;
        let _roots = crate::gc_roots::push_roots();
        let next_index = range_bigint_to_obj(index + BigInt::from(1));
        crate::gc_roots::pin_root(next_index);
        (*it).index = next_index;
        Some(range_bigint_to_obj(value))
    }
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
    fn w_range_fields_i64_roundtrip() {
        let r = w_range_new_i64(0, 10, 2);
        unsafe {
            assert_eq!(w_range_fields_i64(r), Some((0, 10, 2)));
        }
    }

    #[test]
    fn long_range_iter_yields_values() {
        let it = w_long_range_iter_new(
            crate::intobject::w_int_new(5),
            crate::intobject::w_int_new(2),
            crate::intobject::w_int_new(3),
        );
        unsafe {
            assert!(is_long_range_iter(it));
            let v0 = w_long_range_iter_next(it).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v0), 5);
            let v1 = w_long_range_iter_next(it).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v1), 7);
            let v2 = w_long_range_iter_next(it).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(v2), 9);
            assert!(w_long_range_iter_next(it).is_none());
        }
    }

    #[test]
    fn iter_routes_increment_overflow_to_longrange() {
        unsafe {
            // Word-fit bounds, but the forward `start + length*step` overflows
            // i64, so a machine-int iterator would loop forever — must use the
            // bignum iterator instead.
            let fwd = w_range_new_i64(i64::MAX - 5, i64::MAX, 10);
            assert!(is_long_range_iter(w_range_iter(fwd)));
            // The reversed walk's one-past `start - step` underflows i64 here,
            // so `reversed()` must likewise use the bignum iterator.
            let rev = w_range_new_i64(i64::MIN, i64::MIN + 50, 10);
            assert!(is_long_range_iter(w_range_reversed(rev)));
            // A plain range stays on the machine-int (JIT) iterator both ways.
            let n = w_range_new_i64(0, 10, 2);
            assert!(is_range_iter(w_range_iter(n)));
            assert!(is_range_iter(w_range_reversed(n)));
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

/// The wrapped sequence the iterator walks.
///
/// # Safety
/// `obj` must point to a valid `W_SeqIterator`.
#[inline]
pub unsafe fn w_seq_iter_seq(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SeqIterator)).seq }
}

/// The current cursor position.
///
/// # Safety
/// `obj` must point to a valid `W_SeqIterator`.
#[inline]
pub unsafe fn w_seq_iter_index(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_SeqIterator)).index }
}

/// The captured sequence length.
///
/// # Safety
/// `obj` must point to a valid `W_SeqIterator`.
#[inline]
pub unsafe fn w_seq_iter_length(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_SeqIterator)).length }
}

/// Set the cursor position.
///
/// # Safety
/// `obj` must point to a valid `W_SeqIterator`.
#[inline]
pub unsafe fn w_seq_iter_set_index(obj: PyObjectRef, value: i64) {
    unsafe {
        (*(obj as *mut W_SeqIterator)).index = value;
    }
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
