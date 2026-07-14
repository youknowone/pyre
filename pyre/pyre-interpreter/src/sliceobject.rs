//! Interpreter-level slice helpers.
//!
//! PyPy equivalent: `pypy/objspace/std/sliceobject.py`. Only the helpers
//! that the interpreter layer uses are ported here; the data type for
//! `slice` objects lives in `pyre-object::sliceobject`.

use pyre_object::{PyObjectRef, pyobject::is_none};

/// sliceobject.py:221 `_eval_slice_index(space, w_int)`.
///
/// Returns `w_int.__index__()` as an `i64`, converting to `TypeError`
/// when the object has no `__index__` method.
pub(crate) fn eval_slice_index(w_int: PyObjectRef) -> Result<i64, crate::PyError> {
    match crate::builtins::getindex_w(w_int) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::TypeError => Err(crate::PyError::new(
            crate::PyErrorKind::TypeError,
            "slice indices must be integers or None or have an __index__ method".to_string(),
        )),
        Err(e) => Err(e),
    }
}

/// sliceobject.py:233 `adapt_lower_bound(space, size, w_index)`.
///
/// Converts `w_index` via `__index__`, normalizes negatives against
/// `size`, and clamps at zero.
pub fn adapt_lower_bound(size: i64, w_index: PyObjectRef) -> Result<i64, crate::PyError> {
    let mut index = eval_slice_index(w_index)?;
    if index < 0 {
        // `eval_slice_index` clamps an out-of-word index to `i64::MIN`, so fold
        // by `size` without overflowing before flooring at 0.
        index = index.saturating_add(size);
        if index < 0 {
            index = 0;
        }
    }
    debug_assert!(index >= 0);
    Ok(index)
}

/// sliceobject.py:242 `unwrap_start_stop(space, size, w_start, w_end)`.
///
/// Returns `(start, end)` after negative-index normalization. `None`
/// maps to `(0, size)`.
pub fn unwrap_start_stop(
    size: i64,
    w_start: PyObjectRef,
    w_end: PyObjectRef,
) -> Result<(i64, i64), crate::PyError> {
    let start = if unsafe { is_none(w_start) } {
        0
    } else {
        adapt_lower_bound(size, w_start)?
    };
    let end = if unsafe { is_none(w_end) } {
        debug_assert!(size >= 0);
        size
    } else {
        adapt_lower_bound(size, w_end)?
    };
    Ok((start, end))
}

/// sliceobject.py:130 `W_SliceObject.unpack(space)`.
///
/// Evaluates `start`/`stop`/`step` through `__index__` **without** reading
/// the container length: an `__index__` method may mutate the container, so
/// the length must be consulted only afterwards (`_unpack_slice`). `None`
/// endpoints map to the open-ended sentinels that [`slice_adjust_indices`]
/// then clamps. A zero `step` raises `ValueError`.
pub(crate) fn slice_unpack(
    w_start: PyObjectRef,
    w_stop: PyObjectRef,
    w_step: PyObjectRef,
) -> Result<(i64, i64, i64), crate::PyError> {
    let step = if unsafe { is_none(w_step) } {
        1
    } else {
        let step = eval_slice_index(w_step)?;
        if step == 0 {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "slice step cannot be zero".to_string(),
            ));
        }
        step
    };
    let start = if unsafe { is_none(w_start) } {
        if step < 0 { i64::MAX } else { 0 }
    } else {
        eval_slice_index(w_start)?
    };
    let stop = if unsafe { is_none(w_stop) } {
        if step < 0 { i64::MIN } else { i64::MAX }
    } else {
        eval_slice_index(w_stop)?
    };
    Ok((start, stop, step))
}

/// sliceobject.py:139 `W_SliceObject.adjust_indices(start, stop, step, length)`.
///
/// Pure arithmetic: clamps the unpacked `(start, stop, step)` against
/// `length`, clipping out-of-bounds endpoints consistently with
/// extended-slice handling, and returns the triple plus the resulting
/// `slicelength`.
pub(crate) fn slice_adjust_indices(
    mut start: i64,
    mut stop: i64,
    step: i64,
    length: i64,
) -> (i64, i64, i64, i64) {
    if start < 0 {
        start += length;
        if start < 0 {
            start = if step < 0 { -1 } else { 0 };
        }
    } else if start >= length {
        start = if step < 0 { length - 1 } else { length };
    }
    if stop < 0 {
        stop += length;
        if stop < 0 {
            stop = if step < 0 { -1 } else { 0 };
        }
    } else if stop >= length {
        stop = if step < 0 { length - 1 } else { length };
    }
    let slicelength = if (step < 0 && stop >= start) || (step > 0 && start >= stop) {
        0
    } else if step < 0 {
        (stop - start + 1) / step + 1
    } else {
        (stop - start - 1) / step + 1
    };
    (start, stop, step, slicelength)
}

/// sliceobject.py:170 `W_SliceObject.indices3(space, length)`.
///
/// Computes the `(start, stop, step)` triple for a slice over a sequence
/// of `length` items — [`slice_unpack`] then [`slice_adjust_indices`]. A
/// zero `step` raises `ValueError`.
pub fn indices3(
    w_start: PyObjectRef,
    w_stop: PyObjectRef,
    w_step: PyObjectRef,
    length: i64,
) -> Result<(i64, i64, i64), crate::PyError> {
    let (start, stop, step) = slice_unpack(w_start, w_stop, w_step)?;
    let (start, stop, step, _) = slice_adjust_indices(start, stop, step, length);
    Ok((start, stop, step))
}
