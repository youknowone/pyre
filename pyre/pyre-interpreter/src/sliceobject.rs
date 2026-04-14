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
fn eval_slice_index(w_int: PyObjectRef) -> Result<i64, crate::PyError> {
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
        index += size;
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
