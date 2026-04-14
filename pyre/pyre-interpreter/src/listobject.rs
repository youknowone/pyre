//! Interpreter-level list helpers that need access to `space.eq_w`.
//!
//! PyPy equivalent: method bodies on `ListStrategy.find_or_count`
//! (`pypy/objspace/std/listobject.py:941`) and the descr callers that
//! propagate exceptions from `space.eq_w`. The raw list container
//! (`W_ListObject`) and typed fast paths live in
//! `pyre-object::listobject`; the generic `__eq__` loop lives here where
//! the object space is available.

use pyre_object::{PyObjectRef, listobject::ListFindFast};

use crate::PyError;

/// Outcome of `W_ListObject.find_or_count`. PyPy signals "not found" via
/// a Python `ValueError`; pyre keeps the unwrap into `i64` at the
/// descr-level callers so the rare not-found case doesn't allocate a
/// `PyError` inside the inner loop.
pub enum FindOrCountResult {
    Index(i64),
    Count(i64),
    NotFound,
}

/// `baseobjspace.py:823` `space.eq_w(w_obj1, w_obj2)` —
///     `return self.eq(w_obj1, w_obj2) is self.w_True`.
///
/// Returns a `Result<bool, PyError>` so that `__eq__` exceptions
/// propagate up to the list descr caller, matching PyPy. The int-like
/// and str fast paths live inside `baseobjspace::compare`.
fn eq_w(a: PyObjectRef, b: PyObjectRef) -> Result<bool, PyError> {
    if a == b {
        return Ok(true);
    }
    let result = crate::baseobjspace::compare(a, b, crate::baseobjspace::CompareOp::Eq)?;
    Ok(crate::baseobjspace::is_true(result))
}

/// `listobject.py:417` `W_ListObject.find_or_count`.
///
/// Dispatches to the current strategy's typed fast path via
/// `w_list_find_or_count_fast`; when that signals `NeedsGeneric`, runs
/// the generic `ListStrategy.find_or_count` loop
/// (`listobject.py:941-957`) using `space.eq_w`.
pub fn w_list_find_or_count(
    obj: PyObjectRef,
    w_item: PyObjectRef,
    start: i64,
    stop: i64,
    count: bool,
) -> Result<FindOrCountResult, PyError> {
    match unsafe {
        pyre_object::listobject::w_list_find_or_count_fast(obj, w_item, start, stop, count)
    } {
        ListFindFast::Found(i) => return Ok(FindOrCountResult::Index(i)),
        ListFindFast::Count(n) => return Ok(FindOrCountResult::Count(n)),
        ListFindFast::NotFound => return Ok(FindOrCountResult::NotFound),
        ListFindFast::NeedsGeneric => {}
    }
    // listobject.py:941-957 ListStrategy.find_or_count:
    //     while i < stop and i < w_list.length():
    //         if space.eq_w(w_list.getitem(i), w_item):
    //             ...
    //         i += 1
    //     raise ValueError / return count
    let mut i = start.max(0);
    let mut result: i64 = 0;
    while i < stop && i < unsafe { pyre_object::w_list_len(obj) } as i64 {
        let w_curr = match unsafe { pyre_object::w_list_getitem(obj, i) } {
            Some(v) => v,
            None => break,
        };
        if eq_w(w_curr, w_item)? {
            if count {
                result += 1;
            } else {
                return Ok(FindOrCountResult::Index(i));
            }
        }
        i += 1;
    }
    if count {
        Ok(FindOrCountResult::Count(result))
    } else {
        Ok(FindOrCountResult::NotFound)
    }
}

/// `listobject.py:782` `W_ListObject.descr_remove`.
///
/// Runs `find_or_count(value, 0, sys.maxint)`, pops at the returned
/// index when still within bounds (listobject.py:791 guard against
/// `eq_w`-triggered mutations), raises `ValueError` otherwise.
pub fn w_list_remove(obj: PyObjectRef, w_value: PyObjectRef) -> Result<(), PyError> {
    let i = match w_list_find_or_count(obj, w_value, 0, i64::MAX, false)? {
        FindOrCountResult::Index(i) => i,
        FindOrCountResult::NotFound => {
            return Err(PyError::new(
                crate::PyErrorKind::ValueError,
                format!("list.remove(): {} is not in list", unsafe {
                    crate::display::py_repr(w_value)
                }),
            ));
        }
        FindOrCountResult::Count(_) => unreachable!("find_or_count with count=false returns Count"),
    };
    // listobject.py:791: `if i < self.length():  # otherwise list was mutated`
    let length = unsafe { pyre_object::w_list_len(obj) } as i64;
    if i < length {
        unsafe {
            pyre_object::listobject::w_list_pop(obj, i);
        }
    }
    Ok(())
}
