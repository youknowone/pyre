//! W_CellObject — Python `cell` type for closures.
//!
//! A cell holds a reference to a single value. Closures use cells to
//! share mutable bindings between an outer function and its nested
//! inner functions.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// Python cell object.
///
/// Layout: `[ob_type: *const PyType | contents: PyObjectRef]`
/// `contents` is `PY_NULL` when the cell is empty.
#[pyre_class("cell", type_id = 15, static_name = "CELL")]
pub struct W_CellObject {
    pub contents: PyObjectRef,
}

/// Allocate a new cell wrapping `value`.
/// Pass `PY_NULL` for an empty cell.
pub fn w_cell_new(value: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the `lltype::malloc_typed` call below. `value` is a live
    // PyObjectRef root that must survive a potential collection inside
    // the allocation point once the malloc body swaps to a
    // managed allocator.
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(value);
    W_CellObject::allocate(W_CellObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        contents: value,
    })
}

/// Check if an object is a cell.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_cell(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CELL_TYPE) }
}

/// Get the value stored in a cell.
///
/// # Safety
/// `obj` must point to a valid `W_CellObject`.
#[inline]
pub unsafe fn w_cell_get(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_CellObject)).contents }
}

/// Set the value stored in a cell.
///
/// # Safety
/// `obj` must point to a valid `W_CellObject`.
#[inline]
pub unsafe fn w_cell_set(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut W_CellObject)).contents = value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_create_empty() {
        let cell = w_cell_new(PY_NULL);
        unsafe {
            assert!(is_cell(cell));
            assert!(w_cell_get(cell).is_null());
        }
    }

    #[test]
    fn test_cell_create_with_value() {
        let value = 0xDEAD as PyObjectRef;
        let cell = w_cell_new(value);
        unsafe {
            assert!(is_cell(cell));
            assert_eq!(w_cell_get(cell), value);
        }
    }

    #[test]
    fn test_cell_set() {
        let cell = w_cell_new(PY_NULL);
        let value = 0xBEEF as PyObjectRef;
        unsafe {
            w_cell_set(cell, value);
            assert_eq!(w_cell_get(cell), value);
        }
    }

    /// Guard against drift between the constant colocated with
    /// `W_CellObject` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_INT/W_FLOAT/FUNCTION trip-wire
    /// tests.
    #[test]
    fn w_cell_gc_type_id_matches_descr() {
        assert_eq!(W_CELL_GC_TYPE_ID, 15);
        assert_eq!(
            <W_CellObject as crate::lltype::GcType>::type_id(),
            W_CELL_GC_TYPE_ID
        );
        assert_eq!(
            <W_CellObject as crate::lltype::GcType>::SIZE,
            W_CELL_OBJECT_SIZE
        );
    }
}
