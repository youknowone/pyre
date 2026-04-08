//! W_CellObject — Python `cell` type for closures.
//!
//! A cell holds a reference to a single value. Closures use cells to
//! share mutable bindings between an outer function and its nested
//! inner functions.

use crate::pyobject::*;

/// Type descriptor for cell objects.
pub static CELL_TYPE: PyType = PyType { tp_name: "cell" };

/// Python cell object.
///
/// Layout: `[ob_type: *const PyType | contents: PyObjectRef]`
/// `contents` is `PY_NULL` when the cell is empty.
#[repr(C)]
pub struct W_CellObject {
    pub ob_header: PyObject,
    pub contents: PyObjectRef,
}

/// Field offset of `contents` within `W_CellObject`.
pub const CELL_CONTENTS_OFFSET: usize = std::mem::offset_of!(W_CellObject, contents);

/// Allocate a new cell wrapping `value`.
/// Pass `PY_NULL` for an empty cell.
pub fn w_cell_new(value: PyObjectRef) -> PyObjectRef {
    let obj = Box::new(W_CellObject {
        ob_header: PyObject {
            ob_type: &CELL_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        contents: value,
    });
    Box::into_raw(obj) as PyObjectRef
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
}
