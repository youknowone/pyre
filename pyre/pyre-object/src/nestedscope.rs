//! `pypy/interpreter/nestedscope.py` — Python `cell` type for closures.
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
pub struct Cell {
    pub contents: PyObjectRef,
}

/// Allocate a new cell wrapping `value`.
/// Pass `PY_NULL` for an empty cell.
pub fn w_cell_new(value: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): `value`
    // is a live GC pointer that must survive — and be relocated by — the
    // collection the GC malloc below may trigger. `pin_root` records it in
    // the shadow stack so the moving collector keeps it alive and rewrites
    // the slot; we read the relocated address back after the malloc.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(value);

    let header = PyObject {
        ob_type: &CELL_TYPE as *const PyType,
        w_class: get_instantiate(&CELL_TYPE),
    };
    // Route through the managed allocator like `w_list_new`/`w_tuple_new`.
    // A cell whose `contents` is reachable only through this cell (e.g. a
    // closure cellvar) must itself be GC-traced; a `malloc_typed`
    // (`std::alloc`) cell is invisible to `is_managed_heap_object`, so the
    // mark-sweep skips it and the only-reachable-via-cell value is swept.
    let raw = crate::gc_hook::try_gc_alloc_stable(W_CELL_GC_TYPE_ID, W_CELL_OBJECT_SIZE)
        .filter(|p| !p.is_null());
    let value = crate::gc_roots::shadow_stack_get(save_point);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut Cell,
                Cell {
                    ob: header,
                    contents: value,
                },
            );
        }
        // The cell lives in old-gen (`try_gc_alloc_stable`); `contents` may
        // still be a nursery object. Register the cell so the next minor
        // collection scans it (incminimark.py:1495 write_barrier) and
        // relocates a young value held only by `contents`.
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    Cell::allocate(Cell {
        ob: header,
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
/// `obj` must point to a valid `Cell`.
#[inline]
pub unsafe fn w_cell_get(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Cell)).contents }
}

/// Set the value stored in a cell.
///
/// # Safety
/// `obj` must point to a valid `Cell`.
#[inline]
pub unsafe fn w_cell_set(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut Cell)).contents = value }
    // The cell is an old-gen (`try_gc_alloc_stable`) object; storing a
    // possibly-nursery `value` into it needs the incminimark write barrier
    // (incminimark.py:1495) so the next minor collection scans the cell and
    // relocates the young value held only by `contents`.
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
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
    /// `Cell` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_INT/W_FLOAT/FUNCTION trip-wire
    /// tests.
    #[test]
    fn w_cell_gc_type_id_matches_descr() {
        assert_eq!(W_CELL_GC_TYPE_ID, 15);
        assert_eq!(
            <Cell as crate::lltype::GcType>::type_id(),
            W_CELL_GC_TYPE_ID
        );
        assert_eq!(<Cell as crate::lltype::GcType>::SIZE, W_CELL_OBJECT_SIZE);
    }
}
