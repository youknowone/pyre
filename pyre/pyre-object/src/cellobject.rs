//! W_CellObject — Python `cell` type for closures.
//!
//! A cell holds a reference to a single value. Closures use cells to
//! share mutable bindings between an outer function and its nested
//! inner functions.

use crate::pyobject::*;

/// Type descriptor for cell objects.
pub static CELL_TYPE: PyType = crate::pyobject::new_pytype("cell");

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

/// GC type id assigned to `W_CellObject` at JitDriver init time. Held as
/// a constant alongside the struct (rather than runtime-queried) so the
/// allocation hook can reach it without a back-channel, mirroring
/// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID` / `FUNCTION_GC_TYPE_ID`.
/// `pyre/pyre-jit/src/eval.rs` asserts the same id is returned by
/// `gc.register_type(...)` so any drift panics on startup.
pub const W_CELL_GC_TYPE_ID: u32 = 15;

/// Fixed payload size used by `gct_fv_gc_malloc`'s `c_size`
/// (`framework.py:811`).
pub const W_CELL_OBJECT_SIZE: usize = std::mem::size_of::<W_CellObject>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace
/// during minor collection. `ob.w_class` is intentionally absent,
/// mirroring how W_IntObject / W_FloatObject leave the typeptr-shaped
/// header field out of their `gc_ptr_offsets` (W_TypeObject instances
/// are static-region and not subject to nursery relocation).
pub const W_CELL_GC_PTR_OFFSETS: [usize; 1] = [CELL_CONTENTS_OFFSET];

impl crate::lltype::GcType for W_CellObject {
    const TYPE_ID: u32 = W_CELL_GC_TYPE_ID;
    const SIZE: usize = W_CELL_OBJECT_SIZE;
}

/// Allocate a new cell wrapping `value`.
/// Pass `PY_NULL` for an empty cell.
pub fn w_cell_new(value: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the `lltype::malloc_typed` call below. `value` is a live
    // PyObjectRef root that must survive a potential collection inside
    // the allocation point once Phase 2 swaps the malloc body to a
    // managed allocator.
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(value);

    crate::lltype::malloc_typed(W_CellObject {
        ob_header: PyObject {
            ob_type: &CELL_TYPE as *const PyType,
            w_class: get_instantiate(&CELL_TYPE),
        },
        contents: value,
    }) as PyObjectRef
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
            <W_CellObject as crate::lltype::GcType>::TYPE_ID,
            W_CELL_GC_TYPE_ID
        );
        assert_eq!(
            <W_CellObject as crate::lltype::GcType>::SIZE,
            W_CELL_OBJECT_SIZE
        );
    }
}
