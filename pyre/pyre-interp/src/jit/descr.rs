//! Field descriptors for JIT IR operations.
//!
//! GetfieldGcI, GetfieldGcR, and SetfieldGc require a `DescrRef`
//! carrying field offset, size, and type information. This module
//! provides a concrete `PyreFieldDescr` implementing majit's
//! `FieldDescr` trait for pyre's `#[repr(C)]` object layout.

use std::sync::Arc;

use majit_ir::{ArrayDescr, Descr, DescrRef, FieldDescr, Type};

const FIELD_DESCR_TAG: u32 = 0x1000_0000;
const ARRAY_DESCR_TAG: u32 = 0x2000_0000;

fn type_bits(tp: Type) -> u32 {
    match tp {
        Type::Int => 0,
        Type::Ref => 1,
        Type::Float => 2,
        Type::Void => 3,
    }
}

fn stable_field_index(offset: usize, field_size: usize, field_type: Type, signed: bool) -> u32 {
    FIELD_DESCR_TAG
        | (((offset as u32) & 0x000f_ffff) << 4)
        | (((field_size as u32) & 0x7) << 1)
        | ((signed as u32) << 3)
        | type_bits(field_type)
}

fn stable_array_index(base_size: usize, item_size: usize, item_type: Type, signed: bool) -> u32 {
    ARRAY_DESCR_TAG
        | (((base_size as u32) & 0x0000_0fff) << 12)
        | (((item_size as u32) & 0x0000_00ff) << 4)
        | ((signed as u32) << 3)
        | type_bits(item_type)
}

/// Concrete field descriptor for pyre object fields.
#[derive(Debug)]
pub struct PyreFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
}

/// Concrete array descriptor for pointer-backed runtime arrays.
#[derive(Debug)]
pub struct PyreArrayDescr {
    base_size: usize,
    item_size: usize,
    item_type: Type,
    signed: bool,
}

impl Descr for PyreFieldDescr {
    fn index(&self) -> u32 {
        stable_field_index(self.offset, self.field_size, self.field_type, self.signed)
    }

    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }
}

impl Descr for PyreArrayDescr {
    fn index(&self) -> u32 {
        stable_array_index(self.base_size, self.item_size, self.item_type, self.signed)
    }

    fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
        Some(self)
    }
}

impl FieldDescr for PyreFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }
    fn field_size(&self) -> usize {
        self.field_size
    }
    fn field_type(&self) -> Type {
        self.field_type
    }
    fn is_field_signed(&self) -> bool {
        self.signed
    }
}

impl ArrayDescr for PyreArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }

    fn item_size(&self) -> usize {
        self.item_size
    }

    fn type_id(&self) -> u32 {
        0
    }

    fn item_type(&self) -> Type {
        self.item_type
    }

    fn is_item_signed(&self) -> bool {
        self.signed
    }
}

/// Create a field descriptor for an object field.
pub fn make_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
    })
}

/// Create an array descriptor for a pointer-backed array field.
pub fn make_array_descr(
    base_size: usize,
    item_size: usize,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreArrayDescr {
        base_size,
        item_size,
        item_type,
        signed,
    })
}

// ── Range iterator field descriptors ─────────────────────────────────

use pyre_object::floatobject::FLOAT_FLOATVAL_OFFSET;
use pyre_object::rangeobject::{
    RANGE_ITER_CURRENT_OFFSET, RANGE_ITER_STEP_OFFSET, RANGE_ITER_STOP_OFFSET,
};
use pyre_object::{
    BOOL_BOOLVAL_OFFSET, DICT_LEN_OFFSET, INT_INTVAL_OFFSET, PYOBJECT_ARRAY_HEAP_CAP_OFFSET,
    PYOBJECT_ARRAY_LEN_OFFSET, STR_LEN_OFFSET, W_ListObject, W_TupleObject,
};
use pyre_runtime::{PYNAMESPACE_VALUES_LEN_OFFSET, PYNAMESPACE_VALUES_OFFSET};

/// Field descriptor for `W_RangeIterator.current` (i64, signed).
pub fn range_iter_current_descr() -> DescrRef {
    make_field_descr(RANGE_ITER_CURRENT_OFFSET, 8, Type::Int, true)
}

/// Field descriptor for `W_RangeIterator.stop` (i64, signed).
pub fn range_iter_stop_descr() -> DescrRef {
    make_field_descr(RANGE_ITER_STOP_OFFSET, 8, Type::Int, true)
}

/// Field descriptor for `W_RangeIterator.step` (i64, signed).
pub fn range_iter_step_descr() -> DescrRef {
    make_field_descr(RANGE_ITER_STEP_OFFSET, 8, Type::Int, true)
}

pub fn list_items_ptr_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, items),
        8,
        Type::Int,
        false,
    )
}

pub fn list_items_len_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, items) + PYOBJECT_ARRAY_LEN_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn tuple_items_ptr_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_TupleObject, items),
        8,
        Type::Int,
        false,
    )
}

pub fn tuple_items_len_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_TupleObject, items) + PYOBJECT_ARRAY_LEN_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn list_items_heap_cap_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, items) + PYOBJECT_ARRAY_HEAP_CAP_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn int_intval_descr() -> DescrRef {
    make_field_descr(INT_INTVAL_OFFSET, 8, Type::Int, true)
}

pub fn bool_boolval_descr() -> DescrRef {
    make_field_descr(BOOL_BOOLVAL_OFFSET, 1, Type::Int, false)
}

pub fn float_floatval_descr() -> DescrRef {
    make_field_descr(FLOAT_FLOATVAL_OFFSET, 8, Type::Float, false)
}

pub fn str_len_descr() -> DescrRef {
    make_field_descr(STR_LEN_OFFSET, 8, Type::Int, false)
}

pub fn dict_len_descr() -> DescrRef {
    make_field_descr(DICT_LEN_OFFSET, 8, Type::Int, false)
}

pub fn namespace_values_ptr_descr() -> DescrRef {
    make_field_descr(PYNAMESPACE_VALUES_OFFSET, 8, Type::Int, false)
}

pub fn namespace_values_len_descr() -> DescrRef {
    make_field_descr(PYNAMESPACE_VALUES_LEN_OFFSET, 8, Type::Int, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_descr_indices_are_stable_and_distinct() {
        let a = make_field_descr(8, 8, Type::Int, false);
        let b = make_field_descr(8, 8, Type::Int, false);
        let c = make_field_descr(16, 8, Type::Int, false);

        assert_eq!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
    }

    #[test]
    fn test_array_descr_indices_are_stable_and_distinct() {
        let a = make_array_descr(0, 8, Type::Int, false);
        let b = make_array_descr(0, 8, Type::Int, false);
        let c = make_array_descr(0, 8, Type::Ref, false);

        assert_eq!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
    }
}
