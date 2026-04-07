//! Field descriptors for JIT IR operations.
//!
//! GetfieldGcI, GetfieldGcR, and SetfieldGc require a `DescrRef`
//! carrying field offset, size, and type information. This module
//! provides a concrete `PyreFieldDescr` implementing majit's
//! `FieldDescr` trait for pyre's `#[repr(C)]` object layout.

use std::sync::Arc;

use majit_ir::{ArrayDescr, Descr, DescrRef, FieldDescr, SizeDescr, Type};

const FIELD_DESCR_TAG: u32 = 0x1000_0000;
const ARRAY_DESCR_TAG: u32 = 0x2000_0000;
const SIZE_DESCR_TAG: u32 = 0x3000_0000;

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
/// RPython FieldDescr: describes a field in a GC/raw struct.
#[derive(Debug)]
pub struct PyreFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
    /// RPython: is_immutable_field(). Immutable fields survive cache invalidation.
    immutable: bool,
    /// RPython: _is_quasi_immutable(). Fields that rarely change but CAN change.
    /// When read during tracing, emits QUASIIMMUT_FIELD + GUARD_NOT_INVALIDATED.
    /// If mutated at runtime, invalidates all compiled loops watching this field.
    quasi_immutable: bool,
    /// RPython descr.py:227 — field name for heaptracker.py:66 filtering.
    name: &'static str,
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

    /// PyPy FieldDescr.is_always_pure(): immutable fields survive cache invalidation.
    fn is_always_pure(&self) -> bool {
        self.immutable
    }

    fn is_quasi_immutable(&self) -> bool {
        self.quasi_immutable
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
    fn field_name(&self) -> &str {
        self.name
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
        immutable: false,
        quasi_immutable: false,
        name: "",
    })
}

pub fn make_field_descr_full(
    _index: u32,
    offset: usize,
    field_size: usize,
    field_type: Type,
    immutable: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed: false,
        immutable,
        quasi_immutable: false,
        name: "",
    })
}

/// Create a field descriptor for an immutable field (RPython is_immutable_field).
/// Cache entries for immutable fields survive call invalidation.
pub fn make_immutable_field_descr(
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
        immutable: true,
        quasi_immutable: false,
        name: "",
    })
}

/// Create a field descriptor for a quasi-immutable field.
/// When read during tracing, emits QUASIIMMUT_FIELD + GUARD_NOT_INVALIDATED.
pub fn make_quasi_immutable_field_descr(
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
        immutable: false,
        quasi_immutable: true,
        name: "",
    })
}

/// Concrete size descriptor for fixed-size object allocations.
#[derive(Debug)]
pub struct PyreSizeDescr {
    obj_size: usize,
    type_id: u32,
    /// descr.get_vtable() parity: ob_type pointer for NewWithVtable.
    /// optimize_new_with_vtable reads this to set VirtualInfo.known_class.
    vtable: usize,
}

/// GC type id for the `rclass.OBJECT` root — pyre's static `INSTANCE_TYPE`
/// PyType (`tp_name = "object"`). All `PyObject`-layout subclasses chain
/// their `parent` field to this id so `assign_inheritance_ids`
/// (normalizecalls.py:373-389) emits a `subclassrange_{min,max}` covering
/// every descendant. `GUARD_SUBCLASS(obj, &INSTANCE_TYPE)` then succeeds
/// for any `is_object` instance via `int_between(root.min, obj_typeid.min,
/// root.max)` (rclass.py:1133-1137 `ll_issubclass`).
pub const OBJECT_GC_TYPE_ID: u32 = 0;
pub const W_INT_GC_TYPE_ID: u32 = 1;
pub const W_FLOAT_GC_TYPE_ID: u32 = 2;
/// GC type id for JitFrame (jitframe.py:49 register_custom_trace_hook).
pub const JITFRAME_GC_TYPE_ID: u32 = 3;

impl Descr for PyreSizeDescr {
    fn index(&self) -> u32 {
        SIZE_DESCR_TAG | (self.obj_size as u32 & 0x0FFF_FFFF)
    }

    fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
        Some(self)
    }
}

impl SizeDescr for PyreSizeDescr {
    fn size(&self) -> usize {
        self.obj_size
    }

    fn type_id(&self) -> u32 {
        self.type_id
    }

    fn vtable(&self) -> usize {
        self.vtable
    }

    fn is_immutable(&self) -> bool {
        false
    }
}

/// Create a size descriptor for a fixed-size object.
pub fn make_size_descr(obj_size: usize) -> DescrRef {
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id: 0,
        vtable: 0,
    })
}

pub fn make_size_descr_with_type(obj_size: usize, type_id: u32) -> DescrRef {
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id,
        vtable: 0,
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

use pyre_interpreter::{PYNAMESPACE_VALUES_LEN_OFFSET, PYNAMESPACE_VALUES_OFFSET};
use pyre_object::floatobject::{FLOAT_FLOATVAL_OFFSET, W_FloatObject};
use pyre_object::intobject::W_IntObject;
use pyre_object::pyobject::OB_TYPE_OFFSET;
use pyre_object::rangeobject::{
    RANGE_ITER_CURRENT_OFFSET, RANGE_ITER_STEP_OFFSET, RANGE_ITER_STOP_OFFSET,
};
use pyre_object::{
    BOOL_BOOLVAL_OFFSET, DICT_LEN_OFFSET, FLOAT_ARRAY_HEAP_CAP_OFFSET, FLOAT_ARRAY_LEN_OFFSET,
    FLOAT_ARRAY_PTR_OFFSET, INT_ARRAY_HEAP_CAP_OFFSET, INT_ARRAY_LEN_OFFSET, INT_ARRAY_PTR_OFFSET,
    INT_INTVAL_OFFSET, PYOBJECT_ARRAY_HEAP_CAP_OFFSET, PYOBJECT_ARRAY_LEN_OFFSET, STR_LEN_OFFSET,
    W_ListObject, W_TupleObject,
};
use pyre_object::{FLOAT_TYPE, INT_TYPE};

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

pub fn list_strategy_descr() -> DescrRef {
    // RPython parity: list strategy is immutable once set at allocation.
    // Marking immutable lets the heap cache survive calls (boxing etc.),
    // eliminating repeated GetfieldGcI(strategy) + GuardValue(strategy)
    // within the same loop body.
    make_immutable_field_descr(
        std::mem::offset_of!(W_ListObject, strategy),
        1,
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

pub fn list_int_items_ptr_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_PTR_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn list_int_items_len_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_LEN_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn list_int_items_heap_cap_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_HEAP_CAP_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn list_float_items_ptr_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_PTR_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn list_float_items_len_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_LEN_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn list_float_items_heap_cap_descr() -> DescrRef {
    make_field_descr(
        std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_HEAP_CAP_OFFSET,
        8,
        Type::Int,
        false,
    )
}

pub fn tuple_items_ptr_descr() -> DescrRef {
    make_immutable_field_descr(
        std::mem::offset_of!(W_TupleObject, items),
        8,
        Type::Int,
        false,
    )
}

pub fn tuple_items_len_descr() -> DescrRef {
    make_immutable_field_descr(
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
    make_immutable_field_descr(INT_INTVAL_OFFSET, 8, Type::Int, true)
}

pub fn bool_boolval_descr() -> DescrRef {
    make_immutable_field_descr(BOOL_BOOLVAL_OFFSET, 1, Type::Int, false)
}

pub fn float_floatval_descr() -> DescrRef {
    make_immutable_field_descr(FLOAT_FLOATVAL_OFFSET, 8, Type::Float, false)
}

pub fn str_len_descr() -> DescrRef {
    make_immutable_field_descr(STR_LEN_OFFSET, 8, Type::Int, false)
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

// ── Object header & allocation descriptors ──────────────────────────

/// Field descriptor for ob_type (PyObject.ob_type pointer) — immutable.
/// heaptracker.py:66: `if name == 'typeptr': continue`
pub fn ob_type_descr() -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset: OB_TYPE_OFFSET,
        field_size: 8,
        field_type: Type::Int,
        signed: false,
        immutable: true,
        quasi_immutable: false,
        name: "typeptr",
    })
}

/// Size descriptor for W_IntObject allocation via NewWithVtable.
/// vtable = &INT_TYPE (ob_type for virtual materialization).
pub fn w_int_size_descr() -> DescrRef {
    Arc::new(PyreSizeDescr {
        obj_size: std::mem::size_of::<W_IntObject>(),
        type_id: W_INT_GC_TYPE_ID,
        vtable: &INT_TYPE as *const _ as usize,
    })
}

/// Size descriptor for W_FloatObject allocation via NewWithVtable.
/// vtable = &FLOAT_TYPE (ob_type for virtual materialization).
pub fn w_float_size_descr() -> DescrRef {
    Arc::new(PyreSizeDescr {
        obj_size: std::mem::size_of::<W_FloatObject>(),
        type_id: W_FLOAT_GC_TYPE_ID,
        vtable: &FLOAT_TYPE as *const _ as usize,
    })
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
