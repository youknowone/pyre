//! Field descriptors for JIT IR operations.
//!
//! GetfieldGcI, GetfieldGcR, and SetfieldGc require a `DescrRef`
//! carrying field offset, size, and type information. This module
//! provides a concrete `PyreFieldDescr` implementing majit's
//! `FieldDescr` trait for pyre's `#[repr(C)]` object layout.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;

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
    index_in_parent: usize,
    parent_descr: Option<Weak<dyn Descr>>,
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
    fn index_in_parent(&self) -> usize {
        self.index_in_parent
    }
    fn get_parent_descr(&self) -> Option<DescrRef> {
        self.parent_descr
            .as_ref()
            .and_then(|parent| parent.upgrade())
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
        index_in_parent: 0,
        parent_descr: None,
    })
}

/// Create a field descr with an explicit parent SizeDescr.
///
/// RPython parity: `fielddescr.get_parent_descr()` returns the owning
/// struct's SizeDescr, enabling `info.py:180 init_fields(parent_descr,
/// index)`. Without parent_descr, `descr_index()` falls back to
/// `stable_field_index` (a hash) instead of `index_in_parent` (a small
/// sequential index), causing OOM in `ensure_field_descr_slot`.
///
/// The `index_in_parent` is computed by scanning the parent SizeDescr's
/// `all_field_descrs` for a matching offset.
pub fn make_field_descr_with_parent(
    parent: DescrRef,
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    // Derive index_in_parent from the parent SizeDescr's field list.
    let index = parent
        .as_size_descr()
        .and_then(|sd| {
            sd.all_field_descrs()
                .iter()
                .enumerate()
                .find(|(_, fd)| fd.as_field_descr().map_or(false, |f| f.offset() == offset))
                .map(|(i, _)| i)
        })
        .unwrap_or(0);
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
        immutable: false,
        quasi_immutable: false,
        name: "",
        index_in_parent: index,
        parent_descr: Some(Arc::downgrade(&parent)),
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
        index_in_parent: 0,
        parent_descr: None,
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
        index_in_parent: 0,
        parent_descr: None,
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
        index_in_parent: 0,
        parent_descr: None,
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
    all_field_descrs: Vec<Arc<dyn FieldDescr>>,
}

struct PyreObjectDescrGroup {
    size_descr: Arc<PyreSizeDescr>,
}

/// GC type id for the `rclass.OBJECT` root — pyre's static `INSTANCE_TYPE`
/// PyType (`name = "object"`). All `PyObject`-layout subclasses chain
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

fn field_descr_from_group(group: &PyreObjectDescrGroup, index: usize) -> DescrRef {
    let field_descr = group
        .size_descr
        .all_field_descrs
        .get(index)
        .expect("field descriptor index out of bounds")
        .clone();
    field_descr
}

fn build_object_descr_group(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
) -> PyreObjectDescrGroup {
    let size_descr = Arc::new_cyclic(|weak_size: &Weak<PyreSizeDescr>| {
        let parent_descr: Weak<dyn Descr> = weak_size.clone();
        let all_field_descrs: Vec<Arc<dyn FieldDescr>> = fields
            .iter()
            .enumerate()
            .map(
                |(
                    index_in_parent,
                    &(name, offset, field_size, field_type, signed, immutable, quasi_immutable),
                )| {
                    Arc::new(PyreFieldDescr {
                        offset,
                        field_size,
                        field_type,
                        signed,
                        immutable,
                        quasi_immutable,
                        name,
                        index_in_parent,
                        parent_descr: Some(parent_descr.clone()),
                    }) as Arc<dyn FieldDescr>
                },
            )
            .collect();
        PyreSizeDescr {
            obj_size,
            type_id,
            vtable,
            all_field_descrs,
        }
    });
    PyreObjectDescrGroup { size_descr }
}

static W_INT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<W_IntObject>(),
        W_INT_GC_TYPE_ID,
        &INT_TYPE as *const _ as usize,
        &[(
            "W_IntObject.intval",
            INT_INTVAL_OFFSET,
            8,
            Type::Int,
            true,
            true,
            false,
        )],
    )
});

static W_FLOAT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<W_FloatObject>(),
        W_FLOAT_GC_TYPE_ID,
        &FLOAT_TYPE as *const _ as usize,
        &[(
            "W_FloatObject.floatval",
            FLOAT_FLOATVAL_OFFSET,
            8,
            Type::Float,
            false,
            true,
            false,
        )],
    )
});

static W_BOOL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_object::boolobject::W_BoolObject>(),
        0,
        &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
        &[(
            "W_BoolObject.boolval",
            BOOL_BOOLVAL_OFFSET,
            1,
            Type::Int,
            false,
            true,
            false,
        )],
    )
});

static RANGE_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_object::rangeobject::W_RangeIterator>(),
        0,
        &pyre_object::rangeobject::RANGE_ITER_TYPE as *const _ as usize,
        &[
            (
                "W_RangeIterator.current",
                RANGE_ITER_CURRENT_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "W_RangeIterator.stop",
                RANGE_ITER_STOP_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "W_RangeIterator.step",
                RANGE_ITER_STEP_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
    )
});

static W_LIST_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<W_ListObject>(),
        0,
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        &[
            (
                "W_ListObject.items.ptr",
                std::mem::offset_of!(W_ListObject, items),
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.strategy",
                std::mem::offset_of!(W_ListObject, strategy),
                1,
                Type::Int,
                false,
                true,
                false,
            ),
            (
                "W_ListObject.items.len",
                std::mem::offset_of!(W_ListObject, items) + PYOBJECT_ARRAY_LEN_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.int_items.ptr",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_PTR_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.int_items.len",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_LEN_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.int_items.heap_cap",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_HEAP_CAP_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.float_items.ptr",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_PTR_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.float_items.len",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_LEN_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.float_items.heap_cap",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_HEAP_CAP_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.items.heap_cap",
                std::mem::offset_of!(W_ListObject, items) + PYOBJECT_ARRAY_HEAP_CAP_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
        ],
    )
});

static W_TUPLE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<W_TupleObject>(),
        0,
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        &[
            (
                "W_TupleObject.items.ptr",
                std::mem::offset_of!(W_TupleObject, items),
                8,
                Type::Int,
                false,
                true,
                false,
            ),
            (
                "W_TupleObject.items.len",
                std::mem::offset_of!(W_TupleObject, items) + PYOBJECT_ARRAY_LEN_OFFSET,
                8,
                Type::Int,
                false,
                true,
                false,
            ),
        ],
    )
});

static PYNAMESPACE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_interpreter::PyNamespace>(),
        0,
        0,
        &[
            (
                "PyNamespace.values.ptr",
                PYNAMESPACE_VALUES_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "PyNamespace.values.len",
                PYNAMESPACE_VALUES_LEN_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
        ],
    )
});

static PYFRAME_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
        0,
        0,
        &[
            (
                "PyFrame.locals_cells_stack_w",
                crate::frame_layout::PYFRAME_LOCALS_CELLS_STACK_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "PyFrame.valuestackdepth",
                crate::frame_layout::PYFRAME_VALUESTACKDEPTH_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "PyFrame.next_instr",
                crate::frame_layout::PYFRAME_NEXT_INSTR_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "PyFrame.code",
                crate::frame_layout::PYFRAME_CODE_OFFSET,
                8,
                Type::Ref,
                true,
                false,
                false,
            ),
            (
                "PyFrame.namespace",
                crate::frame_layout::PYFRAME_NAMESPACE_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
    )
});

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
    fn all_field_descrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.all_field_descrs
    }
    /// descr.py SizeDescr.is_object: every PyreSizeDescr that ships a
    /// vtable corresponds to a Python object (W_IntObject / W_ListObject /
    /// W_RangeIterator / …). `ensure_ptr_info_arg0` (optimizer.py:480)
    /// uses this to dispatch InstancePtrInfo vs StructPtrInfo.
    fn is_object(&self) -> bool {
        self.vtable != 0
    }
}

/// Create a size descriptor for a fixed-size object.
pub fn make_size_descr(obj_size: usize) -> DescrRef {
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id: 0,
        vtable: 0,
        all_field_descrs: Vec::new(),
    })
}

pub fn make_size_descr_with_type(obj_size: usize, type_id: u32) -> DescrRef {
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id,
        vtable: 0,
        all_field_descrs: Vec::new(),
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

/// Field descriptor for `PyObject.w_class` (Ref, mutable).
///
/// PyObject layout: [ob_type(8)] [w_class(8)]
/// The w_class field holds the Python class for all object types.
///
/// RPython parity: jit.promote(w_obj.__class__) reads typeptr via
/// getfield_gc_r then GUARD_VALUE. This is the pyre equivalent — a
/// field read on the common PyObject header.
///
/// Mutable because __class__ assignment can change it.
pub fn w_class_descr() -> DescrRef {
    make_field_descr(pyre_object::pyobject::W_CLASS_OFFSET, 8, Type::Ref, false)
}

/// Alias for backward compatibility — same as w_class_descr().
pub fn instance_w_type_descr() -> DescrRef {
    w_class_descr()
}

/// Field descriptor for `W_RangeIterator.current` (i64, signed).
pub fn range_iter_current_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 0)
}

/// Field descriptor for `W_RangeIterator.stop` (i64, signed).
pub fn range_iter_stop_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 1)
}

/// Field descriptor for `W_RangeIterator.step` (i64, signed).
pub fn range_iter_step_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 2)
}

pub fn list_items_ptr_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 0)
}

pub fn list_strategy_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 1)
}

pub fn list_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 2)
}

pub fn list_int_items_ptr_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 3)
}

pub fn list_int_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 4)
}

pub fn list_int_items_heap_cap_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 5)
}

pub fn list_float_items_ptr_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 6)
}

pub fn list_float_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 7)
}

pub fn list_float_items_heap_cap_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 8)
}

pub fn tuple_items_ptr_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 0)
}

pub fn tuple_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 1)
}

pub fn list_items_heap_cap_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 9)
}

pub fn int_intval_descr() -> DescrRef {
    field_descr_from_group(&W_INT_DESCR_GROUP, 0)
}

pub fn bool_boolval_descr() -> DescrRef {
    field_descr_from_group(&W_BOOL_DESCR_GROUP, 0)
}

pub fn float_floatval_descr() -> DescrRef {
    field_descr_from_group(&W_FLOAT_DESCR_GROUP, 0)
}

pub fn str_len_descr() -> DescrRef {
    make_immutable_field_descr(STR_LEN_OFFSET, 8, Type::Int, false)
}

pub fn dict_len_descr() -> DescrRef {
    make_field_descr(DICT_LEN_OFFSET, 8, Type::Int, false)
}

pub fn namespace_values_ptr_descr() -> DescrRef {
    field_descr_from_group(&PYNAMESPACE_DESCR_GROUP, 0)
}

pub fn namespace_values_len_descr() -> DescrRef {
    field_descr_from_group(&PYNAMESPACE_DESCR_GROUP, 1)
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
        index_in_parent: 0,
        parent_descr: None,
    })
}

/// Size descriptor for W_IntObject allocation via NewWithVtable.
/// vtable = &INT_TYPE (ob_type for virtual materialization).
pub fn w_int_size_descr() -> DescrRef {
    W_INT_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_FloatObject allocation via NewWithVtable.
/// vtable = &FLOAT_TYPE (ob_type for virtual materialization).
pub fn w_float_size_descr() -> DescrRef {
    W_FLOAT_DESCR_GROUP.size_descr.clone()
}

pub fn pyframe_locals_cells_stack_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 0)
}

pub fn pyframe_stack_depth_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 1)
}

pub fn pyframe_next_instr_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 2)
}

pub fn pyframe_code_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 3)
}

pub fn pyframe_namespace_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 4)
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

/// Minimal call descriptor for CALL_I (int-returning function call).
/// resume.py:1124-1132: allocate_raw_buffer uses callinfo_for_oopspec
/// to get a calldescr for OS_RAW_MALLOC_VARSIZE_CHAR. pyre uses a
/// simple array-descr-shaped descriptor since the backend only needs
/// to know the return type (Int) and argument layout.
pub fn make_call_descr_int() -> DescrRef {
    make_array_descr(0, 8, Type::Int, false)
}
