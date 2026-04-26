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

// PRE-EXISTING-ADAPTATION: tag bits in the high nibble of the descr
// index discriminate Field/Array/Size descrs. RPython stores all descrs
// in `setup_descrs`'s flat `all_descrs` list (descr.py:25-47) and
// recovers the type via `isinstance` on the descr object. Pyre cannot
// downcast `Arc<dyn Descr>` to a specific concrete trait via type id,
// so the index itself encodes the discriminant.
//
// The Field tag is also load-bearing for `FieldIndexDescr` in
// `optimizeopt/virtualize.rs:1620-1654` — that synthetic descriptor
// reconstructs `offset`/`field_size`/`field_type`/`signed` from the
// packed bits. Replacing the tag with a flat counter is contingent on
// that synthetic descriptor being replaced with a real
// `Arc<dyn FieldDescr>` lookup.
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
/// `all_fielddescrs` for a matching offset.
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
            sd.all_fielddescrs()
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
    /// descr.py:72 `self.all_fielddescrs = all_fielddescrs`.
    all_fielddescrs: Vec<Arc<dyn FieldDescr>>,
    /// descr.py:71 `self.gc_fielddescrs = gc_fielddescrs` — precomputed
    /// subset of `all_fielddescrs` via `is_pointer_field()`
    /// (heaptracker.py:94-95 + :70 filter).
    gc_fielddescrs: Vec<Arc<dyn FieldDescr>>,
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
// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID` live in `pyre-object`
// alongside the `W_IntObject` / `W_FloatObject` structs they describe,
// so `pyre-object`'s host-side allocators can reach them without a
// back-channel. Re-exported here for existing call sites.
pub use pyre_object::floatobject::W_FLOAT_GC_TYPE_ID;
pub use pyre_object::intobject::W_INT_GC_TYPE_ID;
/// GC type id for JitFrame (jitframe.py:49 register_custom_trace_hook).
pub const JITFRAME_GC_TYPE_ID: u32 = 3;
/// GC type id for JitVirtualRef (virtualref.py — JIT_VIRTUAL_REF).
pub const VREF_GC_TYPE_ID: u32 = 4;
/// GC type id for W_BoolObject. `bool` inherits from `int` per
/// `objectobject.py W_BoolObject.typedef`, so this chains to
/// `W_INT_GC_TYPE_ID` as its parent via `TypeInfo::object_subclass`
/// (heaptracker.py:23-30 setup_cache_gcstruct2vtable — one typeid per
/// distinct STRUCT, not per root layout).
pub const W_BOOL_GC_TYPE_ID: u32 = 5;
/// GC type id for W_RangeIterator. Inherits from `object`
/// (rangeobject.rs:10 RANGE_ITER_TYPE).
pub const RANGE_ITER_GC_TYPE_ID: u32 = 6;
// `W_LIST_GC_TYPE_ID` / `W_TUPLE_GC_TYPE_ID` live in `pyre-object`
// alongside their structs (matching W_INT/W_FLOAT pattern); re-exported
// here for existing call sites.
pub use pyre_object::listobject::W_LIST_GC_TYPE_ID;
/// GC type id for the variable-length backing block of `PyObjectArray`
/// (the list/tuple items storage). Shape matches `rlist.py:84,116`
/// `GcArray(OBJECTPTR)` — a `T_IS_VARSIZE` block with an 8-byte
/// single-slot `capacity` header (= upstream's GcArray length header,
/// rlist.py:251 `len(l.items)`) followed by inline `PyObjectRef`
/// items. Registered via `TypeInfo::varsize(8, 8, 0,
/// items_have_gc_ptrs=true, [])` so the GC walks each item slot as a
/// Ref (`gctypelayout.py:266-291 T_IS_VARSIZE / T_IS_GCARRAY_OF_GCPTR`);
/// live list length is stored on the enclosing `W_ListObject` wrapper
/// (`PyObjectArray.len`) to match rlist.py:116 `("length", Signed)`.
///
// `PY_OBJECT_ARRAY_GC_TYPE_ID` lives in `pyre-object` alongside the
// `ItemsBlock` struct it describes (matching W_INT/W_FLOAT/W_LIST/
// W_TUPLE pattern). Re-exported here for existing call sites.
pub use pyre_object::object_array::PY_OBJECT_ARRAY_GC_TYPE_ID;
pub use pyre_object::tupleobject::W_TUPLE_GC_TYPE_ID;
// GC type ids for `W_SpecialisedTupleObject_{ii,ff,oo}` live in
// `pyre-object` alongside the structs they describe; re-exported here
// for existing call sites. See
// `pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_*_GC_TYPE_ID`.
pub use pyre_object::specialisedtupleobject::{
    SPECIALISED_TUPLE_FF_GC_TYPE_ID, SPECIALISED_TUPLE_II_GC_TYPE_ID,
    SPECIALISED_TUPLE_OO_GC_TYPE_ID,
};
// `BUILTIN_CODE_GC_TYPE_ID` lives in `pyre-interpreter::gateway`
// alongside the `BuiltinCode` struct it describes. `FUNCTION_GC_TYPE_ID`
// lives in `pyre-interpreter::function` for the same reason and covers
// `Function`, `BuiltinFunction`, and `FunctionWithFixedCode` (the
// latter two are Rust type aliases of `Function`). Re-exported here
// for the JIT registration site (`pyre-jit/src/eval.rs`).
pub use pyre_interpreter::function::FUNCTION_GC_TYPE_ID;
pub use pyre_interpreter::gateway::BUILTIN_CODE_GC_TYPE_ID;
// `W_CELL_GC_TYPE_ID` lives in `pyre-object::cellobject` alongside the
// `W_CellObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::cellobject::W_CELL_GC_TYPE_ID;
// `W_METHOD_GC_TYPE_ID` lives in `pyre-object::methodobject` alongside
// the `W_MethodObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::methodobject::W_METHOD_GC_TYPE_ID;
// `W_SLICE_GC_TYPE_ID` lives in `pyre-object::sliceobject` alongside
// the `W_SliceObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::sliceobject::W_SLICE_GC_TYPE_ID;
// `W_SUPER_GC_TYPE_ID` lives in `pyre-object::superobject` alongside
// the `W_SuperObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::superobject::W_SUPER_GC_TYPE_ID;
// `W_PROPERTY_GC_TYPE_ID` / `W_STATICMETHOD_GC_TYPE_ID` /
// `W_CLASSMETHOD_GC_TYPE_ID` live in `pyre-object::propertyobject`
// alongside their structs. Re-exported for the JIT registration site.
pub use pyre_object::propertyobject::{
    W_CLASSMETHOD_GC_TYPE_ID, W_PROPERTY_GC_TYPE_ID, W_STATICMETHOD_GC_TYPE_ID,
};
// `W_UNION_GC_TYPE_ID` lives in `pyre-object::unionobject` alongside
// the `W_UnionType` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::unionobject::W_UNION_GC_TYPE_ID;
// `W_SEQ_ITER_GC_TYPE_ID` lives in `pyre-object::rangeobject`
// alongside the `W_SeqIterator` struct it describes. Re-exported for
// the JIT registration site.
pub use pyre_object::rangeobject::W_SEQ_ITER_GC_TYPE_ID;
// `W_COUNT_GC_TYPE_ID` / `W_REPEAT_GC_TYPE_ID` live in
// `pyre-object::itertoolsmodule` alongside the `W_Count` /
// `W_Repeat` structs they describe. Re-exported for the JIT
// registration site.
pub use pyre_object::itertoolsmodule::{W_COUNT_GC_TYPE_ID, W_REPEAT_GC_TYPE_ID};
// `W_MEMBER_GC_TYPE_ID` lives in `pyre-object::memberobject`
// alongside the `W_MemberDescr` struct it describes. Re-exported for
// the JIT registration site.
pub use pyre_object::memberobject::W_MEMBER_GC_TYPE_ID;
// `W_BYTES_GC_TYPE_ID` lives in `pyre-object::bytesobject` alongside
// the `W_BytesObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::bytesobject::W_BYTES_GC_TYPE_ID;
// `W_BYTEARRAY_GC_TYPE_ID` lives in `pyre-object::bytearrayobject`
// alongside the `W_BytearrayObject` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::bytearrayobject::W_BYTEARRAY_GC_TYPE_ID;
// `W_DICT_GC_TYPE_ID` lives in `pyre-object::dictobject` alongside
// the `W_DictObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::dictobject::W_DICT_GC_TYPE_ID;
// `W_SET_GC_TYPE_ID` lives in `pyre-object::setobject` alongside the
// `W_SetObject` struct it describes (covers both `set` and
// `frozenset` PyTypes — same Rust struct). Re-exported for the JIT
// registration site.
pub use pyre_object::setobject::W_SET_GC_TYPE_ID;
// `W_EXCEPTION_GC_TYPE_ID` lives in `pyre-object::excobject`
// alongside the `W_ExceptionObject` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::excobject::W_EXCEPTION_GC_TYPE_ID;
// `W_GENERATOR_GC_TYPE_ID` lives in `pyre-object::generatorobject`
// alongside the `W_GeneratorObject` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::generatorobject::W_GENERATOR_GC_TYPE_ID;
// `W_TYPE_GC_TYPE_ID` lives in `pyre-object::typeobject` alongside
// the `W_TypeObject` struct it describes. Re-exported for the JIT
// registration site. (`TYPE_TYPE` is in `all_foreign_pytypes()` but
// the foreign-pytype loop's `sizeof(PyObject)` approximation would
// drastically under-count the W_TypeObject payload.)
pub use pyre_object::typeobject::W_TYPE_GC_TYPE_ID;
// `W_STR_GC_TYPE_ID` / `W_LONG_GC_TYPE_ID` / `W_MODULE_GC_TYPE_ID`
// live alongside their structs in
// `pyre-object::{strobject, longobject, moduleobject}`. Re-exported
// for the JIT registration site. `W_InstanceObject` shares
// `OBJECT_GC_TYPE_ID` with the `object` root (see comment on the
// struct) so it has no separate id.
pub use pyre_object::longobject::W_LONG_GC_TYPE_ID;
pub use pyre_object::moduleobject::W_MODULE_GC_TYPE_ID;
pub use pyre_object::strobject::W_STR_GC_TYPE_ID;

fn field_descr_from_group(group: &PyreObjectDescrGroup, index: usize) -> DescrRef {
    let field_descr = group
        .size_descr
        .all_fielddescrs
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
        let all_fielddescrs: Vec<Arc<dyn FieldDescr>> = fields
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
        // descr.py:123-126 precompute both lists; `gc_fielddescrs` is
        // `all_fielddescrs(only_gc=True)` per heaptracker.py:94-95.
        let gc_fielddescrs: Vec<Arc<dyn FieldDescr>> = all_fielddescrs
            .iter()
            .filter(|fd| fd.is_pointer_field())
            .cloned()
            .collect();
        PyreSizeDescr {
            obj_size,
            type_id,
            vtable,
            all_fielddescrs,
            gc_fielddescrs,
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
        W_BOOL_GC_TYPE_ID,
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
        RANGE_ITER_GC_TYPE_ID,
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
    // Upstream `rpython/rtyper/lltypesystem/rlist.py:116`
    //     GcStruct("list", ("length", Signed), ("items", Ptr(ITEMARRAY)))
    // The parity-field pair is `(length, items)`. `strategy` +
    // `int_items` / `float_items` are pyre-only PRE-EXISTING-
    // ADAPTATIONs for the PyPy interp-level strategy split.
    build_object_descr_group(
        std::mem::size_of::<W_ListObject>(),
        W_LIST_GC_TYPE_ID,
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        &[
            // rlist.py:116 `("length", Signed)`. Mutable: Object-strategy
            // push/pop/insert/remove/drain update it.
            (
                "W_ListObject.length",
                std::mem::offset_of!(W_ListObject, length),
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            // rlist.py:116 `("items", Ptr(GcArray(OBJECTPTR)))`. Points
            // at the `ItemsBlock` GcArray body. Mutable: re-pointed when
            // the Object-strategy storage is reallocated
            // (`list.object_grow` → `grow_list_items_block`) or when the
            // strategy switches.
            (
                "W_ListObject.items",
                std::mem::offset_of!(W_ListObject, items),
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                // `W_ListObject.strategy` is MUTABLE: `switch_to_object_strategy`
                // flips it from Integer/Float to Object when an
                // incompatible item is stored. A trace that folded
                // `strategy == Float` at trace-time into a constant would
                // then read from `float_items.ptr` (empty after the
                // switch) and dereference garbage — spectral_norm n=10
                // SIGSEGV root cause diagnosed in
                // memory/spectral_norm_small_n_crash_2026_04_17.md.
                //
                // Upstream PyPy handles this with a quasi-immutable flag
                // + invalidate_compiled_code hook on strategy change;
                // pyre has no such hook yet, so `strategy` stays
                // plain-mutable. NEW-DEVIATION — strategy split itself
                // is a pyre-only adaptation vs rlist.py.
                "W_ListObject.strategy",
                std::mem::offset_of!(W_ListObject, strategy),
                1,
                Type::Int,
                false,
                false,
                false,
            ),
            // Integer-strategy typed storage (pyre-only
            // PRE-EXISTING-ADAPTATION vs listobject.py's
            // IntegerListStrategy at the interp level — upstream keeps
            // the unwrap inline and doesn't add a separate backing
            // array).
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
            // Float-strategy typed storage.
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
        ],
    )
});

static W_TUPLE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `pypy/objspace/std/tupleobject.py:376-390` `W_TupleObject` stores
    // `wrappeditems: list` with `_immutable_fields_ =
    // ['wrappeditems[*]']`. After translation this becomes
    // `Ptr(GcArray(OBJECTPTR))`; `wrappeditems[*]` flows into both
    // the field descr (`immutable: true`) AND the GcArray contents
    // (read via `getfield_gc_pure_r`). Length comes from the GcArray
    // header via `arraylen_gc(items_block)` — no inline length cache.
    build_object_descr_group(
        std::mem::size_of::<W_TupleObject>(),
        W_TUPLE_GC_TYPE_ID,
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        &[
            // `Ptr(GcArray(OBJECTPTR))` — wrappeditems body. Immutable.
            (
                "W_TupleObject.wrappeditems",
                std::mem::offset_of!(W_TupleObject, wrappeditems),
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
    )
});

static SPECIALISED_TUPLE_II_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `specialisedtupleobject.py:34` `_immutable_fields_ = ['value0',
    // 'value1']` — both fields immutable. Inline-field shape, no array
    // indirection.
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group(
        std::mem::size_of::<W_SpecialisedTupleObject_ii>(),
        SPECIALISED_TUPLE_II_GC_TYPE_ID,
        &SPECIALISED_TUPLE_II_TYPE as *const _ as usize,
        &[
            (
                "W_SpecialisedTupleObject_ii.value0",
                SPECIALISED_TUPLE_II_VALUE0_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
            (
                "W_SpecialisedTupleObject_ii.value1",
                SPECIALISED_TUPLE_II_VALUE1_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
        ],
    )
});

static SPECIALISED_TUPLE_FF_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group(
        std::mem::size_of::<W_SpecialisedTupleObject_ff>(),
        SPECIALISED_TUPLE_FF_GC_TYPE_ID,
        &SPECIALISED_TUPLE_FF_TYPE as *const _ as usize,
        &[
            (
                "W_SpecialisedTupleObject_ff.value0",
                SPECIALISED_TUPLE_FF_VALUE0_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            (
                "W_SpecialisedTupleObject_ff.value1",
                SPECIALISED_TUPLE_FF_VALUE1_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
        ],
    )
});

static SPECIALISED_TUPLE_OO_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group(
        std::mem::size_of::<W_SpecialisedTupleObject_oo>(),
        SPECIALISED_TUPLE_OO_GC_TYPE_ID,
        &SPECIALISED_TUPLE_OO_TYPE as *const _ as usize,
        &[
            (
                "W_SpecialisedTupleObject_oo.value0",
                SPECIALISED_TUPLE_OO_VALUE0_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "W_SpecialisedTupleObject_oo.value1",
                SPECIALISED_TUPLE_OO_VALUE1_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
    )
});

static DICT_STORAGE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_interpreter::DictStorage>(),
        0,
        0,
        &[
            (
                "DictStorage.values.ptr",
                DICT_STORAGE_VALUES_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "DictStorage.values.len",
                DICT_STORAGE_VALUES_LEN_OFFSET,
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
                "PyFrame.last_instr",
                crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "PyFrame.pycode",
                crate::frame_layout::PYFRAME_PYCODE_OFFSET,
                8,
                Type::Ref,
                true,
                false,
                false,
            ),
            (
                "PyFrame.w_globals",
                crate::frame_layout::PYFRAME_W_GLOBALS_OFFSET,
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
    fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.all_fielddescrs
    }
    fn gc_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.gc_fielddescrs
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
        all_fielddescrs: Vec::new(),
        gc_fielddescrs: Vec::new(),
    })
}

pub fn make_size_descr_with_type(obj_size: usize, type_id: u32) -> DescrRef {
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id,
        vtable: 0,
        all_fielddescrs: Vec::new(),
        gc_fielddescrs: Vec::new(),
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

use pyre_interpreter::{DICT_STORAGE_VALUES_LEN_OFFSET, DICT_STORAGE_VALUES_OFFSET};
use pyre_object::floatobject::{FLOAT_FLOATVAL_OFFSET, W_FloatObject};
use pyre_object::intobject::W_IntObject;
use pyre_object::pyobject::OB_TYPE_OFFSET;
use pyre_object::rangeobject::{
    RANGE_ITER_CURRENT_OFFSET, RANGE_ITER_STEP_OFFSET, RANGE_ITER_STOP_OFFSET,
};
use pyre_object::{
    BOOL_BOOLVAL_OFFSET, DICT_LEN_OFFSET, FLOAT_ARRAY_HEAP_CAP_OFFSET, FLOAT_ARRAY_LEN_OFFSET,
    FLOAT_ARRAY_PTR_OFFSET, INT_ARRAY_HEAP_CAP_OFFSET, INT_ARRAY_LEN_OFFSET, INT_ARRAY_PTR_OFFSET,
    INT_INTVAL_OFFSET, STR_LEN_OFFSET, W_ListObject, W_TupleObject,
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

/// rlist.py:116 `l.length` — live length of a list under the Object
/// strategy. Under Integer/Float strategies this field is 0 and
/// consumers must dispatch on `list.strategy` first.
pub fn list_length_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 0)
}

/// rlist.py:116 `l.items: Ptr(GcArray(OBJECTPTR))` — pointer to the
/// `ItemsBlock` GcArray body. Callers that need items[i] must combine
/// with the `PY_OBJECT_ARRAY` array descr (item_size=8, Ref,
/// base_size=`ITEMS_BLOCK_ITEMS_OFFSET`); callers that need capacity
/// must issue `ArraylenGc` against the same array descr.
pub fn list_items_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 1)
}

pub fn list_strategy_descr() -> DescrRef {
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

/// `Ptr(GcArray(OBJECTPTR))` — `wrappeditems` body per
/// `tupleobject.py:381` `_immutable_fields_ = ['wrappeditems[*]']`.
/// Immutable. Length comes from `arraylen_gc(items_block,
/// pyobject_gcarray_descr)` against the GcArray header — no
/// `tuple_length_descr` exists per upstream tupleobject.py:376-390
/// (`W_TupleObject` carries `wrappeditems` only).
pub fn tuple_wrappeditems_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ii.value0` — inline `i64` per
/// `specialisedtupleobject.py:34-44`. Immutable.
pub fn specialised_tuple_ii_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ii.value1` — inline `i64`. Immutable.
pub fn specialised_tuple_ii_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 1)
}

/// `W_SpecialisedTupleObject_ff.value0` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ff.value1` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 1)
}

/// `W_SpecialisedTupleObject_oo.value0` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_oo.value1` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 1)
}

/// `ItemsBlock.capacity` — the GcArray length header at offset 0 of
/// an `ItemsBlock`, matching `rlist.py:84/251` `len(l.items)`
/// (allocated capacity, not live length). Immutable: once a block is
/// allocated the capacity is fixed; resize allocates a fresh block.
/// Callers combine `list_items_descr()` / `tuple_wrappeditems_descr()`
/// → `ItemsBlock*` with this descr to read the block's allocated size.
pub fn items_block_capacity_descr() -> DescrRef {
    make_immutable_field_descr(0, 8, Type::Int, false)
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

pub fn dict_storage_values_ptr_descr() -> DescrRef {
    field_descr_from_group(&DICT_STORAGE_DESCR_GROUP, 0)
}

pub fn dict_storage_values_len_descr() -> DescrRef {
    field_descr_from_group(&DICT_STORAGE_DESCR_GROUP, 1)
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

/// Size descriptor for W_BoolObject allocation via NewWithVtable.
/// vtable = &BOOL_TYPE; type_id = 0 (bool reuses the OBJECT root id).
pub fn w_bool_size_descr() -> DescrRef {
    W_BOOL_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_RangeIterator allocation via NewWithVtable.
/// vtable = &RANGE_ITER_TYPE; type_id = 0.
pub fn w_range_iter_size_descr() -> DescrRef {
    RANGE_ITER_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_FloatObject allocation via NewWithVtable.
/// vtable = &FLOAT_TYPE (ob_type for virtual materialization).
pub fn w_float_size_descr() -> DescrRef {
    W_FLOAT_DESCR_GROUP.size_descr.clone()
}

/// Cached SizeDescr for the host PyFrame virtualizable.
///
/// RPython's `GcCache.get_size_descr()` returns a stable descriptor
/// object for a given struct. Pyre keeps the PyFrame descriptors in the
/// `PYFRAME_DESCR_GROUP` singleton, so callers that need the parent
/// SizeDescr for `VirtualizableInfo::finalize_arc` must reuse that
/// cached Arc instead of allocating a fresh ephemeral `SizeDescr`.
pub fn pyframe_size_descr() -> DescrRef {
    PYFRAME_DESCR_GROUP.size_descr.clone()
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

pub fn pyframe_dict_storage_descr() -> DescrRef {
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

/// resume.py:1124-1132: allocate_raw_buffer uses
/// callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR) to get the calldescr.
pub fn make_raw_malloc_calldescr() -> DescrRef {
    majit_ir::make_raw_malloc_calldescr()
}

/// descr.py:273 ArrayDescr for array-of-structs (FLAG_STRUCT).
/// resume.py:749: allocate_array(self.size, self.arraydescr, clear=True).
pub fn make_struct_array_descr(descr_index: u32, base_size: usize, item_size: usize) -> DescrRef {
    use majit_ir::descr::{ArrayFlag, SimpleArrayDescr};
    Arc::new(SimpleArrayDescr::with_flag(
        descr_index,
        base_size,
        item_size,
        0,
        Type::Void,
        ArrayFlag::Struct,
    ))
}

/// descr.py:384 InteriorFieldDescr for SETINTERIORFIELD_GC.
/// assert arraydescr.flag == FLAG_STRUCT.
/// llmodel.py:648-665: bh_setinteriorfield_gc_{i,r,f} computes
/// offset = arraydescr.basesize + itemindex * itemsize + fielddescr.offset.
pub fn make_interior_field_descr(
    array_descr_index: u32,
    base_size: usize,
    item_size: usize,
    field_offset: usize,
    field_size: usize,
    field_type: u8, // 0=ref, 1=int, 2=float
    field_descr_index: u32,
) -> DescrRef {
    use majit_ir::descr::{
        ArrayFlag, SimpleArrayDescr, SimpleFieldDescr, SimpleInteriorFieldDescr,
    };
    let tp = match field_type {
        0 => Type::Ref,
        2 => Type::Float,
        _ => Type::Int,
    };
    // descr.py:387: assert arraydescr.flag == FLAG_STRUCT
    let array_descr = Arc::new(SimpleArrayDescr::with_flag(
        array_descr_index,
        base_size,
        item_size,
        0,
        Type::Void,
        ArrayFlag::Struct,
    ));
    let field_descr = Arc::new(SimpleFieldDescr::new(
        field_descr_index,
        field_offset,
        field_size,
        tp,
        true, // immutable (struct fields in array-of-struct)
    ));
    Arc::new(SimpleInteriorFieldDescr::new(
        field_descr_index,
        array_descr,
        field_descr,
    ))
}
