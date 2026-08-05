//! Field descriptors for JIT IR operations.
//!
//! GetfieldGcI, GetfieldGcR, and SetfieldGc require a `DescrRef`
//! carrying field offset, size, and type information. This module
//! provides a concrete `PyreFieldDescr` implementing majit's
//! `FieldDescr` trait for pyre's `#[repr(C)]` object layout.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::{
    ArrayDescr, Descr, DescrRef, FieldDescr, IndexMapExt, JitCodeDescr, SizeDescr, SwitchDescr,
    Type,
};

// TODO: tag bits in the high nibble of the descr
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
const HEADERLESS_SIZE_OWNER_MARKER: &str = "__majit_headerless_size__";

// Reserved, hand-assigned indices for mutable-cell payload fields. The
// runtime `HeapCache` keys entries by `descr.index()`; `stable_field_index`
// derives that key from `(offset, field_size, field_type, signed)` alone, so
// two distinct structs sharing a field layout collapse onto one `CacheEntry`.
// `IntMutableCell.intvalue` (offset 16 / 8 / Int) lands on the exact slot of
// `W_IntObject.intval` (signed) or `W_ListObject.length` (unsigned) — no
// `signed` value avoids both. The read-only LOAD fold tolerated the collision,
// but the store fold's `SetfieldGc` plus the const-cell `last_const_box`
// heuristic thrash the shared `CacheEntry` against colliding list/int reads.
// A reserved index above the FIELD/ARRAY/SIZE tag ranges gives the cell
// payload a private `CacheEntry`; no code inspects the tag bits of `index()`.
const CELL_DESCR_TAG: u32 = 0x4000_0000;
const INT_MUTABLE_CELL_VALUE_INDEX: u32 = CELL_DESCR_TAG;

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
    /// `effectinfo.py:465 compute_bitstrings` ei_index. `u32::MAX` until
    /// the codewriter publishes its `field_index` (`effectinfo.py:307-311`)
    /// onto this descr.
    ei_index: AtomicU32,
}

/// Structural key for `ARRAY_DESCR_REGISTRY`. Combination of all fields
/// that PyPy treats as part of `ArrayDescr` identity (`descr.py:273-279
/// + lendescr`). Two array descrs sharing this tuple share the same
/// `descr_id`.
///
/// `array_type_id` carries the codewriter lltype-identity proxy
/// (`majit-translate/src/codewriter/call.rs::DescrIndexRegistry::array_index`
/// key) so the runtime registry's identity domain matches PyPy's
/// `gccache._cache_array[ARRAY_OR_STRUCT]` (`descr.py:348-360`) keyed
/// on the actual lltype object: two BhDescr::Array entries that
/// disagree only on the Rust type spelling
/// (e.g. `"Vec<Foo>"` vs `"Vec<Bar>"` with both at `type_id == 0`)
/// land on distinct registry slots, preventing the second
/// `set_ei_index` from clobbering the first.
///
/// `None` for legacy descrs minted by pyre-jit-trace internal
/// factories with no source-level array_type_id context; two `None`
/// entries still collide on the remaining structural tuple just as
/// the pre-bridge baseline did.
#[derive(Hash, Eq, PartialEq, Clone)]
struct ArrayDescrKey {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type_bits: u32,
    signed: bool,
    len_offset: Option<usize>,
    array_type_id: Option<String>,
}

static NEXT_ARRAY_DESCR_ID: AtomicU32 = AtomicU32::new(0);

/// Maximum sequential ARRAY descr id. Bits 0-27 of the index are
/// available below `ARRAY_DESCR_TAG`; bit 28 is reserved by
/// `FIELD_DESCR_TAG`.
const ARRAY_DESCR_ID_MAX: u32 = 1 << 28;

static ARRAY_DESCR_REGISTRY: LazyLock<Mutex<indexmap::IndexMap<ArrayDescrKey, DescrRef>>> =
    LazyLock::new(|| Mutex::new(indexmap::IndexMap::new()));

fn alloc_array_descr_id() -> u32 {
    let id = NEXT_ARRAY_DESCR_ID.fetch_add(1, Ordering::Relaxed);
    assert!(
        id < ARRAY_DESCR_ID_MAX,
        "array descr registry exhausted (>2^28 instances) — index() bit 28 belongs to FIELD_DESCR_TAG"
    );
    id
}

/// `descr.py:241-254 get_type_flag(ARRAY.OF)` — element classification
/// for the runtime array mint.  Preserves the predicates the dropped
/// `PyreArrayDescr` reported: `is_item_signed` (Signed vs Unsigned for
/// ints), `is_array_of_pointers` (Pointer), `is_array_of_floats`
/// (Float).  Non-struct only — struct arrays are minted as
/// `SimpleArrayDescr(FLAG_STRUCT)` in `make_descr_from_bh`.
fn runtime_array_flag(item_type: Type, signed: bool) -> majit_ir::descr::ArrayFlag {
    use majit_ir::descr::ArrayFlag;
    match item_type {
        Type::Ref => ArrayFlag::Pointer,
        Type::Float => ArrayFlag::Float,
        Type::Int if signed => ArrayFlag::Signed,
        Type::Int => ArrayFlag::Unsigned,
        Type::Void => ArrayFlag::Void,
    }
}

fn get_or_create_array_descr(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_offset: Option<usize>,
) -> DescrRef {
    get_or_create_array_descr_with_full_id(
        base_size, item_size, type_id, item_type, signed, len_offset, None,
    )
}

fn get_or_create_array_descr_with_full_id(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_offset: Option<usize>,
    array_type_id: Option<String>,
) -> DescrRef {
    let key = ArrayDescrKey {
        base_size,
        item_size,
        type_id,
        item_type_bits: type_bits(item_type),
        signed,
        len_offset,
        array_type_id,
    };
    let mut cache = ARRAY_DESCR_REGISTRY
        .lock()
        .expect("ARRAY_DESCR_REGISTRY poisoned");
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    // `descr.py:348-378 get_array_descr(gccache, ARRAY)`: when the
    // caller has an `array_type_id` (the codewriter lltype identity
    // proxy), `gc_cache._cache_array[LLType::Array(path_hash(atid))]`
    // is the authoritative cache slot — consult it FIRST so a prior
    // analyzer-side `gc_cache.get_array_descr` mint
    // (`SimpleArrayDescr`) is reused instead of layered under a fresh
    // runtime `SimpleArrayDescr`.  Matches PyPy `cpu.arraydescrof(ARRAY)`
    // per-ARRAY object identity — both analyzer and pyre runtime
    // consumers share one Arc per `LLType::Array(path_hash(atid))`.
    if let Some(ref atid) = key.array_type_id {
        let gc_key = majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(atid));
        if let Some(existing) = majit_ir::descr::gc_cache()
            .lock()
            .unwrap()
            ._cache_array
            .get(&gc_key)
            .cloned()
        {
            // Memoise into the local structural cache so subsequent
            // `get_or_create_array_descr_with_full_id` calls with the
            // same structural key hit the local fast path without
            // re-consulting gc_cache.
            cache.insert(key.clone(), existing.clone());
            return existing;
        }
    }
    let descr_id = alloc_array_descr_id();
    // `array_type_id` Some → `LLType::Array(path_hash(atid))` cache
    // slot (analyzer ↔ runtime convergence path).
    // `array_type_id` None but `type_id != 0` → no codewriter
    // lltype-identity carrier but a stable GC-tid is available
    // (`make_array_descr_with_type` path).  Widening that tid to
    // u64 preserves per-tid identity in `BhDescr::Array.type_id`,
    // matching the behaviour producer sites in `eval.rs` /
    // `assembler.rs` / `jitcode.rs` relied on before the
    // `cache_key()` migration — without this fallback, every
    // `PY_OBJECT_ARRAY_GC_TYPE_ID`-class runtime descr collapsed
    // onto slot 0 at the `BhDescr` boundary.
    // `array_type_id` None and `type_id == 0` → no identity carrier
    // at all (legacy `make_array_descr` no-identity path); stay 0.
    let cache_key = match key.array_type_id.as_deref() {
        Some(atid) => majit_ir::descr::path_hash(atid),
        None if type_id != 0 => type_id as u64,
        None => 0,
    };
    // `descr.py:273-279` ArrayDescr — a single flag-bearing descriptor.
    // The runtime mint produces the same `SimpleArrayDescr` the
    // analyzer / `make_descr_from_bh` path mints, stamping the
    // content-addressed `ARRAY_DESCR_TAG | descr_id` index so `index()`
    // keeps the value the retired `PyreArrayDescr::index()` returned
    // (`ARRAY_DESCR_REGISTRY` still returns one Arc per structural key,
    // preserving `Arc::as_ptr` identity).
    let index = ARRAY_DESCR_TAG | (descr_id & 0x0FFF_FFFF);
    let mut array_descr = majit_ir::descr::SimpleArrayDescr::with_flag(
        index,
        base_size,
        item_size,
        type_id,
        item_type,
        runtime_array_flag(item_type, signed),
    );
    array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
    array_descr.set_cache_key(cache_key);
    let arc: DescrRef = Arc::new(array_descr);
    cache.insert(key.clone(), arc.clone());
    // Publish the freshly-minted SimpleArrayDescr into
    // `gc_cache._cache_array` keyed on `LLType::Array(path_hash(atid))`
    // so later analyzer-side `gc_cache.get_array_descr` cache-hit
    // returns this exact Arc.  Without an `array_type_id` (legacy
    // `make_array_descr` callers), only the local
    // `ARRAY_DESCR_REGISTRY` carries the descr — gc_cache cannot
    // identify it.
    if let Some(ref atid) = key.array_type_id {
        majit_ir::descr_registry::register_keyed_array(
            majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(atid)),
            arc.clone(),
        );
    }
    arc
}

impl Descr for PyreFieldDescr {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn index(&self) -> u32 {
        stable_field_index(self.offset, self.field_size, self.field_type, self.signed)
    }

    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }

    fn set_ei_index(&self, ei_index: u32) {
        self.ei_index.store(ei_index, Ordering::Relaxed);
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
        ei_index: AtomicU32::new(u32::MAX),
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
pub fn make_field_descr_with_parent(parent: DescrRef, offset: usize) -> DescrRef {
    majit_ir::descr::field_descr_from_parent_by_offset(&parent, offset)
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
        ei_index: AtomicU32::new(u32::MAX),
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
        ei_index: AtomicU32::new(u32::MAX),
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
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Concrete size descriptor for fixed-size object allocations.
#[derive(Debug)]
pub struct PyreSizeDescr {
    obj_size: usize,
    type_id: u32,
    /// `_cache_size[LLType::Struct(cache_key)]` 슬롯 키 — `path_hash`로
    /// 만들어진 STRUCT 구조 identity (publish 슬롯과 동일).  `type_id`
    /// 는 `gc.alloc_gc_typed`용 dense u32 GC tid 이고, `cache_key` 는
    /// `descr.py:108-118 cache[STRUCT]`의 lltype-object identity 와 1:1
    /// 대응한다.  `SizeDescr.cache_key()` 가 이 값을 반환해
    /// `bh_size_spec_from_descr` 역방향 reader 가 publish 슬롯과 같은
    /// `LLType::Struct(cache_key)` 로 round-trip 한다.  init 0 은 단발
    /// fixture 용 fall-back (구조 identity 없는 케이스).
    cache_key: u64,
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
    size_descr: Arc<majit_ir::descr::SimpleSizeDescr>,
    /// This group's own fields, in the order its static table declared them.
    ///
    /// Static accessors index by their table position, so they must read this
    /// list. `descr.py:218-239` still makes each Arc shared by
    /// `(STRUCT, fieldname)`, while each SizeDescr keeps its own frozen
    /// positional field list.
    field_descrs: Vec<Arc<majit_ir::descr::SimpleFieldDescr>>,
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
/// GC type id for W_IntRangeIterator. Inherits from `object`
/// (functional.rs:10 RANGE_ITER_TYPE).
pub const RANGE_ITER_GC_TYPE_ID: u32 = 6;
// `W_LIST_GC_TYPE_ID` / `W_TUPLE_GC_TYPE_ID` live in `pyre-object`
// alongside their structs (matching W_INT/W_FLOAT pattern); re-exported
// here for existing call sites.
pub use pyre_object::listobject::W_LIST_GC_TYPE_ID;
/// GC type id for the variable-length backing block of `PyObjectArray`
/// (the list/tuple items storage). Shape matches `rlist.py:84,116`
/// `GcArray(OBJECTPTR)` — a `T_IS_VARSIZE` block with a single-slot
/// `capacity` header (= upstream's GcArray length header,
/// rlist.py:251 `len(l.items)`) followed by inline `PyObjectRef`
/// items. Registered from `pyre_object::ITEMS_BLOCK_TOKEN` with
/// `items_have_gc_ptrs=true` so the GC walks each item slot as a
/// Ref (`gctypelayout.py:266-291 T_IS_VARSIZE / T_IS_GCARRAY_OF_GCPTR`);
/// live list length is stored on the enclosing `W_ListObject` wrapper
/// (`PyObjectArray.len`) to match rlist.py:116 `("length", Signed)`.
///
// Array GC type ids live in `pyre-object` alongside the backing storage
// structs/constants they describe (matching W_INT/W_FLOAT/W_LIST/W_TUPLE
// pattern). Re-exported here for existing call sites.
pub use pyre_object::object_array::{
    GC_FLOAT_ARRAY_GC_TYPE_ID, GC_INT_ARRAY_GC_TYPE_ID, PY_OBJECT_ARRAY_GC_TYPE_ID,
};
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
// `W_CELL_GC_TYPE_ID` lives in `pyre-object::nestedscope` alongside the
// `Cell` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::nestedscope::W_CELL_GC_TYPE_ID;
// `W_METHOD_GC_TYPE_ID` lives in `pyre-object::function` alongside
// the `Method` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::function::W_METHOD_GC_TYPE_ID;
// `W_SLICE_GC_TYPE_ID` lives in `pyre-object::sliceobject` alongside
// the `W_SliceObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::sliceobject::W_SLICE_GC_TYPE_ID;
// `W_SUPER_GC_TYPE_ID` lives in `pyre-object::descriptor` alongside
// the `W_Super` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::descriptor::W_SUPER_GC_TYPE_ID;
// `W_PROPERTY_GC_TYPE_ID` lives in `pyre-object::descriptor`, while
// `W_STATICMETHOD_GC_TYPE_ID` / `W_CLASSMETHOD_GC_TYPE_ID` live in
// `pyre-object::function` alongside their structs. Re-exported for the JIT
// registration site.
pub use pyre_object::descriptor::W_PROPERTY_GC_TYPE_ID;
pub use pyre_object::function::{W_CLASSMETHOD_GC_TYPE_ID, W_STATICMETHOD_GC_TYPE_ID};
// `W_UNION_GC_TYPE_ID` lives in `pyre-object::_pypy_generic_alias` alongside
// the `UnionType` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::_pypy_generic_alias::W_UNION_GC_TYPE_ID;
// `W_SEQ_ITER_GC_TYPE_ID` lives in `pyre-object::iterobject`
// alongside the `W_SeqIterObject` struct it describes. Re-exported for
// the JIT registration site.
pub use pyre_object::iterobject::W_SEQ_ITER_GC_TYPE_ID;
// `W_COUNT_GC_TYPE_ID` / `W_REPEAT_GC_TYPE_ID` live in
// `pyre-object::interp_itertools` alongside the `W_Count` /
// `W_Repeat` structs they describe. Re-exported for the JIT
// registration site.
pub use pyre_object::interp_itertools::{W_COUNT_GC_TYPE_ID, W_REPEAT_GC_TYPE_ID};
// `W_MEMBER_GC_TYPE_ID` lives in `pyre-object::typedef`
// alongside the `W_MemberDescr` struct it describes. Re-exported for
// the JIT registration site.
pub use pyre_object::typedef::W_MEMBER_GC_TYPE_ID;
// `W_BYTES_GC_TYPE_ID` lives in `pyre-object::bytesobject` alongside
// the `W_BytesObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::bytesobject::W_BYTES_GC_TYPE_ID;
// `W_BYTEARRAY_GC_TYPE_ID` lives in `pyre-object::bytearrayobject`
// alongside the `W_BytearrayObject` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::bytearrayobject::W_BYTEARRAY_GC_TYPE_ID;
// `W_DICT_GC_TYPE_ID` lives in `pyre-object::dictmultiobject` alongside
// the `W_DictObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::dictmultiobject::W_DICT_GC_TYPE_ID;
// `W_MODULE_DICT_GC_TYPE_ID` lives in `pyre-object::dictmultiobject`
// alongside the `W_ModuleDictObject` struct it describes (the PyPy
// `dictmultiobject.py:328 W_ModuleDictObject` port).  Re-exported
// for the JIT registration site.
pub use pyre_object::dictmultiobject::W_MODULE_DICT_GC_TYPE_ID;
// `W_SET_GC_TYPE_ID` lives in `pyre-object::setobject` alongside the
// `W_SetObject` struct it describes (covers both `set` and
// `frozenset` PyTypes — same Rust struct). Re-exported for the JIT
// registration site.
pub use pyre_object::setobject::W_SET_GC_TYPE_ID;
// `W_BASE_EXCEPTION_GC_TYPE_ID` lives in `pyre-object::interp_exceptions`
// alongside the `W_BaseException` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::interp_exceptions::W_BASE_EXCEPTION_GC_TYPE_ID;
// `W_GENERATOR_GC_TYPE_ID` lives in `pyre-object::generator`
// alongside the `GeneratorIterator` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::generator::W_GENERATOR_GC_TYPE_ID;
// `W_TYPE_GC_TYPE_ID` lives in `pyre-object::typeobject` alongside
// the `W_TypeObject` struct it describes. Re-exported for the JIT
// registration site. (`TYPE_TYPE` is in `all_foreign_pytypes()` but
// the foreign-pytype loop's `sizeof(PyObject)` approximation would
// drastically under-count the W_TypeObject payload.)
pub use pyre_object::typeobject::W_TYPE_GC_TYPE_ID;
// `W_UNICODE_GC_TYPE_ID` / `W_LONG_GC_TYPE_ID` / `W_MODULE_GC_TYPE_ID`
// live alongside their structs in
// `pyre-object::{unicodeobject, longobject, module}`. Re-exported
// for the JIT registration site. `W_ObjectObject` shares
// `OBJECT_GC_TYPE_ID` with the `object` root (see comment on the
// struct) so it has no separate id.
pub use pyre_object::longobject::W_LONG_GC_TYPE_ID;
pub use pyre_object::module::W_MODULE_GC_TYPE_ID;
// `W_DICT_PROXY_GC_TYPE_ID` lives in `pyre-object::dictproxyobject`
// alongside the `W_DictProxyObject` struct it describes.  Re-exported
// for the JIT registration site so the typeid stays in the
// pyre-jit-trace exports table next to its sibling Module/PyFrame
// entries.
pub use pyre_object::dictproxyobject::W_DICT_PROXY_GC_TYPE_ID;
pub use pyre_object::unicodeobject::W_UNICODE_GC_TYPE_ID;
// `PYFRAME_GC_TYPE_ID` lives in `pyre-interpreter::pyframe` alongside
// the `PyFrame` struct it describes. Re-exported for the JIT
// registration site (`pyre-jit/src/eval.rs`).
// Registered ahead of any future
// `NewWithVtable(PyFrame)` in trace IR.
pub use pyre_interpreter::pyframe::PYFRAME_GC_TYPE_ID;
// Appended tail registrations for PyFrame-owned auxiliary objects. These live
// with their Rust layouts in pyre-interpreter and are re-exported here beside
// PYFRAME_GC_TYPE_ID for the runtime registration census.
pub use pyre_interpreter::pyframe::{FRAME_BLOCK_GC_TYPE_ID, FRAME_DEBUG_DATA_GC_TYPE_ID};

fn field_descr_from_group(group: &PyreObjectDescrGroup, index: usize) -> DescrRef {
    group
        .field_descrs
        .get(index)
        .expect("field descriptor index out of bounds")
        .clone() as DescrRef
}

/// Build a SizeDescr group for a runtime PyObject layout and publish
/// it into `gc_cache._cache_size` under both the simple-name slot
/// AND the crate-stripped def-path slot.  PyPy `cache[STRUCT]`
/// collapses both into a single lltype-object identity; pyre's
/// analyzer currently hashes the simple name (use-site bare
/// identifier — collect_struct_names registers top-level structs by
/// `simple_name`) so that slot is the de-facto convergence point.
/// The def-path slot is published alongside as a forward-compatible
/// alias for the future analyzer use-import resolver (B-5 follow-up):
/// when that lands, analyzer's `owner_root` switches to qualified
/// form and the SAME `Arc<PyreSizeDescr>` is reachable via the
/// qualified hash. `register_keyed_size` keeps one `_cache_size_order`
/// entry per logical SizeDescr while allowing fuller-layout upgrades.
///
/// `def_path` empty (or equal to `simple_name`) → single publish.
fn build_object_descr_group_with_def_path(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
    simple_name: &str,
    def_path: &str,
) -> PyreObjectDescrGroup {
    build_object_descr_group_with_extra_gc_edges(
        obj_size,
        type_id,
        vtable,
        fields,
        simple_name,
        def_path,
        &[],
    )
}

/// `build_object_descr_group_with_def_path` plus GC edges that the
/// positional `fields` census does not name.  `extra_gc_edges` join the
/// `PyObject.w_class` edge every group already carries: they land in
/// `gc_fielddescrs` — which is what `rewrite.py:498-504 clear_gc_fields`
/// walks to zero a fresh object's GC-pointer slots — while staying out of
/// the positional `all_fielddescrs` list that `field_descr_from_group`
/// indexes.
fn build_object_descr_group_with_extra_gc_edges(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
    simple_name: &str,
    def_path: &str,
    extra_gc_edges: &[Arc<dyn FieldDescr>],
) -> PyreObjectDescrGroup {
    let cache_key = if !def_path.is_empty() {
        majit_ir::descr::path_hash(def_path)
    } else if !simple_name.is_empty() {
        majit_ir::descr::path_hash(simple_name)
    } else {
        0
    };
    let specs: Vec<majit_ir::descr::SimpleFieldDescrSpec> = fields
        .iter()
        .enumerate()
        .map(
            |(
                index_in_parent,
                &(field_key, offset, field_size, field_type, signed, immutable, quasi_immutable),
            )| majit_ir::descr::SimpleFieldDescrSpec {
                index: stable_field_index(offset, field_size, field_type, signed),
                field_key: field_key.to_string(),
                name: if simple_name.is_empty() {
                    field_key.to_string()
                } else {
                    format!("{simple_name}.{field_key}")
                },
                offset,
                field_size,
                field_type,
                is_immutable: immutable,
                is_quasi_immutable: quasi_immutable,
                flag: runtime_array_flag(field_type, signed),
                virtualizable: false,
                index_in_parent,
            },
        )
        .collect();
    let mut gc_edges: Vec<Arc<dyn FieldDescr>> = vec![W_CLASS_FIELD_DESCR.clone()];
    gc_edges.extend(extra_gc_edges.iter().cloned());
    let group = majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
        SIZE_DESCR_TAG | (obj_size as u32 & 0x0FFF_FFFF),
        obj_size,
        type_id,
        cache_key,
        vtable,
        true,
        false,
        &specs,
        &gc_edges,
    );
    let field_descrs = group.field_descrs;
    let size_descr = group.size_descr;
    // heaptracker.py:50-73 recurses into the inherited header, and
    // heaptracker.py:70 includes the embedded `PyObject.w_class` GC edge.
    // The factory keeps that extra edge out of the positional list.
    // Dual-publish: register under BOTH the simple-name slot AND
    // (when supplied) the crate-stripped def-path slot.
    //
    // PyPy `cache[STRUCT]` collapses both namespaces into a single
    // lltype-object identity; pyre's analyzer currently hashes the
    // use-site bare identifier (collect_struct_names registers
    // top-level structs at simple-name) so the simple-name slot is
    // the primary cache hit point.  The def-path slot is published
    // alongside as a forward-compatible alias for the future
    // analyzer use-import resolver (B-5 follow-up): when that lands,
    // analyzer's `owner_root` switches to qualified form and the
    // SAME `Arc<PyreSizeDescr>` is reachable via the qualified
    // hash. `register_keyed_size` keeps one `_cache_size_order` entry
    // per logical SizeDescr while allowing fuller-layout upgrades.
    if !simple_name.is_empty() {
        let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(simple_name));
        majit_ir::descr_registry::register_keyed_size(
            key,
            size_descr.clone() as majit_ir::DescrRef,
        );
    }
    if !def_path.is_empty() && def_path != simple_name {
        let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(def_path));
        majit_ir::descr_registry::register_keyed_size(
            key,
            size_descr.clone() as majit_ir::DescrRef,
        );
    }
    PyreObjectDescrGroup {
        size_descr,
        field_descrs,
    }
}

static W_INT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_IntObject>(),
        W_INT_GC_TYPE_ID,
        &INT_TYPE as *const _ as usize,
        &[("intval", INT_INTVAL_OFFSET, 8, Type::Int, true, true, false)],
        "W_IntObject",
        "intobject::W_IntObject",
    )
});

static W_FLOAT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_FloatObject>(),
        W_FLOAT_GC_TYPE_ID,
        &FLOAT_TYPE as *const _ as usize,
        &[
            (
                "floatval",
                FLOAT_FLOATVAL_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            // Native-subclass `__dict__` / `__slots__` GC-pointer slots. The
            // runtime TypeInfo traces both, so they must enter gc_fielddescrs
            // for rewrite.py:498-504 to emit the delayed NULL stores that
            // malloc_zero_filled=false requires; otherwise a NewWithVtable'd
            // exact float carries uninitialised nursery bytes here and the
            // collector traces poison. They follow `floatval` so its stable
            // field index stays 0.
            (
                "w_dict",
                FLOAT_W_DICT_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_slots",
                FLOAT_W_SLOTS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_FloatObject",
        "floatobject::W_FloatObject",
    )
});

static W_LONG_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::longobject::W_LongObject>(),
        pyre_object::longobject::W_LONG_GC_TYPE_ID,
        &pyre_object::pyobject::LONG_TYPE as *const _ as usize,
        &[(
            "value",
            pyre_object::longobject::LONG_VALUE_OFFSET,
            8,
            // The `value` slot is a gc-pointer to the BigInt payload, so it
            // enters `gc_fielddescrs` (the boxing SetfieldGc emits the write
            // barrier). Immutable: a long's payload is set once at creation.
            Type::Ref,
            false,
            true,
            false,
        )],
        "W_LongObject",
        "longobject::W_LongObject",
    )
});

static W_BOOL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::boolobject::W_BoolObject>(),
        W_BOOL_GC_TYPE_ID,
        &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
        &[(
            "intval",
            BOOL_INTVAL_OFFSET,
            8,
            Type::Int,
            false,
            true,
            false,
        )],
        "W_BoolObject",
        "boolobject::W_BoolObject",
    )
});

// RPython `descr.py:218-239 get_field_descr` returns a FieldDescr owned by
// the cached parent SizeDescr.  Keep the Unicode fields in the same
// object-descriptor group as every other runtime PyObject layout: in
// particular, `str_len_descr()` must not mint a detached FieldDescr whose
// weak `parent_descr` immediately upgrades to `None`.  The optimizer's
// `protect_speculative_field` asks that parent for the expected type before a
// pure length read, exactly as `llmodel.py:protect_speculative_field` does.
static W_UNICODE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::unicodeobject::W_UNICODE_OBJECT_SIZE,
        W_UNICODE_GC_TYPE_ID,
        &pyre_object::pyobject::STR_TYPE as *const _ as usize,
        &[
            (
                "value",
                pyre_object::unicodeobject::UNICODE_VALUE_OFFSET,
                std::mem::size_of::<*mut rustpython_wtf8::Wtf8Buf>(),
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "byte_len",
                pyre_object::unicodeobject::UNICODE_BYTE_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                true,
                false,
            ),
            (
                "len",
                pyre_object::unicodeobject::UNICODE_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                true,
                false,
            ),
            (
                "w_slots",
                pyre_object::unicodeobject::UNICODE_W_SLOTS_OFFSET,
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "index_storage",
                pyre_object::unicodeobject::UNICODE_INDEX_STORAGE_OFFSET,
                std::mem::size_of::<*mut pyre_object::rutf8::Utf8IndexStorage>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(pyre_object::unicodeobject::W_UnicodeObject, hash),
                std::mem::size_of::<i64>(),
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_UnicodeObject",
        "unicodeobject::W_UnicodeObject",
    )
});

static RANGE_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::functional::W_IntRangeIterator>(),
        RANGE_ITER_GC_TYPE_ID,
        &pyre_object::functional::RANGE_ITER_TYPE as *const _ as usize,
        &[
            (
                "current",
                RANGE_ITER_CURRENT_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "remaining",
                RANGE_ITER_REMAINING_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "step",
                RANGE_ITER_STEP_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_IntRangeIterator",
        "functional::W_IntRangeIterator",
    )
});

static SEQ_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<pyre_object::iterobject::W_SeqIterObject>(),
        W_SEQ_ITER_GC_TYPE_ID,
        &pyre_object::iterobject::SEQ_ITER_TYPE as *const _ as usize,
        &[
            (
                "seq",
                pyre_object::iterobject::SEQ_ITER_SEQ_OFFSET,
                std::mem::size_of::<pyre_object::PyObjectRef>(),
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "index",
                pyre_object::iterobject::SEQ_ITER_INDEX_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "length",
                pyre_object::iterobject::SEQ_ITER_LENGTH_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "empty_kind",
                pyre_object::iterobject::SEQ_ITER_EMPTY_KIND_OFFSET,
                1,
                Type::Int,
                false,
                false,
                false,
            ),
        ],
        "W_SeqIterObject",
        "iterobject::W_SeqIterObject",
    )
});

/// `stop` carries no accessor of its own — the GET_ITER virtualization derives
/// the cursor from `start` / `step` / `length` — but it is a pointer-shaped
/// slot, so it belongs in the census for the same reason `FUNCTION_DESCR_GROUP`
/// lists every slot: `clear_gc_fields` walks exactly this group's
/// `gc_fielddescrs` to emit a fresh object's delayed NULL stores, and a slot
/// left out would keep recycled nursery bytes that the collector then follows
/// as a child reference.  Its presence also keeps the list in byte-offset
/// order, which is what makes the positional numbering agree with the
/// analyzer's.
static RANGE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_Range>(),
        pyre_object::functional::W_RANGE_GC_TYPE_ID,
        &pyre_object::functional::RANGE_TYPE as *const _ as usize,
        &[
            (
                "start",
                RANGE_START_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            ("stop", RANGE_STOP_OFFSET, 8, Type::Ref, false, true, false),
            ("step", RANGE_STEP_OFFSET, 8, Type::Ref, false, true, false),
            (
                "length",
                RANGE_LENGTH_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
        "W_Range",
        "functional::W_Range",
    )
});

/// The `Function` fields PyPy declares quasi-immutable — `function.py:34-42`
/// `_immutable_fields_ = ['code?', 'w_func_globals?', 'closure?[*]',
/// 'defs_w?[*]', ...]`.
///
/// The `?` makes each field quasi-immutable upstream and `[*]` makes the
/// selected elements immutable after the field has been promoted.  Pyre's
/// setters (`function_set_defaults` and friends) do not yet call
/// `do_force_quasi_immutable`, so marking these descriptors quasi-immutable
/// would leave compiled loops alive after `f.__defaults__ = ...` /
/// `f.__code__ = ...`.  Keep the fields live/mutable for now; the inline-call
/// path pairs every read with a `GuardValue`, which is the sound
/// pre-invalidation equivalent.  A tuple's backing array has its own immutable
/// descriptor and is read with `GetarrayitemGcPureR`.
///
/// The entries are in byte-offset order and each key is the struct's own field
/// name, which is what makes the group's numbering agree with the analyzer's.
/// `index_in_parent` is a field's slot number in the virtual
/// (`virtualize.py:210 fielddescr.get_index()`) and its `PtrInfo._fields` key
/// in the heap optimizer, so two fields sharing one number makes a read of
/// either resolve to the other's cached value and a virtual keep only the
/// later of the two stores.  Two things assign it: `descr.py:218-239` caches a
/// FieldDescr per `(STRUCT, fieldname)` and a cache hit reports the analyzer's
/// number, which counts the struct's own fields past the flattened header;
/// a miss numbers the field by its position in this list.  Keeping the list in
/// offset order — with the inherited `PyObject.w_class` last, since the
/// analyzer's count does not include it — makes both answers the same one.
/// Look fields up by offset (`function_field_descr`) so the accessors below
/// cannot drift out of that order.
///
/// The census is COMPLETE — every pointer-shaped slot of `Function` is listed,
/// not only the four the inline-call path reads.  `emit_make_function_inline`
/// allocates a `Function` with `NewWithVtable`, and the nursery is not
/// zero-filled: `rewrite.py:498-504 clear_gc_fields` emits the delayed NULL
/// stores, and it walks exactly this group's `gc_fielddescrs`.  A pointer slot
/// missing from the census would keep recycled nursery bytes and the collector
/// would follow that word as a child reference.  `can_change_code` is a plain
/// byte, so it gets no NULL store and the emit writes it explicitly.
static FUNCTION_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_interpreter::function as f;
    let field = |key, offset| {
        (
            key,
            offset,
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
            false,
            false,
        )
    };
    build_object_descr_group_with_def_path(
        f::FUNCTION_OBJECT_SIZE,
        FUNCTION_GC_TYPE_ID,
        &pyre_interpreter::FUNCTION_TYPE as *const _ as usize,
        &[
            field("code", f::FUNCTION_CODE_OFFSET),
            // `function.py:33 can_change_code = True`; `False` for the
            // `FunctionWithFixedCode` / `BuiltinFunction` subclasses.  A plain
            // byte, so `clear_gc_fields` leaves it alone and the value a
            // materialized function reports comes from the emit's own store.
            (
                "can_change_code",
                std::mem::offset_of!(pyre_interpreter::function::Function, can_change_code),
                1,
                Type::Int,
                false,
                false,
                false,
            ),
            // `function.py:51 self.name = forcename or code.co_name` — a
            // pointer to the name string's storage.  It is pointer-shaped and
            // the runtime walker visits it, so it belongs in the census; the
            // storage itself may be GC-managed (a mortal function's own box) or
            // permanent (a builtin's `malloc_raw` box, or the code object's
            // borrowed `co_name`), which the walker's managed-heap guard sorts
            // out.
            field("name", f::FUNCTION_NAME_OFFSET),
            field("closure", f::FUNCTION_CLOSURE_OFFSET),
            field("defs_w", f::FUNCTION_DEFS_W_OFFSET),
            field("w_kw_defs", f::FUNCTION_W_KW_DEFS_OFFSET),
            field("w_module", f::FUNCTION_W_MODULE_OFFSET),
            field("w_func_globals_obj", f::FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET),
            field("w_builtins", f::FUNCTION_W_BUILTINS_OFFSET),
            field("w_ann", f::FUNCTION_W_ANN_OFFSET),
            field("w_annotate", f::FUNCTION_W_ANNOTATE_OFFSET),
            field("w_func_dict", f::FUNCTION_W_FUNC_DICT_OFFSET),
            field("w_typeparams", f::FUNCTION_W_TYPEPARAMS_OFFSET),
            field("w_doc", f::FUNCTION_W_DOC_OFFSET),
            field("w_qualname", f::FUNCTION_W_QUALNAME_OFFSET),
            field("w_objclass", f::FUNCTION_W_OBJCLASS_OFFSET),
            field("w_text_signature", f::FUNCTION_W_TEXT_SIGNATURE_OFFSET),
            // `w_new_self` is absent from `FUNCTION_GC_PTR_OFFSETS` (it names a
            // static-region type that is never relocated), but it is still a
            // pointer slot an app-level `__self__` read dereferences, so it
            // needs the census entry that gets it NULLed behind the allocation.
            field("w_new_self", f::FUNCTION_W_NEW_SELF_OFFSET),
            field("w_moduleobj", f::FUNCTION_W_MODULEOBJ_OFFSET),
            // The inline emit can escape a guard and be materialized, so the
            // inherited Python class is a proper virtual field of this group —
            // same reasoning as the `Method` / `W_ListObject` entries.  It sits
            // last, outside the offset ordering, because the analyzer counts
            // only the struct's own fields.
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "Function",
        "function::Function",
    )
});

/// `W_DictObject.keys_version` is pyre's explicit representation of the live
/// strategy-iterator state PyPy carries implicitly
/// (`dictmultiobject.py:807-845`).  Key insertion/removal/strategy replacement
/// bumps it; value-only replacement deliberately does not.  A promoted
/// identity-key lookup can therefore guard this field to pin the resolved
/// entry index while continuing to read that entry's value live.
static W_DICT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::dictmultiobject::W_DICT_OBJECT_SIZE,
        W_DICT_GC_TYPE_ID,
        &pyre_object::pyobject::DICT_TYPE as *const _ as usize,
        &[(
            "keys_version",
            std::mem::offset_of!(pyre_object::dictmultiobject::W_DictObject, keys_version),
            std::mem::size_of::<usize>(),
            Type::Int,
            false,
            false,
            false,
        )],
        "W_DictObject",
        "dictmultiobject::W_DictObject",
    )
});

/// `Method` field layout — `w_function`, `w_self`, `w_class`, `w_module`.
/// All four are Ref slots; the JIT only consumes `w_function` (for guarding
/// which method) and `w_self` (for recovering the receiver `OpRef` discarded
/// by `LOAD_METHOD`). `w_class` and `w_module` are included for layout
/// completeness so the descrs match the struct order — a field the struct
/// declares but this census omits has no `index_in_parent` to rederive, so
/// the two sides that mint its descr disagree on the number.
///
/// `w_function` and `w_self` are marked immutable per
/// `pypy/interpreter/function.py:567`
/// `_Method._immutable_fields_ = ['w_function', 'w_instance']`. `w_class`
/// is not listed there and stays mutable; `w_module` is written after
/// construction by `w_method_set_module` and is mutable for that reason.
static W_METHOD_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::function::{
        METHOD_W_CLASS_OFFSET, METHOD_W_FUNCTION_OFFSET, METHOD_W_MODULE_OFFSET,
        METHOD_W_SELF_OFFSET, W_METHOD_GC_TYPE_ID, W_METHOD_OBJECT_SIZE,
    };
    build_object_descr_group_with_def_path(
        W_METHOD_OBJECT_SIZE,
        W_METHOD_GC_TYPE_ID,
        &pyre_object::function::METHOD_TYPE as *const _ as usize,
        &[
            (
                "w_function",
                METHOD_W_FUNCTION_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "w_self",
                METHOD_W_SELF_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "w_class",
                METHOD_W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "w_module",
                METHOD_W_MODULE_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // The inline bound-method emit (`emit_bound_method_inline`) can
            // escape a guard and be materialized, so the inherited Python
            // class needs to be a proper virtual field of this group — same
            // reasoning as the `PyObject.w_class` entry on W_ListObject.
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "Method",
        "function::Method",
    )
});

/// `pypy/objspace/std/typeobject.py:26-34 ObjectMutableCell`. The single
/// `w_value` field is read LIVE on the module-global cell fast path: a
/// frequently-rewritten global mutates the cell payload in place without
/// bumping `mstrategy.version` (`typeobject.py:55-71 write_cell`), so the
/// JIT cell read must observe the current value and stays MUTABLE (not
/// immutable / quasi-immutable). Cell identity is what the `version?`
/// quasi-immutable guard protects.
static W_OBJECT_MUTABLE_CELL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::celldict::{
        OBJECT_MUTABLE_CELL_TYPE, W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS,
        W_OBJECT_MUTABLE_CELL_GC_TYPE_ID, W_OBJECT_MUTABLE_CELL_OBJECT_SIZE,
    };
    build_object_descr_group_with_def_path(
        W_OBJECT_MUTABLE_CELL_OBJECT_SIZE,
        W_OBJECT_MUTABLE_CELL_GC_TYPE_ID,
        &OBJECT_MUTABLE_CELL_TYPE as *const _ as usize,
        &[(
            "w_value",
            W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS[0],
            8,
            Type::Ref,
            false,
            false,
            false,
        )],
        "ObjectMutableCell",
        "celldict::ObjectMutableCell",
    )
});

static W_LIST_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // Upstream `rpython/rtyper/lltypesystem/rlist.py:116`
    //     GcStruct("list", ("length", Signed), ("items", Ptr(ITEMARRAY)))
    // The parity-field pair is `(length, items)`. `strategy` +
    // `int_items` / `float_items` are pyre-only PRE-EXISTING-
    // ADAPTATIONs for the PyPy interp-level strategy split.
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_ListObject>(),
        W_LIST_GC_TYPE_ID,
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        &[
            // rlist.py:116 `("length", Signed)`. Mutable: Object-strategy
            // push/pop/insert/remove/drain update it.
            (
                // `length` is a `usize` (8 bytes on 64-bit, 4 on wasm32). A
                // hardcoded 8 makes `GetfieldGcI` read the adjacent `items`
                // pointer into the high half on wasm32 → a garbage length →
                // out-of-bounds list access. Same fix as `str_len_descr`; the
                // `usize`/pointer fields below follow suit (the `Type::Ref`
                // fields are safe — read at pointer width regardless of size).
                "length",
                std::mem::offset_of!(W_ListObject, length),
                std::mem::size_of::<usize>(),
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
                "items",
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
                // then read from `float_items.block` (empty after the
                // switch) and dereference garbage — the spectral_norm n=10
                // SIGSEGV root cause.
                //
                // Upstream PyPy handles this with a quasi-immutable flag
                // + invalidate_compiled_code hook on strategy change;
                // pyre has no such hook yet, so `strategy` stays
                // plain-mutable. TODO — strategy split itself
                // is a pyre-only adaptation vs rlist.py.
                "strategy",
                std::mem::offset_of!(W_ListObject, strategy),
                1,
                Type::Int,
                false,
                false,
                false,
            ),
            // Integer-strategy typed storage (pyre-only
            // TODO vs listobject.py's
            // IntegerListStrategy at the interp level — upstream keeps
            // the unwrap inline and doesn't add a separate backing
            // array).
            (
                "int_items.len",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            // Float-strategy typed storage.
            (
                "float_items.len",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_LEN_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            // `Ptr(GcArray(Signed))` / `Ptr(GcArray(Float))` — the typed
            // strategy backing blocks (`erase([int])` / `erase([float])`).
            // Read as a Ref so `GetarrayitemGcI` / `GetarrayitemGcF` address
            // items[i] through the heap cache. Mutable: re-pointed on grow /
            // strategy switch (like `W_ListObject.items`).
            (
                "int_items.block",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_BLOCK_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "float_items.block",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_BLOCK_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // The inline list emit can escape through an exception's args_w
            // slot and be materialized.  Track the inherited Python class in
            // the same parent group as tuple objects so its SetfieldGc is a
            // proper virtual field and materialization reproduces w_list_new.
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // `BaseUserClassMapdict.storage` equivalent for native list
            // subclasses with `__slots__`. Mutable and instance-owned.
            (
                "W_ListObject.w_slots",
                std::mem::offset_of!(W_ListObject, w_slots),
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_ListObject",
        "listobject::W_ListObject",
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
    // Python 3.14's mutable-once `hash` cache is the intentional version
    // delta from that PyPy layout.
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_TupleObject>(),
        W_TUPLE_GC_TYPE_ID,
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        &[
            // `Ptr(GcArray(OBJECTPTR))` — wrappeditems body. Immutable.
            (
                "wrappeditems",
                std::mem::offset_of!(W_TupleObject, wrappeditems),
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(W_TupleObject, hash),
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            // Mapdict's optional instance dictionary for tuple subclasses.
            // Exact tuples leave this mutable GC slot null.
            (
                "W_TupleObject.w_dict",
                std::mem::offset_of!(W_TupleObject, w_dict),
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_TupleObject",
        "tupleobject::W_TupleObject",
    )
});

static SPECIALISED_TUPLE_II_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `specialisedtupleobject.py:34` `_immutable_fields_ = ['value0',
    // 'value1']` — both fields immutable. Inline-field shape, no array
    // indirection.
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_SpecialisedTupleObject_ii>(),
        SPECIALISED_TUPLE_II_GC_TYPE_ID,
        &SPECIALISED_TUPLE_II_TYPE as *const _ as usize,
        &[
            (
                "value0",
                SPECIALISED_TUPLE_II_VALUE0_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
            (
                "value1",
                SPECIALISED_TUPLE_II_VALUE1_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(W_SpecialisedTupleObject_ii, hash),
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_ii",
        "specialisedtupleobject::W_SpecialisedTupleObject_ii",
    )
});

static SPECIALISED_TUPLE_FF_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_SpecialisedTupleObject_ff>(),
        SPECIALISED_TUPLE_FF_GC_TYPE_ID,
        &SPECIALISED_TUPLE_FF_TYPE as *const _ as usize,
        &[
            (
                "value0",
                SPECIALISED_TUPLE_FF_VALUE0_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            (
                "value1",
                SPECIALISED_TUPLE_FF_VALUE1_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(W_SpecialisedTupleObject_ff, hash),
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_ff",
        "specialisedtupleobject::W_SpecialisedTupleObject_ff",
    )
});

static SPECIALISED_TUPLE_OO_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
        std::mem::size_of::<W_SpecialisedTupleObject_oo>(),
        SPECIALISED_TUPLE_OO_GC_TYPE_ID,
        &SPECIALISED_TUPLE_OO_TYPE as *const _ as usize,
        &[
            (
                "value0",
                SPECIALISED_TUPLE_OO_VALUE0_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "value1",
                SPECIALISED_TUPLE_OO_VALUE1_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "hash",
                std::mem::offset_of!(W_SpecialisedTupleObject_oo, hash),
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_oo",
        "specialisedtupleobject::W_SpecialisedTupleObject_oo",
    )
});

static ITEMS_BLOCK_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::object_array::ITEMS_BLOCK_ITEMS_OFFSET,
        0,
        0,
        &[(
            "capacity",
            pyre_object::object_array::ITEMS_BLOCK_LEN_OFFSET,
            std::mem::size_of::<usize>(),
            Type::Int,
            false,
            true,
            false,
        )],
        "ItemsBlock",
        "object_array::ItemsBlock",
    )
});

// The RPython `tuple2` returned by `rbigint.divmod` / `int_divmod`
// (rbigint.py:1002/1050). Not a PyObject — no vtable and no allocation type id,
// because the trace never NEWs one: it arrives as an elidable's result and is
// only read. Both fields are `Type::Ref`, which is what puts them in
// `gc_fielddescrs`, and both are immutable, so the two reads CSE the way
// upstream's pair of `getfield_gc_r` off one `call_pure_r` does.
static RBIGINT_PAIR_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
        pyre_object::longobject::BIGINT_PAIR_SIZE,
        0,
        0,
        &[
            (
                "item0",
                pyre_object::longobject::BIGINT_PAIR_ITEM0_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "item1",
                pyre_object::longobject::BIGINT_PAIR_ITEM1_OFFSET,
                std::mem::size_of::<usize>(),
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
        "RBigIntPair",
        "rbigint::RBigIntPair",
    )
});

// `pypy/objspace/std/sliceobject.py:13` `W_SliceObject._immutable_fields_ =
// ['w_start', 'w_stop', 'w_step']` — all three Ref fields are immutable
// once `__init__` runs.  The `space.newslice(w_start, w_end, w_step)` JIT
// shape allocates the W_SliceObject inline so the optimizer can virtualize
// the three SetfieldGc writes when the slice never escapes (per
// `optimizeopt/virtualize.py optimize_NEW_WITH_VTABLE`).
static W_SLICE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::sliceobject::*;
    build_object_descr_group_with_def_path(
        W_SLICE_OBJECT_SIZE,
        W_SLICE_GC_TYPE_ID,
        &pyre_object::sliceobject::SLICE_TYPE as *const _ as usize,
        &[
            (
                "w_start",
                SLICE_START_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "w_stop",
                SLICE_STOP_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "w_step",
                SLICE_STEP_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
        "W_SliceObject",
        "sliceobject::W_SliceObject",
    )
});

/// `rvirtualizable.py:29` appends `('vable_token', llmemory.GCREF)` to the
/// virtualizable's own fields, so upstream's `gc_fielddescrs` names it and
/// `clear_gc_fields` zeroes the slot on every `new`.  pyre declares
/// `PyFrame.vable_token` as a plain `usize` and the positional census below
/// does not list it, so without this edge a JIT-inlined
/// `NewWithVtable(pyframe_size_descr())` leaves the slot holding recycled
/// nursery bytes — which `emit_force_virtualizable`'s `GETFIELD_GC_R` then
/// reads as a live GC reference (`pyjitpl.py:1148-1158`).
static PYFRAME_VABLE_TOKEN_FIELD_DESCR: LazyLock<Arc<dyn FieldDescr>> = LazyLock::new(|| {
    Arc::new(PyreFieldDescr {
        offset: crate::frame_layout::PYFRAME_VABLE_TOKEN_OFFSET,
        field_size: std::mem::size_of::<usize>(),
        field_type: Type::Ref,
        signed: false,
        immutable: false,
        quasi_immutable: false,
        name: "vable_token",
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
});

static PYFRAME_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_extra_gc_edges(
        std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
        PYFRAME_GC_TYPE_ID,
        // `NewWithVtable` writes this typeptr at `cpu.vtable_offset`
        // (`OB_TYPE_OFFSET = 0`), populating the frame's `ob_header.ob_type`
        // so a JIT-built inline callee frame carries the same `frame` type
        // tag as a `FrameBox`-constructed one.
        &pyre_interpreter::pyframe::FRAME_TYPE as *const _ as usize,
        &[
            (
                "locals_cells_stack_w",
                crate::frame_layout::PYFRAME_LOCALS_CELLS_STACK_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "valuestackdepth",
                crate::frame_layout::PYFRAME_VALUESTACKDEPTH_OFFSET,
                // `usize`, not a fixed 64-bit int.  The other `Type::Int`
                // fields in this table are all `i64`, so 8 is right for
                // them; this one and `last_instr` below are the two that
                // are a machine word wide.  A literal 8 makes the store a
                // byte pair too wide on a 32-bit target, and the overrun
                // lands on `last_instr`, which sits immediately after it.
                std::mem::size_of::<usize>(),
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "last_instr",
                crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET,
                // `isize` — see the width note on `valuestackdepth` above.
                std::mem::size_of::<isize>(),
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "pycode",
                crate::frame_layout::PYFRAME_PYCODE_OFFSET,
                8,
                Type::Ref,
                true,
                false,
                false,
            ),
            // `pyframe.py:49 self.w_globals` — the slot the inline
            // new-PyFrame helper populates from the function's globals dict.
            (
                "PyFrame.w_globals",
                crate::frame_layout::PYFRAME_W_GLOBALS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.debugdata",
                crate::frame_layout::PYFRAME_DEBUGDATA_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.lastblock",
                crate::frame_layout::PYFRAME_LASTBLOCK_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // Inline PyFrame 생성 시 새 frame 의
            // execution_context 슬롯에 caller 의 ec 를 SetfieldGc 로 쓰기 위해
            // 필요. RPython parity 는 interp_jit.py:67 reds=[frame, ec] 의 ec
            // 슬롯과 동등 — pyre 는 ec 를 PyFrame 헤더에 inline 저장.
            (
                "PyFrame.execution_context",
                crate::frame_layout::PYFRAME_EXECUTION_CONTEXT_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.f_generator_nowref",
                crate::frame_layout::PYFRAME_F_GENERATOR_NOWREF_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.w_yielding_from",
                crate::frame_layout::PYFRAME_W_YIELDING_FROM_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.f_backref",
                crate::frame_layout::PYFRAME_F_BACKREF_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.w_builtin",
                crate::frame_layout::PYFRAME_W_BUILTIN_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // `pyframe.py:80 escaped` and its neighbours live in one `u8`, so
            // the traced `mark_as_escaped` is a byte-wide read-or-write, not a
            // word store.  Appended last: the accessors above address this
            // group by index.
            (
                "PyFrame.flags",
                crate::frame_layout::PYFRAME_FLAGS_OFFSET,
                std::mem::size_of::<u8>(),
                Type::Int,
                false,
                false,
                false,
            ),
        ],
        "PyFrame",
        "pyframe::PyFrame",
        &[PYFRAME_VABLE_TOKEN_FIELD_DESCR.clone()],
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

    /// `descr.py:108-118 get_size_descr` cache identity 와 line-by-line
    /// 동등: `register_keyed_size` 가 publish 한 슬롯 키 (현재
    /// `path_hash_stripped_crate(module_path!(), bare_name)` —
    /// 즉 def-path 기반의 `path_hash("module::Bare")`).  매크로 publish
    /// 시 `__majit_type_id()` 에서 계산되어 인스턴스 슬롯에 저장된
    /// 값을 그대로 반환한다.  `bh_size_spec_from_descr` 역방향 reader 는
    /// 이 값을 `BhSizeSpec.type_id` 에 넣고 `simple_descr_group_from_bh_size`
    /// 는 `LLType::Struct(spec.type_id)` 로 publish 슬롯에 round-trip 한다.
    /// `type_id` (dense GC tid) 와 `cache_key` (structural identity) 는
    /// `descr.rs:1928-1934` 트레이트 doc 의 분리 contract 를 따른다.
    fn cache_key(&self) -> u64 {
        self.cache_key
    }

    fn vtable(&self) -> usize {
        self.vtable
    }

    fn w_class_obj(&self) -> Option<i64> {
        w_class_obj_for_vtable(self.vtable)
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
    /// W_IntRangeIterator / …). `ensure_ptr_info_arg0` (optimizer.py:480)
    /// uses this to dispatch InstancePtrInfo vs StructPtrInfo.
    fn is_object(&self) -> bool {
        self.vtable != 0
    }
}

/// Empty-struct fallback for `BhDescr::Size` decode (`make_descr_from_bh`).
/// RPython `descr.py:188 init_size_descr` records an empty
/// `all_fielddescrs` list when the underlying STRUCT has no GC fields, so
/// the consumer-side decoder still needs a constructor that produces a
/// `PyreSizeDescr` with an empty field-list rather than refusing to build
/// one. Producers carrying a populated field-list go through
/// `simple_descr_group_from_bh_size` instead.
pub fn make_size_descr_with_type_and_vtable(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
) -> DescrRef {
    // 빈 fielddescr fallback — `BhDescr::Size` 디코더가 구조 identity
    // 캐리어 없이 호출하는 자리.  `cache_key = 0` 은 round-trip 시
    // `simple_descr_group_from_bh_size` 의 no-identity branch
    // (`descr.rs:2382-2388`) 가 per-call distinct 처리하므로 안전.
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id,
        cache_key: 0,
        vtable,
        all_fielddescrs: Vec::new(),
        gc_fielddescrs: Vec::new(),
    })
}

/// Synthetic `len` field descriptor matching upstream
/// `descr.py:264 FieldDescr("len", ofs, WORD, FLAG_SIGNED)`. Lives at
/// offset 0 of the `Ptr(GcArray(T))` block (FixedObjectArray /
/// pyobject_gcarray layout): items start at `base_size`, so the word
/// before items is the length header. Stored as
/// `SimpleArrayDescr.lendescr` so `gen_initialize_len`
/// (`rewrite.py:565,572`) emits the runtime length store after
/// `CallMallocNurseryVarsize`.
fn array_lendescr_at_offset(offset: usize) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size: std::mem::size_of::<usize>(),
        field_type: Type::Int,
        signed: true,
        immutable: false,
        quasi_immutable: false,
        name: "len",
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Lift `Option<usize>` ↔ `Option<DescrRef>` so `make_array_descr*`
/// callers express nolength/length-prefixed shapes directly. PyPy
/// `descr.py:359-362` decides this from
/// `ARRAY_INSIDE._hints.get('nolength', False)`; the explicit
/// `Option<usize>` is the structural equivalent.
fn maybe_array_lendescr_at_offset(len_offset: Option<usize>) -> Option<DescrRef> {
    len_offset.map(array_lendescr_at_offset)
}

/// Create a fresh ARRAY descriptor without identity carrier.
///
/// `len_offset`: `None` for the `nolength=True` shape (descr.py:360);
/// `Some(off)` for length-prefixed layouts (descr.py:362).
///
/// PyPy `descr.py:348-378 get_array_descr(gccache, ARRAY)` keys
/// `_cache_array[ARRAY_OR_STRUCT]` on the ARRAY object identity, never
/// on its structural shape — two distinct lltype ARRAYs that share
/// `(base_size, item_size, item_type, signed, len_offset)` get
/// distinct `ArrayDescr` Arcs.  Pyre's no-identity-carrier callers
/// (this function: `array_type_id = None`, `type_id = 0`) cannot
/// participate in the keyed cache because they have no ARRAY-object
/// surrogate to hash; the orthodox behaviour is therefore "each call
/// is a distinct ARRAY" — mint fresh `SimpleArrayDescr` per call so
/// shape-coincident-but-logically-distinct ARRAYs receive distinct
/// `descr_id` slots.  Callers that need singleton semantics
/// (`int_array_descr`, `float_array_descr`, `pyobject_array_descr`,
/// …) route through [`make_array_descr_with_full_id`] with a stable
/// identity string instead — the keyed cache canonicalises by that
/// string.
pub fn make_array_descr(
    base_size: usize,
    item_size: usize,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    let descr_id = alloc_array_descr_id();
    // No identity carrier — fresh mint per call (cache_key = 0 means "no
    // cache slot").  Single flag-bearing `SimpleArrayDescr` (descr.py:273-279);
    // index carries the content-addressed `ARRAY_DESCR_TAG | descr_id`.
    let index = ARRAY_DESCR_TAG | (descr_id & 0x0FFF_FFFF);
    let mut array_descr = majit_ir::descr::SimpleArrayDescr::with_flag(
        index,
        base_size,
        item_size,
        0,
        item_type,
        runtime_array_flag(item_type, signed),
    );
    array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
    Arc::new(array_descr)
}

pub fn make_array_descr_with_type(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    get_or_create_array_descr(base_size, item_size, type_id, item_type, signed, len_offset)
}

/// Bridge-only factory that threads the codewriter's `array_type_id`
/// (`majit-translate::codewriter::call::DescrIndexRegistry::array_index`
/// key) into `ArrayDescrKey` so two BhDescr::Array entries with
/// identical structural fields but different lltype spellings receive
/// distinct registry slots — matching upstream
/// `gccache._cache_array[ARRAY_OR_STRUCT]` (`descr.py:348-360`).
pub fn make_array_descr_with_full_id(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
    array_type_id: Option<String>,
) -> DescrRef {
    get_or_create_array_descr_with_full_id(
        base_size,
        item_size,
        type_id,
        item_type,
        signed,
        len_offset,
        array_type_id,
    )
}

// ── Range iterator field descriptors ─────────────────────────────────

use pyre_object::floatobject::{
    FLOAT_FLOATVAL_OFFSET, FLOAT_W_DICT_OFFSET, FLOAT_W_SLOTS_OFFSET, W_FloatObject,
};
use pyre_object::functional::{
    RANGE_ITER_CURRENT_OFFSET, RANGE_ITER_REMAINING_OFFSET, RANGE_ITER_STEP_OFFSET,
    RANGE_LENGTH_OFFSET, RANGE_START_OFFSET, RANGE_STEP_OFFSET, RANGE_STOP_OFFSET, W_Range,
};
use pyre_object::interp_exceptions::{
    EXC_ARGS_W_OFFSET, EXC_KIND_COUNT, EXC_KIND_OFFSET, EXC_W_ATTR_OBJ_OFFSET, EXC_W_CAUSE_OFFSET,
    EXC_W_CODE_OFFSET, EXC_W_CONTEXT_OFFSET, EXC_W_DICT_OFFSET, EXC_W_ENCODING_OFFSET,
    EXC_W_END_OFFSET, EXC_W_ERRNO_OFFSET, EXC_W_FILENAME_OFFSET, EXC_W_FILENAME2_OFFSET,
    EXC_W_GROUP_EXCEPTIONS_OFFSET, EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET, EXC_W_GROUP_MESSAGE_OFFSET,
    EXC_W_IMPORT_MSG_OFFSET, EXC_W_IMPORT_NAME_FROM_OFFSET, EXC_W_IMPORT_PATH_OFFSET,
    EXC_W_NAME_OFFSET, EXC_W_OBJECT_OFFSET, EXC_W_REASON_OFFSET, EXC_W_START_OFFSET,
    EXC_W_STRERROR_OFFSET, EXC_W_SYNTAX_END_LINENO_OFFSET, EXC_W_SYNTAX_END_OFFSET_OFFSET,
    EXC_W_SYNTAX_FILENAME_OFFSET, EXC_W_SYNTAX_LINENO_OFFSET, EXC_W_SYNTAX_METADATA_OFFSET,
    EXC_W_SYNTAX_MSG_OFFSET, EXC_W_SYNTAX_OFFSET_OFFSET, EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET,
    EXC_W_SYNTAX_TEXT_OFFSET, EXC_W_TRACEBACK_OFFSET, EXC_W_VALUE_OFFSET, EXC_W_WEAKREF_OFFSET,
    ExcKind, W_BASE_EXCEPTION_GC_PTR_OFFSETS, W_BASE_EXCEPTION_SIZE, exc_kind_to_pytype,
};
use pyre_object::intobject::W_IntObject;
use pyre_object::pyobject::W_CLASS_OFFSET;
use pyre_object::{
    BOOL_INTVAL_OFFSET, FLOAT_ARRAY_BLOCK_OFFSET, FLOAT_ARRAY_LEN_OFFSET, INT_ARRAY_BLOCK_OFFSET,
    INT_ARRAY_LEN_OFFSET, INT_INTVAL_OFFSET, W_ListObject, W_TupleObject,
};
// Re-import the rest without duplication
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
fn new_w_class_field_descr() -> Arc<dyn FieldDescr> {
    // Named "w_class" so `FieldDescr::is_w_class()` recognises the
    // header field; OptVirtualize must resolve it from the object's
    // class identity rather than indexing it against a virtual's value
    // fields (its `index_in_parent` of 0 would otherwise collide with
    // the first value field, e.g. `W_IntObject.intval`).
    Arc::new(PyreFieldDescr {
        offset: pyre_object::pyobject::W_CLASS_OFFSET,
        // ⚠️`WORD` on paper — the field is a `*mut PyObject`, so 4 bytes on
        // wasm32, and the build-time descr pool already sizes it that way
        // (`call.rs get_type_flag` → `layout::target_word_size()`). Deriving it
        // here to match makes `synth/exception_traceback_loop_forms` lose one
        // iteration's `e.__traceback__` on the wasm backend, so the two
        // universes stay deliberately out of step until that is understood.
        // `state.rs materialize_virtual_object` keys its w_class branch off
        // `field_size == size_of::<*mut PyObject>()`, a guard that therefore
        // never fires on wasm32.
        field_size: 8,
        field_type: Type::Ref,
        signed: false,
        immutable: false,
        quasi_immutable: false,
        name: "w_class",
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// The single `w_class` field descriptor for the shared `PyObject` header.
///
/// `descr.py:218-239 get_field_descr` caches on `(STRUCT, fieldname)`, so a
/// given field is one object for the whole run and every consumer compares it
/// by identity: `heap.py` keys `cached_fields` by the descriptor itself
/// (`heap.rs:860 cached_field_pos_for_descr` matches on `descr_identity`), and
/// so do the short-preamble export and `force_from_effectinfo`. Minting a
/// fresh `Arc` per call gave each `w_class` read in a trace its own identity,
/// so two reads of the same header could never share a cache entry.
static W_CLASS_FIELD_DESCR: LazyLock<Arc<dyn FieldDescr>> = LazyLock::new(new_w_class_field_descr);

pub fn w_class_descr() -> DescrRef {
    W_CLASS_FIELD_DESCR.clone() as DescrRef
}

/// The canonical `w_class` (Python class object) for instances of the type
/// `vtable` names — `get_instantiate(vtable_type)`. Read live (not cached at
/// construction) since the type objects are installed after the descrs are
/// built. `None` before `init_typeobjects()` runs.
///
/// Registered as majit's [`majit_ir::descr::WClassObjFn`] so the generic
/// `SimpleSizeDescr` — which every runtime PyObject group and every
/// blackhole-dispatch size descr is built as — answers `w_class_obj` the same
/// way `PyreSizeDescr` does. Without it `OptVirtualize` cannot fold the
/// `w_class` header read off a `new_with_vtable` virtual and forces the
/// virtual instead.
pub fn w_class_obj_for_vtable(vtable: usize) -> Option<i64> {
    if vtable == 0 {
        return None;
    }
    // A `JitVirtualRef` names itself with a type tag, not a `PyType` address,
    // and has no `w_class` to answer with — reading the tag as a type would
    // dereference a constant.
    if vtable == majit_metainterp::virtualref::JIT_VIRTUAL_REF_VTABLE as usize {
        return None;
    }
    let tp = vtable as *const pyre_object::pyobject::PyType;
    let w_class = unsafe { pyre_object::pyobject::get_instantiate(&*tp) };
    if w_class.is_null() {
        None
    } else {
        Some(w_class as i64)
    }
}

/// Alias for backward compatibility — same as w_class_descr().
pub fn instance_w_type_descr() -> DescrRef {
    w_class_descr()
}

/// Field descriptor for `W_IntRangeIterator.current` (i64, signed).
pub fn range_iter_current_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 0)
}

/// Field descriptor for `W_IntRangeIterator.remaining` (i64, signed).
pub fn range_iter_remaining_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 1)
}

/// Field descriptor for `W_IntRangeIterator.step` (i64, signed).
pub fn range_iter_step_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 2)
}

/// Field descriptor for `W_SeqIterObject.seq`.
pub fn seq_iter_seq_descr() -> DescrRef {
    field_descr_from_group(&SEQ_ITER_DESCR_GROUP, 0)
}

/// Field descriptor for `W_SeqIterObject.index`.
pub fn seq_iter_index_descr() -> DescrRef {
    field_descr_from_group(&SEQ_ITER_DESCR_GROUP, 1)
}

/// Resolve one [`RANGE_DESCR_GROUP`] field by byte offset, so the accessors
/// below stay correct however the census is ordered.  They were positional
/// until `stop` joined the census and shifted every later slot.
fn range_field_descr(offset: usize) -> DescrRef {
    let parent = RANGE_DESCR_GROUP.size_descr.clone() as DescrRef;
    majit_ir::descr::field_descr_from_parent_by_offset(&parent, offset)
}

/// Field descriptor for `W_Range.start` (wrapped PyObjectRef).
pub fn range_start_descr() -> DescrRef {
    range_field_descr(RANGE_START_OFFSET)
}

/// Field descriptor for `W_Range.stop` (wrapped PyObjectRef).
pub fn range_stop_descr() -> DescrRef {
    range_field_descr(RANGE_STOP_OFFSET)
}

/// Field descriptor for `W_Range.step` (wrapped PyObjectRef).
pub fn range_step_descr() -> DescrRef {
    range_field_descr(RANGE_STEP_OFFSET)
}

/// Field descriptor for `W_Range.length` (wrapped PyObjectRef).
pub fn range_length_descr() -> DescrRef {
    range_field_descr(RANGE_LENGTH_OFFSET)
}

/// `Method.w_function` — the underlying function (`Function` or
/// `BuiltinFunction`) bound by `getattr(obj, name)`. Marked immutable
/// per `pypy/interpreter/function.py:567` `_Method._immutable_fields_`,
/// so reads survive cache invalidation across calls. Used by the
/// bound-method specialization in `call_callable_value`.
pub fn method_w_function_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 0)
}

/// Resolve one [`FUNCTION_DESCR_GROUP`] field by byte offset, so the accessors
/// below stay correct however the census is ordered.
fn function_field_descr(offset: usize) -> DescrRef {
    let parent = FUNCTION_DESCR_GROUP.size_descr.clone() as DescrRef;
    majit_ir::descr::field_descr_from_parent_by_offset(&parent, offset)
}

/// Live `Function.defs_w` field used by the positional-default inline path.
/// See [`FUNCTION_DESCR_GROUP`] for why this is deliberately mutable until
/// pyre wires the upstream quasi-immutable invalidation hook.
pub fn function_defs_w_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_DEFS_W_OFFSET)
}

/// Live `Function.code` — the field `Function.getcode()` promotes
/// (`function.py:95 jit.promote(self.code)`).  This is what identifies an
/// inlined body, so the inline lever guards it instead of the function object.
pub fn function_code_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_CODE_OFFSET)
}

/// Live `Function.w_func_globals_obj` — the namespace an inlined callee's
/// LOAD_GLOBAL folds against.
pub fn function_w_globals_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET)
}

/// Live `Function.closure` — the freevar cell tuple threaded into the inlined
/// callee's own frame.
pub fn function_closure_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_CLOSURE_OFFSET)
}

/// `function.py:51 self.name` — the pointer to the name string's storage.
pub fn function_name_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_NAME_OFFSET)
}

/// `Function.w_builtins` — the builtin mapping resolved from the globals
/// once at construction and then frozen (rebinding `globals['__builtins__']`
/// afterwards does not change it).
pub fn function_w_builtins_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_BUILTINS_OFFSET)
}

/// `function.py:54 self.qualname = qualname or self.name` — the wrapped
/// qualified name stamped at construction from the code object.
pub fn function_w_qualname_descr() -> DescrRef {
    function_field_descr(pyre_interpreter::function::FUNCTION_W_QUALNAME_OFFSET)
}

/// `function.py:33 can_change_code = True` — a plain byte, so a fresh
/// allocation does not zero it and an emit must write it.
pub fn function_can_change_code_descr() -> DescrRef {
    function_field_descr(std::mem::offset_of!(
        pyre_interpreter::function::Function,
        can_change_code
    ))
}

/// Inherited `PyObject.w_class` on a `Function` — the Python-level `function`
/// class the constructor's header stamps.  Kept in the Function group (not the
/// standalone `w_class_descr`) so an inline emit's store is a virtual field of
/// the same size descr and materialization reproduces the header.
pub fn function_header_w_class_descr() -> DescrRef {
    function_field_descr(pyre_object::pyobject::W_CLASS_OFFSET)
}

/// Size descriptor for `Function` allocation via `NewWithVtable`
/// (vtable = `&FUNCTION_TYPE`).
pub fn w_function_size_descr() -> DescrRef {
    FUNCTION_DESCR_GROUP.size_descr.clone()
}

pub fn dict_keys_version_descr() -> DescrRef {
    field_descr_from_group(&W_DICT_DESCR_GROUP, 0)
}

/// `Method.w_self` — the receiver object. The bound-method
/// specialization extracts this via `GetfieldGcR` to recover the receiver
/// `OpRef` after `LOAD_METHOD` discarded it (load_method.rs:6334 pushes
/// `null_value` for `is_method` attrs). Immutable per
/// `_Method._immutable_fields_`.
pub fn method_w_self_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 1)
}

/// `Method.w_class` — the class the bound function was found on
/// (`function.py` `Method.w_class`), distinct from the inherited
/// `PyObject.w_class` naming the Method's own type.
pub fn method_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 2)
}

/// `Method.w_module` — `__module__` storage for a bound builtin method,
/// written after construction by `w_method_set_module`. The JIT does not
/// read it; the census entry exists so the struct's field order is complete.
pub fn method_w_module_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 3)
}

/// Inherited `PyObject.w_class` on a `Method` — the Python-level `method`
/// class stamped by `w_method_new`'s header. Kept in the Method group (not
/// the standalone `w_class_descr`) so an inline emit's store is a virtual
/// field of the same size descr and materialization reproduces the header.
pub fn method_header_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 4)
}

/// Size descriptor for `Method` allocation via `NewWithVtable`
/// (vtable = `&METHOD_TYPE`); `w_function` / `w_self` / `w_class` and the
/// inherited header `w_class` are `SetfieldGc`'d after.
pub fn w_method_size_descr() -> DescrRef {
    W_METHOD_DESCR_GROUP.size_descr.clone()
}

/// `typeobject.py:26-34 ObjectMutableCell.w_value` — the boxed payload of
/// a module-global cell, read LIVE on the cell fast path. The value is
/// rewritten in place when a hot global is reassigned without bumping the
/// strategy version, so this descriptor is mutable (not immutable /
/// quasi-immutable); the `version?` guard protects cell identity, not the
/// payload.
pub fn object_mutable_cell_value_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_MUTABLE_CELL_DESCR_GROUP, 0)
}

/// `typeobject.py:37-45 IntMutableCell.intvalue` — the unboxed `Signed`
/// payload of a module-global int cell, read LIVE on the cell fast path.
/// `write_cell` rewrites it in place when a hot int global is reassigned to
/// another int (no strategy-version bump), so this descriptor is mutable
/// (not immutable / quasi-immutable); the `version?` guard protects cell
/// identity, not the payload. Reassigning to a non-int replaces the cell,
/// bumping the version and invalidating the fold.
/// Descriptor for `IntMutableCell.intvalue`.  Minted with a reserved unique
/// [`INT_MUTABLE_CELL_VALUE_INDEX`] rather than `stable_field_index` because
/// that field's `(offset 16, size 8, Int)` layout collides with
/// `W_IntObject.intval` / `W_ListObject.length` in the runtime `HeapCache`
/// key space (which keys by `descr.index()`; see [`CELL_DESCR_TAG`]).
///
/// A SINGLETON `Arc`, one descr per field exactly as upstream's codewriter
/// produces one `FieldDescr` per `IntMutableCell.inst_intvalue`.  The
/// optimizer's `cached_fields` is keyed by `descr_identity`
/// (`Arc::as_ptr`), so the LOAD `getfield_gc_i` and the STORE
/// `setfield_gc_i` MUST share the Arc: with per-call fresh Arcs the store's
/// lazy `setfield` lives in a `CachedField` the load's lookup never finds,
/// the load skips heap.py:67-75 `possible_aliasing_two_infos` entirely, and
/// `force_lazy_sets_for_guard` later flushes the store BELOW the emitted
/// load — reordering a store past a load of the same location (the nested
/// module-loop `i = i + 1; while i < n` reads the pre-increment value and
/// runs one extra iteration).  Distinct cells (`i`/`j`/`k`) do NOT
/// cross-forward under the shared descr: `CachedField` distinguishes
/// structs by the obj operand (`same_box` MUST_ALIAS / UNKNOWN_ALIAS →
/// `force_lazy_set`, heap.py:103-120).  Signed `i64` payload, mutable
/// (`write_cell` rewrites `intvalue` in place for an int->int reassign with
/// no version bump).
pub fn int_mutable_cell_value_descr() -> DescrRef {
    static DESCR: std::sync::OnceLock<DescrRef> = std::sync::OnceLock::new();
    DESCR
        .get_or_init(|| {
            Arc::new(majit_ir::descr::SimpleFieldDescr::new_with_name(
                INT_MUTABLE_CELL_VALUE_INDEX,
                core::mem::offset_of!(pyre_object::celldict::IntMutableCell, intvalue),
                8,
                Type::Int,
                false,
                majit_ir::descr::ArrayFlag::Signed,
                "IntMutableCell.intvalue".to_string(),
                "intvalue".to_string(),
            ))
        })
        .clone()
}

/// Size descriptor for `W_ListObject` allocation via NewWithVtable.
/// vtable = &LIST_TYPE; the Object-strategy fields `length` / `items` /
/// `strategy` are SetField'd after.  `int_items.block` / `float_items.block`
/// are GC-pointer fields of this descr, so `rewrite.py:498-504
/// clear_gc_fields` zeroes them behind the allocation (== empty, never read
/// under the Object strategy); their `len` halves are plain ints and stay at
/// whatever the recycled nursery bytes held, which no strategy reads while the
/// block slot is null.
pub fn w_list_size_descr() -> DescrRef {
    W_LIST_DESCR_GROUP.size_descr.clone()
}

/// `typeobject.py:162 _version_tag` — the method-cache version (`u64`, 8
/// bytes, unsigned) on `W_TypeObject`. The `LOAD_METHOD` fast path reads it
/// through `promote(self.version_tag())` (typeobject.py:506) so the
/// `_pure_lookup_where_with_method_cache` `CALL_PURE_R` folds on a green
/// version.
///
/// Quasi-immutable, per `typeobject.py:177 _immutable_fields_ =
/// ['_version_tag?']`: the read is replaced by a `QUASIIMMUT_FIELD` plus one
/// `GUARD_NOT_INVALIDATED` per trace, and `mutated()` (typeobject.py:285-286)
/// revokes the dependent loops through
/// [`pyre_object::typeobject::w_type_set_version_tag`] instead of the trace
/// re-checking a live value every iteration. A residual call cannot flush a
/// quasi-immutable field, which is the whole point — the interim live read +
/// `guard_value` had to be re-done after every un-inlined call in the body.
///
/// One object per run, for the identity reason documented on
/// [`W_CLASS_FIELD_DESCR`], and load-bearing here: `heap.rs:3274` keys
/// `quasi_immut_cache` on `field_cache_identity`, which is the `Arc` pointer,
/// so a per-call descriptor would miss its own cache on every read.
static TYPE_VERSION_TAG_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    make_quasi_immutable_field_descr(
        core::mem::offset_of!(pyre_object::typeobject::W_TypeObject, version_tag),
        8,
        Type::Int,
        false,
    )
});

pub fn type_version_tag_descr() -> DescrRef {
    TYPE_VERSION_TAG_FIELD_DESCR.clone()
}

/// `celldict.py:32 ModuleDictStrategy.version` — the module-namespace version
/// tag (`u64`, 8 bytes, unsigned) on the strategy box.
///
/// Quasi-immutable, per `celldict.py:34 _immutable_fields_ = ["version?"]`,
/// which is the same declaration `getdictvalue_no_unwrapping` promotes before
/// its elidable lookup (`celldict.py:47-55`). The `LOAD_GLOBAL` / `STORE_GLOBAL`
/// cell folds bake the slot's stored cell as a `ConstPtr` under a
/// `QUASIIMMUT_FIELD` on this field: `_setitem_str_cell_known`
/// (`celldict.py:80-90`) calls `mutated()` before every write that replaces the
/// stored pointer, and an in-place cell write leaves the pointer alone, so the
/// version is exactly the datum that proves the baked address still stands.
///
/// `offset_of!` rather than a literal because `ModuleDictStrategy` is not
/// `repr(C)`.
///
/// One object per run, for the reason spelled out on
/// [`TYPE_VERSION_TAG_FIELD_DESCR`].
static MODULE_DICT_VERSION_FIELD_DESCR: LazyLock<DescrRef> = LazyLock::new(|| {
    make_quasi_immutable_field_descr(
        core::mem::offset_of!(pyre_object::celldict::ModuleDictStrategy, version),
        8,
        Type::Int,
        false,
    )
});

pub fn module_dict_version_descr() -> DescrRef {
    MODULE_DICT_VERSION_FIELD_DESCR.clone()
}

/// `W_ObjectObject` SizeDescr group (`objectobject.rs:34-46`) — the instance
/// layout `[ob_type | w_class | map | storage]`.  Built with a parent SizeDescr
/// (unlike a bare [`make_field_descr`]) so a `getfield_gc` on `map` / `storage`
/// resolves `FieldDescr.get_parent_descr()` in the optimizer's
/// `ensure_ptr_info_arg0` (`optimizer.py:478`); the LOAD_ATTR fold reads these
/// fields inline, so they must carry the owning struct's SizeDescr.
///
/// `map` is an opaque `Int` word (interned immortal map nodes — not a GC ref,
/// stays off `gc_fielddescrs`); `storage` is a `Ref` block pointer (enters
/// `gc_fielddescrs` so a `setfield_gc` emits the write barrier).
///
/// The inherited header `w_class` is a member of the group — not just the
/// standalone `w_class_descr` — because the instantiation emit
/// (`try_walker_inline_type_call`) builds instances with `NewWithVtable`, and
/// the class a `getfield_gc(w_class)` off such a virtual must answer with is
/// the *stored* one.  Every instance shares `INSTANCE_TYPE` as its vtable
/// while its Python class varies per instance, so the vtable-derived fallback
/// (`w_class_obj`, which resolves `INSTANCE_TYPE`'s `get_instantiate` to
/// `object`) is wrong here; only a field the virtual actually tracks gives
/// `OptVirtualize` the right answer, and materialization then reproduces the
/// header.
static W_OBJECT_OBJECT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    let group = build_object_descr_group_with_def_path(
        pyre_object::W_OBJECT_OBJECT_SIZE,
        pyre_object::objectobject::W_OBJECT_OBJECT_GC_TYPE_ID,
        &pyre_object::pyobject::INSTANCE_TYPE as *const _ as usize,
        &[
            (
                "W_ObjectObject.map",
                core::mem::offset_of!(pyre_object::W_ObjectObject, map),
                // `map` is a `*const MapNode` erased to an opaque Int word, so
                // its width is one machine word — 4 bytes on wasm32, 8 on
                // 64-bit. Hardcoding 8 would read/write past the field on
                // wasm32, folding the adjacent `storage` pointer into the high
                // half of a `guard_value(map)` load (and clobbering it on a
                // `setfield_gc(map)` store).
                core::mem::size_of::<usize>(),
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ObjectObject.storage",
                core::mem::offset_of!(pyre_object::W_ObjectObject, storage),
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_ObjectObject",
        "objectobject::W_ObjectObject",
    );
    // `alloc_instance_object` (`objectobject.rs`) allocates every interpreter
    // instance through the stable, non-moving old-gen allocator, because the
    // instance layer reaches an instance through raw pointers it does not root
    // — `store_attr_caching` holds one across the allocation of the storage
    // block it is about to install.  A JIT-emitted instance has to agree: put
    // it in the nursery and the first minor collection inside such a residual
    // moves it, after which the caller finishes writing `map` / `storage` into
    // the dead pre-move copy and the attribute is lost.
    group.size_descr.set_non_moving(true);
    group
});

/// `W_ObjectObject.map` (`objectobject.rs:38`) — the erased `*const MapNode`
/// instance shape pointer, `self.map` of PyPy's `MapdictStorageMixin`
/// (`mapdict.py:907`). Read as an opaque `Int` word so the LOAD_ATTR fast path
/// can `guard_value` it to a constant map (`jit.promote(self.map)`,
/// mapdict.py:905), after which the resolved `storageindex` is a green
/// constant. The map nodes are interned + immortal, so the pointer is a stable
/// identity and the guard need not treat it as a GC ref.
pub fn object_map_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_OBJECT_DESCR_GROUP, 0)
}

/// `W_ObjectObject.storage` (`objectobject.rs:40`) — `self.storage` of
/// `MapdictStorageMixin` (`mapdict.py:910`), a `Ptr(GcArray(OBJECTPTR))` block
/// of attribute values. Read as a `Ref` (the block pointer) so the LOAD_ATTR
/// fast path can then `getarrayitem_gc_r` the value at the green-constant
/// `storageindex` (`mapdict.py:914-916` `_mapdict_read_storage`), mirroring
/// `list_items_descr` → `pyobject_gcarray_descr`. Mutable: STORE_ATTR grows /
/// replaces the block.
pub fn object_storage_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_OBJECT_DESCR_GROUP, 1)
}

/// The instance's own `PyObject.w_class` — the Python class `w_instance_new`
/// stamps into the header. Kept in the instance group (not the standalone
/// `w_class_descr`) so the instantiation emit's store is a virtual field of
/// the same size descr; see [`W_OBJECT_OBJECT_DESCR_GROUP`].
pub fn object_header_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_OBJECT_DESCR_GROUP, 2)
}

/// Size descriptor for a `W_ObjectObject` allocation via `NewWithVtable`
/// (vtable = `&INSTANCE_TYPE`); the header `w_class` and `map` are
/// `SetfieldGc`'d after, and `storage` stays at the allocator's zero (the
/// `_mapdict_init_empty` `storage = None` state).
pub fn w_object_object_size_descr() -> DescrRef {
    W_OBJECT_OBJECT_DESCR_GROUP.size_descr.clone()
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

pub fn list_int_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 3)
}

pub fn list_float_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 4)
}

/// `Ptr(GcArray(Signed))` — the `int_items` backing block (`erase([int])`).
/// Read as a Ref; combine with the GcArray(Signed) array descr
/// (`int_gcarray_descr`) for `GetarrayitemGcI` / `SetarrayitemGc`.
pub fn list_int_items_block_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 5)
}

/// `Ptr(GcArray(Float))` — the `float_items` backing block (`erase([float])`).
/// Read as a Ref; combine with the GcArray(Float) array descr
/// (`float_gcarray_descr`) for `GetarrayitemGcF` / `SetarrayitemGc`.
pub fn list_float_items_block_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 6)
}

pub fn list_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 7)
}

pub fn list_w_slots_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 8)
}

/// `Ptr(GcArray(OBJECTPTR))` — `wrappeditems` body per
/// `tupleobject.py:381` `_immutable_fields_ = ['wrappeditems[*]']`.
/// Immutable. Length comes from `arraylen_gc(items_block,
/// pyobject_gcarray_descr)` against the GcArray header — no
/// `tuple_length_descr` exists per upstream tupleobject.py:376-390
/// (`W_TupleObject` carries no separate length field).
pub fn tuple_wrappeditems_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 0)
}

pub fn tuple_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 1)
}

pub fn tuple_hash_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 2)
}

pub fn tuple_w_dict_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 3)
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

pub fn specialised_tuple_ii_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 2)
}

pub fn specialised_tuple_ii_hash_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 3)
}

/// `W_SpecialisedTupleObject_ff.value0` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ff.value1` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 1)
}

pub fn specialised_tuple_ff_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 2)
}

pub fn specialised_tuple_ff_hash_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 3)
}

/// `W_SpecialisedTupleObject_oo.value0` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_oo.value1` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 1)
}

pub fn specialised_tuple_oo_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 2)
}

pub fn specialised_tuple_oo_hash_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 3)
}

/// `ItemsBlock.capacity` — the GcArray length header at offset 0 of
/// an `ItemsBlock`, matching `rlist.py:84/251` `len(l.items)`
/// (allocated capacity, not live length). Immutable: once a block is
/// allocated the capacity is fixed; resize allocates a fresh block.
/// Callers combine `list_items_descr()` / `tuple_wrappeditems_descr()`
/// → `ItemsBlock*` with this descr to read the block's allocated size.
/// The group parent lets sub-walk reads pass the optimizer's
/// `ensure_ptr_info_arg0` parent lookup.
pub fn items_block_capacity_descr() -> DescrRef {
    field_descr_from_group(&ITEMS_BLOCK_DESCR_GROUP, 0)
}

pub fn int_intval_descr() -> DescrRef {
    field_descr_from_group(&W_INT_DESCR_GROUP, 0)
}

pub fn bool_intval_descr() -> DescrRef {
    field_descr_from_group(&W_BOOL_DESCR_GROUP, 0)
}

pub fn float_floatval_descr() -> DescrRef {
    field_descr_from_group(&W_FLOAT_DESCR_GROUP, 0)
}

/// FieldDescr for `W_LongObject.value` (the `*mut BigInt` gc-pointer), for the
/// inline-NEW boxing of a `jit_w_long_*_raw` result.
pub fn long_value_descr() -> DescrRef {
    field_descr_from_group(&W_LONG_DESCR_GROUP, 0)
}

/// `RBigIntPair.item0` — the quotient half of a divmod `tuple2`.
pub fn rbigint_pair_item0_descr() -> DescrRef {
    field_descr_from_group(&RBIGINT_PAIR_DESCR_GROUP, 0)
}

/// `RBigIntPair.item1` — the remainder half of a divmod `tuple2`.
pub fn rbigint_pair_item1_descr() -> DescrRef {
    field_descr_from_group(&RBIGINT_PAIR_DESCR_GROUP, 1)
}

pub fn str_len_descr() -> DescrRef {
    // Python len(str) returns codepoint count.
    // unicodeobject.py:165 W_UnicodeObject._len() → _length field.
    // `W_UnicodeObject.len` is a `usize`: 8 bytes on 64-bit, 4 on wasm32 — a
    // hardcoded 8 reads the adjacent field into the high half on a 32-bit
    // target (blackhole resume of `len(str)`).
    field_descr_from_group(&W_UNICODE_DESCR_GROUP, 2)
}

// ── Object header & allocation descriptors ──────────────────────────

/// `PyCode.code_ptr` — the host `CodeObject` every code-field getter resolves
/// through (`code_get_field` -> `require_code`).  Read only to prove it is
/// non-null, which is the check the getter would have run.
///
/// The three code descrs below are standalone rather than a positional group:
/// a `PyCode` is never allocated from a trace, so the group's size / GC-edge
/// half would have no consumer, and publishing a partial layout under the
/// live `W_CODE_GC_TYPE_ID` would put a second answer in the registry for a
/// type the collector already describes.
static PYCODE_CODE_PTR_FIELD_DESCR: LazyLock<Arc<dyn FieldDescr>> = LazyLock::new(|| {
    Arc::new(PyreFieldDescr {
        offset: pyre_interpreter::pycode::CODE_PTR_OFFSET,
        field_size: std::mem::size_of::<*const ()>(),
        field_type: Type::Int,
        signed: false,
        immutable: false,
        quasi_immutable: false,
        name: "code_ptr",
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
});

pub fn pycode_code_ptr_descr() -> DescrRef {
    PYCODE_CODE_PTR_FIELD_DESCR.clone() as DescrRef
}

/// `PyCode.w_name` — the realized `co_name` string.  `w_code_name_obj` builds
/// it on first demand and retains it, so the slot IS the getter's value once
/// it is non-null; a null slot declines to the residual, which realizes it.
static PYCODE_W_NAME_FIELD_DESCR: LazyLock<Arc<dyn FieldDescr>> = LazyLock::new(|| {
    Arc::new(PyreFieldDescr {
        offset: pyre_interpreter::pycode::CODE_W_NAME_OFFSET,
        field_size: std::mem::size_of::<pyre_object::PyObjectRef>(),
        field_type: Type::Ref,
        signed: false,
        immutable: false,
        quasi_immutable: false,
        name: "w_name",
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
});

pub fn pycode_w_name_descr() -> DescrRef {
    PYCODE_W_NAME_FIELD_DESCR.clone() as DescrRef
}

/// `PyCode.co_firstlineno_raw` — a signed 32-bit slot, because 3.14's
/// `CodeType` constructor accepts zero and negative first lines that
/// `CodeObject.first_line_number` cannot hold.
static PYCODE_CO_FIRSTLINENO_FIELD_DESCR: LazyLock<Arc<dyn FieldDescr>> = LazyLock::new(|| {
    Arc::new(PyreFieldDescr {
        offset: pyre_interpreter::pycode::CODE_CO_FIRSTLINENO_RAW_OFFSET,
        field_size: std::mem::size_of::<i32>(),
        field_type: Type::Int,
        signed: true,
        immutable: false,
        quasi_immutable: false,
        name: "co_firstlineno_raw",
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
});

pub fn pycode_co_firstlineno_descr() -> DescrRef {
    PYCODE_CO_FIRSTLINENO_FIELD_DESCR.clone() as DescrRef
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

/// Size descriptor for W_IntRangeIterator allocation via NewWithVtable.
/// vtable = &RANGE_ITER_TYPE; type_id = 0.
pub fn w_range_iter_size_descr() -> DescrRef {
    RANGE_ITER_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_Range allocation via NewWithVtable.
/// vtable = &RANGE_TYPE.
pub fn w_range_size_descr() -> DescrRef {
    RANGE_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_FloatObject allocation via NewWithVtable.
/// vtable = &FLOAT_TYPE (ob_type for virtual materialization).
pub fn w_float_size_descr() -> DescrRef {
    W_FLOAT_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_LongObject allocation via NewWithVtable (the inline
/// boxing of a bigint result). vtable = &LONG_TYPE (ob_type).
pub fn w_long_size_descr() -> DescrRef {
    W_LONG_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for canonical `W_TupleObject`.
pub fn w_tuple_size_descr() -> DescrRef {
    W_TUPLE_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_ii`.
pub fn specialised_tuple_ii_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_II_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_ff`.
pub fn specialised_tuple_ff_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_FF_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_oo`.
pub fn specialised_tuple_oo_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_OO_DESCR_GROUP.size_descr.clone()
}

/// SizeDescr + field descrs for `W_BaseException` allocation via
/// NewWithVtable, one set per `ExcKind`.  The vtable (`ob_type`) differs
/// per kind (`exc_kind_to_pytype`), so each kind owns its group; the
/// three SetField'd fields — `kind`, `w_class`, `args_w` — share the
/// same offsets across kinds.  `w_cause`/`w_context`/… stay zeroed by
/// the `NewWithVtable` memzero (PY_NULL), matching
/// `w_exception_new_empty`.
fn build_w_exception_group(kind: ExcKind) -> PyreObjectDescrGroup {
    build_object_descr_group_with_def_path(
        W_BASE_EXCEPTION_SIZE,
        W_BASE_EXCEPTION_GC_TYPE_ID,
        exc_kind_to_pytype(kind) as *const _ as usize,
        &[
            // `kind` is a `u8` tag (1 byte, unsigned).
            (
                "W_BaseException.kind",
                EXC_KIND_OFFSET,
                1,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_class",
                W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.args_w",
                EXC_ARGS_W_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // `w_context` (`__context__`): a GC pointer slot.  Written by
            // the RAISE_VARARGS `__context__` chaining lowering
            // (`exc.w_context = ec.sys_exc_value`) so the optimizer can
            // track it on the virtual exception; carried at field index 3.
            (
                "W_BaseException.w_context",
                EXC_W_CONTEXT_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // The runtime flattens the subclass-specific exception fields
            // onto W_BaseException and its TypeInfo traces every one of them.
            // Keep them in gc_fielddescrs so rewrite.py:498-504 emits the
            // delayed NULL stores required by malloc_zero_filled=false.  They
            // follow the four optimizer-visible fields above so the stable
            // kind/w_class/args_w/w_context indices do not change.
            (
                "W_BaseException.w_cause",
                EXC_W_CAUSE_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_traceback",
                EXC_W_TRACEBACK_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_object",
                EXC_W_OBJECT_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_start",
                EXC_W_START_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_end",
                EXC_W_END_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_reason",
                EXC_W_REASON_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_encoding",
                EXC_W_ENCODING_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_errno",
                EXC_W_ERRNO_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_strerror",
                EXC_W_STRERROR_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_filename",
                EXC_W_FILENAME_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_filename2",
                EXC_W_FILENAME2_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_code",
                EXC_W_CODE_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_value",
                EXC_W_VALUE_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_exc_name",
                EXC_W_NAME_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_attr_obj",
                EXC_W_ATTR_OBJ_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_import_path",
                EXC_W_IMPORT_PATH_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_import_name_from",
                EXC_W_IMPORT_NAME_FROM_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_import_msg",
                EXC_W_IMPORT_MSG_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_dict",
                EXC_W_DICT_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_weakreflifeline",
                EXC_W_WEAKREF_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_msg",
                EXC_W_SYNTAX_MSG_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_filename",
                EXC_W_SYNTAX_FILENAME_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_lineno",
                EXC_W_SYNTAX_LINENO_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_offset",
                EXC_W_SYNTAX_OFFSET_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_text",
                EXC_W_SYNTAX_TEXT_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_end_lineno",
                EXC_W_SYNTAX_END_LINENO_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_end_offset",
                EXC_W_SYNTAX_END_OFFSET_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_print_file_and_line",
                EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_syntax_metadata",
                EXC_W_SYNTAX_METADATA_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_group_message",
                EXC_W_GROUP_MESSAGE_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_group_exceptions",
                EXC_W_GROUP_EXCEPTIONS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "W_BaseException.w_group_exceptions_repr",
                EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        // Empty name: the per-kind vtable means a shared "W_BaseException"
        // name-registry slot would be first-write-wins and lose the other
        // kinds' vtables.  NewWithVtable embeds the SizeDescr in the op, so
        // the name-registry publish is not needed here.
        "",
        "",
    )
}

static W_BASE_EXCEPTION_DESCR_CACHE: LazyLock<Mutex<Vec<Option<PyreObjectDescrGroup>>>> =
    LazyLock::new(|| Mutex::new((0..EXC_KIND_COUNT).map(|_| None).collect()));

/// Field descrs for the exception construction emit: `(size, kind,
/// w_class, args_w)`.  Built and cached per `ExcKind` on first use.
pub fn w_exception_descrs(kind: ExcKind) -> (DescrRef, DescrRef, DescrRef, DescrRef) {
    let idx = kind as u8 as usize;
    let mut cache = W_BASE_EXCEPTION_DESCR_CACHE.lock().unwrap();
    if cache[idx].is_none() {
        cache[idx] = Some(build_w_exception_group(kind));
    }
    let group = cache[idx].as_ref().unwrap();
    (
        group.size_descr.clone() as DescrRef,
        field_descr_from_group(group, 0),
        field_descr_from_group(group, 1),
        field_descr_from_group(group, 2),
    )
}

/// Field descr for `W_BaseException.w_context` (the `__context__`
/// slot), index 3 of the per-kind exception descr group.  Used by the
/// RAISE_VARARGS `__context__` chaining lowering; shares the same parent
/// `SizeDescr` as the `NewWithVtable` emit so the optimizer recognises
/// the store as a field of the virtual exception.
pub fn w_exception_context_descr(kind: ExcKind) -> DescrRef {
    let idx = kind as u8 as usize;
    let mut cache = W_BASE_EXCEPTION_DESCR_CACHE.lock().unwrap();
    if cache[idx].is_none() {
        cache[idx] = Some(build_w_exception_group(kind));
    }
    let group = cache[idx].as_ref().unwrap();
    field_descr_from_group(group, 3)
}

/// Field descr for `W_BaseException.w_dict` (the lazily allocated instance
/// dictionary), the last slot of the per-kind exception descr group.  The
/// LOAD_METHOD method-cache fold reads it to pin the receiver at "carries no
/// instance dictionary", which is what makes the folded descriptor safe: a
/// later `e.<name> = ...` allocates the dictionary and side-exits.
pub fn w_exception_dict_descr(kind: ExcKind) -> DescrRef {
    let idx = kind as u8 as usize;
    let mut cache = W_BASE_EXCEPTION_DESCR_CACHE.lock().unwrap();
    if cache[idx].is_none() {
        cache[idx] = Some(build_w_exception_group(kind));
    }
    let group = cache[idx].as_ref().unwrap();
    // Located by offset rather than by a hand-counted position.  The field
    // list is edited by hand, and naming the wrong index does not fail to
    // compile: it silently reads a neighbouring slot that stays null for an
    // ordinary subclass, which turns the shadowing guard below into a no-op
    // and lets compiled code keep calling a method an instance attribute has
    // already shadowed.
    let field = group
        .field_descrs
        .iter()
        .position(|d| d.offset() == pyre_object::interp_exceptions::EXC_W_DICT_OFFSET)
        .expect("exception descr group has no w_dict field");
    field_descr_from_group(group, field)
}

/// Field descriptor for `W_BaseException.w_traceback`, sharing the
/// per-kind exception allocation descriptor with the other exception slots.
pub fn w_exception_traceback_descr(kind: ExcKind) -> DescrRef {
    let idx = kind as u8 as usize;
    let mut cache = W_BASE_EXCEPTION_DESCR_CACHE.lock().unwrap();
    if cache[idx].is_none() {
        cache[idx] = Some(build_w_exception_group(kind));
    }
    field_descr_from_group(cache[idx].as_ref().unwrap(), 5)
}

static PYTRACEBACK_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_interpreter::pytraceback::{
        PYTRACEBACK_FRAME_OFFSET, PYTRACEBACK_GC_TYPE_ID, PYTRACEBACK_LASTI_OFFSET,
        PYTRACEBACK_LINENO_OFFSET, PYTRACEBACK_OBJECT_SIZE, PYTRACEBACK_TYPE,
        PYTRACEBACK_W_CODE_OFFSET, PYTRACEBACK_W_NEXT_OFFSET,
    };

    let group = build_object_descr_group_with_def_path(
        PYTRACEBACK_OBJECT_SIZE,
        PYTRACEBACK_GC_TYPE_ID,
        &PYTRACEBACK_TYPE as *const _ as usize,
        &[
            (
                "PyTraceback.w_class",
                W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyTraceback.frame",
                PYTRACEBACK_FRAME_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyTraceback.lasti",
                PYTRACEBACK_LASTI_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "PyTraceback.w_next",
                PYTRACEBACK_W_NEXT_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyTraceback.lineno",
                PYTRACEBACK_LINENO_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "PyTraceback.w_code",
                PYTRACEBACK_W_CODE_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "",
        "",
    );
    // `w_pytraceback_new` allocates traceback nodes non-moving because raw
    // `*mut PyTraceback` readers and the exception `w_traceback` chain keep
    // bare pointers. A nursery allocation would move the node at the next
    // minor collection while those copies retain its old address.
    group.size_descr.set_non_moving(true);
    group
});

pub fn pytraceback_size_descr() -> DescrRef {
    PYTRACEBACK_DESCR_GROUP.size_descr.clone()
}

pub fn pytraceback_field_descr(index: usize) -> DescrRef {
    field_descr_from_group(&PYTRACEBACK_DESCR_GROUP, index)
}

/// Field descriptor for `PyTraceback.frame`, the node's own frame slot read by
/// `descr_get_tb_frame`.
pub fn pytraceback_frame_descr() -> DescrRef {
    let index = PYTRACEBACK_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == pyre_interpreter::pytraceback::PYTRACEBACK_FRAME_OFFSET)
        .expect("PyTraceback descr group has no frame field");
    field_descr_from_group(&PYTRACEBACK_DESCR_GROUP, index)
}

/// Field descriptor for `PyTraceback.w_next`, the chain link `descr_get_next`
/// reads.  Located by offset rather than by a hand-counted position, so a
/// later edit to the field list above cannot silently repoint this at a
/// neighbouring slot.
pub fn pytraceback_w_next_descr() -> DescrRef {
    let index = PYTRACEBACK_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == pyre_interpreter::pytraceback::PYTRACEBACK_W_NEXT_OFFSET)
        .expect("PyTraceback descr group has no w_next field");
    field_descr_from_group(&PYTRACEBACK_DESCR_GROUP, index)
}

/// Field descriptor for `PyTraceback.lineno`, the line `descr_get_tb_lineno`
/// reports.  Located by offset for the same reason as
/// [`pytraceback_w_next_descr`].
pub fn pytraceback_lineno_descr() -> DescrRef {
    let index = PYTRACEBACK_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == pyre_interpreter::pytraceback::PYTRACEBACK_LINENO_OFFSET)
        .expect("PyTraceback descr group has no lineno field");
    field_descr_from_group(&PYTRACEBACK_DESCR_GROUP, index)
}

/// Cached field descriptor for a raw reference slot selected by the
/// exception attribute fold.  Indices are those of `build_w_exception_group`;
/// no parallel descriptor is constructed.
pub fn w_exception_slot_descr(
    kind: ExcKind,
    slot: pyre_interpreter::baseobjspace::ExceptionAttrSlot,
) -> DescrRef {
    let idx = kind as u8 as usize;
    let mut cache = W_BASE_EXCEPTION_DESCR_CACHE.lock().unwrap();
    if cache[idx].is_none() {
        cache[idx] = Some(build_w_exception_group(kind));
    }
    let field_index = match slot {
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args => 2,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Context => 3,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Cause => 4,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Errno => 11,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Strerror => 12,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Filename => 13,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Filename2 => 14,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Code => 15,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback => 5,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeObject => 6,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeStart => 7,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeEnd => 8,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeReason => 9,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeEncoding => 10,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::Name => 16,
        pyre_interpreter::baseobjspace::ExceptionAttrSlot::AttrObj => 17,
    };
    field_descr_from_group(cache[idx].as_ref().unwrap(), field_index)
}

/// Field descr for `ExecutionContext::sys_exc_value`, used by the JIT
/// lowering of PUSH_EXC_INFO / POP_EXCEPT to GETFIELD_GC_R / SETFIELD_GC.
///
/// `field_type = Ref` so the optimizer tracks the value as a GC
/// reference (virtual-defer + correct resume), but the `flag` is
/// deliberately NON-pointer so `is_pointer_field()` is false and the GC
/// rewrite emits NO write barrier (`rewrite.rs handle_write_barrier_setfield`
/// gates the barrier on `is_pointer_field() && val_is_ref`).  That is
/// correct here: the EC is a non-GC `Rc`-owned struct whose
/// `sys_exc_value` slot is forwarded directly as a GC root every
/// collection (`eval::walk_pyframe_roots`), so the generational
/// remembered-set barrier is unnecessary.  A single cached Arc gives the
/// PUSH and POP ops the same `descr_identity`, so the heap optimizer
/// dead-store-eliminates a balanced save/restore (and the stored
/// exception, if it never otherwise escapes, stays virtual and DCEs).
pub fn ec_sys_exc_value_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_SYS_EXC_VALUE_OFFSET)
}

/// Field descr for `ExecutionContext::topframeref`, used by the JIT lowering
/// of `executioncontext.py:88-89 enter` / `:96-97 leave` at an inlined call:
/// `frame.f_backref = self.topframeref` reads it and
/// `self.topframeref = jit.virtual_ref(frame)` writes it back.
///
/// Same group as [`ec_sys_exc_value_descr`] — one struct, one identity, so the
/// heap optimizer can forward an `enter` store to the matching `leave` read
/// and dead-store-eliminate a balanced pair whose frame never escaped.
pub fn ec_topframeref_descr() -> DescrRef {
    ec_field_descr(pyre_interpreter::EC_TOPFRAMEREF_OFFSET)
}

/// Resolve one `EC_DESCR_GROUP` field by byte offset.  The group stamps
/// `index_in_parent` as the field's rank by offset, so the positional order of
/// `field_descrs` follows the offsets rather than the declaration order; look
/// the field up by offset so the two accessors above cannot swap.
fn ec_field_descr(offset: usize) -> DescrRef {
    let parent = EC_DESCR_GROUP.size_descr.clone() as DescrRef;
    majit_ir::descr::field_descr_from_parent_by_offset(&parent, offset)
}

/// The `ExecutionContext` field group.  `type_id 0 + vtable 0` →
/// `SimpleSizeDescr::is_object() == false`, so the optimizer builds a
/// StructPtrInfo for the (non-GC) EC pointer.  Both fields are Ref-typed
/// (ref value tracking) but Unsigned-flagged (`is_pointer_field()` is false →
/// no write barrier), which is correct because each slot is forwarded
/// directly as a GC root every collection (`eval::walk_pyframe_roots`), so the
/// generational remembered-set barrier is unnecessary.
static EC_DESCR_GROUP: LazyLock<majit_ir::descr::SimpleDescrGroup> = LazyLock::new(|| {
    use majit_ir::descr::{ArrayFlag, SimpleFieldDescrSpec};
    let field = |index: u32, field_key: &str, offset: usize| SimpleFieldDescrSpec {
        index,
        // descr.py:220-233 cache key is the bare fieldname; `name` is the
        // qualified display form.
        field_key: field_key.to_string(),
        name: format!("ExecutionContext.{field_key}"),
        offset,
        field_size: std::mem::size_of::<pyre_object::PyObjectRef>(),
        field_type: Type::Ref,
        is_immutable: false,
        is_quasi_immutable: false,
        flag: ArrayFlag::Unsigned,
        virtualizable: false,
        // Stamped below, once the specs are in offset order.
        index_in_parent: 0,
    };
    let mut specs = vec![
        field(
            0,
            "sys_exc_value",
            pyre_interpreter::EC_SYS_EXC_VALUE_OFFSET,
        ),
        field(1, "topframeref", pyre_interpreter::EC_TOPFRAMEREF_OFFSET),
    ];
    // `index_in_parent` is the field's rank by byte offset
    // (`jitcode/assembler.rs:625`), and once a parent SizeDescr is bound it is
    // also the `PtrInfo._fields` slot key the heap optimizer reads
    // (`optimizeopt/heap.rs` `field_slot_index`), so the two fields must not
    // share a rank — a shared rank makes a read of one field resolve to the
    // cached value of the other.  `ExecutionContext` is `repr(Rust)`, so rank by
    // the actual offsets instead of the declaration order; sorting the specs
    // first keeps the rank, the spec order and the `all_fielddescrs` positional
    // order one numbering, as `.enumerate()` gives
    // `build_object_descr_group_with_def_path`.
    specs.sort_by_key(|spec| spec.offset);
    for (index_in_parent, spec) in specs.iter_mut().enumerate() {
        spec.index_in_parent = index_in_parent;
    }
    majit_ir::descr::make_simple_descr_group(u32::MAX, pyre_interpreter::EC_SIZE, 0, 0, &specs)
});

/// Size descriptor for W_SliceObject allocation via NewWithVtable.
/// vtable = &SLICE_TYPE (ob_type for virtual materialization).
/// Mirrors `pypy/objspace/std/objspace.py:385` `space.newslice` →
/// `W_SliceObject(w_start, w_end, w_step)` allocation shape.
pub fn w_slice_size_descr() -> DescrRef {
    W_SLICE_DESCR_GROUP.size_descr.clone()
}

/// `W_SliceObject.w_start` — `Ptr(W_Root)` per
/// `sliceobject.py:13` `_immutable_fields_ = ['w_start', ...]`. Immutable.
pub fn slice_w_start_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 0)
}

/// `W_SliceObject.w_stop` — `Ptr(W_Root)`. Immutable.
pub fn slice_w_stop_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 1)
}

/// `W_SliceObject.w_step` — `Ptr(W_Root)`. Immutable.
pub fn slice_w_step_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 2)
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

/// R3.3b prep: canonical `PyFrame.w_globals` slot
/// (PYFRAME_W_GLOBALS_OFFSET).  Used by
/// `emit_new_pyframe_inline_self_recursive` to populate the
/// W_DictObject sibling so trace-time chases observe a non-null
/// PyObjectRef.  `PyFrame.w_globals` is the single globals slot;
/// the raw dict-storage accessor has been retired.
pub fn pyframe_w_globals_obj_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 4)
}

/// rewrite.py:665-695 handle_call_assembler scalar field read for the
/// `debugdata` slot of the virtualizable expansion (Phase D-1 prereq).
pub fn pyframe_debugdata_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 5)
}

/// rewrite.py:665-695 handle_call_assembler scalar field read for the
/// `lastblock` slot of the virtualizable expansion (Phase D-1 prereq).
pub fn pyframe_lastblock_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 6)
}

/// PyFrame.execution_context FieldDescr.
/// inline PyFrame 생성 시 caller 의 ec 를 새 frame 으로 SetfieldGc 하기 위해.
/// 호출 사이트는 `helpers.rs::emit_new_pyframe_inline*`.
pub fn pyframe_execution_context_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 7)
}

pub fn pyframe_f_generator_nowref_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 8)
}

pub fn pyframe_w_yielding_from_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 9)
}

pub fn pyframe_f_backref_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 10)
}

/// `PyFrame.flags` — the byte carrying `FLAG_ESCAPED`.  Read-or-written by the
/// traced `tb_frame` fold to reproduce the getter's `mark_as_escaped()`.
/// Located by offset, so appending another field cannot repoint it.
pub fn pyframe_flags_descr() -> DescrRef {
    let index = PYFRAME_DESCR_GROUP
        .field_descrs
        .iter()
        .position(|d| d.offset() == crate::frame_layout::PYFRAME_FLAGS_OFFSET)
        .expect("PyFrame descr group has no flags field");
    field_descr_from_group(&PYFRAME_DESCR_GROUP, index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_len_descr_keeps_its_unicode_parent_alive() {
        let descr = str_len_descr();
        let field = descr.as_field_descr().expect("str len FieldDescr");
        assert_eq!(
            field.offset(),
            pyre_object::unicodeobject::UNICODE_LEN_OFFSET
        );
        assert!(field.is_immutable());

        let parent = field
            .get_parent_descr()
            .expect("W_UnicodeObject.len must retain its parent SizeDescr");
        let size = parent
            .as_size_descr()
            .expect("Unicode field parent must be a SizeDescr");
        assert_eq!(
            size.size(),
            pyre_object::unicodeobject::W_UNICODE_OBJECT_SIZE
        );
        assert_eq!(size.type_id(), W_UNICODE_GC_TYPE_ID);
        let parent_field = size.all_fielddescrs()[2].clone() as DescrRef;
        assert!(std::sync::Arc::ptr_eq(&parent_field, &descr));
    }

    #[test]
    fn pyobject_size_descrs_include_inherited_w_class_gc_field() {
        let descr = w_int_size_descr();
        let size = descr.as_size_descr().expect("W_IntObject SizeDescr");
        assert!(
            size.gc_fielddescrs()
                .iter()
                .any(|fd| fd.offset() == W_CLASS_OFFSET),
            "malloc_zero_filled=False requires the inherited PyObject.w_class edge"
        );
    }

    #[test]
    fn jit_emitted_raw_pointer_objects_are_non_moving() {
        let traceback_descr = pytraceback_size_descr();
        let traceback_size = traceback_descr
            .as_size_descr()
            .expect("PyTraceback SizeDescr");
        assert!(
            traceback_size.non_moving(),
            "raw traceback pointers are not rewritten when a minor collection moves an object"
        );

        let instance_descr = w_object_object_size_descr();
        let instance_size = instance_descr
            .as_size_descr()
            .expect("W_ObjectObject SizeDescr");
        assert!(
            instance_size.non_moving(),
            "raw instance pointers can survive across allocation without being rooted"
        );

        let storage_descr = crate::state::mapdict_storage_gcarray_descr();
        let storage_array = storage_descr
            .as_array_descr()
            .expect("mapdict storage ArrayDescr");
        assert!(
            storage_array.non_moving(),
            "the mapdict custom tracer marks raw storage pointers but cannot rewrite them"
        );
    }

    #[test]
    fn pyframe_size_descr_clears_the_vable_token_slot() {
        let descr = pyframe_size_descr();
        let size = descr.as_size_descr().expect("PyFrame SizeDescr");
        assert!(
            size.gc_fielddescrs()
                .iter()
                .any(|fd| fd.offset() == crate::frame_layout::PYFRAME_VABLE_TOKEN_OFFSET),
            "emit_force_virtualizable reads vable_token with GETFIELD_GC_R, so \
             clear_gc_fields must zero it on a JIT-inlined frame allocation"
        );
    }

    #[test]
    fn exception_size_descr_clears_every_runtime_traced_gc_field() {
        let (descr, _, _, _) = w_exception_descrs(ExcKind::ValueError);
        let size = descr.as_size_descr().expect("W_BaseException SizeDescr");
        let mut actual: Vec<usize> = size.gc_fielddescrs().iter().map(|fd| fd.offset()).collect();
        actual.sort_unstable();
        actual.dedup();

        let mut expected = W_BASE_EXCEPTION_GC_PTR_OFFSETS.to_vec();
        expected.push(W_CLASS_OFFSET);
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_field_descr_indices_are_stable_and_distinct() {
        let a = make_field_descr(8, 8, Type::Int, false);
        let b = make_field_descr(8, 8, Type::Int, false);
        let c = make_field_descr(16, 8, Type::Int, false);

        assert_eq!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
    }

    #[test]
    fn test_array_descr_indices_are_distinct_per_call() {
        // PyPy `descr.py:350-351 cache[ARRAY_OR_STRUCT]` keys on ARRAY
        // object identity; `make_array_descr` callers without an
        // identity carrier (`array_type_id = None`) each produce a
        // distinct ARRAY → distinct `descr_id`.  Singleton semantics
        // require routing through `make_array_descr_with_full_id` with
        // a stable identity string instead.
        let a = make_array_descr(0, 8, None, Type::Int, false);
        let b = make_array_descr(0, 8, None, Type::Int, false);
        let c = make_array_descr(0, 8, None, Type::Ref, false);

        assert_ne!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
        assert_ne!(b.index(), c.index());
    }

    #[test]
    fn test_array_descr_with_full_id_singleton_per_identity() {
        // `descr.py:348-378 get_array_descr` cache hit on
        // `LLType::Array(path_hash(atid))` returns the existing Arc
        // — `make_array_descr_with_full_id` with the same identity
        // string is a singleton.
        let a = crate::descr::make_array_descr_with_full_id(
            0,
            8,
            0,
            None,
            Type::Int,
            false,
            Some("pyre::test_singleton_id".to_string()),
        );
        let b = crate::descr::make_array_descr_with_full_id(
            0,
            8,
            0,
            None,
            Type::Int,
            false,
            Some("pyre::test_singleton_id".to_string()),
        );
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "same identity carrier must collapse to the same Arc"
        );
    }

    #[test]
    fn make_call_descr_from_bh_round_trips_most_general_effectinfo() {
        use majit_ir::EffectInfo;
        use majit_translate::jitcode::BhCallDescr;

        let bh = BhCallDescr::from_arg_classes("r".to_string(), 'r', EffectInfo::MOST_GENERAL);

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Ref]);
        assert_eq!(call.result_type(), Type::Ref);
        assert_eq!(call.result_size(), 8);
        assert!(!call.is_result_signed());
        assert_eq!(call.get_extra_info(), &EffectInfo::MOST_GENERAL);
        assert!(call.get_extra_info().check_can_raise(false));
    }

    #[test]
    fn make_call_descr_from_bh_round_trips_cannot_raise_effectinfo() {
        use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex};
        use majit_translate::jitcode::BhCallDescr;

        let extra_info = EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None);
        let bh = BhCallDescr::from_arg_classes("ir".to_string(), 'v', extra_info.clone());

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Int, Type::Ref]);
        assert_eq!(call.result_type(), Type::Void);
        assert_eq!(call.result_size(), 0);
        assert_eq!(call.get_extra_info(), &extra_info);
        assert!(!call.get_extra_info().check_can_raise(false));
    }

    #[test]
    fn make_call_descr_from_bh_preserves_singlefloat_result_layout() {
        use majit_ir::EffectInfo;
        use majit_translate::jitcode::{BhCallDescr, CallResultErasedKey};

        let bh = BhCallDescr::from_arg_classes("S".to_string(), 'S', EffectInfo::MOST_GENERAL);

        assert_eq!(bh.arg_classes, "S");
        assert_eq!(bh.result_type, 'S');
        assert_eq!(bh.result_size, 4);
        assert!(!bh.result_signed);
        assert_eq!(bh.result_erased, CallResultErasedKey::SingleFloat);

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Int]);
        assert_eq!(call.result_type(), Type::Int);
        // descr.py:524-526 `get_result_type()` parity — the raw 'S' char
        // must survive the BhCallDescr -> CallDescr conversion, so
        // downstream consumers can distinguish singlefloat from a real
        // int result.  pyre's `result_class()` returns the raw char
        // (matches `descr.py:526 get_result_type()`); the normalized
        // form per descr.py:527-532 (collapsing 'S' → 'i') is not yet
        // exposed as a separate method but the underlying `result_type`
        // is already `Type::Int`, which is the normalized view.
        assert_eq!(call.result_class(), 'S');
        assert_eq!(call.result_size(), 4);
        assert!(!call.is_result_signed());
    }

    #[test]
    fn make_descr_from_bh_field_preserves_parent_name_index() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::{BhDescr, BhFieldSpec, BhSizeSpec};

        let parent = BhSizeSpec {
            size: 24,
            type_id: 7,
            vtable: 0,
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: vec![
                BhFieldSpec {
                    index: 0,
                    field_key: "next".into(),
                    name: "Cell.next".into(),
                    offset: 8,
                    field_size: 8,
                    field_type: Type::Ref,
                    field_flag: ArrayFlag::Pointer,
                    is_field_signed: false,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent: 0,
                },
                BhFieldSpec {
                    index: 1,
                    field_key: "value".into(),
                    name: "Cell.value".into(),
                    offset: 16,
                    field_size: 8,
                    field_type: Type::Int,
                    field_flag: ArrayFlag::Signed,
                    is_field_signed: true,
                    is_immutable: true,
                    is_quasi_immutable: false,
                    index_in_parent: 1,
                },
            ],
        };

        let descr = make_descr_from_bh(&BhDescr::Field {
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
            field_flag: ArrayFlag::Signed,
            is_field_signed: true,
            is_immutable: true,
            is_quasi_immutable: false,
            index_in_parent: 1,
            parent: Some(parent),
            name: "value".into(),
            owner: "Cell".into(),
        });
        let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");

        assert_eq!(field.field_name(), "Cell.value");
        assert_eq!(field.index_in_parent(), 1);
        assert_eq!(field.offset(), 16);
        assert!(field.is_immutable());
        let parent = field
            .get_parent_descr()
            .expect("FieldDescr.parent_descr must be preserved");
        let size = parent
            .as_size_descr()
            .expect("parent_descr must be a SizeDescr");
        assert_eq!(size.size(), 24);
        assert_eq!(size.type_id(), 7);
        assert_eq!(size.all_fielddescrs().len(), 2);
        assert_eq!(size.all_fielddescrs()[1].field_name(), "Cell.value");
    }

    #[test]
    fn make_descr_from_bh_items_block_capacity_with_parent_is_canonical() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::{BhDescr, BhFieldSpec, BhSizeSpec};

        let capacity = BhFieldSpec {
            index: 0,
            field_key: "capacity".into(),
            name: "ItemsBlock.capacity".into(),
            offset: 0,
            field_size: std::mem::size_of::<usize>(),
            field_type: Type::Int,
            field_flag: ArrayFlag::Unsigned,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
        };
        let parent = BhSizeSpec {
            size: std::mem::size_of::<usize>(),
            type_id: 3938139489201595032,
            vtable: 0,
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: vec![capacity.clone()],
        };
        let descr = make_descr_from_bh(&BhDescr::Field {
            offset: capacity.offset,
            field_size: capacity.field_size,
            field_type: capacity.field_type,
            field_flag: capacity.field_flag,
            is_field_signed: capacity.is_field_signed,
            is_immutable: capacity.is_immutable,
            is_quasi_immutable: capacity.is_quasi_immutable,
            index_in_parent: capacity.index_in_parent,
            parent: Some(parent),
            name: "capacity".into(),
            owner: "ItemsBlock".into(),
        });
        let canonical = items_block_capacity_descr();

        assert!(
            std::sync::Arc::ptr_eq(&descr, &canonical),
            "the build-time parent must not mint a second capacity FieldDescr",
        );
        assert_eq!(descr.index(), canonical.index());
    }

    #[test]
    fn make_descr_from_bh_bridges_codewriter_int_items_leaves_to_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::BhDescr;

        // Shape `_handle_list_call` emits and the codewriter assembler
        // round-trips: dotted nested name, owner `W_ListObject`, offset 0
        // (unresolved, because `W_ListObject` is not in the codewriter
        // struct layouts), no parent. Without the bridge these mint a
        // SimpleFieldDescr at offset 0 (the list header).
        for (name, expected, ty) in [
            ("int_items.len", list_int_items_len_descr(), Type::Int),
            ("int_items.block", list_int_items_block_descr(), Type::Ref),
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 0,
                field_size: 8,
                field_type: ty,
                field_flag: ArrayFlag::Signed,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 0,
                parent: None,
                name: name.into(),
                owner: "W_ListObject".into(),
            });
            let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");
            assert_ne!(field.offset(), 0, "{name} bridged to the list header");
            assert_eq!(
                field.offset(),
                expected.as_field_descr().unwrap().offset(),
                "{name} offset must match the W_LIST_DESCR_GROUP entry",
            );
        }
    }

    #[test]
    fn make_descr_from_bh_bridges_codewriter_bare_items_aliases_to_block_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::BhDescr;

        // A bare `int_items` / `float_items` read (the `w_list_append` body
        // reads the typed-storage struct base before reaching `.ptr`/`.len`)
        // must bridge to the same canonical `.block` group entry as the dotted
        // `.block` leaf — a populated parent_descr and the `.block` offset, not
        // the offset-0 list header.
        for (name, expected) in [
            ("int_items", list_int_items_block_descr()),
            ("float_items", list_float_items_block_descr()),
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 0,
                field_size: 8,
                field_type: Type::Ref,
                field_flag: ArrayFlag::Signed,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 0,
                parent: None,
                name: name.into(),
                owner: "W_ListObject".into(),
            });
            let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");
            assert!(
                field.get_parent_descr().is_some(),
                "{name} must carry a parent_descr after the bridge",
            );
            assert_eq!(
                field.offset(),
                expected.as_field_descr().unwrap().offset(),
                "{name} offset must match the `.block` W_LIST_DESCR_GROUP entry",
            );
        }
    }

    #[test]
    fn make_descr_from_bh_bridges_codewriter_list_header_fields_to_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::BhDescr;

        // The `w_list_append` body reads `list.{strategy,length,items}`
        // directly. The codewriter mints these as a `SimpleFieldDescr` with no
        // parent_descr backreference, which `ensure_ptr_info_arg0` rejects.
        // The bridge must route them to the canonical group entries so
        // `get_parent_descr()` is populated (and the offset matches).
        for (name, expected, ty) in [
            ("strategy", list_strategy_descr(), Type::Int),
            ("length", list_length_descr(), Type::Int),
            ("items", list_items_descr(), Type::Ref),
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 0,
                field_size: 8,
                field_type: ty,
                field_flag: ArrayFlag::Signed,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 0,
                parent: None,
                name: name.into(),
                owner: "W_ListObject".into(),
            });
            let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");
            assert!(
                field.get_parent_descr().is_some(),
                "{name} must carry a parent_descr after the bridge",
            );
            assert_eq!(
                field.offset(),
                expected.as_field_descr().unwrap().offset(),
                "{name} offset must match the W_LIST_DESCR_GROUP entry",
            );
        }
    }

    #[test]
    fn list_and_tuple_subclass_storage_is_cleared_by_allocation_descrs() {
        let list_size = w_list_size_descr();
        let list_gc_offsets: Vec<_> = list_size
            .as_size_descr()
            .unwrap()
            .gc_fielddescrs()
            .iter()
            .map(|field| field.offset())
            .collect();
        assert!(list_gc_offsets.contains(&std::mem::offset_of!(W_ListObject, w_slots)));

        let tuple_size = w_tuple_size_descr();
        let tuple_gc_offsets: Vec<_> = tuple_size
            .as_size_descr()
            .unwrap()
            .gc_fielddescrs()
            .iter()
            .map(|field| field.offset())
            .collect();
        assert!(tuple_gc_offsets.contains(&std::mem::offset_of!(W_TupleObject, w_dict)));
    }

    #[test]
    fn make_descr_from_bh_bridges_codewriter_box_payload_fields_to_group() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::BhDescr;

        // A codewriter-lowered body reads a box payload through the producer's
        // struct-layout descr: `W_IntObject` is modeled with a header, so its
        // `intval` lands at `index_in_parent` = 2.  The walker materializes the
        // same box through the single-field `W_*_DESCR_GROUP` (`index` = 0).
        // The bridge must return the group entry so the optimizer's
        // identity-keyed virtual-field cache matches the box-creation descr.
        for (owner, name, expected, ty) in [
            ("W_IntObject", "intval", int_intval_descr(), Type::Int),
            (
                "pyre_object::intobject::W_IntObject",
                "intval",
                int_intval_descr(),
                Type::Int,
            ),
            ("W_BoolObject", "intval", bool_intval_descr(), Type::Int),
            (
                "W_FloatObject",
                "floatval",
                float_floatval_descr(),
                Type::Float,
            ),
        ] {
            let descr = make_descr_from_bh(&BhDescr::Field {
                offset: 16,
                field_size: 8,
                field_type: ty,
                field_flag: ArrayFlag::Signed,
                is_field_signed: true,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 2,
                parent: None,
                name: name.into(),
                owner: owner.into(),
            });
            assert!(
                std::sync::Arc::ptr_eq(&descr, &expected),
                "{owner}.{name} must bridge to the canonical group entry Arc",
            );
        }
    }

    /// A `PyObject.w_class` `BhDescr` describing the same access as the
    /// canonical header descr, for both owner spellings the codewriter emits.
    fn w_class_bh(owner: &str, field_size: usize) -> majit_translate::jitcode::BhDescr {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::BhDescr;

        BhDescr::Field {
            offset: pyre_object::pyobject::W_CLASS_OFFSET,
            field_size,
            field_type: Type::Ref,
            field_flag: ArrayFlag::Signed,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            // slot 0 is `ob_type`; `w_class` is slot 1 of the header.
            index_in_parent: 1,
            parent: None,
            name: "w_class".into(),
            owner: owner.into(),
        }
    }

    /// The shared `PyObject` header's `w_class` bridges to the same descr the
    /// walker pins a value's class through, so a codewriter-lowered subclass
    /// test (`is_plain_int1`) reads the header the walker already guarded
    /// instead of emitting a second, uncacheable read of the same offset.
    #[test]
    fn make_descr_from_bh_bridges_pyobject_w_class_to_the_walker_descr() {
        let canonical = w_class_descr();
        let width = W_CLASS_FIELD_DESCR.field_size();

        for owner in ["PyObject", "pyre_object::pyobject::PyObject"] {
            let descr = make_descr_from_bh(&w_class_bh(owner, width));
            assert!(
                std::sync::Arc::ptr_eq(&descr, &canonical),
                "{owner}.w_class must bridge to the walker's w_class descr Arc",
            );
        }
    }

    /// …but only when the two spellings describe the same access. The canonical
    /// descr hardcodes an 8-byte width while the codewriter sizes a pointer by
    /// `target_word_size()`, so on a 32-bit target the incoming descr is a
    /// narrower load at the same offset. Bridging there would widen the read
    /// over the adjacent payload, so the mismatch declines instead.
    #[test]
    fn make_descr_from_bh_declines_w_class_bridge_on_a_width_mismatch() {
        let canonical = w_class_descr();
        let narrower = W_CLASS_FIELD_DESCR.field_size() / 2;

        for owner in ["PyObject", "pyre_object::pyobject::PyObject"] {
            let descr = make_descr_from_bh(&w_class_bh(owner, narrower));
            assert!(
                !std::sync::Arc::ptr_eq(&descr, &canonical),
                "{owner}.w_class must not bridge to a descr of a different width",
            );
            assert_eq!(
                descr.as_field_descr().map(|f| f.field_size()),
                Some(narrower),
                "the declined descr must keep the width the codewriter asked for",
            );
        }
    }

    #[test]
    fn make_descr_from_bh_struct_array_preserves_type_and_interior_fields() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::{BhDescr, BhFieldSpec, BhInteriorFieldSpec, BhSizeSpec};

        let fields = vec![
            BhFieldSpec {
                index: 0,
                field_key: "x".into(),
                name: "Point.x".into(),
                offset: 0,
                field_size: 8,
                field_type: Type::Int,
                field_flag: ArrayFlag::Signed,
                is_field_signed: true,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 0,
            },
            BhFieldSpec {
                index: 1,
                field_key: "y".into(),
                name: "Point.y".into(),
                offset: 8,
                field_size: 8,
                field_type: Type::Float,
                field_flag: ArrayFlag::Float,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 1,
            },
        ];
        let owner = BhSizeSpec {
            size: 16,
            type_id: 11,
            vtable: 0,
            is_gc_managed: true,
            headerless: false,
            all_fielddescrs: fields.clone(),
        };
        let interior_fields = vec![
            BhInteriorFieldSpec {
                index: 0,
                field: fields[0].clone(),
                owner: owner.clone(),
            },
            BhInteriorFieldSpec {
                index: 1,
                field: fields[1].clone(),
                owner,
            },
        ];

        let descr = make_descr_from_bh(&BhDescr::Array {
            base_size: 8,
            itemsize: 16,
            len_offset: Some(0),
            type_id: 42,
            gc_type_id: 0,
            item_type: Type::Ref,
            is_array_of_pointers: false,
            is_array_of_structs: true,
            is_item_signed: false,
            // GcArray of Point structs (header-carrying), so the descr keeps
            // the GUARD_GC_TYPE gate.
            is_gc_managed: true,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields,
        });
        let array = descr.as_array_descr().expect("Array BhDescr -> ArrayDescr");

        assert!(array.is_array_of_structs());
        // `type_id` is the dense sequential GC tid allocated by
        // `GcCache::init_array_descr` (analog of `gc.py:544-549
        // GcLLDescr_framework.init_array_descr` + `gctypelayout.py:301-357
        // TypeLayoutBuilder.get_type_id`).  Exact value depends on the
        // global allocator state — test-suite ordering is non-deterministic
        // so we only assert it is non-zero (tid 0 reserved per
        // `gctypelayout.py:328-331`).  The structural identity that
        // round-trips through `BhDescr::Array.type_id` (path_hash payload)
        // lives in `cache_key` (descr.rs:2120-2131), independent of the
        // GC tid.
        assert_ne!(array.type_id(), 0);
        assert_eq!(array.cache_key(), 42);
        assert_eq!(array.item_type(), Type::Ref);
        let interior = array
            .get_all_interiorfielddescrs()
            .expect("struct array must preserve interior field descrs");
        assert_eq!(interior.len(), 2);
        let second = interior[1]
            .as_interior_field_descr()
            .expect("interior field descr shape");
        assert_eq!(second.field_descr().field_name(), "Point.y");
        let parent = second
            .field_descr()
            .get_parent_descr()
            .expect("interior field parent_descr must be preserved");
        assert_eq!(parent.as_size_descr().unwrap().size(), 16);
    }
}

/// CallDescr for `pyre_object::longobject::jit_w_long_fits_int(obj) -> i64`.
/// `rbigint.fits_int()` is not annotated `@jit.elidable` upstream; it is only
/// used as a cannot-raise runtime guard before the elidable `toint()` call.
pub fn make_jit_w_long_fits_int_calldescr() -> DescrRef {
    majit_ir::make_call_descr(
        vec![Type::Ref],
        Type::Int,
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    )
}

/// CallDescr for `pyre_object::longobject::jit_w_long_toint(obj) -> i64`.
/// `W_LongObject.toint()` (longobject.py:138) → `rbigint.toint()`
/// (rbigint.py:465) — `EF_ELIDABLE_CANNOT_RAISE` because the caller
/// emits a fits_int GUARD_TRUE before invoking; OverflowError is
/// statically unreachable post-guard.
pub fn make_jit_w_long_toint_calldescr() -> DescrRef {
    majit_ir::make_call_descr(
        vec![Type::Ref],
        Type::Int,
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::ElidableCannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    )
}

fn simple_field_spec_from_bh(
    spec: &majit_translate::jitcode::BhFieldSpec,
) -> majit_ir::descr::SimpleFieldDescrSpec {
    majit_ir::descr::SimpleFieldDescrSpec {
        index: spec.index,
        field_key: spec.field_key().to_string(),
        name: spec.name.clone(),
        offset: spec.offset,
        field_size: spec.field_size,
        field_type: spec.field_type,
        is_immutable: spec.is_immutable,
        is_quasi_immutable: spec.is_quasi_immutable,
        flag: spec.field_flag,
        virtualizable: false,
        index_in_parent: spec.index_in_parent,
    }
}

/// `descr.py:108-118 get_size_descr` cache parity.
///
/// PyPy `gc_cache._cache_size[STRUCT]` keys on the **STRUCT object
/// identity**, not on its layout — two distinct RPython STRUCTs that
/// share `(size, vtable, fieldlist)` get distinct `SizeDescr` Arcs.
/// Pyre's analogue of "STRUCT identity" is `BhSizeSpec.type_id`
/// (`jit_struct.rs:__majit_type_id` → `path_hash(module_path::TypeName)`):
/// every struct type has a unique `type_id`, and two RPython STRUCTs
/// with coincidentally-identical layout end up with distinct
/// `type_id`s.  Keying the cache on `type_id` alone matches PyPy's
/// per-type identity, where structural-equality keying (the prior
/// `BhSizeSpec`-by-value variant) would have collapsed identity for
/// layout-coincident-but-logically-distinct structs.
///
/// `spec.type_id == 0` is the legacy fallback path
/// (`assembler.rs:2244 bh_size_spec_from_callcontrol` stamps zero
/// when the analyzer-time callcontrol has no host-type carrier).
/// Without a STRUCT-identity carrier we MUST NOT key the cache by
/// the zero sentinel — different STRUCTs with `type_id == 0` would
/// alias onto the first one inserted (`or_insert` "first wins"),
/// silently mixing their field tables.  PyPy's `_cache_size[STRUCT]`
/// never aliases distinct STRUCTs; absent a real identity carrier,
/// the closest orthodox behaviour is "each call is a distinct
/// STRUCT" — mint fresh per call.
fn simple_descr_group_from_bh_size(
    spec: &majit_translate::jitcode::BhSizeSpec,
) -> majit_ir::descr::SimpleDescrGroup {
    let field_specs: Vec<_> = spec
        .all_fielddescrs
        .iter()
        .map(simple_field_spec_from_bh)
        .collect();

    if spec.type_id == 0 {
        // No STRUCT-identity carrier — mint fresh per call so distinct
        // type_id-less STRUCTs don't collapse onto the first-inserted
        // descr group.  Per-STRUCT caching kicks in only when callers
        // route through a real `type_id` source.
        //
        // The missing key says nothing about the STRUCT's shape, so carry
        // `is_gc_managed`/`headerless` here exactly as the keyed arm below
        // does.  `bh_size_spec_from_descr` (`majit-translate`) reads both
        // straight off a descr while taking `type_id` from `cache_key()`, so
        // a raw or header-less parent with no key reaches this arm; the
        // defaulting factory would give it back a GC header it does not have.
        return majit_ir::descr::make_simple_descr_group_with_flags(
            u32::MAX,
            spec.size,
            spec.type_id as u32,
            spec.vtable as usize,
            spec.is_gc_managed,
            spec.headerless,
            &field_specs,
        );
    }
    // `descr.py:108-118 get_size_descr` + `:218-239 get_field_descr`
    // keyed publish: GcCache is the sole owner/cache for this STRUCT.
    majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
        u32::MAX,
        spec.size,
        spec.type_id as u32,
        spec.type_id,
        spec.vtable as usize,
        spec.is_gc_managed,
        spec.headerless,
        &field_specs,
        &[],
    )
}

fn field_descr_from_bh_field(
    field: &majit_translate::jitcode::BhFieldSpec,
    parent: Option<&majit_translate::jitcode::BhSizeSpec>,
) -> DescrRef {
    if let Some(parent) = parent {
        // `descr.py:218-239 get_field_descr` cache-hit: when the parent
        // STRUCT is published in `_cache_size`, walk its
        // `all_fielddescrs()` directly so the returned Arc is the
        // runtime `PyreFieldDescr` (or analyzer-published
        // `SimpleFieldDescr`).  Both back-reference the same parent
        // SizeDescr via `parent_descr` (descr.py:200), so the
        // an adapter wrapper is unnecessary on this path — analyzer
        // raw-set Arcs and runtime allocator descrs share
        // one identity slot.
        if parent.type_id != 0 {
            let key = majit_ir::descr::LLType::Struct(parent.type_id);
            let field_key = field.field_key().to_string();
            let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
            // `descr.py:220-221 cache[STRUCT][fieldname]` hit.
            if let Some(fd) = gc
                ._cache_field
                .get(&key)
                .and_then(|inner| inner.get(&field_key))
            {
                return fd.clone() as DescrRef;
            }
            // Miss with the parent STRUCT published: mint through
            // `descr.py:225-238 get_field_descr` so this resolution and the
            // walker's (`pyjitpl/dispatch.rs field_descr_ref_from_bh`) share
            // one Arc.  `descr.py:238 parent_descr = get_size_descr(STRUCT)`
            // needs the `_cache_size` slot, so only take this route when it
            // is populated.
            if gc._cache_size.contains_key(&key) {
                let fd = gc.get_field_descr(
                    key,
                    &field_key,
                    Some(field.name.as_str()),
                    field.offset,
                    field.field_size,
                    field.field_type,
                    field.is_immutable,
                    field.is_quasi_immutable,
                    field.field_flag,
                    field.index,
                    false,
                    field.index_in_parent,
                );
                return fd as DescrRef;
            }
        }
        // Cache miss / non-keyed parent — fall back to the descr group
        // field itself; the keyed path minted it through get_field_descr.
        let group = simple_descr_group_from_bh_size(parent);
        if let Some((pos, _)) =
            parent.all_fielddescrs.iter().enumerate().find(|(_, spec)| {
                spec.offset == field.offset && spec.field_key() == field.field_key()
            })
        {
            if let Some(descr) = group.field_descrs.get(pos) {
                return descr.clone() as DescrRef;
            }
        }
    }

    let descr = majit_ir::descr::SimpleFieldDescr::new_with_name(
        field.index,
        field.offset,
        field.field_size,
        field.field_type,
        field.is_immutable,
        field.field_flag,
        field.name.clone(),
        field.field_key().to_string(),
    )
    .with_quasi_immutable(field.is_quasi_immutable);
    let arc: DescrRef = Arc::new(descr);
    // descr.py:225-235 `get_field_descr` cache-miss path — register the
    // freshly-minted field descr so `compute_bitstrings` enumerates it.
    majit_ir::descr_registry::register_field(arc.clone());
    arc
}

fn bh_field_cache_key(owner: &str, name: &str) -> String {
    if owner.is_empty() {
        return name.to_string();
    }
    let prefix = format!("{owner}.");
    name.strip_prefix(&prefix).unwrap_or(name).to_string()
}

/// Keyed sibling: accepts the u64 `cache_key` (= `path_hash(array_type_id)`)
/// so the freshly-minted `SimpleArrayDescr` lands in
/// `gc_cache._cache_array[LLType::Array(cache_key)]` in addition to
/// the snapshot order Vec.  Mirrors PyPy `cpu.arraydescrof(ARRAY)`
/// per-ARRAY cache identity (`descr.py:348-378`).  `cache_key == 0`
/// is the no-identity sentinel — registers via the non-keyed path.
pub fn make_struct_array_descr_full_keyed(
    descr_index: u32,
    base_size: usize,
    item_size: usize,
    len_offset: Option<usize>,
    type_id: u32,
    cache_key: u64,
    item_type: Type,
    interior_fields: &[majit_translate::jitcode::BhInteriorFieldSpec],
) -> DescrRef {
    use majit_ir::descr::{ArrayFlag, LLType, SimpleArrayDescr, gc_cache, try_downcast_arc};
    // `descr.py:348-378 get_array_descr(gccache, ARRAY)` cache-or-mint:
    // an `LLType::Array(cache_key)` cache hit returns the existing Arc
    // (the `SimpleArrayDescr` in the slot — from a prior analyzer call
    // or from a runtime mint via `make_array_descr_with_full_id`); only
    // a miss mints a fresh
    // descr.  Matches PyPy `cpu.arraydescrof(ARRAY)` per-ARRAY object
    // identity — both pyre runtime mint sites and analyzer share a
    // single Arc per cache key.  `cache_key == 0` is the no-identity
    // sentinel (legacy non-keyed callers) — mint locally, no cache
    // publication.
    let array_descr_dyn: DescrRef = if cache_key != 0 {
        let array_key = LLType::Array(cache_key);
        let cached = gc_cache().lock().unwrap().get_array_descr(
            array_key.clone(),
            base_size,
            item_size,
            ArrayFlag::Struct,
            item_type,
            len_offset.is_none(),
            len_offset.unwrap_or(0),
            false,
            '\x00',
        );
        // PyPy `gc.py:544-549 init_array_descr` stamps `descr.tid`
        // from `layoutbuilder.get_type_id(A)` — a dense sequential
        // GC type id.  Pyre does not yet port the layoutbuilder
        // analog, so the cache-hit branch only
        // updates the per-trace `descr_index` and leaves
        // `SimpleArrayDescr.type_id` at its mint default (0, set in
        // `get_array_descr` at descr.rs:515).  The
        // `BhDescr::Array.type_id` payload threaded through this
        // helper is the producer-side `path_hash(array_type_id)` and
        // already lands in `SimpleArrayDescr.cache_key` via the
        // `get_array_descr` cache-miss-mint stamp at descr.rs:526-528
        // — structural identity (`cache_key`) is decoupled from GC tid
        // (`type_id`) per the trait doc at descr.rs:2120-2131.  Runtime
        // registrations (`SimpleArrayDescr` from the runtime mint
        // factories) carry their real GC tid at mint and win the cache
        // slot.
        cached.set_index(descr_index);
        cached
    } else {
        // No cache identity — local mint.  Two `cache_key == 0`
        // entries are intentionally distinct STRUCTs sharing the
        // no-identity sentinel; per-`make_array_descr` legacy callers
        // rely on this.
        let mut raw_array_descr = SimpleArrayDescr::with_flag(
            descr_index,
            base_size,
            item_size,
            type_id,
            item_type,
            ArrayFlag::Struct,
        );
        raw_array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
        let arc: DescrRef = Arc::new(raw_array_descr);
        majit_ir::descr_registry::register_array(arc.clone());
        arc
    };
    if interior_fields.is_empty() {
        return array_descr_dyn;
    }

    // Upcast the cached array descr to `Arc<dyn ArrayDescr>` for
    // `SimpleInteriorFieldDescr.array_descr` storage.  The cache slot
    // always holds a concrete `SimpleArrayDescr` (analyzer mint,
    // gc_cache internal mint, or runtime mint); downcast to the
    // concrete Arc type, then upcast.
    let array_descr_for_interior: Arc<dyn majit_ir::descr::ArrayDescr> =
        try_downcast_arc::<SimpleArrayDescr>(array_descr_dyn.clone())
            .expect("array descriptor cache must hold SimpleArrayDescr for struct arrays");

    let mut descrs: Vec<DescrRef> = Vec::new();
    for interior in interior_fields {
        let owner_group = simple_descr_group_from_bh_size(&interior.owner);
        let field_pos = interior
            .owner
            .all_fielddescrs
            .iter()
            .position(|field| {
                field.index_in_parent == interior.field.index_in_parent
                    && field.name == interior.field.name
            })
            .unwrap_or(interior.field.index_in_parent);
        if let Some(field_descr) = owner_group.field_descrs.get(field_pos) {
            // `descr.py:423-438 get_interiorfield_descr` cache-or-mint
            // is keyed on the outer ARRAY's lltype identity.  When the
            // outer array carries `cache_key != 0`, route through the
            // keyed `_cache_interiorfield[(LLType::Array(cache_key),
            // name, "")]` so both analyzer and runtime share one Arc
            // per `(ARRAY, name)` tuple.  With `cache_key == 0`
            // (no-identity outer array) PyPy has NO "merge several
            // ARRAYs' interiors into one slot" behavior — local mint
            // a fresh `SimpleInteriorFieldDescr` per call so distinct
            // no-identity arrays do not alias on their interior field
            // descrs.
            //
            // Bare interior field name (`spec.name`) is the cache key per
            // `descr.py:221 cache[STRUCT][fieldname]` shape.
            let bare_name = interior
                .field
                .name
                .rsplit_once('.')
                .map(|(_, n)| n.to_string())
                .unwrap_or_else(|| interior.field.name.clone());
            let field_dyn: Arc<dyn majit_ir::descr::FieldDescr> = field_descr.clone();
            let ifd: DescrRef = if cache_key != 0 {
                gc_cache().lock().unwrap().get_interiorfield_descr(
                    LLType::Array(cache_key),
                    bare_name,
                    String::new(),
                    array_descr_for_interior.clone(),
                    field_dyn,
                )
            } else {
                Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
                    u32::MAX,
                    array_descr_for_interior.clone(),
                    field_dyn,
                )) as DescrRef
            };
            // Per-trace `interior.index` stamp matches the analyzer's
            // `cc.interiorfielddescrof` codewriter idx convention.
            ifd.set_index(interior.index);
            descrs.push(ifd);
        }
    }

    // `descr.py:372-375 arraydescr.all_interiorfielddescrs = descrs`
    // set-once via OnceLock.  Cache-hit case: a prior populator already
    // set the list; our set is a no-op which is the desired semantic.
    array_descr_for_interior.set_all_interiorfielddescrs(descrs);
    array_descr_dyn
}

/// Concrete `JitCodeDescr` adapter for `inline_call_*` opcodes.
///
/// RPython parity: `JitCode(AbstractDescr)` carries `fnaddr` +
/// `calldescr` + the callee's bytecode body and is emitted directly as
/// the descr operand of `inline_call_*`. The codewriter side surfaces
/// this as `BhDescr::JitCode { jitcode_index, fnaddr, calldescr }`
/// (`majit-translate/src/codewriter/jitcode.rs:667`); the trace-side
/// walker (`jitcode_dispatch.rs::WalkContext`) consumes
/// `&[Arc<dyn Descr>]` and queries `as_jitcode_descr()` /
/// `jitcode_index()`.
///
/// `PyreJitCodeDescr` bridges those two layers: production callers
/// build a `Vec<DescrRef>` from the codewriter's `BhDescr` pool via
/// [`make_descr_from_bh`] (each `BhDescr::JitCode` wraps in this
/// struct so the walker's `as_jitcode_descr() -> Some(&self)` cast
/// succeeds; Field/Array/Size become `PyreFieldDescr` /
/// `SimpleArrayDescr` / `PyreSizeDescr`; `Call` becomes a
/// `MetaCallDescr` carrying the codewriter's `EffectInfo`).
///
/// Tests in `jitcode_dispatch.rs` previously used a `TestJitCodeDescr`
/// duplicate of this shape — production code now goes through the same
/// type so the test fixture can be progressively replaced without
/// behaviour drift.
#[derive(Debug)]
pub struct PyreJitCodeDescr {
    jitcode_index: usize,
}

impl PyreJitCodeDescr {
    /// Build a `PyreJitCodeDescr` with the given runtime jitcode index.
    /// `jitcode_index` indexes into the runtime's all-jitcodes table
    /// (`pyre-jit-trace/src/jitcode_runtime.rs::ALL_JITCODES`); the
    /// walker's `sub_jitcode_lookup` resolves it to the callee's body.
    pub fn new(jitcode_index: usize) -> Self {
        Self { jitcode_index }
    }
}

impl Descr for PyreJitCodeDescr {
    fn as_jitcode_descr(&self) -> Option<&dyn JitCodeDescr> {
        Some(self)
    }
}

impl JitCodeDescr for PyreJitCodeDescr {
    fn jitcode_index(&self) -> usize {
        self.jitcode_index
    }
}

/// Build a `DescrRef` carrying a `PyreJitCodeDescr`. Production callers
/// use this when materializing the descr pool from a codewriter
/// `&[BhDescr]` (`BhDescr::JitCode { jitcode_index, .. }` → this
/// adapter).
pub fn make_jitcode_descr(jitcode_index: usize) -> DescrRef {
    Arc::new(PyreJitCodeDescr::new(jitcode_index))
}

/// Trace-side `SwitchDictDescr` adapter. The bytecode blackhole keeps
/// `BhDescr::Switch` directly; the MIFrame walker needs an `Arc<dyn Descr>`
/// slot for the same `Assembler.descrs` index.
#[derive(Debug)]
pub struct PyreSwitchDescr {
    dict: std::collections::HashMap<i64, usize>,
    const_keys_in_order: Vec<i64>,
}

impl PyreSwitchDescr {
    pub fn new(dict: std::collections::HashMap<i64, usize>) -> Self {
        let mut const_keys_in_order: Vec<i64> = dict.keys().copied().collect();
        const_keys_in_order.sort_unstable();
        Self {
            dict,
            const_keys_in_order,
        }
    }
}

impl Descr for PyreSwitchDescr {
    fn repr(&self) -> String {
        let entries = self
            .const_keys_in_order
            .iter()
            .map(|key| {
                let target = self
                    .dict
                    .get(key)
                    .expect("const_keys_in_order must mirror SwitchDictDescr.dict");
                format!("{key}: {target}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("<SwitchDictDescr {{{entries}}}>")
    }

    fn as_switch_descr(&self) -> Option<&dyn SwitchDescr> {
        Some(self)
    }
}

impl SwitchDescr for PyreSwitchDescr {
    fn lookup(&self, value: i64) -> Option<usize> {
        self.dict.get(&value).copied()
    }

    fn const_keys_in_order(&self) -> &[i64] {
        &self.const_keys_in_order
    }
}

#[cfg(test)]
mod switch_descr_tests {
    use super::*;

    #[test]
    fn pyre_switch_descr_repr_matches_rpython_switchdictdescr() {
        let descr = PyreSwitchDescr::new(std::collections::HashMap::from([(9, 23), (5, 17)]));

        assert_eq!(
            <PyreSwitchDescr as Descr>::repr(&descr),
            "<SwitchDictDescr {5: 17, 9: 23}>"
        );
        assert_eq!(descr.const_keys_in_order(), &[5, 9]);
    }
}

/// Trace-side adapter for pyre's Rust-vtable method descriptor.
#[derive(Debug)]
pub struct PyreVtableMethodDescr {
    trait_root: String,
    method_name: String,
}

impl PyreVtableMethodDescr {
    pub fn new(trait_root: String, method_name: String) -> Self {
        Self {
            trait_root,
            method_name,
        }
    }
}

impl Descr for PyreVtableMethodDescr {
    fn repr(&self) -> String {
        format!(
            "VtableMethodDescr({}::{})",
            self.trait_root, self.method_name
        )
    }
}

/// `assembler.py:23 Assembler.descrs` parity adapter — translate one
/// codewriter-side `BhDescr` slot (`majit-translate/src/codewriter/jitcode.rs`)
/// into the matching trace-side `Arc<dyn Descr>` so trace ops emitted
/// by the walker (`crate::jitcode_dispatch::dispatch_via_miframe`) can carry
/// real-content descrs instead of `make_fail_descr` placeholders.
///
/// RPython parity: in upstream the metainterp + blackhole interpreter
/// share one `metainterp_sd.all_descrs` list — the same Python object
/// is the field/array/call descr regardless of which path is reading
/// it. pyre carries the codewriter-side typed list (`BhDescr`) and the
/// trait-side `Arc<dyn Descr>` view (`DescrRef`) as separate Rust
/// types because `Arc<dyn Descr>` cannot be downcast safely; this
/// adapter is the single point that bridges them.
///
/// Every branch builds the same descriptor kind carried by the
/// codewriter-side `BhDescr`:
/// * `Field` — `offset`, `field_size`, `field_type`, signedness, and
///   immutable/quasi-immutable flags are preserved.
/// * `Array` — `base_size`, `itemsize`, `type_id`, item type, signedness,
///   and array-of-structs classification are preserved.
/// * `Size` — `size`, `type_id`, and `vtable` are preserved.
/// * `Call` — `BhCallDescr.arg_classes` (e.g. `"iR"`) maps to
///   `Vec<Type>` per char (`i`->Int, `r`->Ref, `f`->Float; `R`/`I`/`F`
///   var-list markers split into the per-arg base type), and
///   `result_type` (one of `'i','r','f','v'`) maps to the `Type` of
///   the call result. `extra_info` is threaded into
///   `make_call_descr_with_effect`, preserving RPython `call.py:320`
///   effectinfo_from_writeanalyze parity for descr cache keys and
///   residual-call classification.
/// * `Switch` / `VableField` / `VableArray` / `VtableMethod` — trace-side
///   adapters preserve the descriptor slot instead of substituting a
///   fail-descr placeholder.
pub fn make_descr_from_bh(bh: &majit_translate::jitcode::BhDescr) -> DescrRef {
    use majit_translate::jitcode::BhDescr;
    match bh {
        BhDescr::Field {
            offset,
            field_size,
            field_type,
            field_flag,
            is_field_signed,
            is_immutable,
            is_quasi_immutable,
            index_in_parent,
            parent,
            name,
            owner,
            ..
        } => {
            let field_key = bh_field_cache_key(owner, name);
            // #171 codewriter descr-bridge: `_handle_list_call`
            // (codewriter/jtransform.rs) lowers Integer-strategy list
            // ops to fields on the dotted nested names
            // `int_items.{len,block}` (owner `W_ListObject`).
            // `bh_field_name` treats the dotted name as already-qualified,
            // and `W_ListObject` is a runtime Rust type absent from the
            // codewriter struct layouts, so the assembler's `fielddescrof`
            // leaves these at offset 0 (the list header). Map the leaves to
            // the canonical `W_LIST_DESCR_GROUP` entries the walker-native
            // list specializations already use so an assembled codewriter
            // list body addresses `IntArray.{len,block}` rather
            // than the header.
            //
            // This runs BEFORE the parent-group lookup below, not after: when
            // the codewriter DOES model the parent struct, that lookup answers
            // with the parent group's own entry for the same offset, and the
            // field ends up carrying two descrs — the parent group's for a
            // codewriter-lowered body, `W_LIST_DESCR_GROUP`'s for the
            // walker-native specializations.  The heapcache and the optimizer's
            // heap pass both key on descr identity, so the split silently
            // breaks aliasing: the `w_list_append` sub-walk's
            // `SetfieldGc(int_items.len)` does not invalidate the `len(xs)`
            // read that follows it, which then folds to the pre-append length
            // (one skipped `list.pop(0)` per compiled loop entry). One field is
            // one descr — `metainterp_sd.all_descrs` has no second entry for a
            // field just because a different interpreter reached it.
            // Same split, one struct up: the shared `PyObject` header's
            // `w_class`. The walker pins a value's Python-level class by
            // reading offset 8 through `w_class_descr()`; a codewriter-lowered
            // body testing the same header — `is_plain_int1` reading
            // `value.w_class` (listobject.rs) — reaches it through the modelled
            // `PyObject` parent, whose group entry is a different identity for
            // the same field. The pinned constant then never reached the
            // second read, so the strict subclass test stayed symbolic and
            // re-emitted the load plus its null and equality tests.
            //
            // Like the leaves below this runs BEFORE the parent-group lookup,
            // which would otherwise answer with the parent's own entry.
            //
            // Only when the two spellings describe the same memory access.
            // `new_w_class_field_descr` hardcodes `field_size: 8` while the
            // codewriter sizes a pointer field by `layout::target_word_size()`
            // (`call.rs get_type_flag`), so on wasm32 the incoming descr is a
            // 4-byte load and the canonical one an 8-byte load at the same
            // offset. Merging them there would widen the read over four bytes
            // of the adjacent payload. That size split is deliberate and
            // documented at `new_w_class_field_descr`; until it is resolved the
            // bridge declines rather than papering over it, leaving those
            // targets exactly as they were before the bridge existed.
            if name.as_str() == "w_class"
                && matches!(
                    owner.as_str(),
                    "PyObject" | "pyre_object::pyobject::PyObject"
                )
            {
                let canonical = &*W_CLASS_FIELD_DESCR;
                if canonical.offset() == *offset
                    && canonical.field_size() == *field_size
                    && canonical.field_type() == *field_type
                {
                    return w_class_descr();
                }
            }
            if owner.as_str() == "W_ListObject" {
                match name.as_str() {
                    "int_items.len" => return list_int_items_len_descr(),
                    "int_items.block" => return list_int_items_block_descr(),
                    "float_items.len" => return list_float_items_len_descr(),
                    "float_items.block" => return list_float_items_block_descr(),
                    // A bare `int_items` / `float_items` read addresses the
                    // typed-storage struct base, which is its first field
                    // (`block`, `INT_ARRAY_BLOCK_OFFSET == 0`) — the same
                    // offset + `Ref` type as the `.block` leaf above. The
                    // `w_list_append` body reads `int_items` directly before
                    // reaching `.len`; bridge it to the canonical
                    // `.block` group entry so the read resolves a parent_descr.
                    "int_items" => return list_int_items_block_descr(),
                    "float_items" => return list_float_items_block_descr(),
                    // The `w_list_append` body's `match list.strategy` reads the
                    // header `strategy` field directly.  The codewriter resolves
                    // its offset but produces a `SimpleFieldDescr` with no
                    // `parent_descr` backreference, which the optimizer's
                    // `ensure_ptr_info_arg0` rejects.  Bridge it to the canonical
                    // group entry (correct offset + parent_descr) like the
                    // `int_items.*` leaves above.
                    "strategy" => return list_strategy_descr(),
                    "length" => return list_length_descr(),
                    "items" => return list_items_descr(),
                    "w_slots" => return list_w_slots_descr(),
                    _ => {}
                }
            }
            // #171 object-strategy capacity read: `list.obj_capacity` lowers
            // to getfield_gc_r(items) + getfield_gc_i(block.capacity). The
            // block's offset-0 GcArray length header IS the allocated
            // capacity (immutable for the block's lifetime).
            //
            // This must precede the generic parent-group lookup. Charon gives
            // `ItemsBlock.capacity` a real parent, whose local field index is
            // zero; rebuilding that group would therefore return a distinct
            // index-0 descriptor. Walker-native list promotion seeds the
            // capacity under the canonical descriptor's stable index, and
            // RPython has only one FieldDescr identity for this field.
            if owner.as_str() == "ItemsBlock" && name.as_str() == "capacity" {
                return items_block_capacity_descr();
            }
            if owner.as_str() == "W_TupleObject" && name.as_str() == "w_dict" {
                return tuple_w_dict_descr();
            }
            if let Some(parent) = parent {
                if parent.type_id != 0 {
                    let key = majit_ir::descr::LLType::Struct(parent.type_id);
                    if let Some(fd) = majit_ir::descr::gc_cache()
                        .lock()
                        .unwrap()
                        ._cache_field
                        .get(&key)
                        .and_then(|inner| inner.get(&field_key))
                    {
                        return fd.clone() as DescrRef;
                    }
                    let group = simple_descr_group_from_bh_size(parent);
                    if let Some((pos, _)) =
                        parent.all_fielddescrs.iter().enumerate().find(|(_, spec)| {
                            spec.offset == *offset && spec.field_key() == field_key
                        })
                    {
                        // `pos` is the slot this reader will index; the descr's
                        // own `index_in_parent` (`descr.py:228`) is what every
                        // later consumer indexes by
                        // (`optimizeopt/info.rs force_box`).  They are two
                        // producers' answers to one question and nothing else
                        // compares them: the cache-hit return above and
                        // `get_field_descr`'s own `derive_index_in_parent` both
                        // resolve by NAME, so a stale rank is silently replaced
                        // rather than reported.  Count it where it is still
                        // visible.
                        majit_ir::descr::census_attached_index(pos, *index_in_parent);
                        if let Some(descr) = group.field_descrs.get(pos) {
                            return descr.clone() as DescrRef;
                        }
                    }
                }
            }
            // #171 codewriter descr-bridge: a codewriter-lowered body reads a
            // box payload (`W_IntObject.intval` / `W_BoolObject.intval` /
            // `W_FloatObject.floatval`) through the producer's struct-layout
            // `SimpleFieldDescr` (header modeled, `index_in_parent` = 2).  The
            // walker materializes those same boxes through the single-field
            // `W_*_DESCR_GROUP` (`index_in_parent` = 0).  Both address the same
            // offset, but the optimizer's virtual-field cache matches on descr
            // identity, so a sub-walk read with the producer descr misses the
            // virtual built with the group descr — the box is forced and the
            // field read folds to uninitialized storage.  Return the canonical
            // group entry so the read matches the box-creation descr and folds
            // to the carried unboxed value (the payload is genuinely immutable
            // post-construction, matching the group entry's flag).
            match (owner.as_str(), name.as_str()) {
                ("W_IntObject" | "pyre_object::intobject::W_IntObject", "intval") => {
                    return int_intval_descr();
                }
                ("W_BoolObject" | "pyre_object::boolobject::W_BoolObject", "intval") => {
                    return bool_intval_descr();
                }
                ("W_FloatObject" | "pyre_object::floatobject::W_FloatObject", "floatval") => {
                    return float_floatval_descr();
                }
                _ => {}
            }
            let full_name = if owner.is_empty() || name.contains('.') {
                name.clone()
            } else {
                format!("{owner}.{name}")
            };
            // RPython `descr.py:214 FieldDescr.get_index()` returns
            // the value `heaptracker.get_fielddescr_index_in(STRUCT,
            // name)` recorded into `FieldDescr.index` at construction
            // time (`descr.py:200`).  Pyre's `BhDescr::Field` carries
            // that as `index_in_parent`; thread it through as
            // `BhFieldSpec.index` so the `parent` matching fallback
            // produces a `SimpleFieldDescr` whose `index()` matches the
            // upstream value rather than a `u32::MAX` sentinel.
            let field = majit_translate::jitcode::BhFieldSpec {
                index: *index_in_parent as u32,
                field_key,
                name: full_name,
                offset: *offset,
                field_size: *field_size,
                field_type: *field_type,
                field_flag: *field_flag,
                is_field_signed: *is_field_signed,
                is_immutable: *is_immutable,
                is_quasi_immutable: *is_quasi_immutable,
                index_in_parent: *index_in_parent,
            };
            field_descr_from_bh_field(&field, parent.as_ref())
        }
        BhDescr::Array {
            base_size,
            itemsize,
            len_offset,
            type_id,
            item_type,
            is_array_of_structs,
            is_item_signed,
            ei_index,
            array_type_id,
            interior_fields,
            ..
        } => {
            // Old serialized BhDescr values default `gc_type_id` to zero.
            // Resolve that legacy structural identity through the GC cache
            // before publishing a runtime descriptor, exactly as the
            // blackhole/materialization paths do.
            let resolved_gc_type_id = bh.resolve_gc_tid();
            let descr = if *is_array_of_structs {
                // `descr.py:348-378 get_array_descr(gccache, ARRAY)`:
                // the u64 `*type_id` from `BhDescr::Array` is the cache
                // key (`path_hash` of the producer-side `array_type_id`,
                // see `BhSizeSpec.type_id` doc); thread it into the
                // keyed factory so `gc_cache._cache_array[LLType::Array(
                // cache_key)]` is populated and subsequent lookups
                // resolve to the same Arc.  The u32 truncation for the
                // SimpleArrayDescr gc tid is a TODO
                // (gc tid should come from `init_array_descr`
                // sequential allocation).
                make_struct_array_descr_full_keyed(
                    u32::MAX,
                    *base_size,
                    *itemsize,
                    *len_offset,
                    resolved_gc_type_id,
                    *type_id,
                    *item_type,
                    interior_fields,
                )
            } else {
                // `descr.py:348-360 gccache._cache_array[ARRAY_OR_STRUCT]`
                // is keyed on lltype object identity; thread the
                // codewriter `array_type_id` across the BhDescr
                // boundary into the runtime `ArrayDescrKey` so two
                // BhDescr::Array entries that disagree only on the
                // Rust type spelling don't collapse to the same
                // registry slot (`set_ei_index` clobber).
                make_array_descr_with_full_id(
                    *base_size,
                    *itemsize,
                    resolved_gc_type_id,
                    *len_offset,
                    *item_type,
                    *is_item_signed,
                    array_type_id.clone(),
                )
            };
            // `effectinfo.py:465 compute_bitstrings` republish: the
            // codewriter-side `array_index` carried across the BhDescr
            // boundary lands on the runtime descr so heap.rs
            // `force_from_effectinfo` (`heap.py:537-571`) reads the
            // same bitstring slot the producer wrote.
            if *ei_index != u32::MAX {
                descr.set_ei_index(*ei_index);
            }
            descr
        }
        BhDescr::Size {
            size,
            type_id,
            vtable,
            owner,
            all_fielddescrs,
            is_gc_managed,
            ..
        } => {
            // `descr.py:108-118 get_size_descr` cache-hit semantics:
            // when the producer's `type_id` matches a runtime publish
            // (`build_object_descr_group_with_def_path` →
            // `register_keyed_size`), the published `Arc<PyreSizeDescr>`
            // is the canonical object identity for that STRUCT.  Return
            // it directly so analyzer side-tables, runtime allocations,
            // and BhDescr round-trip all share one `Arc<dyn Descr>` —
            // matching PyPy `cache[STRUCT]` per-tuple identity.  Falls
            // through to the legacy mint path on cache miss (transient
            // empty-field `bh_new` descrs from `pyre-jit/src/eval.rs`,
            // test fixtures, etc.).
            if *type_id != 0 {
                let key = majit_ir::descr::LLType::Struct(*type_id);
                let hit = majit_ir::descr::gc_cache()
                    .lock()
                    .unwrap()
                    ._cache_size
                    .get(&key)
                    .cloned();
                if let Some(descr) = hit {
                    return descr;
                }
            }
            // RPython `descr.py:120 get_size_descr` → `:188 init_size_descr`
            // populates `SizeDescr.all_fielddescrs` (and the
            // `gc_fielddescrs` subset) from
            // `heaptracker.all_fielddescrs(STRUCT)` so consumers like
            // `info.py:180 init_fields` (`optimizeopt/info.rs:1989`)
            // see the full struct field list off the descr without a
            // round-trip through the codewriter.  When the producer
            // shipped a non-empty `all_fielddescrs`, build the parent
            // `SimpleSizeDescr` via the cyclic `make_simple_descr_group`
            // path so `Arc<SimpleFieldDescr>` parents back-reference
            // the same `SimpleSizeDescr` (`descr.py:200` parent slot).
            // The transient short-lived `BhDescr::Size` constructed in
            // `pyre-jit/src/eval.rs` (`bh_new` / `bh_new_with_vtable`
            // dispatch) carries an empty list and falls through to the
            // bare ctor, which is the existing test-helper shape.
            let headerless = owner == HEADERLESS_SIZE_OWNER_MARKER;
            if all_fielddescrs.is_empty() && !headerless {
                // TODO: `make_size_descr_with_type_and_vtable`
                // takes the u32 gc tid; `*type_id` is the u64 cache key.
                // Truncate `as u32` until gc_cache routing.
                make_size_descr_with_type_and_vtable(*size, *type_id as u32, *vtable as usize)
            } else {
                let spec = majit_translate::jitcode::BhSizeSpec {
                    size: *size,
                    type_id: *type_id,
                    vtable: *vtable,
                    is_gc_managed: *is_gc_managed,
                    headerless,
                    all_fielddescrs: all_fielddescrs.clone(),
                };
                simple_descr_group_from_bh_size(&spec).size_descr.clone()
            }
        }
        BhDescr::InteriorField { array, field } => {
            // `descr.py:388 InteriorFieldDescr(arraydescr, fielddescr)`:
            // recompose the array + field sub-descrs.  The nested
            // BhDescrs are always `BhDescr::Array` / `BhDescr::Field`
            // (`BhDescr::from_interior_field_descr`); a short stream is an
            // encoder bug surfaced here rather than silently mis-typed.
            let (base_size, itemsize, len_offset, type_id, item_type, interior_fields) =
                match array.as_ref() {
                    BhDescr::Array {
                        base_size,
                        itemsize,
                        len_offset,
                        type_id,
                        item_type,
                        interior_fields,
                        ..
                    } => (
                        *base_size,
                        *itemsize,
                        *len_offset,
                        *type_id,
                        *item_type,
                        interior_fields,
                    ),
                    other => panic!(
                        "BhDescr::InteriorField array slot must be BhDescr::Array, got {other:?}"
                    ),
                };
            // `descr.py:430` `get_interiorfield_descr` builds the interior
            // descr from `get_array_descr(gc_ll_descr, ARRAY)` — the SAME
            // cached arraydescr the `BhDescr::Array` arm restores.  Route the
            // outer struct array through the identical keyed factory so the
            // rebuilt `InteriorFieldDescr.arraydescr` shares the
            // `gc_cache._cache_array[LLType::Array(cache_key)]` identity rather
            // than a fresh local mint, and re-attaches `all_interiorfielddescrs`
            // (`descr.py:372-375`).  `type_id` is the producer-side cache key
            // (`ArrayDescr.cache_key` = `path_hash(array_type_id)`); the u32
            // gc-tid slot is ignored on the cache path (`init_array_descr`
            // stamps the real tid), keeping ARRAY identity cache key separate
            // from GC layout tid.  On a cache MISS the factory requests
            // `ArrayFlag::Struct`, so the minted arraydescr is FLAG_STRUCT; on a
            // cache HIT the existing slot is reused verbatim, so the resolved
            // arraydescr is re-checked against `descr.py:389`'s
            // `assert arraydescr.flag == FLAG_STRUCT` below.
            let array_dyn = make_struct_array_descr_full_keyed(
                u32::MAX,
                base_size,
                itemsize,
                len_offset,
                type_id as u32,
                type_id,
                item_type,
                interior_fields,
            );
            // The cache slot always holds a concrete `SimpleArrayDescr`
            // (analyzer / gc_cache / runtime mint).
            let array_descr: Arc<dyn majit_ir::descr::ArrayDescr> =
                match majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleArrayDescr>(
                    array_dyn,
                ) {
                    Ok(simple) => simple,
                    Err(other) => panic!(
                        "BhDescr::InteriorField array resolved to an unknown ArrayDescr type: {other:?}"
                    ),
                };
            // `descr.py:389 InteriorFieldDescr.__init__`:
            //   assert arraydescr.flag == FLAG_STRUCT
            // The factory requests `ArrayFlag::Struct` on a cache MISS, but a
            // cache HIT returns the existing `LLType::Array(cache_key)` slot
            // verbatim — which may be a non-struct array (a contaminated cache
            // key).  Reject those here rather than silently rebuilding the
            // InteriorFieldDescr around a non-struct array.
            assert!(
                array_descr.is_array_of_structs(),
                "BhDescr::InteriorField arraydescr must be FLAG_STRUCT (descr.py:389)"
            );
            let (offset, field_size, field_type, field_flag, index_in_parent, name) =
                match field.as_ref() {
                    BhDescr::Field {
                        offset,
                        field_size,
                        field_type,
                        field_flag,
                        index_in_parent,
                        name,
                        ..
                    } => (
                        *offset,
                        *field_size,
                        *field_type,
                        *field_flag,
                        *index_in_parent,
                        name.clone(),
                    ),
                    other => panic!(
                        "BhDescr::InteriorField field slot must be BhDescr::Field, got {other:?}"
                    ),
                };
            // Bare interior field name (`descr.py:221 cache[STRUCT][fieldname]`
            // shape) — the `get_interiorfield_descr` cache key the
            // `make_struct_array_descr_full_keyed` interior loop used.
            let bare_name = name
                .rsplit_once('.')
                .map(|(_, n)| n.to_string())
                .unwrap_or_else(|| name.clone());
            // `descr.py:393` `InteriorFieldDescr.get_index()` delegates to
            // `fielddescr.get_index()` (= `index_in_parent`), read back by
            // `info.py:573-594` getinteriorfield/setinteriorfield_virtual.
            let field_descr: Arc<dyn majit_ir::descr::FieldDescr> = Arc::new(
                majit_ir::descr::SimpleFieldDescr::new_with_name(
                    u32::MAX,
                    offset,
                    field_size,
                    field_type,
                    false,
                    field_flag,
                    name.clone(),
                    name,
                )
                .with_index_in_parent(index_in_parent),
            );
            // `descr.py:423-438 get_interiorfield_descr` cache-or-mint keyed on
            // the outer ARRAY identity, so the analyzer's
            // `cc.interiorfielddescrof`, the array's `all_interiorfielddescrs`,
            // and this restore share one `Arc` per `(ARRAY, name)` tuple — the
            // `make_struct_array_descr_full_keyed` interior loop above already
            // populated `_cache_interiorfield` (with the per-trace
            // `Descr::index()`) for `cache_key != 0`, so this hits and returns
            // that same descr.  `cache_key == 0` is the legacy no-identity
            // sentinel — local mint a fresh `SimpleInteriorFieldDescr` (the
            // `u32::MAX` slot is the "no setup_descrs index assigned" default;
            // `get_index()` comes from `field_descr.index_in_parent` above).
            if type_id != 0 {
                majit_ir::descr::gc_cache()
                    .lock()
                    .unwrap()
                    .get_interiorfield_descr(
                        majit_ir::descr::LLType::Array(type_id),
                        bare_name,
                        String::new(),
                        array_descr,
                        field_descr,
                    )
            } else {
                Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
                    u32::MAX,
                    array_descr,
                    field_descr,
                ))
            }
        }
        BhDescr::Call { calldescr } => make_call_descr_from_bh(calldescr),
        BhDescr::JitCode { jitcode_index, .. } => make_jitcode_descr(*jitcode_index),
        BhDescr::Switch { dict, .. } => Arc::new(PyreSwitchDescr::new(dict.clone())),
        BhDescr::VableField { index } => majit_ir::descr::vable_static_field_descr(*index as u16),
        BhDescr::VableArray { index } => majit_ir::descr::vable_array_field_descr(*index as u16),
        BhDescr::VtableMethod {
            trait_root,
            method_name,
        } => Arc::new(PyreVtableMethodDescr::new(
            trait_root.clone(),
            method_name.clone(),
        )),
    }
}

/// Look a serialized raw-set member back up in the process-global gccache.
///
/// **Lookup only — never mint.** `descr.py:218-239 get_field_descr` is
/// cache-or-mint because upstream calls it from `heaptracker.all_fielddescrs
/// (STRUCT)`, which always has the whole `STRUCT` in hand and therefore mints
/// the parent `SizeDescr` with its complete `all_fielddescrs` list
/// (`descr.py:188 init_size_descr`). A serialized member carries only its own
/// slot, so minting through it would publish a parent with an empty field
/// list, win `_cache_field` by first-write, and leave
/// `SizeDescr.all_fielddescrs` and `_cache_field` describing different
/// objects — breaking the positional invariant `heaptracker.py:76-101
/// get_fielddescr_index_in` establishes and `optimizeopt/info.rs force_box`
/// asserts.
///
/// Outcome of looking one serialized raw-set member back up.
enum SetMemberLookup {
    /// The gccache holds the slot the analyzer minted through.
    Resolved(majit_ir::DescrRef),
    /// The *container* (`STRUCT` / `ARRAY`) is absent from this process's
    /// descr universe entirely, so the member is dropped from the raw set.
    ///
    /// **Upstream cannot reach this.** `cpu.*descrof` and `compute_bitstrings`
    /// share one gccache in one process, so `setup_descrs` (`descr.py:25-47`)
    /// snapshots the very cache the raw-set descrs were minted into and
    /// `effectinfo.py:465-499` only ever unions descrs that are already there.
    /// A member that fails to resolve is therefore not a legitimate projection
    /// but a divergence, and its count is a badness counter held at zero
    /// (`descr_set_absent` on the `[jit-stats]` line) — pyre's form of the bare
    /// `assert`s upstream states this class of condition with.
    ///
    /// Two publishes are what hold it at zero, both inside the `Once` in
    /// `jitcode_runtime::rehydrate_build_descr_raw_sets` and both ahead of the
    /// first lookup: `publish_runtime_descr_groups` for the module-scope
    /// groups (`W_IntObject.intval` was landing here until it did), and
    /// `publish_effect_info_descr_mints` for the slots no opcode names, which
    /// `descrs.bin` — upstream's `opcode_descrs`, not its `all_descrs` — does
    /// not carry.
    ///
    /// The arm stays because the answer is still reachable in principle, and a
    /// dropped member is a real hazard when it is: "absent" is evaluated once,
    /// when the `BhCallDescr` is materialised
    /// (`jitcode_runtime.rs rehydrated_call_descr_ref`), so a container
    /// registered afterwards would leave that EI permanently claiming "not
    /// written" for a field the callee does write, and the heapcache would not
    /// invalidate a read across the call. [`stale_absent_containers`] re-asks
    /// the question at exit and gates that residue at zero too.
    AbsentContainer,
    /// The container IS published but not under this member's key.  A real
    /// runtime descr for the field may exist under a different spelling, so
    /// dropping the member would answer "not written" for a field that is;
    /// the caller must fall back to the wildcard instead.
    Ambiguous,
}

/// Look a serialized raw-set member back up in the process-global gccache.
///
/// **Lookup only — never mint.** `descr.py:218-239 get_field_descr` is
/// cache-or-mint because upstream calls it from `heaptracker.all_fielddescrs
/// (STRUCT)`, which always has the whole `STRUCT` in hand and therefore mints
/// the parent `SizeDescr` with its complete `all_fielddescrs` list
/// (`descr.py:188 init_size_descr`). A serialized member carries only its own
/// slot, so minting through it would publish a parent with an empty field
/// list, win `_cache_field` by first-write, and leave
/// `SizeDescr.all_fielddescrs` and `_cache_field` describing different
/// objects — breaking the positional invariant `heaptracker.py:76-101
/// get_fielddescr_index_in` establishes and `optimizeopt/info.rs force_box`
/// asserts.
/// Ledger of every serialized raw-set member that came back
/// [`SetMemberLookup::Ambiguous`] — its container *is* published in this
/// process's descr universe, just not under the key the analyzer minted
/// through.
///
/// `Ambiguous` is the two-spellings defect by construction: both sides named
/// the same container and disagreed on how to spell it.  Its consumer degrades
/// the whole `EffectInfo` to `EF_RANDOM_EFFECTS`, so every entry below is a
/// residual call that stopped invalidating precisely and became a whole-heap
/// barrier.  `AbsentContainer` is the other non-answer and is equally a
/// divergence rather than a projection — see [`SetMemberLookup`] — but it is
/// the *missing table* defect, not the *two spellings* one, so the two are
/// counted apart.
///
/// This is the gate for struct-identity work; a converged-field-descr count is
/// not.  A field-descr census only counts the descrs a particular run happens
/// to mint, so deleting a call site greens it without joining anything,
/// whereas the members here are fixed by `descrs.bin` and stay `Ambiguous`
/// until the two spellings actually collapse onto one key.
/// Force every module-scope runtime descr group into the gccache.
///
/// `descr.py:25-47 setup_descrs` publishes the whole descr universe before
/// anything consumes it, and upstream gets that for free: `cpu.*descrof` runs
/// during codewriting, so by the time `compute_bitstrings`
/// (`effectinfo.py:484-489`) walks the raw sets, every `STRUCT` the program
/// mentions already has its slot.  Pyre splits those two phases across a
/// build-time analyzer and a runtime process, and the runtime groups are
/// `LazyLock`s that publish only when a trace first touches the type — so the
/// raw-set members were being resolved against a universe that was still
/// mostly empty, and answered [`SetMemberLookup::AbsentContainer`] for
/// containers this process does register, just later.
///
/// Publishing here restores the upstream order.  It must run *before* the
/// build-time `make_descr_from_bh` loop: those specs carry `vtable: 0` and the
/// inherited header fields, so on a shared key they would otherwise displace
/// the runtime publish that owns the only copy of the vtable (see the
/// authority rule in `majit-ir` `register_keyed_size`).
pub(crate) fn publish_runtime_descr_groups() {
    // `PyreObjectDescrGroup`'s constructor is what registers into the
    // gccache, so dereferencing the `LazyLock` is the publish.
    let _published = [
        &*W_INT_DESCR_GROUP,
        &*W_FLOAT_DESCR_GROUP,
        &*W_LONG_DESCR_GROUP,
        &*W_BOOL_DESCR_GROUP,
        &*W_UNICODE_DESCR_GROUP,
        &*RANGE_ITER_DESCR_GROUP,
        &*SEQ_ITER_DESCR_GROUP,
        &*RANGE_DESCR_GROUP,
        &*W_METHOD_DESCR_GROUP,
        &*W_OBJECT_MUTABLE_CELL_DESCR_GROUP,
        &*W_LIST_DESCR_GROUP,
        &*W_TUPLE_DESCR_GROUP,
        &*SPECIALISED_TUPLE_II_DESCR_GROUP,
        &*SPECIALISED_TUPLE_FF_DESCR_GROUP,
        &*SPECIALISED_TUPLE_OO_DESCR_GROUP,
        &*ITEMS_BLOCK_DESCR_GROUP,
        &*W_SLICE_DESCR_GROUP,
        &*PYFRAME_DESCR_GROUP,
        &*W_OBJECT_OBJECT_DESCR_GROUP,
        &*PYTRACEBACK_DESCR_GROUP,
    ];
}

/// Fill the gccache slots that only an `EffectInfo` raw set names, from the
/// mint arguments the analyzer recorded at its own `get_*_descr` call.
///
/// This is the far half of `descr.py`'s mint-or-hit: `get_field_descr`
/// (`:218-239`), `get_array_descr` (`:348-378`) and `get_interiorfield_descr`
/// (`:404-437`) all *construct* on a cache miss, and upstream reaches every one
/// of these slots because `compute_bitstrings` (`effectinfo.py:465-499`) unions
/// descr objects out of the same gccache `setup_descrs` (`descr.py:25-47`) then
/// snapshots.  Pyre carries only the assembler's opcode table across the
/// build/runtime split (`pyjitpl.py:2261 setup_descrs(asm.descrs)`), so a descr
/// minted purely to fill a raw set — named by no opcode — arrives with nothing
/// to resolve against.
///
/// Each entry is published under **the member's own key**, never a key
/// re-derived from a struct name: the analyzer prefers a source-attached
/// identity token over `path_hash(canonical_struct_name(..))` when it has one,
/// so re-deriving would land on a different slot than the one the member
/// names.
///
/// Runs **last**, after both the runtime groups and the `make_descr_from_bh`
/// loop (`jitcode_runtime.rs rehydrate_build_descr_raw_sets`), and writes only
/// into slots those left empty.  Going first would pre-fill a slot the loop
/// then asks for, with the analyzer's `index_in_parent` numbering rather than
/// the established producer's — the two disagree on any `#[pyre_class]`
/// struct, by the two injected header words.  Publishing the leftovers keeps
/// the established answers untouched.
///
/// The slots left empty are the ones no opcode names, which is why
/// `descrs.bin` — RPython's `opcode_descrs`, not its `all_descrs` — does not
/// carry them.  Upstream has that population too: `effectinfo.py:300-322`
/// builds every raw set through `cpu.fielddescrof` / `arraydescrof` /
/// `interiorfielddescrof`, a cache-or-mint into the gccache that runs whether
/// or not any operation names the field, and `descr.py:25-47 setup_descrs`
/// then snapshots that gccache into `all_descrs`.  So a raw-set-only descr
/// referenced by no trace op is normal, not a sign of a mis-keyed publish.
pub(crate) fn publish_effect_info_descr_mints(entries: &[majit_ir::effectinfo::DescrMintEntry]) {
    use majit_ir::descr::LLType;
    use majit_ir::effectinfo::{DescrMintSpec, DescrSetMember};

    for entry in entries {
        match (&entry.member, &entry.spec) {
            (
                DescrSetMember::Field {
                    struct_id,
                    field_name,
                },
                spec,
            ) => {
                mint_field(LLType::Struct(*struct_id), field_name, spec);
            }
            (DescrSetMember::Array { array_id }, spec) => {
                mint_array(LLType::Array(*array_id), spec);
            }
            (
                DescrSetMember::InteriorField { array_id, name },
                DescrMintSpec::InteriorField {
                    array,
                    field_struct_id,
                    field_name,
                    field,
                },
            ) => {
                let array_key = LLType::Array(*array_id);
                let Some(array_descr) = mint_array(array_key.clone(), array)
                    .and_then(majit_ir::descr::descr_arc_as_array_descr)
                else {
                    continue;
                };
                // `descr.py:435 fielddescr = get_field_descr(gc_ll_descr,
                // REALARRAY.OF, name)` — the element struct's own slot, then
                // `:436 InteriorFieldDescr(arraydescr, fielddescr)`.
                let Some(field_descr) =
                    mint_field(LLType::Struct(*field_struct_id), field_name, field)
                else {
                    continue;
                };
                let _ = majit_ir::descr::gc_cache()
                    .lock()
                    .unwrap()
                    .get_interiorfield_descr(
                        array_key,
                        name.clone(),
                        String::new(),
                        array_descr,
                        field_descr,
                    );
            }
            // A member paired with a spec of another shape cannot describe its
            // own slot; leaving it unpublished keeps it counted as absent
            // rather than filling the slot with the wrong layout.
            _ => {}
        }
    }
}

/// `descr.py:234-238` — bind the parent size descr, then mint the field.
fn mint_field(
    struct_key: majit_ir::descr::LLType,
    field_name: &str,
    spec: &majit_ir::effectinfo::DescrMintSpec,
) -> Option<std::sync::Arc<dyn majit_ir::descr::FieldDescr>> {
    let majit_ir::effectinfo::DescrMintSpec::Field {
        struct_size,
        offset,
        field_size,
        field_type,
        flag,
        is_immutable,
        is_quasi_immutable,
        index_in_parent,
    } = spec
    else {
        return None;
    };
    let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
    // Only an empty slot is this function's business: an unresolvable member is
    // by definition one nothing has filled, and a filled slot already belongs
    // to whoever published it. `get_field_descr` would return that owner's
    // descr anyway (`descr.py:220-221`), so taking the hit through it changes
    // nothing except to offer a second opinion on the layout — and the two
    // producers do not agree on one field of it. `index_in_parent` here is the
    // analyzer's `field_pos`, which counts the header words at offsets 0 and 8
    // that `heaptracker.py:60-71 all_fielddescrs` skips, so it reads two higher
    // than the runtime publish's for the same field.
    if let Some(existing) = gc
        ._cache_field
        .get(&struct_key)
        .and_then(|inner| inner.get(field_name))
    {
        return Some(existing.clone() as std::sync::Arc<dyn majit_ir::descr::FieldDescr>);
    }
    // `descr.py:238 fielddescr.parent_descr = get_size_descr(gccache, STRUCT,
    // vtable)`; the analyzer has no vtable surface, so 0 — a runtime publish
    // that carries one wins the slot either way.
    let _parent = gc.get_size_descr(struct_key.clone(), *struct_size, 0, false);
    Some(gc.get_field_descr(
        struct_key,
        field_name,
        None,
        *offset,
        *field_size,
        *field_type,
        *is_immutable,
        *is_quasi_immutable,
        *flag,
        u32::MAX,
        false,
        *index_in_parent,
    ))
}

/// `descr.py:353-370` — mint the array descr, including its lendescr.
fn mint_array(
    array_key: majit_ir::descr::LLType,
    spec: &majit_ir::effectinfo::DescrMintSpec,
) -> Option<majit_ir::DescrRef> {
    let majit_ir::effectinfo::DescrMintSpec::Array {
        base_size,
        item_size,
        flag,
        item_type,
        nolength,
        length_offset,
        is_pure,
        concrete_type,
    } = spec
    else {
        return None;
    };
    Some(majit_ir::descr::gc_cache().lock().unwrap().get_array_descr(
        array_key,
        *base_size,
        *item_size,
        *flag,
        *item_type,
        *nolength,
        *length_offset,
        *is_pure,
        *concrete_type,
    ))
}

/// `ambiguous` empty alone is not enough to call the gate green: a member
/// whose container is not registered answers `AbsentContainer` instead, so a
/// run where `absent` dwarfs `resolved` has an empty `ambiguous` set
/// vacuously, because nothing was ever joined.  The ledger carries all three
/// outcomes so the denominator is visible next to the two failures.
#[derive(Default)]
pub struct SetMemberLedger {
    pub resolved: usize,
    /// Container keys no lookup could reach, by name.  Kept as a set rather
    /// than a count because a spelling miss hides here too: a container the
    /// runtime *did* publish, under a different key, is absent from both
    /// `_cache_field` and `_cache_size` at the analyzer's key and so reads
    /// exactly like a container this process never traced.
    pub absent: std::collections::BTreeSet<String>,
    pub ambiguous: std::collections::BTreeSet<String>,
    /// The dropped members themselves, so
    /// [`stale_absent_containers`] can ask the question again later.
    absent_members: Vec<majit_ir::effectinfo::DescrSetMember>,
}

static SET_MEMBER_LEDGER: LazyLock<Mutex<SetMemberLedger>> =
    LazyLock::new(|| Mutex::new(SetMemberLedger::default()));

fn set_member_label(m: &majit_ir::effectinfo::DescrSetMember) -> String {
    use majit_ir::effectinfo::DescrSetMember;
    match m {
        DescrSetMember::Field {
            struct_id,
            field_name,
        } => format!("Struct({struct_id:#018x}).{field_name}"),
        DescrSetMember::Array { array_id } => format!("Array({array_id:#018x})"),
        DescrSetMember::InteriorField { array_id, name } => {
            format!("Array({array_id:#018x}).{name}")
        }
    }
}

fn record_set_member_lookup(m: &majit_ir::effectinfo::DescrSetMember, out: &SetMemberLookup) {
    let mut ledger = SET_MEMBER_LEDGER.lock().unwrap();
    match out {
        SetMemberLookup::Resolved(_) => ledger.resolved += 1,
        SetMemberLookup::AbsentContainer => {
            let label = set_member_label(m);
            if ledger.absent.insert(label) {
                ledger.absent_members.push(m.clone());
            }
        }
        SetMemberLookup::Ambiguous => {
            let label = set_member_label(m);
            ledger.ambiguous.insert(label);
        }
    }
}

/// Snapshot of the [`SET_MEMBER_LEDGER`].
pub fn set_member_ledger() -> SetMemberLedger {
    let ledger = SET_MEMBER_LEDGER.lock().unwrap();
    SetMemberLedger {
        resolved: ledger.resolved,
        absent: ledger.absent.clone(),
        ambiguous: ledger.ambiguous.clone(),
        absent_members: Vec::new(),
    }
}

/// Re-ask the [`SetMemberLookup::AbsentContainer`] question against the
/// universe as it stands *now*, and return the members that have since become
/// reachable.
///
/// Every entry is a live instance of the gap documented on that variant: the
/// member was dropped from its raw set when the `BhCallDescr` was
/// materialised, the container was registered afterwards, and the frozen
/// `EffectInfo` still claims the callee does not touch it. Upstream cannot
/// reach this state — `effectinfo.py` builds its sets through
/// `cpu.fielddescrof` / `cpu.arraydescrof` in the same process that owns the
/// descr cache, so the container is always registered by construction.
///
/// An empty result is the evidence that the gap is not being hit; a non-empty
/// one names exactly which containers to publish eagerly next.
pub fn stale_absent_containers() -> Vec<String> {
    let members = SET_MEMBER_LEDGER.lock().unwrap().absent_members.clone();
    members
        .iter()
        .filter(|m| !matches!(descr_from_set_member(m), SetMemberLookup::AbsentContainer))
        .map(set_member_label)
        .collect()
}

fn descr_from_set_member(m: &majit_ir::effectinfo::DescrSetMember) -> SetMemberLookup {
    use majit_ir::descr::{LLType, gc_cache};

    match m {
        majit_ir::effectinfo::DescrSetMember::Field {
            struct_id,
            field_name,
            ..
        } => {
            let struct_key = LLType::Struct(*struct_id);
            let gc = gc_cache().lock().unwrap();
            match gc._cache_field.get(&struct_key) {
                Some(inner) => match inner.get(field_name.as_str()) {
                    Some(fd) => SetMemberLookup::Resolved(fd.clone() as majit_ir::DescrRef),
                    None => SetMemberLookup::Ambiguous,
                },
                // No field map and no size slot: nothing in this process
                // ever named the struct.
                None if !gc._cache_size.contains_key(&struct_key) => {
                    SetMemberLookup::AbsentContainer
                }
                None => SetMemberLookup::Ambiguous,
            }
        }
        majit_ir::effectinfo::DescrSetMember::Array { array_id, .. } => {
            match gc_cache()
                .lock()
                .unwrap()
                ._cache_array
                .get(&LLType::Array(*array_id))
            {
                Some(ad) => SetMemberLookup::Resolved(ad.clone()),
                None => SetMemberLookup::AbsentContainer,
            }
        }
        majit_ir::effectinfo::DescrSetMember::InteriorField { array_id, name, .. } => {
            let gc = gc_cache().lock().unwrap();
            let array_key = LLType::Array(*array_id);
            match gc
                ._cache_interiorfield
                .get(&(array_key.clone(), name.clone(), String::new()))
            {
                Some(d) => SetMemberLookup::Resolved(d.clone()),
                None if !gc._cache_array.contains_key(&array_key) => {
                    SetMemberLookup::AbsentContainer
                }
                None => SetMemberLookup::Ambiguous,
            }
        }
    }
}

/// Fill the six `_*_descrs_*` raw sets from `descr_set_keys`, canonicalising
/// exactly as `effectinfo.py:128-145 frozenset_or_none` /
/// `canonicalize_descr_set`.
///
/// All six sets or none of them: `effectinfo.py:149-162` makes them `None`
/// **iff** the EI is `EF_RANDOM_EFFECTS`, and `compute_bitstrings`
/// (`effectinfo.py:484-489`) asserts that biconditional before deciding
/// whether to clear the bitstrings.  A half-populated EI would clear them
/// while `extraeffect` still claims concrete effects, and the next
/// `check_readonly_descr_field` would then read a `None` bitstring.
pub fn rehydrate_effect_info(ei: &mut majit_ir::EffectInfo) {
    // `effectinfo.py:285-292` wildcard: the shape to fall back to whenever
    // the concrete sets cannot be rebuilt faithfully.  Conservative in the
    // sound direction — `has_random_effects()` makes every heap consumer
    // assume the call touched everything.
    fn degrade(ei: &mut majit_ir::EffectInfo) {
        ei.extraeffect = majit_ir::ExtraEffect::RandomEffects;
        // effectinfo.py:364-365 — the wildcard forces can_collect.
        ei.can_collect = true;
        // `call.py:284-286` states it outright: "random_effects implies
        // can_invalidate".  `effectinfo.py:271-273 MOST_GENERAL` is built with
        // `can_invalidate=True`, and so is pyre's own
        // `EffectInfo::MOST_GENERAL`.  Without this a degraded EI is a shape
        // upstream cannot construct — random effects with
        // `check_can_invalidate()` still false — and `check_can_invalidate`
        // is read on a path `has_random_effects()` does not cover
        // (`heap.py:457-459`; `_seen_guard_not_invalidated` is otherwise only
        // reset in `__init__`, `heap.py:341`), so a quasi-immutable guard
        // would survive a call the wildcard says invalidates everything.
        ei.can_invalidate = true;
        ei._readonly_descrs_fields = None;
        ei._write_descrs_fields = None;
        ei._readonly_descrs_arrays = None;
        ei._write_descrs_arrays = None;
        ei._readonly_descrs_interiorfields = None;
        ei._write_descrs_interiorfields = None;
        ei.readonly_descrs_fields = None;
        ei.write_descrs_fields = None;
        ei.readonly_descrs_arrays = None;
        ei.write_descrs_arrays = None;
        ei.readonly_descrs_interiorfields = None;
        ei.write_descrs_interiorfields = None;
        ei.single_write_descr_array = None;
    }

    let Some(keys) = ei.descr_set_keys.as_ref() else {
        // No serialized key channel.  For a build-time EI that means the
        // codewriter already emitted the `EF_RANDOM_EFFECTS` wildcard, so
        // the six raw sets are `None` and there is nothing to rebuild.  Any
        // other shape lost its sets somewhere the invariant above does not
        // cover; restore the wildcard rather than leave a concrete
        // `extraeffect` pointing at absent sets.
        if ei.extraeffect != majit_ir::ExtraEffect::RandomEffects {
            degrade(ei);
        }
        return;
    };
    let resolve = |members: &[majit_ir::effectinfo::DescrSetMember]| {
        let mut out = Vec::with_capacity(members.len());
        // Walk the WHOLE set even once it is known to be incomplete: the
        // ledger behind `record_set_member_lookup` is what
        // `descr_set_absent` / `descr_set_ambiguous` report, and those are
        // gated in `check.py`.  Returning at the first non-answer would stop
        // counting the rest of the set, so the gate's own numbers would shrink
        // with the defect they exist to measure — by an order-dependent
        // amount, which is worse than useless.
        let mut complete = true;
        for m in members {
            let looked_up = descr_from_set_member(m);
            record_set_member_lookup(m, &looked_up);
            match looked_up {
                SetMemberLookup::Resolved(d) => out.push(d),
                // `effectinfo.py:479-494 compute_bitstrings` unions EVERY
                // member of every raw set; a member there is a live descr
                // object, so upstream has no third answer.  Both of pyre's
                // non-answers therefore mean the same thing — this set
                // cannot be completed — and the only sound reply is the
                // `EF_RANDOM_EFFECTS` wildcard.  Keeping a *concrete* set
                // with the member omitted is what breaks the contract
                // `optimizeopt/heap.rs force_from_effectinfo` relies on:
                // out-of-range `bitcheck` returns false (`bitstring.py:16-20`
                // parity), so the omission reads as "the callee does not
                // write this field" and the optimizer keeps a stale heap
                // cache entry across a call that does write it.
                //
                // Absence is not even stable — the descr universe keeps
                // growing after the raw sets freeze, so a container absent
                // at rehydrate time can arrive later while the omission
                // lives on.  Closing the serialized universe (the
                // build-time `DescrMintSpec` channel) is what makes this
                // arm unreachable rather than merely rare.
                SetMemberLookup::AbsentContainer | SetMemberLookup::Ambiguous => complete = false,
            }
        }
        complete.then(|| majit_ir::effectinfo::canonicalize_descr_set(out))
    };
    let resolved = (
        resolve(&keys.readonly_fields),
        resolve(&keys.write_fields),
        resolve(&keys.readonly_arrays),
        resolve(&keys.write_arrays),
        resolve(&keys.readonly_interiorfields),
        resolve(&keys.write_interiorfields),
    );
    let (
        Some(readonly_fields),
        Some(write_fields),
        Some(readonly_arrays),
        Some(write_arrays),
        Some(readonly_interiorfields),
        Some(write_interiorfields),
    ) = resolved
    else {
        degrade(ei);
        return;
    };
    ei._readonly_descrs_fields = Some(readonly_fields);
    ei._write_descrs_fields = Some(write_fields);
    ei._readonly_descrs_arrays = Some(readonly_arrays);
    // effectinfo.py:201-206 single_write_descr_array — also `serde(skip)`,
    // and read in production by `heap.rs force_from_effectinfo`, so it is
    // re-derived from the set that just came back rather than left `None`.
    ei.single_write_descr_array = match write_arrays.as_slice() {
        [only] => Some(only.clone()),
        _ => None,
    };
    ei._write_descrs_arrays = Some(write_arrays);
    ei._readonly_descrs_interiorfields = Some(readonly_interiorfields);
    ei._write_descrs_interiorfields = Some(write_interiorfields);
}

/// `BhCallDescr` -> `CallDescr` adapter. RPython parity: codewriter
/// `Assembler.descrs` carries the same `CallDescr` instance the
/// metainterp pulls during op recording. pyre keeps the codewriter-side
/// call descr as serializable fields and rebuilds a `MetaCallDescr` on
/// demand here, preserving the per-call-site `EffectInfo`.
///
/// `arg_classes` is RPython `CallDescr.arg_classes`: one char per non-void
/// function argument. Uppercase `I/R/F` are assembler list markers and must not
/// appear here.
pub fn make_call_descr_from_bh(bh: &majit_translate::jitcode::BhCallDescr) -> DescrRef {
    let arg_types: Vec<Type> = bh
        .arg_classes
        .chars()
        .filter_map(|c| match c {
            'i' | 'S' => Some(Type::Int),
            'r' => Some(Type::Ref),
            'f' | 'L' => Some(Type::Float),
            _ => None,
        })
        .collect();
    let result_type = match bh.result_type {
        'i' | 'S' => Type::Int,
        'r' => Type::Ref,
        'f' | 'L' => Type::Float,
        _ => Type::Void,
    };
    // call.py:320 effectinfo_from_writeanalyze parity: the descr consumed
    // by pyjitpl/residual-call recording must expose the same EffectInfo
    // that the codewriter classified for this call site.
    //
    // descr.py:524-526 `get_result_type()` parity — preserve the raw
    // `bh.result_type` char ('i'/'r'/'f'/'v'/'S'/'L') so downstream
    // consumers (`bhimpl_call_*` dispatch, `is_result_signed`) can
    // recover the original singlefloat/longlong classification that the
    // normalized `Type` collapses.
    majit_ir::descr::make_call_descr_full_with_result_class(
        u32::MAX,
        arg_types,
        result_type,
        bh.result_type,
        bh.result_signed,
        bh.result_size,
        bh.extra_info.clone(),
    )
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
    majit_ir::descr_registry::register_array(array_descr.clone() as DescrRef);
    let field_descr = Arc::new(SimpleFieldDescr::new(
        field_descr_index,
        field_offset,
        field_size,
        tp,
        true, // immutable (struct fields in array-of-struct)
    ));
    majit_ir::descr_registry::register_field(field_descr.clone() as DescrRef);
    let interior: DescrRef = Arc::new(SimpleInteriorFieldDescr::new(
        field_descr_index,
        array_descr,
        field_descr,
    ));
    majit_ir::descr_registry::register_interior_field(interior.clone());
    interior
}

#[cfg(test)]
mod set_member_lookup_tests {
    use super::*;
    use majit_ir::effectinfo::DescrSetMember;

    /// Distinguishing the two non-resolving outcomes is the whole gate:
    /// `AbsentContainer` is the projection onto a universe that never named the
    /// container, `Ambiguous` is the container being named under a different
    /// key. Only the second is a defect, and it is the one `check.py` gates at
    /// zero.
    ///
    /// Both assertions turn on one key each, so they hold whatever else the
    /// process-global gccache already carries.
    #[test]
    fn an_unnamed_container_is_absent_and_a_named_one_missing_the_field_is_ambiguous() {
        let published = 0x7e57_0000_0000_0001u64;
        let never_named = 0x7e57_0000_0000_0002u64;

        majit_ir::descr::gc_cache()
            .lock()
            .unwrap()
            .register_keyed_size(
                majit_ir::descr::LLType::Struct(published),
                Arc::new(majit_ir::descr::SimpleSizeDescr::with_vtable(
                    u32::MAX,
                    32,
                    0,
                    0,
                )) as DescrRef,
            );

        let member = |struct_id| DescrSetMember::Field {
            struct_id,
            field_name: "no_such_field".to_string(),
        };
        assert!(matches!(
            descr_from_set_member(&member(never_named)),
            SetMemberLookup::AbsentContainer
        ));
        assert!(matches!(
            descr_from_set_member(&member(published)),
            SetMemberLookup::Ambiguous
        ));
    }

    /// The label is what `PYRE_DESCR_SPELLING_GATE` prints, and it has to name
    /// the container in the same hex form `descrs.bin` keys on so an entry can
    /// be grepped straight back to a producer.
    #[test]
    fn labels_carry_the_container_key_and_member_name() {
        assert_eq!(
            set_member_label(&DescrSetMember::Field {
                struct_id: 0xa3af111df5325ac5,
                field_name: "items".to_string(),
            }),
            "Struct(0xa3af111df5325ac5).items"
        );
        assert_eq!(
            set_member_label(&DescrSetMember::Array {
                array_id: 0x0000000000000010
            }),
            "Array(0x0000000000000010)"
        );
    }

    /// The point of carrying the mint arguments across the build/runtime split:
    /// a container this process would never otherwise name resolves anyway,
    /// because the publish takes the same `descr.py:224-238` miss branch the
    /// analyzer took. Without it the member reads `AbsentContainer` — which is
    /// the pre-state this asserts first, on the same key.
    #[test]
    fn publishing_the_recorded_mint_turns_an_absent_member_into_a_resolved_one() {
        use majit_ir::effectinfo::{DescrMintEntry, DescrMintSpec};

        let struct_id = 0x7e57_0000_0000_0003u64;
        let member = DescrSetMember::Field {
            struct_id,
            field_name: "carried".to_string(),
        };
        assert!(
            matches!(
                descr_from_set_member(&member),
                SetMemberLookup::AbsentContainer
            ),
            "nothing has named this container yet"
        );

        publish_effect_info_descr_mints(&[DescrMintEntry {
            member: member.clone(),
            spec: DescrMintSpec::Field {
                struct_size: 24,
                offset: 16,
                field_size: 8,
                field_type: majit_ir::value::Type::Int,
                flag: majit_ir::descr::ArrayFlag::Signed,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 1,
            },
        }]);

        let SetMemberLookup::Resolved(descr) = descr_from_set_member(&member) else {
            panic!("the published slot must resolve");
        };
        let field = descr
            .as_field_descr()
            .expect("a Field member resolves to a FieldDescr");
        assert_eq!(field.offset(), 16);
        assert_eq!(field.field_size(), 8);
        assert_eq!(field.index_in_parent(), 1);
    }
}
