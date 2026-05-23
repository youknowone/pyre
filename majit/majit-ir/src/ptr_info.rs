//! Building blocks for `PtrInfo` — the pointer-analysis info type
//! attached to each `_forwarded` slot. Hosted in `majit-ir` so the
//! `Forwarded` move that follows can reference these types without
//! a `majit-metainterp → majit-ir` circular dep.
//!
//! Pure data + leaf methods only. Methods that need `Op` / `OptContext`
//! / `BoxRef` from `majit-metainterp` live as extension traits in
//! `metainterp::optimizeopt::info`.

use crate::field_entry::FieldEntry;
use crate::intbound::IntBound;
use crate::rawbuffer::{RawBuffer, RawBufferError};
use crate::{DescrRef, GcRef, OpRef, RdVirtualInfo};

/// info.py: `AbstractVirtualPtrInfo` (RPython base class hint). Pyre
/// hoists only the fields shared by every Virtual* variant so each
/// `PtrInfo::Virtual*` carries a single embedded slot instead of N
/// independent copies of the same field set.
///
/// `descr` and `_is_virtual` are NOT lifted here:
///   - `descr` is variant-specific (SizeDescr for Virtual, ArrayDescr
///     for VirtualArray, etc.) — RPython's `_attrs_` is a hint to the
///     translator's slot allocator, not a parity constraint on the
///     storage *type*. Each pyre variant keeps its own typed `descr`.
///   - `_is_virtual` collapses into the pyre enum tag itself
///     (`PtrInfo::Virtual(_)` IS the truthy carrier of `_is_virtual`);
///     no separate slot is needed.
///
/// `make_virtual_info` (resume.py:307-315) reads `cached_vinfo` to
/// dedup RdVirtualInfo allocations across multiple finish() calls
/// referencing the same virtual. `RefCell` provides interior
/// mutability so the immutable-receiver accessor can populate the
/// cache on first miss.
#[derive(Clone, Debug, Default)]
pub struct AbstractVirtualPtrInfo {
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<RdVirtualInfo>>>,
}

impl AbstractVirtualPtrInfo {
    pub fn new() -> Self {
        Self {
            cached_vinfo: std::cell::RefCell::new(None),
        }
    }
}

/// vstring.py:50-140: StrPtrInfo
#[derive(Clone, Debug)]
pub struct StrPtrInfo {
    /// vstring.py: self.lenbound — IntBound for string length.
    pub lenbound: Option<IntBound>,
    /// vstring.py:53 self.lgtop — cached length OpRef (set by getstrlen).
    /// After force_box, this preserves the computed length so subsequent
    /// STRLEN queries reuse it instead of emitting a new STRLEN op.
    pub lgtop: Option<OpRef>,
    /// vstring.py: self.mode — 0 = mode_string, 1 = mode_unicode.
    pub mode: u8,
    /// vstring.py: self.length — known exact length (-1 if unknown).
    pub length: i32,
    /// vstring.py: subclass-specific state
    /// (`VStringPlainInfo` / `VStringSliceInfo` / `VStringConcatInfo`).
    pub variant: VStringVariant,
    /// info.py:91-92: last_guard_pos
    pub last_guard_pos: i32,
    /// info.py:124-128 `AbstractVirtualPtrInfo._cached_vinfo` — inherited
    /// through `StrPtrInfo(AbstractVirtualPtrInfo)` (vstring.py:50,55).
    /// Lifted into `AbstractVirtualPtrInfo` per RPython `_attrs_`
    /// inheritance contract; `make_virtual_info` dedups across finish()
    /// calls by comparing fieldnums (resume.py:309-314).
    pub avpi: AbstractVirtualPtrInfo,
}

impl StrPtrInfo {
    /// vstring.py:168 / 227 / 278 `is_virtual()` on the string ptrinfo classes.
    pub fn is_virtual(&self) -> bool {
        match &self.variant {
            VStringVariant::Ptr => false,
            VStringVariant::Plain(_) | VStringVariant::Slice(_) => true,
            VStringVariant::Concat(info) => info._is_virtual,
        }
    }
}

/// vstring.py:142-334 subclass state carried by `StrPtrInfo`.
#[derive(Clone, Debug)]
pub enum VStringVariant {
    /// Non-virtual base `StrPtrInfo`.
    Ptr,
    /// vstring.py:142 `VStringPlainInfo`.
    Plain(VStringPlainInfo),
    /// vstring.py:214 `VStringSliceInfo`.
    Slice(VStringSliceInfo),
    /// vstring.py:266 `VStringConcatInfo`.
    Concat(VStringConcatInfo),
}

/// vstring.py:142-212 `VStringPlainInfo`
#[derive(Clone, Debug)]
pub struct VStringPlainInfo {
    pub _chars: Vec<Option<OpRef>>,
}

/// vstring.py:214-264 `VStringSliceInfo`
#[derive(Clone, Debug)]
pub struct VStringSliceInfo {
    pub s: OpRef,
    pub start: OpRef,
    pub lgtop: OpRef,
}

/// vstring.py:266-334 `VStringConcatInfo`
#[derive(Clone, Debug)]
pub struct VStringConcatInfo {
    pub vleft: OpRef,
    pub vright: OpRef,
    pub _is_virtual: bool,
}

/// A virtual object whose allocation has been removed.
///
/// Fields are tracked as OpRefs to the operations that produce their values.
///
/// ## Invariant: `fields` NEVER contains typeptr (offset 0)
///
/// Matches RPython upstream: `heaptracker.py:66-67 all_fielddescrs()` skips
/// `typeptr`, so `info.py:180 AbstractStructPtrInfo.init_fields` sizes
/// `_fields` with typeptr excluded from the indexable range. The typeptr
/// (offset 0) is tracked separately via `known_class` and emitted by the
/// GC rewriter's `gen_initialize_vtable` path (rewrite.py:479-484), NOT
/// from the force-path field loop.
///
/// Enforced by:
/// - `virtualize.rs optimize_setfield_gc` Virtual arm: runtime check that
///   returns early on `offset == Some(0)` before calling `set_field`.
/// - `virtualize.rs force_virtual_instance`: `debug_assert_no_typeptr`
///   at the entry of the field-emit loop.
/// - `virtualstate.rs export_single_value`:
///   `debug_assert_no_typeptr` on the fields collection boundary.
#[derive(Clone, Debug)]
pub struct VirtualInfo {
    /// The size descriptor of this object.
    pub descr: DescrRef,
    /// Known class (if any).
    pub known_class: Option<GcRef>,
    /// ob_type field descriptor for force path. In RPython the vtable is
    /// set by allocate_with_vtable, not as a struct field. pyre stores
    /// ob_type at offset 0 explicitly. This descr lets force emit
    /// SetfieldGc(ob_type) without polluting `fields` (which feeds rd_virtuals).
    pub ob_type_descr: Option<DescrRef>,
    /// Field values: `(field_descr_index, value_opref)`.
    /// **Invariant**: never contains typeptr (offset 0) — see struct-level docs.
    pub fields: Vec<(u32, OpRef)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py:124-128 `AbstractVirtualPtrInfo._cached_vinfo` inherited
    /// state. Lifted into `AbstractVirtualPtrInfo` per RPython `_attrs_`
    /// inheritance — see the shared-struct doc above.
    pub avpi: AbstractVirtualPtrInfo,
}

/// A virtual array.
#[derive(Clone, Debug)]
pub struct VirtualArrayInfo {
    /// The array descriptor.
    pub descr: DescrRef,
    /// Whether this was created by NewArrayClear (zero-initialized).
    pub clear: bool,
    /// Element values.
    pub items: Vec<OpRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

/// A non-virtual object with cached field info.
///
/// Mirrors RPython's InstancePtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct InstancePtrInfo {
    /// Best-known instance descriptor, if any.
    pub descr: Option<DescrRef>,
    /// Known class pointer, if guarded exactly.
    pub known_class: Option<GcRef>,
    /// info.py:175 _fields — cached field values.
    /// RPython stores both normal Boxes and PreambleOp sentinels in the
    /// same list. Rust mirrors this with `Vec<(u32, FieldEntry)>`.
    pub fields: Vec<(u32, FieldEntry)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual GC struct with cached field info.
///
/// Mirrors RPython's StructPtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct StructPtrInfo {
    /// Exact struct descriptor.
    pub descr: DescrRef,
    /// info.py:175 _fields — cached field values (same as InstancePtrInfo).
    pub fields: Vec<(u32, FieldEntry)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual GC array with cached item info and lenbound.
///
/// Mirrors RPython's ArrayPtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct ArrayPtrInfo {
    /// Exact array descriptor.
    pub descr: DescrRef,
    /// Known bounds on the array length.
    pub lenbound: IntBound,
    /// info.py:579 _items — cached item values for constant indices.
    /// RPython stores both normal Boxes and PreambleOp sentinels.
    pub items: Vec<FieldEntry>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A virtual struct (no vtable).
#[derive(Clone, Debug)]
pub struct VirtualStructInfo {
    /// The size descriptor.
    pub descr: DescrRef,
    /// Field values: (field_index, value, optional original field descriptor).
    pub fields: Vec<(u32, OpRef)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

/// A virtual array of structs (interior field access pattern).
///
/// Mirrors RPython's VArrayStructInfo where each array element
/// is a fixed-size struct with named fields. Used for RPython arrays
/// with complex item types (e.g., hash table entries with key+value fields).
#[derive(Clone, Debug)]
pub struct VirtualArrayStructInfo {
    /// The array descriptor (arraydescr).
    pub descr: DescrRef,
    /// Per-element fields: outer Vec = elements, inner Vec = (field_descr_index, value_opref).
    pub element_fields: Vec<Vec<(u32, OpRef)>>,
    /// resume.py VArrayStructInfo.fielddescrs — InteriorFieldDescr per field.
    /// Used by _number_virtuals to extract item_size/field_offset/field_size.
    pub fielddescrs: Vec<DescrRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

/// info.py:RawSlicePtrInfo — alias view into a parent virtual raw buffer.
///
/// Created by `make_virtual_raw_slice` (virtualize.py:60-65) when an
/// `INT_ADD(rawbuf, const_offset)` is folded against a virtual raw buffer.
/// Reads / writes through a slice add `offset` to the requested byte
/// offset and forward to the parent buffer.
#[derive(Clone, Debug)]
pub struct VirtualRawSliceInfo {
    /// Slice offset relative to the parent buffer's base. Signed because
    /// `info.py:460 RawSlicePtrInfo.__init__(offset, parent)` accepts an
    /// unbounded RPython int — `optimize_INT_ADD` folds the addend as a
    /// signed `getint()` and a negative addend is a valid (if rare)
    /// slice base.
    pub offset: i64,
    /// OpRef of the parent VirtualRawBuffer (or another VirtualRawSlice
    /// — `optimize_int_add` flattens chained slices when the underlying
    /// info is `VirtualRawBufferInfo`/`VirtualRawSliceInfo`).
    pub parent: OpRef,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

/// info.py:386 RawBufferPtrInfo — pointer info for virtual raw memory.
///
/// RPython stores the byte-write tracking in a separate `RawBuffer` object
/// (`self.buffer = RawBuffer(cpu, None)` in info.py:392-393). Rust mirrors
/// that by keeping the rawbuffer.py parallel-list state in `buffer`, while
/// this struct owns the RawBufferPtrInfo metadata.
#[derive(Clone, Debug)]
pub struct VirtualRawBufferInfo {
    /// info.py:390 self.func — raw malloc function pointer.
    pub func: i64,
    /// info.py:391 self.size — size of the virtual raw buffer.
    pub size: usize,
    /// info.py:387/392 self.buffer — rawbuffer.py RawBuffer.
    pub buffer: RawBuffer,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py:420: calldescr for CALL_I(func, size) raw malloc.
    /// Saved from the original CALL_I op during virtualization.
    pub calldescr: Option<DescrRef>,
    /// info.py `_cached_vinfo` — see AbstractVirtualPtrInfo.
    pub avpi: AbstractVirtualPtrInfo,
}

impl VirtualRawBufferInfo {
    /// virtualize.py:52-58 creates RawBufferPtrInfo(cpu, func, size),
    /// whose constructor initializes `self.buffer = RawBuffer(cpu, None)`.
    pub fn new(func: i64, size: usize, calldescr: Option<DescrRef>) -> Self {
        Self {
            func,
            size,
            buffer: RawBuffer::new(),
            last_guard_pos: -1,
            calldescr,
            avpi: AbstractVirtualPtrInfo::new(),
        }
    }

    /// info.py:403-410 RawBufferPtrInfo.getitem_raw delegates to RawBuffer.
    pub fn read_value(
        &self,
        offset: i64,
        length: usize,
        descr: &DescrRef,
    ) -> Result<OpRef, RawBufferError> {
        self.buffer.read_value(offset, length, descr)
    }

    /// info.py:412-415 RawBufferPtrInfo.setitem_raw delegates to RawBuffer.
    pub fn write_value(
        &mut self,
        offset: i64,
        length: usize,
        descr: DescrRef,
        value: OpRef,
    ) -> Result<(), RawBufferError> {
        self.buffer.write_value(offset, length, descr, value)
    }
}

/// Tracked field state for a virtualizable object (interpreter frame).
///
/// Mirrors RPython's virtualizable handling in the optimizer:
/// the frame already exists on the heap, but during JIT execution its
/// fields are kept in registers. The optimizer tracks the current value
/// of each field so that redundant setfield/getfield ops are eliminated.
///
/// When the virtualizable is "forced" (escapes to non-JIT code), field
/// values are written back to the heap via SETFIELD_RAW ops.
#[derive(Clone, Debug)]
pub struct VirtualizableFieldState {
    /// Tracked static field values: (field_descr_index, current_value_opref).
    /// Indices correspond to VirtualizableInfo::static_fields order.
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors: (field_descr_index, original_descr).
    /// Used to emit correct SetfieldRaw ops when forcing.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// Tracked array field values: (array_field_index, element_values).
    /// Indices correspond to VirtualizableInfo::array_fields order.
    pub arrays: Vec<(u32, Vec<OpRef>)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}
