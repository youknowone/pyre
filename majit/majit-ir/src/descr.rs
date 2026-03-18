/// Descriptor traits for the JIT IR.
///
/// Translated from rpython/jit/metainterp/history.py (AbstractDescr)
/// and rpython/jit/backend/llsupport/descr.py.
///
/// Descriptors carry type metadata needed by the optimizer and backend
/// for field access, array access, function calls, and guard failures.
use std::sync::Arc;

use crate::value::Type;

/// Opaque reference to a descriptor, shared across the JIT pipeline.
pub type DescrRef = Arc<dyn Descr>;

/// Base trait for all descriptors.
///
/// Mirrors rpython/jit/metainterp/history.py AbstractDescr.
pub trait Descr: Send + Sync + std::fmt::Debug {
    /// Unique index of this descriptor (for serialization).
    /// Returns u32::MAX if not assigned.
    fn index(&self) -> u32 {
        u32::MAX
    }

    /// Human-readable representation for debugging.
    fn repr(&self) -> String {
        format!("{:?}", self)
    }

    // ── Downcasting helpers ──

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        None
    }
    fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
        None
    }
    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        None
    }
    fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
        None
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        None
    }
    fn as_interior_field_descr(&self) -> Option<&dyn InteriorFieldDescr> {
        None
    }

    /// Whether the field/array described is always pure (immutable).
    fn is_always_pure(&self) -> bool {
        false
    }

    /// Whether this descriptor marks a loop version guard.
    ///
    /// Loop version guards have their alternative path compiled immediately
    /// after the main loop, rather than lazily on failure.
    fn is_loop_version(&self) -> bool {
        false
    }

    /// intbounds.py: descr.is_integer_bounded() / get_integer_min/max.
    /// Returns (field_size_bytes, is_signed) if this is a field descriptor.
    /// Used by intbounds to narrow GETFIELD result bounds.
    fn field_size_and_sign(&self) -> (usize, bool) {
        if let Some(fd) = self.as_field_descr() {
            (fd.field_size(), fd.is_field_signed())
        } else {
            (0, false)
        }
    }
}

/// Descriptor for guard failures — carries resume information.
///
/// Mirrors rpython/jit/metainterp/history.py AbstractFailDescr.
pub trait FailDescr: Descr {
    /// Index in the fail descr table.
    fn fail_index(&self) -> u32;

    /// The types of the fail arguments.
    fn fail_arg_types(&self) -> &[Type];

    /// Whether this fail descriptor represents a FINISH exit.
    fn is_finish(&self) -> bool {
        false
    }

    /// Identifier of the compiled trace that owns this exit.
    ///
    /// Backends that lower loops and bridges as separate compiled traces use
    /// this to let the frontend distinguish root-loop exits from bridge exits.
    fn trace_id(&self) -> u64 {
        0
    }

    /// Whether the given exit slot should be treated as a real GC root.
    ///
    /// Backends may override this to distinguish rooted refs from opaque
    /// handles that reuse `Type::Ref`, such as FORCE_TOKEN values.
    fn is_gc_ref_slot(&self, slot: usize) -> bool {
        matches!(self.fail_arg_types().get(slot), Some(Type::Ref))
    }

    /// Exit slot indices that carry opaque force-token handles.
    fn force_token_slots(&self) -> &[usize] {
        &[]
    }
}

/// Descriptor for a fixed-size struct/object allocation.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py SizeDescr.
pub trait SizeDescr: Descr {
    /// Total size in bytes.
    fn size(&self) -> usize;

    /// Type ID (for GC header).
    fn type_id(&self) -> u32;

    /// Whether this is an immutable object.
    fn is_immutable(&self) -> bool;

    /// Whether this is an object (has vtable).
    fn is_object(&self) -> bool {
        false
    }

    /// Vtable address, if is_object().
    fn vtable(&self) -> usize {
        0
    }

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!("SizeDescr(size={}, type_id={})", self.size(), self.type_id())
    }

    /// Field descriptors for fields containing GC pointers.
    fn gc_field_descrs(&self) -> &[Arc<dyn FieldDescr>] {
        &[]
    }

    /// All field descriptors (not just GC pointer ones).
    /// descr.py: get_all_fielddescrs()
    fn all_field_descrs(&self) -> &[Arc<dyn FieldDescr>] {
        self.gc_field_descrs() // default: same as gc_field_descrs
    }

    /// Number of fields.
    fn num_fields(&self) -> usize {
        self.all_field_descrs().len()
    }
}

/// Descriptor for a field within a struct.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py FieldDescr.
pub trait FieldDescr: Descr {
    /// Byte offset from the start of the struct.
    fn offset(&self) -> usize;

    /// Size of the field in bytes.
    fn field_size(&self) -> usize;

    /// Type of value stored in this field.
    fn field_type(&self) -> Type;

    /// Whether this is a pointer field (needs GC tracking).
    fn is_pointer_field(&self) -> bool {
        self.field_type() == Type::Ref
    }

    /// Whether this is a float field.
    fn is_float_field(&self) -> bool {
        self.field_type() == Type::Float
    }

    /// Whether reads from this field are signed.
    fn is_field_signed(&self) -> bool {
        true
    }

    /// Whether this field is immutable (never written after object creation).
    ///
    /// Immutable field reads from a constant object can be folded to constants,
    /// and their cached values survive cache invalidation by calls/side effects.
    /// Delegates to `Descr::is_always_pure()` by default.
    fn is_immutable(&self) -> bool {
        self.is_always_pure()
    }

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!(
            "FieldDescr(offset={}, size={}, type={:?})",
            self.offset(),
            self.field_size(),
            self.field_type()
        )
    }

    /// descr.py: index_in_parent — position within parent struct.
    fn index_in_parent(&self) -> usize {
        0
    }

    /// descr.py: sort_key() — for ordering field descriptors.
    fn sort_key(&self) -> usize {
        self.offset()
    }
}

/// Descriptor for an array type.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py ArrayDescr.
pub trait ArrayDescr: Descr {
    /// Size of the fixed header (before array items).
    fn base_size(&self) -> usize;

    /// Size of each array item in bytes.
    fn item_size(&self) -> usize;

    /// Type ID (for GC header).
    fn type_id(&self) -> u32;

    /// Type of each array item.
    fn item_type(&self) -> Type;

    /// Whether items are GC pointers.
    fn is_array_of_pointers(&self) -> bool {
        self.item_type() == Type::Ref
    }

    /// Whether items are floats.
    fn is_array_of_floats(&self) -> bool {
        self.item_type() == Type::Float
    }

    /// Whether integer items should be sign-extended on loads.
    ///
    /// RPython array descriptors distinguish signed from unsigned integer
    /// storage. Backends should ignore this for non-integer item types.
    fn is_item_signed(&self) -> bool {
        true
    }

    /// Descriptor for the length field.
    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        None
    }

    /// Whether items are primitive (integer or float, not pointer).
    /// descr.py: is_array_of_primitives()
    fn is_array_of_primitives(&self) -> bool {
        !self.is_array_of_pointers()
    }

    /// Whether items are structs (array-of-structs pattern).
    /// descr.py: is_array_of_structs()
    fn is_array_of_structs(&self) -> bool {
        false
    }
}

/// Descriptor for a field within an array element (interior pointer).
///
/// Mirrors rpython/jit/backend/llsupport/descr.py InteriorFieldDescr.
pub trait InteriorFieldDescr: Descr {
    fn array_descr(&self) -> &dyn ArrayDescr;
    fn field_descr(&self) -> &dyn FieldDescr;
}

/// Descriptor for a function call.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py CallDescr.
pub trait CallDescr: Descr {
    /// Types of the arguments.
    fn arg_types(&self) -> &[Type];

    /// Type of the return value.
    fn result_type(&self) -> Type;

    /// Size of the return value in bytes.
    fn result_size(&self) -> usize;

    /// Whether the result is a signed integer.
    fn is_result_signed(&self) -> bool {
        true
    }

    /// Target compiled loop token for `CALL_ASSEMBLER_*`, if this call
    /// descriptor represents a nested JIT-to-JIT call.
    fn call_target_token(&self) -> Option<u64> {
        None
    }

    /// Side effect information.
    fn effect_info(&self) -> &EffectInfo;

    /// Argument class string (RPython encoding: 'i'=int, 'r'=ref, 'f'=float).
    /// descr.py: arg_classes
    fn arg_classes(&self) -> String {
        self.arg_types()
            .iter()
            .map(|t| match t {
                Type::Int => 'i',
                Type::Ref => 'r',
                Type::Float => 'f',
                Type::Void => 'v',
            })
            .collect()
    }

    /// Result type as arg class character.
    fn result_class(&self) -> char {
        match self.result_type() {
            Type::Int => 'i',
            Type::Ref => 'r',
            Type::Float => 'f',
            Type::Void => 'v',
        }
    }

    /// Number of arguments.
    fn num_args(&self) -> usize {
        self.arg_types().len()
    }
}

/// Descriptor for `DebugMergePoint` operations — carries source position
/// information at merge points (bytecode boundaries in the traced interpreter).
///
/// Mirrors rpython/jit/metainterp/resoperation.py DebugMergePoint.
/// RPython's meta-interpreter emits these at each bytecode boundary
/// during tracing. They carry:
/// - The JitDriver name (which interpreter generated this trace)
/// - A source-level representation (e.g., "bytecode 42 in function foo")
/// - The call depth (for inlined functions)
///
/// These are used by jitviewer and profiling tools to map compiled code
/// back to the source interpreter's bytecode positions.
#[derive(Clone, Debug)]
pub struct DebugMergePointInfo {
    /// Name of the JitDriver that generated this trace.
    /// E.g., "pypyjit" for PyPy's main interpreter.
    pub jd_name: String,
    /// Source-level representation: a human-readable string identifying
    /// the position in the traced interpreter's code.
    /// E.g., "bytecode LOAD_FAST at offset 12 in function foo".
    pub source_repr: String,
    /// Bytecode index (program counter value) in the traced interpreter.
    pub bytecode_index: i64,
    /// Call depth: 0 for the outermost (root) trace, incremented for
    /// each level of inlined function calls.
    pub call_depth: u32,
}

impl DebugMergePointInfo {
    pub fn new(
        jd_name: impl Into<String>,
        source_repr: impl Into<String>,
        bytecode_index: i64,
        call_depth: u32,
    ) -> Self {
        DebugMergePointInfo {
            jd_name: jd_name.into(),
            source_repr: source_repr.into(),
            bytecode_index,
            call_depth,
        }
    }
}

/// Concrete descriptor wrapping `DebugMergePointInfo` for attachment to IR ops.
#[derive(Debug)]
pub struct DebugMergePointDescr {
    pub info: DebugMergePointInfo,
}

impl DebugMergePointDescr {
    pub fn new(info: DebugMergePointInfo) -> Self {
        DebugMergePointDescr { info }
    }
}

impl Descr for DebugMergePointDescr {
    fn repr(&self) -> String {
        format!(
            "debug_merge_point({}, '{}', pc={}, depth={})",
            self.info.jd_name,
            self.info.source_repr,
            self.info.bytecode_index,
            self.info.call_depth
        )
    }
}

/// Side effect classification for calls.
///
/// Translated from rpython/jit/codewriter/effectinfo.py.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectInfo {
    pub extra_effect: ExtraEffect,
    pub oopspec_index: OopSpecIndex,
    /// effectinfo.py: bitstring_readonly_descrs_fields
    /// Bitset of field descriptor indices that may be read by this call.
    pub readonly_descrs_fields: u64,
    /// effectinfo.py: bitstring_write_descrs_fields
    /// Bitset of field descriptor indices that may be written by this call.
    pub write_descrs_fields: u64,
    /// effectinfo.py: bitstring_readonly_descrs_arrays
    /// Bitset of array descriptor indices that may be read.
    pub readonly_descrs_arrays: u64,
    /// effectinfo.py: bitstring_write_descrs_arrays
    /// Bitset of array descriptor indices that may be written.
    pub write_descrs_arrays: u64,
    /// effectinfo.py: can_invalidate
    pub can_invalidate: bool,
}

impl Default for EffectInfo {
    fn default() -> Self {
        EffectInfo {
            extra_effect: ExtraEffect::CanRaise,
            oopspec_index: OopSpecIndex::None,
            readonly_descrs_fields: 0,
            write_descrs_fields: 0,
            readonly_descrs_arrays: 0,
            write_descrs_arrays: 0,
            can_invalidate: false,
        }
    }
}

/// How a call affects the optimizer's ability to optimize surrounding code.
///
/// Ordered from most optimizable to least.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ExtraEffect {
    /// Pure function that cannot raise — can be eliminated entirely.
    ElidableCannotRaise = 0,
    /// Call once per loop iteration.
    LoopInvariant = 1,
    /// Cannot raise any exception.
    CannotRaise = 2,
    /// Pure but may raise MemoryError.
    ElidableOrMemoryError = 3,
    /// Pure but may raise.
    ElidableCanRaise = 4,
    /// Normal function that can raise.
    CanRaise = 5,
    /// Can force virtualizables/virtual objects.
    ForcesVirtualOrVirtualizable = 6,
    /// Arbitrary effects — optimizer assumes the worst.
    RandomEffects = 7,
}

/// OopSpec index — identifies special-cased operations for the optimizer.
///
/// Translated from rpython/jit/codewriter/effectinfo.py OS_* constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum OopSpecIndex {
    None = 0,
    Arraycopy = 1,
    Str2Unicode = 2,
    ShrinkArray = 3,
    DictLookup = 4,
    ThreadlocalrefGet = 5,
    NotInTrace = 8,
    Arraymove = 9,
    IntPyDiv = 12,
    IntUdiv = 13,
    IntPyMod = 14,
    IntUmod = 15,
    StrConcat = 22,
    StrSlice = 23,
    StrEqual = 24,
    StreqSliceChecknull = 25,
    StreqSliceNonnull = 26,
    StreqSliceChar = 27,
    StreqNonnull = 28,
    StreqNonnullChar = 29,
    StreqChecknullChar = 30,
    StreqLengthok = 31,
    StrCmp = 32,
    UniConcat = 42,
    UniSlice = 43,
    UniEqual = 44,
    UnieqSliceChecknull = 45,
    UnieqSliceNonnull = 46,
    UnieqSliceChar = 47,
    UnieqNonnull = 48,
    UnieqNonnullChar = 49,
    UnieqChecknullChar = 50,
    UnieqLengthok = 51,
    UniCmp = 52,
    LibffiCall = 62,
    LlongInvert = 69,
    LlongAdd = 70,
    LlongSub = 71,
    LlongMul = 72,
    LlongLt = 73,
    LlongLe = 74,
    LlongEq = 75,
    LlongNe = 76,
    LlongGt = 77,
    LlongGe = 78,
    LlongAnd = 79,
    LlongOr = 80,
    LlongLshift = 81,
    LlongRshift = 82,
    LlongXor = 83,
    LlongFromInt = 84,
    LlongToInt = 85,
    LlongFromFloat = 86,
    LlongToFloat = 87,
    LlongUlt = 88,
    LlongUle = 89,
    LlongUgt = 90,
    LlongUge = 91,
    LlongUrshift = 92,
    LlongFromUint = 93,
    LlongUToFloat = 94,
    MathSqrt = 100,
    MathReadTimestamp = 101,
    RawMallocVarsizeChar = 110,
    RawFree = 111,
    StrCopyToRaw = 112,
    UniCopyToRaw = 113,
    JitForceVirtual = 120,
    JitForceVirtualizable = 121,
}

impl EffectInfo {
    pub fn is_elidable(&self) -> bool {
        matches!(
            self.extra_effect,
            ExtraEffect::ElidableCannotRaise
                | ExtraEffect::ElidableOrMemoryError
                | ExtraEffect::ElidableCanRaise
        )
    }

    pub fn is_loopinvariant(&self) -> bool {
        self.extra_effect == ExtraEffect::LoopInvariant
    }

    pub fn can_raise(&self) -> bool {
        self.extra_effect >= ExtraEffect::ElidableCanRaise
    }

    /// effectinfo.py: check_can_invalidate()
    /// Whether this call can invalidate compiled code (quasi-immutable mutations).
    pub fn can_invalidate(&self) -> bool {
        self.can_invalidate
    }

    /// Whether this call can trigger GC collection.
    /// RPython: `effectinfo.can_collect()`
    pub fn can_collect(&self) -> bool {
        self.extra_effect >= ExtraEffect::CanRaise
    }

    /// effectinfo.py: check_forces_virtual_or_virtualizable()
    /// Whether this call forces virtualizables or virtual objects.
    pub fn forces_virtual_or_virtualizable(&self) -> bool {
        self.extra_effect >= ExtraEffect::ForcesVirtualOrVirtualizable
    }

    /// Whether this call has random effects (worst case).
    pub fn has_random_effects(&self) -> bool {
        self.extra_effect == ExtraEffect::RandomEffects
    }

    /// Whether the oopspec identifies a special-cased operation.
    pub fn has_oopspec(&self) -> bool {
        self.oopspec_index != OopSpecIndex::None
    }

    /// effectinfo.py: check_can_raise(ignore_memoryerror)
    /// Whether this call can raise exceptions (optionally ignoring MemoryError).
    pub fn check_can_raise(&self, ignore_memoryerror: bool) -> bool {
        if ignore_memoryerror {
            // ElidableOrMemoryError can only raise MemoryError
            matches!(
                self.extra_effect,
                ExtraEffect::ElidableCanRaise
                    | ExtraEffect::CanRaise
                    | ExtraEffect::ForcesVirtualOrVirtualizable
                    | ExtraEffect::RandomEffects
            )
        } else {
            self.can_raise()
        }
    }

    /// effectinfo.py: is_call_release_gil()
    /// Whether this call releases the GIL (for FFI calls).
    pub fn is_call_release_gil(&self) -> bool {
        false // Not applicable in our Rust runtime
    }

    /// Const-compatible constructor for static initialization.
    pub const fn const_new(extra_effect: ExtraEffect, oopspec_index: OopSpecIndex) -> Self {
        EffectInfo {
            extra_effect,
            oopspec_index,
            readonly_descrs_fields: 0,
            write_descrs_fields: 0,
            readonly_descrs_arrays: 0,
            write_descrs_arrays: 0,
            can_invalidate: false,
        }
    }

    /// Create a new EffectInfo with the given effect and oopspec.
    pub fn new(extra_effect: ExtraEffect, oopspec_index: OopSpecIndex) -> Self {
        EffectInfo {
            extra_effect,
            oopspec_index,
            ..Default::default()
        }
    }

    /// Create an EffectInfo for a pure, elidable operation.
    pub fn elidable() -> Self {
        EffectInfo {
            extra_effect: ExtraEffect::ElidableCannotRaise,
            ..Default::default()
        }
    }

    /// Create an EffectInfo for a side-effecting operation.
    pub fn side_effecting() -> Self {
        EffectInfo {
            extra_effect: ExtraEffect::RandomEffects,
            ..Default::default()
        }
    }

    // ── Bitstring check methods (effectinfo.py parity) ──

    /// effectinfo.py: check_readonly_descr_field(fielddescr)
    /// Check if this call may read the given field descriptor.
    pub fn check_readonly_descr_field(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.readonly_descrs_fields & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py: check_write_descr_field(fielddescr)
    /// Check if this call may write the given field descriptor.
    pub fn check_write_descr_field(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.write_descrs_fields & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py: check_readonly_descr_array(arraydescr)
    pub fn check_readonly_descr_array(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.readonly_descrs_arrays & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py: check_write_descr_array(arraydescr)
    pub fn check_write_descr_array(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.write_descrs_arrays & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py: check_is_elidable()
    pub fn check_is_elidable(&self) -> bool {
        self.is_elidable()
    }
}

// ── Concrete descriptor implementations (descr.py) ──

/// Simple concrete FieldDescr for use by pyre-jit and tests.
#[derive(Debug, Clone)]
pub struct SimpleFieldDescr {
    index: u32,
    offset: usize,
    field_size: usize,
    field_type: Type,
    is_immutable: bool,
    is_signed: bool,
}

impl SimpleFieldDescr {
    pub fn new(
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_immutable: bool,
    ) -> Self {
        SimpleFieldDescr {
            index,
            offset,
            field_size,
            field_type,
            is_immutable,
            is_signed: true,
        }
    }

    pub fn with_signed(mut self, signed: bool) -> Self {
        self.is_signed = signed;
        self
    }
}

impl Descr for SimpleFieldDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn is_always_pure(&self) -> bool {
        self.is_immutable
    }
    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }
}

impl FieldDescr for SimpleFieldDescr {
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
        self.is_signed
    }
    fn is_immutable(&self) -> bool {
        self.is_immutable
    }
}

/// Simple concrete SizeDescr.
#[derive(Debug, Clone)]
pub struct SimpleSizeDescr {
    index: u32,
    size: usize,
    type_id: u32,
    is_immutable: bool,
}

impl SimpleSizeDescr {
    pub fn new(index: u32, size: usize, type_id: u32) -> Self {
        SimpleSizeDescr {
            index,
            size,
            type_id,
            is_immutable: false,
        }
    }
}

impl Descr for SimpleSizeDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
        Some(self)
    }
}

impl SizeDescr for SimpleSizeDescr {
    fn size(&self) -> usize {
        self.size
    }
    fn type_id(&self) -> u32 {
        self.type_id
    }
    fn is_immutable(&self) -> bool {
        self.is_immutable
    }
}

/// Simple concrete ArrayDescr.
#[derive(Debug, Clone)]
pub struct SimpleArrayDescr {
    index: u32,
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    is_signed: bool,
}

impl SimpleArrayDescr {
    pub fn new(
        index: u32,
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
    ) -> Self {
        SimpleArrayDescr {
            index,
            base_size,
            item_size,
            type_id,
            item_type,
            is_signed: true,
        }
    }
}

impl Descr for SimpleArrayDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
        Some(self)
    }
}

impl ArrayDescr for SimpleArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }
    fn item_size(&self) -> usize {
        self.item_size
    }
    fn type_id(&self) -> u32 {
        self.type_id
    }
    fn item_type(&self) -> Type {
        self.item_type
    }
    fn is_item_signed(&self) -> bool {
        self.is_signed
    }
}

/// Simple concrete InteriorFieldDescr.
#[derive(Debug, Clone)]
pub struct SimpleInteriorFieldDescr {
    index: u32,
    array_descr: std::sync::Arc<SimpleArrayDescr>,
    field_descr: std::sync::Arc<SimpleFieldDescr>,
}

impl SimpleInteriorFieldDescr {
    pub fn new(
        index: u32,
        array_descr: std::sync::Arc<SimpleArrayDescr>,
        field_descr: std::sync::Arc<SimpleFieldDescr>,
    ) -> Self {
        SimpleInteriorFieldDescr {
            index,
            array_descr,
            field_descr,
        }
    }
}

impl Descr for SimpleInteriorFieldDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn as_interior_field_descr(&self) -> Option<&dyn InteriorFieldDescr> {
        Some(self)
    }
}

impl InteriorFieldDescr for SimpleInteriorFieldDescr {
    fn array_descr(&self) -> &dyn ArrayDescr {
        self.array_descr.as_ref()
    }
    fn field_descr(&self) -> &dyn FieldDescr {
        self.field_descr.as_ref()
    }
}

/// Simple concrete CallDescr for non-test use.
#[derive(Debug, Clone)]
pub struct SimpleCallDescr {
    index: u32,
    arg_types: Vec<Type>,
    result_type: Type,
    result_size: usize,
    effect: EffectInfo,
}

impl SimpleCallDescr {
    pub fn new(
        index: u32,
        arg_types: Vec<Type>,
        result_type: Type,
        result_size: usize,
        effect: EffectInfo,
    ) -> Self {
        SimpleCallDescr {
            index,
            arg_types,
            result_type,
            result_size,
            effect,
        }
    }
}

impl Descr for SimpleCallDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
}

impl CallDescr for SimpleCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        self.result_size
    }
    fn effect_info(&self) -> &EffectInfo {
        &self.effect
    }
}

/// Simple concrete FailDescr for guard failure descriptors.
#[derive(Debug, Clone)]
pub struct SimpleFailDescr {
    index: u32,
    fail_index: u32,
    fail_arg_types: Vec<Type>,
    is_finish: bool,
    trace_id: u64,
}

impl SimpleFailDescr {
    pub fn new(index: u32, fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        SimpleFailDescr {
            index,
            fail_index,
            fail_arg_types,
            is_finish: false,
            trace_id: 0,
        }
    }

    pub fn finish(index: u32, fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        SimpleFailDescr {
            index,
            fail_index,
            fail_arg_types,
            is_finish: true,
            trace_id: 0,
        }
    }

    pub fn with_trace_id(mut self, trace_id: u64) -> Self {
        self.trace_id = trace_id;
        self
    }
}

impl Descr for SimpleFailDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for SimpleFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        self.is_finish
    }
    fn trace_id(&self) -> u64 {
        self.trace_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FFI call surface parity tests (rpython/jit/metainterp/test/test_fficall.py) ──

    /// Concrete CallDescr for testing.
    #[derive(Debug)]
    struct TestCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
        result_size: usize,
        result_signed: bool,
        effect: EffectInfo,
    }

    impl Descr for TestCallDescr {
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            self.result_size
        }
        fn is_result_signed(&self) -> bool {
            self.result_signed
        }
        fn effect_info(&self) -> &EffectInfo {
            &self.effect
        }
    }

    #[test]
    fn test_call_descr_stores_arg_types_and_result() {
        // Parity with test_simple_call_int: CallDescr correctly stores arg types and result type
        let descr = TestCallDescr {
            arg_types: vec![Type::Int, Type::Int],
            result_type: Type::Int,
            result_size: 8,
            result_signed: true,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.arg_types(), &[Type::Int, Type::Int]);
        assert_eq!(descr.result_type(), Type::Int);
        assert_eq!(descr.result_size(), 8);
        assert!(descr.is_result_signed());
    }

    #[test]
    fn test_call_descr_float_args() {
        // Parity with test_simple_call_float
        let descr = TestCallDescr {
            arg_types: vec![Type::Float, Type::Float],
            result_type: Type::Float,
            result_size: 8,
            result_signed: false,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.arg_types(), &[Type::Float, Type::Float]);
        assert_eq!(descr.result_type(), Type::Float);
    }

    #[test]
    fn test_call_descr_void_result() {
        // Parity with test_returns_none
        let descr = TestCallDescr {
            arg_types: vec![Type::Int, Type::Int],
            result_type: Type::Void,
            result_size: 0,
            result_signed: false,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.result_type(), Type::Void);
        assert_eq!(descr.result_size(), 0);
    }

    #[test]
    fn test_call_descr_many_arguments() {
        // Parity with test_many_arguments: various argument counts
        for count in [0, 6, 20] {
            let arg_types = vec![Type::Int; count];
            let descr = TestCallDescr {
                arg_types,
                result_type: Type::Int,
                result_size: 8,
                result_signed: true,
                effect: EffectInfo::default(),
            };
            assert_eq!(descr.arg_types().len(), count);
        }
    }

    #[test]
    fn test_call_descr_ref_result() {
        let descr = TestCallDescr {
            arg_types: vec![Type::Ref],
            result_type: Type::Ref,
            result_size: 8,
            result_signed: false,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.arg_types(), &[Type::Ref]);
        assert_eq!(descr.result_type(), Type::Ref);
    }

    #[test]
    fn test_call_descr_downcasts_via_trait() {
        let descr: Arc<dyn Descr> = Arc::new(TestCallDescr {
            arg_types: vec![Type::Int],
            result_type: Type::Int,
            result_size: 8,
            result_signed: true,
            effect: EffectInfo::default(),
        });
        let cd = descr.as_call_descr().expect("should downcast to CallDescr");
        assert_eq!(cd.arg_types(), &[Type::Int]);
        assert_eq!(cd.result_type(), Type::Int);
    }

    #[test]
    fn test_call_target_token_default_none() {
        let descr = TestCallDescr {
            arg_types: vec![],
            result_type: Type::Void,
            result_size: 0,
            result_signed: false,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.call_target_token(), None);
    }

    #[test]
    fn test_effect_info_default_can_raise() {
        let ei = EffectInfo::default();
        assert_eq!(ei.extra_effect, ExtraEffect::CanRaise);
        assert_eq!(ei.oopspec_index, OopSpecIndex::None);
        assert!(ei.can_raise());
        assert!(!ei.is_elidable());
        assert!(!ei.is_loopinvariant());
    }

    #[test]
    fn test_effect_info_elidable_variants() {
        let elidable_effects = [
            ExtraEffect::ElidableCannotRaise,
            ExtraEffect::ElidableOrMemoryError,
            ExtraEffect::ElidableCanRaise,
        ];
        for effect in elidable_effects {
            let ei = EffectInfo {
                extra_effect: effect,
                oopspec_index: OopSpecIndex::None,
            };
            assert!(ei.is_elidable(), "expected elidable for {effect:?}");
        }

        let non_elidable = [
            ExtraEffect::CannotRaise,
            ExtraEffect::CanRaise,
            ExtraEffect::LoopInvariant,
            ExtraEffect::ForcesVirtualOrVirtualizable,
            ExtraEffect::RandomEffects,
        ];
        for effect in non_elidable {
            let ei = EffectInfo {
                extra_effect: effect,
                oopspec_index: OopSpecIndex::None,
            };
            assert!(!ei.is_elidable(), "expected non-elidable for {effect:?}");
        }
    }

    #[test]
    fn test_effect_info_can_raise_ordering() {
        // ExtraEffect ordering: effects >= ElidableCanRaise can raise
        let cannot_raise = [
            ExtraEffect::ElidableCannotRaise,
            ExtraEffect::LoopInvariant,
            ExtraEffect::CannotRaise,
            ExtraEffect::ElidableOrMemoryError,
        ];
        for effect in cannot_raise {
            let ei = EffectInfo {
                extra_effect: effect,
                oopspec_index: OopSpecIndex::None,
            };
            assert!(!ei.can_raise(), "expected cannot raise for {effect:?}");
        }

        let can_raise = [
            ExtraEffect::ElidableCanRaise,
            ExtraEffect::CanRaise,
            ExtraEffect::ForcesVirtualOrVirtualizable,
            ExtraEffect::RandomEffects,
        ];
        for effect in can_raise {
            let ei = EffectInfo {
                extra_effect: effect,
                oopspec_index: OopSpecIndex::None,
            };
            assert!(ei.can_raise(), "expected can raise for {effect:?}");
        }
    }

    #[test]
    fn test_effect_info_loop_invariant() {
        let ei = EffectInfo {
            extra_effect: ExtraEffect::LoopInvariant,
            oopspec_index: OopSpecIndex::None,
        };
        assert!(ei.is_loopinvariant());
        assert!(!ei.is_elidable());
        assert!(!ei.can_raise());
    }

    #[test]
    fn test_effect_info_libffi_call_oopspec() {
        // FFI calls use LibffiCall oopspec index
        let ei = EffectInfo {
            extra_effect: ExtraEffect::CanRaise,
            oopspec_index: OopSpecIndex::LibffiCall,
        };
        assert_eq!(ei.oopspec_index, OopSpecIndex::LibffiCall);
        assert!(ei.can_raise());
    }

    #[test]
    fn test_effect_info_forces_virtual() {
        // Parity: calls that force virtualizable objects
        let ei = EffectInfo {
            extra_effect: ExtraEffect::ForcesVirtualOrVirtualizable,
            oopspec_index: OopSpecIndex::JitForceVirtualizable,
        };
        assert!(ei.can_raise());
        assert!(!ei.is_elidable());
        assert_eq!(ei.oopspec_index, OopSpecIndex::JitForceVirtualizable);
    }

    #[test]
    fn test_call_release_gil_opcodes_exist() {
        use crate::resoperation::OpCode;
        // Parity with test_fficall.py: CallReleaseGil opcodes for all return types
        let int_op = OpCode::call_release_gil_for_type(Type::Int);
        assert_eq!(int_op, OpCode::CallReleaseGilI);

        let float_op = OpCode::call_release_gil_for_type(Type::Float);
        assert_eq!(float_op, OpCode::CallReleaseGilF);

        let ref_op = OpCode::call_release_gil_for_type(Type::Ref);
        assert_eq!(ref_op, OpCode::CallReleaseGilR);

        let void_op = OpCode::call_release_gil_for_type(Type::Void);
        assert_eq!(void_op, OpCode::CallReleaseGilN);
    }

    #[test]
    fn test_fail_descr_trait() {
        #[derive(Debug)]
        struct TestFailDescr {
            index: u32,
            arg_types: Vec<Type>,
        }
        impl Descr for TestFailDescr {
            fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
                Some(self)
            }
        }
        impl FailDescr for TestFailDescr {
            fn fail_index(&self) -> u32 {
                self.index
            }
            fn fail_arg_types(&self) -> &[Type] {
                &self.arg_types
            }
        }

        let fd = TestFailDescr {
            index: 7,
            arg_types: vec![Type::Int, Type::Ref],
        };
        assert_eq!(fd.fail_index(), 7);
        assert_eq!(fd.fail_arg_types(), &[Type::Int, Type::Ref]);
        assert!(!fd.is_finish());
        assert_eq!(fd.trace_id(), 0);
        // Ref slot is a GC ref
        assert!(fd.is_gc_ref_slot(1));
        // Int slot is not
        assert!(!fd.is_gc_ref_slot(0));
    }

    #[test]
    fn test_debug_merge_point_descr_repr() {
        let info = DebugMergePointInfo::new("testjit", "bytecode LOAD at 12", 12, 0);
        let descr = DebugMergePointDescr::new(info);
        let repr = descr.repr();
        assert!(repr.contains("testjit"));
        assert!(repr.contains("bytecode LOAD at 12"));
        assert!(repr.contains("pc=12"));
        assert!(repr.contains("depth=0"));
    }
}

// ── Factory functions (descr.py: get_field_descr, get_size_descr, etc.) ──

/// Create a field descriptor with the given layout.
pub fn make_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    std::sync::Arc::new(
        SimpleFieldDescr::new(0, offset, field_size, field_type, false).with_signed(signed),
    )
}

/// Create a field descriptor with explicit index and immutability.
pub fn make_field_descr_full(
    index: u32,
    offset: usize,
    field_size: usize,
    field_type: Type,
    is_immutable: bool,
) -> DescrRef {
    std::sync::Arc::new(SimpleFieldDescr::new(
        index,
        offset,
        field_size,
        field_type,
        is_immutable,
    ))
}

/// Create a size descriptor.
pub fn make_size_descr(size: usize) -> DescrRef {
    std::sync::Arc::new(SimpleSizeDescr::new(0, size, 0))
}

/// Create a size descriptor with explicit index and type_id.
pub fn make_size_descr_full(index: u32, size: usize, type_id: u32) -> DescrRef {
    std::sync::Arc::new(SimpleSizeDescr::new(index, size, type_id))
}

/// Create an array descriptor.
pub fn make_array_descr(base_size: usize, item_size: usize, item_type: Type) -> DescrRef {
    std::sync::Arc::new(SimpleArrayDescr::new(0, base_size, item_size, 0, item_type))
}

/// Create an array descriptor with explicit index and type_id.
pub fn make_array_descr_full(
    index: u32,
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
) -> DescrRef {
    std::sync::Arc::new(SimpleArrayDescr::new(
        index, base_size, item_size, type_id, item_type,
    ))
}

/// Create a call descriptor.
pub fn make_call_descr(arg_types: Vec<Type>, result_type: Type, effect: EffectInfo) -> DescrRef {
    std::sync::Arc::new(SimpleCallDescr::new(
        0,
        arg_types,
        result_type,
        match result_type {
            Type::Int | Type::Ref => 8,
            Type::Float => 8,
            Type::Void => 0,
        },
        effect,
    ))
}

/// Create a call descriptor with explicit index.
pub fn make_call_descr_full(
    index: u32,
    arg_types: Vec<Type>,
    result_type: Type,
    result_size: usize,
    effect: EffectInfo,
) -> DescrRef {
    std::sync::Arc::new(SimpleCallDescr::new(
        index,
        arg_types,
        result_type,
        result_size,
        effect,
    ))
}

/// Create a fail descriptor.
pub fn make_fail_descr(fail_index: u32, fail_arg_types: Vec<Type>) -> DescrRef {
    std::sync::Arc::new(SimpleFailDescr::new(0, fail_index, fail_arg_types))
}

/// Create a finish descriptor.
pub fn make_finish_descr(fail_index: u32, fail_arg_types: Vec<Type>) -> DescrRef {
    std::sync::Arc::new(SimpleFailDescr::finish(0, fail_index, fail_arg_types))
}

// ── descr.py: unpack helpers ──

/// descr.py: unpack_fielddescr(descr)
/// Extract offset and type from a field descriptor.
pub fn unpack_fielddescr(descr: &DescrRef) -> Option<(usize, usize, Type)> {
    let fd = descr.as_field_descr()?;
    Some((fd.offset(), fd.field_size(), fd.field_type()))
}

/// descr.py: unpack_arraydescr(descr)
/// Extract base size, item size, and type from an array descriptor.
pub fn unpack_arraydescr(descr: &DescrRef) -> Option<(usize, usize, Type)> {
    let ad = descr.as_array_descr()?;
    Some((ad.base_size(), ad.item_size(), ad.item_type()))
}

/// descr.py: unpack_interiorfielddescr(descr)
/// Extract array and field info from an interior field descriptor.
pub fn unpack_interiorfielddescr(descr: &DescrRef) -> Option<(usize, usize, usize, usize, Type)> {
    let ifd = descr.as_interior_field_descr()?;
    let ad = ifd.array_descr();
    let fd = ifd.field_descr();
    Some((
        ad.base_size(),
        ad.item_size(),
        fd.offset(),
        fd.field_size(),
        fd.field_type(),
    ))
}
