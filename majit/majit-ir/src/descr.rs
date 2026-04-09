/// Descriptor traits for the JIT IR.
///
/// Translated from rpython/jit/metainterp/history.py (AbstractDescr)
/// and rpython/jit/backend/llsupport/descr.py.
///
/// Descriptors carry type metadata needed by the optimizer and backend
/// for field access, array access, function calls, and guard failures.
use std::sync::Arc;

use crate::OpRef;
use crate::value::Type;
use serde::{Deserialize, Serialize};

/// Opaque reference to a descriptor, shared across the JIT pipeline.
pub type DescrRef = Arc<dyn Descr>;

/// backend/*/regalloc.py: LABEL/JUMP arg location payload attached to
/// TargetToken descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetArgLoc {
    Reg {
        regnum: u8,
        is_xmm: bool,
    },
    Frame {
        position: usize,
        ebp_offset: i32,
        is_float: bool,
    },
    Ebp {
        ebp_offset: i32,
        is_float: bool,
    },
    Immed {
        value: i64,
        is_float: bool,
    },
    Addr {
        base: u8,
        index: u8,
        scale: u8,
        offset: i32,
    },
}

/// history.py: TargetToken backend-visible state.
pub trait LoopTargetDescr: Descr {
    fn token_id(&self) -> u64;
    fn is_preamble_target(&self) -> bool;
    fn ll_loop_code(&self) -> usize;
    fn set_ll_loop_code(&self, loop_code: usize);
    fn target_arglocs(&self) -> Vec<TargetArgLoc>;
    fn set_target_arglocs(&self, arglocs: Vec<TargetArgLoc>);
}

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

    /// compile.py: clone() — create a subtype-preserving copy with a fresh
    /// fail_index. Returns None if this descriptor type doesn't support cloning.
    /// RPython: `olddescr.clone()` preserves the concrete type
    /// (ResumeGuardDescr, CompileLoopVersionDescr, etc.).
    fn clone_descr(&self) -> Option<DescrRef> {
        None
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
    fn as_loop_target_descr(&self) -> Option<&dyn LoopTargetDescr> {
        None
    }

    /// Whether the field/array described is always pure (immutable).
    fn is_always_pure(&self) -> bool {
        false
    }

    /// Whether the field is quasi-immutable (rarely changes but can).
    /// quasiimmut.py: fields marked _immutable_fields_ = ['x?']
    fn is_quasi_immutable(&self) -> bool {
        false
    }

    /// Whether this descriptor marks a loop version guard.
    ///
    /// Loop version guards have their alternative path compiled immediately
    /// after the main loop, rather than lazily on failure.
    fn is_loop_version(&self) -> bool {
        false
    }

    /// Whether this descriptor refers to a virtualizable field.
    ///
    /// Virtualizable fields (e.g. linked-list head/size) are not force-emitted
    /// at guards; they go into pendingfields instead, matching RPython's
    /// treatment of virtualizable fields in force_lazy_sets_for_guard.
    fn is_virtualizable(&self) -> bool {
        false
    }

    /// compile.py: isinstance(resumekey, ResumeAtPositionDescr).
    /// Guards created during loop unrolling / short preamble inlining
    /// return true. When bridge compilation starts from such a guard,
    /// inline_short_preamble is set to false.
    fn is_resume_at_position(&self) -> bool {
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

    /// history.py:137-139: exits_early()
    /// Is this guard a guard_early_exit or moved before one?
    fn exits_early(&self) -> bool {
        false
    }

    /// history.py:141-143: loop_version()
    /// Should a loop version be compiled out of this guard?
    fn loop_version(&self) -> bool {
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

    /// compile.py:741-745: read status for must_compile.
    fn get_status(&self) -> u64 {
        0
    }

    /// compile.py:786-788: start_compiling — set ST_BUSY_FLAG.
    fn start_compiling(&self) {}

    /// compile.py:790-795: done_compiling — clear ST_BUSY_FLAG.
    fn done_compiling(&self) {}

    /// compile.py:750: check ST_BUSY_FLAG.
    fn is_compiling(&self) -> bool {
        false
    }

    /// history.py:143-147 / schedule.py:654-655 — attach vector resume info
    /// to a guard descriptor. Non-guard fail descriptors ignore this.
    fn attach_vector_info(&self, _info: AccumVectorInfo) {}

    /// Read back any attached vector resume info.
    fn vector_info(&self) -> Vec<AccumVectorInfo> {
        Vec::new()
    }
}

/// resume.py:65-80: AccumInfo — metadata attached to guard descriptors
/// so deoptimization can reconstruct vector accumulators.
///
/// Two distinct OpRefs following RPython's separation:
///   - `variable`: original scalar accumulator (resume.py:29 getoriginal(),
///     used for type inference)
///   - `vector_loc`: vector SSA result holding the accumulated vector
///     (regalloc.py:350 accuminfo.location, used by backend for lane reduction)
#[derive(Debug, Clone)]
pub struct AccumVectorInfo {
    pub failargs_pos: usize,
    /// resume.py:29: the original scalar variable (getoriginal()).
    pub variable: OpRef,
    /// regalloc.py:350: vector register/SSA where the accumulated vector lives.
    /// Backend reads this for extractlane + reduction at guard exit.
    pub vector_loc: OpRef,
    pub operator: char,
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
        format!(
            "SizeDescr(size={}, type_id={})",
            self.size(),
            self.type_id()
        )
    }

    /// Field descriptors for fields containing GC pointers.
    fn gc_field_descrs(&self) -> &[Arc<dyn FieldDescr>] {
        &[]
    }

    /// All field descriptors (not just GC pointer ones).
    /// descr.py: get_all_interiorfielddescrs()
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
    /// descr.py / FieldDescr.get_parent_descr() — the SizeDescr of the
    /// containing struct/object that owns this field. PyPy
    /// optimizer.py:478-484 reads this in `ensure_ptr_info_arg0` to
    /// dispatch a fresh GETFIELD/SETFIELD/QUASIIMMUT_FIELD opinfo to
    /// `InstancePtrInfo` (when `parent_descr.is_object()`) or
    /// `StructPtrInfo` (otherwise). Default returns `None`; field
    /// descriptors that don't carry a backreference fall through to the
    /// generic path and the Rust port's `ensure_ptr_info_arg0` panics
    /// rather than installing a malformed PtrInfo.
    ///
    /// Returns a `DescrRef` (rather than `Arc<dyn SizeDescr>`) so callers
    /// can store it in `PtrInfo::Instance.descr` / `PtrInfo::Struct.descr`
    /// directly. The caller reads `parent.as_size_descr().is_object()` to
    /// pick the Instance vs Struct constructor.
    fn parent_descr(&self) -> Option<DescrRef> {
        None
    }

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

    /// descr.py:227 — field name. Format is either:
    /// - `"STRUCT.fieldname"` (from codewriter: descr.py:227)
    /// - `"typeptr"` (from pyre tracer: ob_type_descr)
    /// - `""` (unnamed/dynamic field descriptors)
    fn field_name(&self) -> &str {
        ""
    }

    /// heaptracker.py:66: `if name == 'typeptr': continue`
    ///
    /// RPython filters typeptr by raw field name BEFORE creating
    /// descriptors (heaptracker.py:60-67). In majit, descriptors are
    /// already created, so we check the name at use time.
    ///
    /// Handles both formats:
    /// - `"typeptr"` (pyre tracer ob_type_descr)
    /// - `"STRUCT.typeptr"` (codewriter format, descr.py:227)
    fn is_typeptr(&self) -> bool {
        let name = self.field_name();
        name == "typeptr" || name.ends_with(".typeptr")
    }

    /// descr.py: sort_key() — for ordering field descriptors.
    fn sort_key(&self) -> usize {
        self.offset()
    }
}

/// RPython: descr.py FLAG_* constants for array element type classification.
///
/// ```python
/// FLAG_POINTER  = 'P'  # GC pointer (Ptr to gc obj)
/// FLAG_FLOAT    = 'F'  # Float or longlong
/// FLAG_UNSIGNED = 'U'  # Unsigned integer
/// FLAG_SIGNED   = 'S'  # Signed integer
/// FLAG_STRUCT   = 'X'  # Inline struct (array-of-structs)
/// FLAG_VOID     = 'V'  # Void
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArrayFlag {
    /// RPython: FLAG_POINTER = 'P'
    Pointer,
    /// RPython: FLAG_FLOAT = 'F'
    Float,
    /// RPython: FLAG_UNSIGNED = 'U'
    Unsigned,
    /// RPython: FLAG_SIGNED = 'S'
    Signed,
    /// RPython: FLAG_STRUCT = 'X'
    Struct,
    /// RPython: FLAG_VOID = 'V'
    Void,
}

impl ArrayFlag {
    /// RPython: get_type_flag(TYPE) (descr.py:241-254).
    ///
    /// When only the IR type is known (no concrete Rust type string),
    /// `Type::Int` maps to `Unsigned` — RPython's default for unknown
    /// integer types (descr.py:254: `return FLAG_UNSIGNED`).
    /// Use `get_type_flag()` in call.rs for precise signed/unsigned
    /// classification from concrete type names.
    pub fn from_item_type(item_type: Type, is_struct: bool) -> Self {
        if is_struct {
            return ArrayFlag::Struct;
        }
        match item_type {
            Type::Ref => ArrayFlag::Pointer,
            Type::Float => ArrayFlag::Float,
            // RPython: default for unresolved integer type is FLAG_UNSIGNED
            // (descr.py:254). Callers with concrete type info should use
            // get_type_flag() for FLAG_SIGNED/FLAG_UNSIGNED distinction.
            Type::Int => ArrayFlag::Unsigned,
            Type::Void => ArrayFlag::Void,
        }
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
    /// descr.py: is_array_of_structs() → self.flag == FLAG_STRUCT
    fn is_array_of_structs(&self) -> bool {
        false
    }

    /// descr.py:291 ArrayDescr.get_all_fielddescrs() →
    /// all_interiorfielddescrs. For array-of-structs, returns
    /// interior field descriptors.
    fn get_all_interiorfielddescrs(&self) -> Option<&[DescrRef]> {
        None
    }

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!(
            "ArrayDescr(base={}, item={}, type={:?})",
            self.base_size(),
            self.item_size(),
            self.item_type()
        )
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

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!(
            "CallDescr(args={}, result={:?})",
            self.arg_classes(),
            self.result_type()
        )
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectInfo {
    pub extra_effect: ExtraEffect,
    pub oopspec_index: OopSpecIndex,
    /// effectinfo.py: bitstring_readonly_descrs_fields
    pub readonly_descrs_fields: u64,
    /// effectinfo.py: bitstring_write_descrs_fields
    pub write_descrs_fields: u64,
    /// effectinfo.py: bitstring_readonly_descrs_arrays
    pub readonly_descrs_arrays: u64,
    /// effectinfo.py: bitstring_write_descrs_arrays
    pub write_descrs_arrays: u64,
    /// effectinfo.py: bitstring_readonly_descrs_interiorfields
    /// Bitset of interior field descriptor indices that may be read.
    /// effectinfo.py:327-340: interiorfield reads also set array read bits.
    pub readonly_descrs_interiorfields: u64,
    /// effectinfo.py: bitstring_write_descrs_interiorfields
    pub write_descrs_interiorfields: u64,
    /// effectinfo.py: can_invalidate
    pub can_invalidate: bool,
    /// effectinfo.py:194: can_collect — whether this call can trigger GC collection.
    /// RPython: set by collect_analyzer.analyze(op, self.seen_gc).
    pub can_collect: bool,
    /// effectinfo.py:201-206: single_write_descr_array
    #[serde(skip)]
    pub single_write_descr_array: Option<DescrRef>,
}

/// Manual PartialEq: single_write_descr_array is excluded (like RPython's
/// cache key which also excludes it — effectinfo.py:155-164).
impl PartialEq for EffectInfo {
    fn eq(&self, other: &Self) -> bool {
        self.extra_effect == other.extra_effect
            && self.oopspec_index == other.oopspec_index
            && self.readonly_descrs_fields == other.readonly_descrs_fields
            && self.write_descrs_fields == other.write_descrs_fields
            && self.readonly_descrs_arrays == other.readonly_descrs_arrays
            && self.write_descrs_arrays == other.write_descrs_arrays
            && self.readonly_descrs_interiorfields == other.readonly_descrs_interiorfields
            && self.write_descrs_interiorfields == other.write_descrs_interiorfields
            && self.can_invalidate == other.can_invalidate
            && self.can_collect == other.can_collect
    }
}

impl Eq for EffectInfo {}

impl Default for EffectInfo {
    fn default() -> Self {
        EffectInfo {
            extra_effect: ExtraEffect::CanRaise,
            oopspec_index: OopSpecIndex::None,
            readonly_descrs_fields: 0,
            write_descrs_fields: 0,
            readonly_descrs_arrays: 0,
            write_descrs_arrays: 0,
            readonly_descrs_interiorfields: 0,
            write_descrs_interiorfields: 0,
            single_write_descr_array: None,
            can_invalidate: false,
            // RPython effectinfo.py:125: can_collect=True default
            can_collect: true,
        }
    }
}

/// How a call affects the optimizer's ability to optimize surrounding code.
///
/// Ordered from most optimizable to least.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    /// effectinfo.py: check_can_collect() — whether this call can trigger GC collection.
    /// RPython: stored as field from collect_analyzer result.
    pub fn can_collect(&self) -> bool {
        self.can_collect
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
            readonly_descrs_interiorfields: 0,
            write_descrs_interiorfields: 0,
            single_write_descr_array: None,
            can_invalidate: false,
            can_collect: true,
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

    /// effectinfo.py:223-224: check_readonly_descr_interiorfield
    /// NOTE: this is not used so far (matches RPython comment)
    pub fn check_readonly_descr_interiorfield(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.readonly_descrs_interiorfields & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py:227-228: check_write_descr_interiorfield
    /// NOTE: this is not used so far (matches RPython comment)
    pub fn check_write_descr_interiorfield(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.write_descrs_interiorfields & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py: check_is_elidable()
    pub fn check_is_elidable(&self) -> bool {
        self.is_elidable()
    }

    /// effectinfo.py:201-206: set single_write_descr_array.
    ///
    /// Builder: attaches the actual array DescrRef for ARRAYCOPY/ARRAYMOVE
    /// unrolling. RPython sets this in `EffectInfo.__new__()` when
    /// `_write_descrs_arrays` has exactly one element.
    pub fn with_single_write_descr_array(mut self, descr: DescrRef) -> Self {
        self.single_write_descr_array = Some(descr);
        self
    }

    /// effectinfo.py:201-206: auto-set single_write_descr_array.
    ///
    /// If `write_descrs_arrays` has exactly one bit set and a matching
    /// array DescrRef is provided, store it for ARRAYCOPY/ARRAYMOVE.
    pub fn set_single_write_descr_array(&mut self, descr: DescrRef) {
        let w = self.write_descrs_arrays;
        if w != 0 && w.is_power_of_two() {
            self.single_write_descr_array = Some(descr);
        }
    }
}

// ── Concrete descriptor implementations (descr.py) ──

/// Simple concrete FieldDescr for use by pyre-jit and tests.
/// RPython: `FieldDescr(name, offset, size, flag, index_in_parent, is_pure)`.
#[derive(Debug, Clone)]
pub struct SimpleFieldDescr {
    index: u32,
    /// RPython: FieldDescr.name — e.g. "MyStruct.field_name"
    name: String,
    offset: usize,
    field_size: usize,
    field_type: Type,
    is_immutable: bool,
    is_signed: bool,
    virtualizable: bool,
    /// descr.py: FieldDescr.parent_descr — backreference to the SizeDescr
    /// of the containing struct/object. Required by
    /// `OptContext::ensure_ptr_info_arg0` to dispatch Instance vs Struct
    /// PtrInfo per `optimizer.py:478-484`.
    parent_descr: Option<DescrRef>,
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
            name: String::new(),
            offset,
            field_size,
            field_type,
            is_immutable,
            is_signed: true,
            virtualizable: false,
            parent_descr: None,
        }
    }

    /// RPython: FieldDescr(name, offset, size, flag, index_in_parent, is_pure).
    /// `name` format: `"STRUCT.fieldname"` (descr.py:227).
    /// `is_signed`: RPython flag == FLAG_SIGNED (descr.py:241-254).
    pub fn new_with_name(
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_immutable: bool,
        is_signed: bool,
        name: String,
    ) -> Self {
        SimpleFieldDescr {
            index,
            name,
            offset,
            field_size,
            field_type,
            is_immutable,
            is_signed,
            virtualizable: false,
            parent_descr: None,
        }
    }

    pub fn with_signed(mut self, signed: bool) -> Self {
        self.is_signed = signed;
        self
    }

    pub fn with_virtualizable(mut self, virtualizable: bool) -> Self {
        self.virtualizable = virtualizable;
        self
    }

    /// Builder: attach a parent SizeDescr backreference. Required when the
    /// descriptor will be used as the `op.descr` of a GETFIELD/SETFIELD/
    /// QUASIIMMUT_FIELD that flows through `ensure_ptr_info_arg0`.
    pub fn with_parent_descr(mut self, parent: DescrRef) -> Self {
        self.parent_descr = Some(parent);
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
    fn is_virtualizable(&self) -> bool {
        self.virtualizable
    }
    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }
}

impl FieldDescr for SimpleFieldDescr {
    fn parent_descr(&self) -> Option<DescrRef> {
        self.parent_descr.clone()
    }
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
    fn field_name(&self) -> &str {
        &self.name
    }
}

/// Simple concrete SizeDescr.
#[derive(Debug, Clone)]
pub struct SimpleSizeDescr {
    index: u32,
    size: usize,
    type_id: u32,
    is_immutable: bool,
    vtable: usize,
}

impl SimpleSizeDescr {
    pub fn new(index: u32, size: usize, type_id: u32) -> Self {
        SimpleSizeDescr {
            index,
            size,
            type_id,
            is_immutable: false,
            vtable: 0,
        }
    }

    pub fn with_vtable(index: u32, size: usize, type_id: u32, vtable: usize) -> Self {
        SimpleSizeDescr {
            index,
            size,
            type_id,
            is_immutable: false,
            vtable,
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
    fn is_object(&self) -> bool {
        self.vtable != 0
    }
    fn vtable(&self) -> usize {
        self.vtable
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
    /// RPython: descr.py ArrayDescr.flag — element type classification.
    flag: ArrayFlag,
    /// RPython: descr.py ArrayDescr.all_interiorfielddescrs.
    /// For array-of-structs, contains interior field descriptors.
    all_interiorfielddescrs: Option<Vec<DescrRef>>,
}

impl SimpleArrayDescr {
    pub fn new(
        index: u32,
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
    ) -> Self {
        let flag = ArrayFlag::from_item_type(item_type, false);
        SimpleArrayDescr {
            index,
            base_size,
            item_size,
            type_id,
            item_type,
            flag,
            all_interiorfielddescrs: None,
        }
    }

    /// RPython: ArrayDescr with explicit flag (for struct arrays).
    pub fn with_flag(
        index: u32,
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
        flag: ArrayFlag,
    ) -> Self {
        SimpleArrayDescr {
            index,
            base_size,
            item_size,
            type_id,
            item_type,
            flag,
            all_interiorfielddescrs: None,
        }
    }

    /// RPython: arraydescr.all_interiorfielddescrs = descrs
    pub fn set_all_interiorfielddescrs(&mut self, descrs: Vec<DescrRef>) {
        self.all_interiorfielddescrs = Some(descrs);
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
        self.flag == ArrayFlag::Signed
    }
    /// RPython: descr.py ArrayDescr.is_array_of_pointers()
    fn is_array_of_pointers(&self) -> bool {
        self.flag == ArrayFlag::Pointer
    }
    /// RPython: descr.py ArrayDescr.is_array_of_floats()
    fn is_array_of_floats(&self) -> bool {
        self.flag == ArrayFlag::Float
    }
    /// RPython: descr.py ArrayDescr.is_array_of_structs()
    fn is_array_of_structs(&self) -> bool {
        self.flag == ArrayFlag::Struct
    }
    /// RPython: descr.py ArrayDescr.is_array_of_primitives()
    fn is_array_of_primitives(&self) -> bool {
        matches!(
            self.flag,
            ArrayFlag::Float | ArrayFlag::Signed | ArrayFlag::Unsigned
        )
    }
    /// RPython: descr.py ArrayDescr.get_all_interiorfielddescrs()
    fn get_all_interiorfielddescrs(&self) -> Option<&[DescrRef]> {
        self.all_interiorfielddescrs.as_deref()
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
#[derive(Debug)]
pub struct SimpleFailDescr {
    index: u32,
    fail_index: u32,
    fail_arg_types: Vec<Type>,
    is_finish: bool,
    trace_id: u64,
    /// schedule.py:654: vector accumulation info attached during vectorization.
    vector_info: std::cell::UnsafeCell<Vec<AccumVectorInfo>>,
}

impl Clone for SimpleFailDescr {
    fn clone(&self) -> Self {
        SimpleFailDescr {
            index: self.index,
            fail_index: self.fail_index,
            fail_arg_types: self.fail_arg_types.clone(),
            is_finish: self.is_finish,
            trace_id: self.trace_id,
            vector_info: std::cell::UnsafeCell::new(unsafe { (&*self.vector_info.get()).clone() }),
        }
    }
}

// Safety: JIT is single-threaded (RPython GIL equivalent). UnsafeCell
// replaces Mutex for rd_vector_info — no concurrent access.
unsafe impl Send for SimpleFailDescr {}
unsafe impl Sync for SimpleFailDescr {}

impl SimpleFailDescr {
    pub fn new(index: u32, fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        SimpleFailDescr {
            index,
            fail_index,
            fail_arg_types,
            is_finish: false,
            trace_id: 0,
            vector_info: std::cell::UnsafeCell::new(Vec::new()),
        }
    }

    pub fn finish(index: u32, fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        SimpleFailDescr {
            index,
            fail_index,
            fail_arg_types,
            is_finish: true,
            trace_id: 0,
            vector_info: std::cell::UnsafeCell::new(Vec::new()),
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
    fn attach_vector_info(&self, info: AccumVectorInfo) {
        unsafe { &mut *self.vector_info.get() }.push(info);
    }
    fn vector_info(&self) -> Vec<AccumVectorInfo> {
        unsafe { &mut *self.vector_info.get() }.clone()
    }
}

/// effectinfo.py: CallInfoCollection — maps oopspec indices to
/// (calldescr, func_ptr) pairs. Used to look up the implementation
/// of special-cased operations (arraycopy, string ops, etc.).
#[derive(Debug, Clone, Default)]
pub struct CallInfoCollection {
    /// RPython: `_callinfo_for_oopspec` — {oopspecindex: (calldescr, func_as_int)}.
    entries: std::collections::HashMap<OopSpecIndex, (DescrRef, u64)>,
    /// majit extension: func_as_int → function name.
    /// RPython derives names from `func.ptr._obj._name` at `see_raw_object` time.
    /// Since majit has no function pointers, we store the name at `add()` time.
    func_names: std::collections::HashMap<u64, String>,
}

impl CallInfoCollection {
    pub fn new() -> Self {
        Self::default()
    }

    /// effectinfo.py: add(oopspecindex, calldescr, func_as_int)
    pub fn add(&mut self, oopspec: OopSpecIndex, calldescr: DescrRef, func_addr: u64) {
        self.entries.insert(oopspec, (calldescr, func_addr));
    }

    /// Register the name for a function address.
    /// RPython: the name is `func.ptr._obj._name`, extracted by `see_raw_object`.
    /// In majit, we must store it explicitly since we have no pointer linkage.
    pub fn register_func_name(&mut self, func_addr: u64, name: String) {
        self.func_names.insert(func_addr, name);
    }

    /// effectinfo.py: has_oopspec(oopspecindex)
    pub fn has_oopspec(&self, oopspec: OopSpecIndex) -> bool {
        self.entries.contains_key(&oopspec)
    }

    /// Get the calldescr for an oopspec.
    pub fn get(&self, oopspec: OopSpecIndex) -> Option<&(DescrRef, u64)> {
        self.entries.get(&oopspec)
    }

    /// effectinfo.py: all_function_addresses_as_int()
    pub fn all_function_addresses(&self) -> Vec<u64> {
        self.entries.values().map(|(_, addr)| *addr).collect()
    }

    /// Look up function name by address.
    /// RPython: `see_raw_object(func.ptr)` derives name from `func.ptr._obj._name`.
    pub fn func_name(&self, addr: u64) -> Option<&str> {
        self.func_names.get(&addr).map(String::as_str)
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
            };
            assert!(ei.can_raise(), "expected can raise for {effect:?}");
        }
    }

    #[test]
    fn test_effect_info_loop_invariant() {
        let ei = EffectInfo {
            extra_effect: ExtraEffect::LoopInvariant,
            oopspec_index: OopSpecIndex::None,
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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

/// Create a size descriptor with vtable (for NEW_WITH_VTABLE objects).
pub fn make_size_descr_with_vtable(
    index: u32,
    size: usize,
    type_id: u32,
    vtable: usize,
) -> DescrRef {
    std::sync::Arc::new(SimpleSizeDescr::with_vtable(index, size, type_id, vtable))
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
