/// JIT IR operations, faithfully translated from rpython/jit/metainterp/resoperation.py.
///
/// Operations with multiple result types (e.g., SAME_AS/1/ifr) are expanded
/// into type-suffixed variants (SameAsI, SameAsR, SameAsF).
///
/// Naming convention: CamelCase variant name, with type suffix I/R/F/N where applicable.
use smallvec::SmallVec;

use crate::descr::DescrRef;
use crate::value::Type;

/// Index into an operation list, used as a reference to an operation's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpRef(pub u32);

impl OpRef {
    pub const NONE: OpRef = OpRef(u32::MAX);

    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }
}

/// A single IR operation.
#[derive(Clone, Debug)]
pub struct Op {
    pub opcode: OpCode,
    pub args: SmallVec<[OpRef; 3]>,
    pub descr: Option<DescrRef>,
    /// Index of this op in the trace (set by the trace builder).
    pub pos: OpRef,
}

impl Op {
    pub fn new(opcode: OpCode, args: &[OpRef]) -> Self {
        Op {
            opcode,
            args: SmallVec::from_slice(args),
            descr: None,
            pos: OpRef::NONE,
        }
    }

    pub fn with_descr(opcode: OpCode, args: &[OpRef], descr: DescrRef) -> Self {
        Op {
            opcode,
            args: SmallVec::from_slice(args),
            descr: Some(descr),
            pos: OpRef::NONE,
        }
    }

    pub fn arg(&self, idx: usize) -> OpRef {
        self.args[idx]
    }

    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    pub fn result_type(&self) -> Type {
        self.opcode.result_type()
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.opcode.is_guard() {
            write!(f, "{:?}(", self.opcode)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "v{}", arg.0)?;
            }
            write!(f, ")")
        } else if self.opcode.result_type() != Type::Void {
            write!(f, "v{} = {:?}(", self.pos.0, self.opcode)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "v{}", arg.0)?;
            }
            write!(f, ")")
        } else {
            write!(f, "{:?}(", self.opcode)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "v{}", arg.0)?;
            }
            write!(f, ")")
        }
    }
}

/// Format a trace (list of ops) with optional constants for debugging.
pub fn format_trace(ops: &[Op], constants: &std::collections::HashMap<u32, i64>) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for op in ops {
        // Replace known constants in display
        write!(out, "  ").unwrap();
        if op.opcode.is_guard() {
            write!(out, "{:?}(", op.opcode).unwrap();
        } else if op.opcode.result_type() != Type::Void {
            write!(out, "v{} = {:?}(", op.pos.0, op.opcode).unwrap();
        } else {
            write!(out, "{:?}(", op.opcode).unwrap();
        }
        for (i, arg) in op.args.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            if let Some(&val) = constants.get(&arg.0) {
                write!(out, "{val}").unwrap();
            } else {
                write!(out, "v{}", arg.0).unwrap();
            }
        }
        writeln!(out, ")").unwrap();
    }
    out
}

/// All JIT IR opcodes.
///
/// Faithfully mirrors rpython/jit/metainterp/resoperation.py `_oplist`.
/// Operations that produce typed results are expanded with suffixes:
///   _I (int), _R (ref/pointer), _F (float), _N (void/none)
///
/// Boundary markers (e.g., _GUARD_FIRST) are not included as enum variants;
/// instead, classification is done via methods on OpCode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum OpCode {
    // ── Final ──
    Jump = 0,
    Finish,

    Label,

    // ── Guards (foldable) ──
    GuardTrue,
    GuardFalse,
    VecGuardTrue,
    VecGuardFalse,
    GuardValue,
    GuardClass,
    GuardNonnull,
    GuardIsnull,
    GuardNonnullClass,
    GuardGcType,
    GuardIsObject,
    GuardSubclass,
    // ── Guards (non-foldable) ──
    GuardNoException,
    GuardException,
    GuardNoOverflow,
    GuardOverflow,
    GuardNotForced,
    GuardNotForced2,
    GuardNotInvalidated,
    GuardFutureCondition,
    GuardAlwaysFails,

    // ── Always pure: integer arithmetic ──
    IntAdd,
    IntSub,
    IntMul,
    UintMulHigh,
    IntAnd,
    IntOr,
    IntXor,
    IntRshift,
    IntLshift,
    UintRshift,
    IntSignext,

    // ── Always pure: float arithmetic ──
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatTrueDiv,
    FloatNeg,
    FloatAbs,

    // ── Always pure: casts ──
    CastFloatToInt,
    CastIntToFloat,
    CastFloatToSinglefloat,
    CastSinglefloatToFloat,
    ConvertFloatBytesToLonglong,
    ConvertLonglongBytesToFloat,

    // ── Always pure: vector arithmetic ──
    VecIntAdd,
    VecIntSub,
    VecIntMul,
    VecIntAnd,
    VecIntOr,
    VecIntXor,
    VecFloatAdd,
    VecFloatSub,
    VecFloatMul,
    VecFloatTrueDiv,
    VecFloatNeg,
    VecFloatAbs,

    // ── Always pure: vector comparisons / casts ──
    VecFloatEq,
    VecFloatNe,
    VecFloatXor,
    VecIntIsTrue,
    VecIntNe,
    VecIntEq,
    VecIntSignext,
    VecCastFloatToSinglefloat,
    VecCastSinglefloatToFloat,
    VecCastFloatToInt,
    VecCastIntToFloat,

    // ── Always pure: vector pack/unpack ──
    VecI,
    VecF,
    VecUnpackI,
    VecUnpackF,
    VecPackI,
    VecPackF,
    VecExpandI,
    VecExpandF,

    // ── Always pure: integer comparisons ──
    IntLt,
    IntLe,
    IntEq,
    IntNe,
    IntGt,
    IntGe,
    UintLt,
    UintLe,
    UintGt,
    UintGe,

    // ── Always pure: float comparisons ──
    FloatLt,
    FloatLe,
    FloatEq,
    FloatNe,
    FloatGt,
    FloatGe,

    // ── Always pure: unary int ──
    IntIsZero,
    IntIsTrue,
    IntNeg,
    IntInvert,
    IntForceGeZero,

    // ── Always pure: identity / cast ──
    SameAsI,
    SameAsR,
    SameAsF,
    CastPtrToInt,
    CastIntToPtr,

    // ── Always pure: pointer comparisons ──
    PtrEq,
    PtrNe,
    InstancePtrEq,
    InstancePtrNe,
    NurseryPtrIncrement,

    // ── Always pure: array/string length, getitem ──
    ArraylenGc,
    Strlen,
    Strgetitem,
    GetarrayitemGcPureI,
    GetarrayitemGcPureR,
    GetarrayitemGcPureF,
    Unicodelen,
    Unicodegetitem,

    // ── Always pure: backend-specific loads ──
    LoadFromGcTable,
    LoadEffectiveAddress,

    // ── No side effect (but not always pure) ──
    GcLoadI,
    GcLoadR,
    GcLoadF,
    GcLoadIndexedI,
    GcLoadIndexedR,
    GcLoadIndexedF,

    // ── Raw loads ──
    GetarrayitemGcI,
    GetarrayitemGcR,
    GetarrayitemGcF,
    GetarrayitemRawI,
    GetarrayitemRawF,
    RawLoadI,
    RawLoadF,
    VecLoadI,
    VecLoadF,

    // ── No side effect: field/interior access ──
    GetinteriorfieldGcI,
    GetinteriorfieldGcR,
    GetinteriorfieldGcF,
    GetfieldGcI,
    GetfieldGcR,
    GetfieldGcF,
    GetfieldRawI,
    GetfieldRawR,
    GetfieldRawF,

    // ── No side effect: pure field access (immutable) ──
    GetfieldGcPureI,
    GetfieldGcPureR,
    GetfieldGcPureF,

    // ── Allocation ──
    New,
    NewWithVtable,
    NewArray,
    NewArrayClear,
    Newstr,
    Newunicode,

    // ── No side effect: misc ──
    ForceToken,
    VirtualRefI,
    VirtualRefR,
    Strhash,
    Unicodehash,

    // ── Side effects: GC stores ──
    GcStore,
    GcStoreIndexed,

    // ── Side effects: misc ──
    IncrementDebugCounter,

    // ── Raw stores ──
    SetarrayitemGc,
    SetarrayitemRaw,
    RawStore,
    VecStore,

    // ── Side effects: field/interior stores ──
    SetinteriorfieldGc,
    SetinteriorfieldRaw,
    SetfieldGc,
    ZeroArray,
    SetfieldRaw,
    Strsetitem,
    Unicodesetitem,

    // ── GC write barriers ──
    CondCallGcWb,
    CondCallGcWbArray,

    // ── Debug ──
    DebugMergePoint,
    EnterPortalFrame,
    LeavePortalFrame,
    JitDebug,

    // ── Testing only ──
    EscapeI,
    EscapeR,
    EscapeF,
    EscapeN,
    ForceSpill,

    // ── Misc side effects ──
    VirtualRefFinish,
    Copystrcontent,
    Copyunicodecontent,
    QuasiimmutField,
    AssertNotNone,
    RecordExactClass,
    RecordExactValueR,
    RecordExactValueI,
    Keepalive,
    SaveException,
    SaveExcClass,
    RestoreException,

    // ── Calls (can raise) ──
    CallI,
    CallR,
    CallF,
    CallN,
    CondCallN,
    CondCallValueI,
    CondCallValueR,
    CallAssemblerI,
    CallAssemblerR,
    CallAssemblerF,
    CallAssemblerN,
    CallMayForceI,
    CallMayForceR,
    CallMayForceF,
    CallMayForceN,
    CallLoopinvariantI,
    CallLoopinvariantR,
    CallLoopinvariantF,
    CallLoopinvariantN,
    CallReleaseGilI,
    CallReleaseGilF,
    CallReleaseGilN,
    CallPureI,
    CallPureR,
    CallPureF,
    CallPureN,
    CheckMemoryError,
    CallMallocNursery,
    CallMallocNurseryVarsize,
    CallMallocNurseryVarsizeFrame,
    RecordKnownResult,

    // ── Overflow ──
    IntAddOvf,
    IntSubOvf,
    IntMulOvf,
}

// ── Boundary constants for category classification ──
// These correspond to the _FIRST/_LAST markers in resoperation.py.

const FINAL_FIRST: u16 = OpCode::Jump as u16;
const FINAL_LAST: u16 = OpCode::Finish as u16;

const GUARD_FIRST: u16 = OpCode::GuardTrue as u16;
const GUARD_FOLDABLE_FIRST: u16 = OpCode::GuardTrue as u16;
const GUARD_FOLDABLE_LAST: u16 = OpCode::GuardSubclass as u16;
const GUARD_LAST: u16 = OpCode::GuardAlwaysFails as u16;

const ALWAYS_PURE_FIRST: u16 = OpCode::IntAdd as u16;
const ALWAYS_PURE_LAST: u16 = OpCode::LoadEffectiveAddress as u16;

const NOSIDEEFFECT_FIRST: u16 = OpCode::IntAdd as u16; // same as ALWAYS_PURE_FIRST
const NOSIDEEFFECT_LAST: u16 = OpCode::Unicodehash as u16;

const MALLOC_FIRST: u16 = OpCode::New as u16;
const MALLOC_LAST: u16 = OpCode::Newunicode as u16;

const RAW_LOAD_FIRST: u16 = OpCode::GetarrayitemGcI as u16;
const RAW_LOAD_LAST: u16 = OpCode::VecLoadF as u16;

const RAW_STORE_FIRST: u16 = OpCode::SetarrayitemGc as u16;
const RAW_STORE_LAST: u16 = OpCode::VecStore as u16;

const JIT_DEBUG_FIRST: u16 = OpCode::DebugMergePoint as u16;
const JIT_DEBUG_LAST: u16 = OpCode::JitDebug as u16;

const CALL_FIRST: u16 = OpCode::CallI as u16;
const CALL_LAST: u16 = OpCode::RecordKnownResult as u16;

const CANRAISE_FIRST: u16 = OpCode::CallI as u16;
const CANRAISE_LAST: u16 = OpCode::IntMulOvf as u16;

const OVF_FIRST: u16 = OpCode::IntAddOvf as u16;
const OVF_LAST: u16 = OpCode::IntMulOvf as u16;

impl OpCode {
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    // ── Category classification (mirrors rop.is_* static methods) ──

    pub fn is_final(self) -> bool {
        let n = self.as_u16();
        FINAL_FIRST <= n && n <= FINAL_LAST
    }

    pub fn is_guard(self) -> bool {
        let n = self.as_u16();
        GUARD_FIRST <= n && n <= GUARD_LAST
    }

    pub fn is_foldable_guard(self) -> bool {
        let n = self.as_u16();
        GUARD_FOLDABLE_FIRST <= n && n <= GUARD_FOLDABLE_LAST
    }

    pub fn is_always_pure(self) -> bool {
        let n = self.as_u16();
        ALWAYS_PURE_FIRST <= n && n <= ALWAYS_PURE_LAST
    }

    pub fn has_no_side_effect(self) -> bool {
        let n = self.as_u16();
        NOSIDEEFFECT_FIRST <= n && n <= NOSIDEEFFECT_LAST
    }

    pub fn is_malloc(self) -> bool {
        let n = self.as_u16();
        MALLOC_FIRST <= n && n <= MALLOC_LAST
    }

    pub fn is_call(self) -> bool {
        let n = self.as_u16();
        CALL_FIRST <= n && n <= CALL_LAST
    }

    pub fn can_raise(self) -> bool {
        let n = self.as_u16();
        CANRAISE_FIRST <= n && n <= CANRAISE_LAST
    }

    pub fn can_malloc(self) -> bool {
        self.is_call() || self.is_malloc()
    }

    pub fn is_ovf(self) -> bool {
        let n = self.as_u16();
        OVF_FIRST <= n && n <= OVF_LAST
    }

    pub fn is_raw_load(self) -> bool {
        let n = self.as_u16();
        RAW_LOAD_FIRST < n && n < RAW_LOAD_LAST
    }

    pub fn is_raw_store(self) -> bool {
        let n = self.as_u16();
        RAW_STORE_FIRST < n && n < RAW_STORE_LAST
    }

    pub fn is_jit_debug(self) -> bool {
        let n = self.as_u16();
        JIT_DEBUG_FIRST <= n && n <= JIT_DEBUG_LAST
    }

    pub fn is_comparison(self) -> bool {
        self.is_always_pure() && self.returns_bool()
    }

    pub fn is_guard_exception(self) -> bool {
        matches!(self, OpCode::GuardException | OpCode::GuardNoException)
    }

    pub fn is_guard_overflow(self) -> bool {
        matches!(self, OpCode::GuardOverflow | OpCode::GuardNoOverflow)
    }

    pub fn is_same_as(self) -> bool {
        matches!(self, OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF)
    }

    pub fn is_getfield(self) -> bool {
        matches!(
            self,
            OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF
        )
    }

    pub fn is_getarrayitem(self) -> bool {
        matches!(
            self,
            OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
        )
    }

    pub fn is_plain_call(self) -> bool {
        matches!(
            self,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        )
    }

    pub fn is_call_assembler(self) -> bool {
        matches!(
            self,
            OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN
        )
    }

    pub fn is_call_may_force(self) -> bool {
        matches!(
            self,
            OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN
        )
    }

    pub fn is_call_pure(self) -> bool {
        matches!(
            self,
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN
        )
    }

    pub fn is_call_release_gil(self) -> bool {
        matches!(
            self,
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN
        )
    }

    pub fn is_call_loopinvariant(self) -> bool {
        matches!(
            self,
            OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        )
    }

    pub fn is_cond_call_value(self) -> bool {
        matches!(self, OpCode::CondCallValueI | OpCode::CondCallValueR)
    }

    pub fn is_label(self) -> bool {
        matches!(self, OpCode::Label)
    }

    pub fn is_vector_arithmetic(self) -> bool {
        matches!(
            self,
            OpCode::VecIntAdd
                | OpCode::VecIntSub
                | OpCode::VecIntMul
                | OpCode::VecIntAnd
                | OpCode::VecIntOr
                | OpCode::VecIntXor
                | OpCode::VecFloatAdd
                | OpCode::VecFloatSub
                | OpCode::VecFloatMul
                | OpCode::VecFloatTrueDiv
                | OpCode::VecFloatNeg
                | OpCode::VecFloatAbs
        )
    }

    /// Expected number of arguments, or None for variadic.
    pub fn arity(self) -> Option<u8> {
        OPARITY[self.as_u16() as usize]
    }

    /// Whether this operation takes a descriptor.
    pub fn has_descr(self) -> bool {
        OPWITHDESCR[self.as_u16() as usize]
    }

    /// Whether this operation produces a boolean result.
    pub fn returns_bool(self) -> bool {
        OPBOOL[self.as_u16() as usize]
    }

    /// Result type of this operation.
    pub fn result_type(self) -> Type {
        OPRESTYPE[self.as_u16() as usize]
    }

    /// Name of this operation (for debugging).
    pub fn name(self) -> &'static str {
        OPNAME[self.as_u16() as usize]
    }
}

// ── Typed dispatch helpers (mirrors rop.*_for_descr) ──

impl OpCode {
    pub fn call_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallI,
            Type::Ref => OpCode::CallR,
            Type::Float => OpCode::CallF,
            Type::Void => OpCode::CallN,
        }
    }

    pub fn call_pure_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallPureI,
            Type::Ref => OpCode::CallPureR,
            Type::Float => OpCode::CallPureF,
            Type::Void => OpCode::CallPureN,
        }
    }

    pub fn call_may_force_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallMayForceI,
            Type::Ref => OpCode::CallMayForceR,
            Type::Float => OpCode::CallMayForceF,
            Type::Void => OpCode::CallMayForceN,
        }
    }

    pub fn call_assembler_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallAssemblerI,
            Type::Ref => OpCode::CallAssemblerR,
            Type::Float => OpCode::CallAssemblerF,
            Type::Void => OpCode::CallAssemblerN,
        }
    }

    pub fn call_loopinvariant_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallLoopinvariantI,
            Type::Ref => OpCode::CallLoopinvariantR,
            Type::Float => OpCode::CallLoopinvariantF,
            Type::Void => OpCode::CallLoopinvariantN,
        }
    }

    pub fn same_as_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::SameAsI,
            Type::Ref => OpCode::SameAsR,
            Type::Float => OpCode::SameAsF,
            Type::Void => unreachable!("same_as has no void variant"),
        }
    }

    pub fn getfield_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetfieldGcR,
            Type::Float => OpCode::GetfieldGcF,
            _ => OpCode::GetfieldGcI,
        }
    }

    pub fn getarrayitem_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetarrayitemGcR,
            Type::Float => OpCode::GetarrayitemGcF,
            _ => OpCode::GetarrayitemGcI,
        }
    }
}

// ── Boolean inverse/reflex tables (from resoperation.py) ──

impl OpCode {
    /// Returns the boolean inverse of a comparison, e.g. INT_EQ -> INT_NE.
    pub fn bool_inverse(self) -> Option<OpCode> {
        match self {
            OpCode::IntEq => Some(OpCode::IntNe),
            OpCode::IntNe => Some(OpCode::IntEq),
            OpCode::IntLt => Some(OpCode::IntGe),
            OpCode::IntGe => Some(OpCode::IntLt),
            OpCode::IntGt => Some(OpCode::IntLe),
            OpCode::IntLe => Some(OpCode::IntGt),
            OpCode::UintLt => Some(OpCode::UintGe),
            OpCode::UintGe => Some(OpCode::UintLt),
            OpCode::UintGt => Some(OpCode::UintLe),
            OpCode::UintLe => Some(OpCode::UintGt),
            OpCode::FloatEq => Some(OpCode::FloatNe),
            OpCode::FloatNe => Some(OpCode::FloatEq),
            OpCode::FloatLt => Some(OpCode::FloatGe),
            OpCode::FloatGe => Some(OpCode::FloatLt),
            OpCode::FloatGt => Some(OpCode::FloatLe),
            OpCode::FloatLe => Some(OpCode::FloatGt),
            OpCode::PtrEq => Some(OpCode::PtrNe),
            OpCode::PtrNe => Some(OpCode::PtrEq),
            _ => None,
        }
    }

    /// Returns the reflexive form of a comparison (swap operands),
    /// e.g. INT_LT -> INT_GT.
    pub fn bool_reflex(self) -> Option<OpCode> {
        match self {
            OpCode::IntEq => Some(OpCode::IntEq),
            OpCode::IntNe => Some(OpCode::IntNe),
            OpCode::IntLt => Some(OpCode::IntGt),
            OpCode::IntGe => Some(OpCode::IntLe),
            OpCode::IntGt => Some(OpCode::IntLt),
            OpCode::IntLe => Some(OpCode::IntGe),
            OpCode::UintLt => Some(OpCode::UintGt),
            OpCode::UintGe => Some(OpCode::UintLe),
            OpCode::UintGt => Some(OpCode::UintLt),
            OpCode::UintLe => Some(OpCode::UintGe),
            OpCode::FloatEq => Some(OpCode::FloatEq),
            OpCode::FloatNe => Some(OpCode::FloatNe),
            OpCode::FloatLt => Some(OpCode::FloatGt),
            OpCode::FloatGe => Some(OpCode::FloatLe),
            OpCode::FloatGt => Some(OpCode::FloatLt),
            OpCode::FloatLe => Some(OpCode::FloatGe),
            OpCode::PtrEq => Some(OpCode::PtrEq),
            OpCode::PtrNe => Some(OpCode::PtrNe),
            _ => None,
        }
    }

    /// Maps a scalar op to its vector equivalent, e.g. INT_ADD -> VEC_INT_ADD.
    pub fn to_vector(self) -> Option<OpCode> {
        match self {
            OpCode::IntAdd => Some(OpCode::VecIntAdd),
            OpCode::IntSub => Some(OpCode::VecIntSub),
            OpCode::IntMul => Some(OpCode::VecIntMul),
            OpCode::IntAnd => Some(OpCode::VecIntAnd),
            OpCode::IntOr => Some(OpCode::VecIntOr),
            OpCode::IntXor => Some(OpCode::VecIntXor),
            OpCode::FloatAdd => Some(OpCode::VecFloatAdd),
            OpCode::FloatSub => Some(OpCode::VecFloatSub),
            OpCode::FloatMul => Some(OpCode::VecFloatMul),
            OpCode::FloatTrueDiv => Some(OpCode::VecFloatTrueDiv),
            OpCode::FloatAbs => Some(OpCode::VecFloatAbs),
            OpCode::FloatNeg => Some(OpCode::VecFloatNeg),
            OpCode::FloatEq => Some(OpCode::VecFloatEq),
            OpCode::FloatNe => Some(OpCode::VecFloatNe),
            OpCode::IntIsTrue => Some(OpCode::VecIntIsTrue),
            OpCode::IntEq => Some(OpCode::VecIntEq),
            OpCode::IntNe => Some(OpCode::VecIntNe),
            OpCode::IntSignext => Some(OpCode::VecIntSignext),
            OpCode::CastFloatToSinglefloat => Some(OpCode::VecCastFloatToSinglefloat),
            OpCode::CastSinglefloatToFloat => Some(OpCode::VecCastSinglefloatToFloat),
            OpCode::CastIntToFloat => Some(OpCode::VecCastIntToFloat),
            OpCode::CastFloatToInt => Some(OpCode::VecCastFloatToInt),
            OpCode::GuardTrue => Some(OpCode::VecGuardTrue),
            OpCode::GuardFalse => Some(OpCode::VecGuardFalse),
            _ => None,
        }
    }

    /// The non-overflow version of an overflow op, e.g. INT_ADD_OVF -> INT_ADD.
    pub fn without_overflow(self) -> Option<OpCode> {
        match self {
            OpCode::IntAddOvf => Some(OpCode::IntAdd),
            OpCode::IntSubOvf => Some(OpCode::IntSub),
            OpCode::IntMulOvf => Some(OpCode::IntMul),
            _ => None,
        }
    }
}

// ── Metadata tables ──
// These are generated to match the setup() function in resoperation.py.
// Format: arity (None = variadic), has_descr, returns_bool, result_type, name.

macro_rules! opcode_count {
    () => {
        OpCode::IntMulOvf as usize + 1
    };
}

/// Number of defined opcodes.
pub const OPCODE_COUNT: usize = opcode_count!();

// We use include! or manual arrays. For now, manual tables.
// These tables are indexed by OpCode as u16.

/// Arity: Some(n) for fixed arity, None for variadic.
static OPARITY: [Option<u8>; OPCODE_COUNT] = {
    let mut t = [None; OPCODE_COUNT];
    use OpCode::*;
    // Variadic ops (arity = *)
    // Jump, Finish, Label, DebugMergePoint, JitDebug, Escape*, all Calls, CondCall*, RecordKnownResult
    // are variadic -> None (already default)

    // Fixed arity ops
    macro_rules! set {
        ($op:ident, $a:expr) => {
            t[$op as usize] = Some($a);
        };
    }
    // Guards
    set!(GuardTrue, 1);
    set!(GuardFalse, 1);
    set!(VecGuardTrue, 1);
    set!(VecGuardFalse, 1);
    set!(GuardValue, 2);
    set!(GuardClass, 2);
    set!(GuardNonnull, 1);
    set!(GuardIsnull, 1);
    set!(GuardNonnullClass, 2);
    set!(GuardGcType, 2);
    set!(GuardIsObject, 1);
    set!(GuardSubclass, 2);
    set!(GuardNoException, 0);
    set!(GuardException, 1);
    set!(GuardNoOverflow, 0);
    set!(GuardOverflow, 0);
    set!(GuardNotForced, 0);
    set!(GuardNotForced2, 0);
    set!(GuardNotInvalidated, 0);
    set!(GuardFutureCondition, 0);
    set!(GuardAlwaysFails, 0);
    // Arithmetic (binary)
    set!(IntAdd, 2);
    set!(IntSub, 2);
    set!(IntMul, 2);
    set!(UintMulHigh, 2);
    set!(IntAnd, 2);
    set!(IntOr, 2);
    set!(IntXor, 2);
    set!(IntRshift, 2);
    set!(IntLshift, 2);
    set!(UintRshift, 2);
    set!(IntSignext, 2);
    set!(FloatAdd, 2);
    set!(FloatSub, 2);
    set!(FloatMul, 2);
    set!(FloatTrueDiv, 2);
    set!(FloatNeg, 1);
    set!(FloatAbs, 1);
    // Casts (unary)
    set!(CastFloatToInt, 1);
    set!(CastIntToFloat, 1);
    set!(CastFloatToSinglefloat, 1);
    set!(CastSinglefloatToFloat, 1);
    set!(ConvertFloatBytesToLonglong, 1);
    set!(ConvertLonglongBytesToFloat, 1);
    // Vector arithmetic (binary/unary)
    set!(VecIntAdd, 2);
    set!(VecIntSub, 2);
    set!(VecIntMul, 2);
    set!(VecIntAnd, 2);
    set!(VecIntOr, 2);
    set!(VecIntXor, 2);
    set!(VecFloatAdd, 2);
    set!(VecFloatSub, 2);
    set!(VecFloatMul, 2);
    set!(VecFloatTrueDiv, 2);
    set!(VecFloatNeg, 1);
    set!(VecFloatAbs, 1);
    set!(VecFloatEq, 2);
    set!(VecFloatNe, 2);
    set!(VecFloatXor, 2);
    set!(VecIntIsTrue, 1);
    set!(VecIntNe, 2);
    set!(VecIntEq, 2);
    set!(VecIntSignext, 2);
    set!(VecCastFloatToSinglefloat, 1);
    set!(VecCastSinglefloatToFloat, 1);
    set!(VecCastFloatToInt, 1);
    set!(VecCastIntToFloat, 1);
    set!(VecI, 0);
    set!(VecF, 0);
    set!(VecUnpackI, 3);
    set!(VecUnpackF, 3);
    set!(VecPackI, 4);
    set!(VecPackF, 4);
    set!(VecExpandI, 1);
    set!(VecExpandF, 1);
    // Comparisons
    set!(IntLt, 2);
    set!(IntLe, 2);
    set!(IntEq, 2);
    set!(IntNe, 2);
    set!(IntGt, 2);
    set!(IntGe, 2);
    set!(UintLt, 2);
    set!(UintLe, 2);
    set!(UintGt, 2);
    set!(UintGe, 2);
    set!(FloatLt, 2);
    set!(FloatLe, 2);
    set!(FloatEq, 2);
    set!(FloatNe, 2);
    set!(FloatGt, 2);
    set!(FloatGe, 2);
    // Unary int
    set!(IntIsZero, 1);
    set!(IntIsTrue, 1);
    set!(IntNeg, 1);
    set!(IntInvert, 1);
    set!(IntForceGeZero, 1);
    // Identity/cast
    set!(SameAsI, 1);
    set!(SameAsR, 1);
    set!(SameAsF, 1);
    set!(CastPtrToInt, 1);
    set!(CastIntToPtr, 1);
    // Pointer comparisons
    set!(PtrEq, 2);
    set!(PtrNe, 2);
    set!(InstancePtrEq, 2);
    set!(InstancePtrNe, 2);
    set!(NurseryPtrIncrement, 2);
    // Array/string length
    set!(ArraylenGc, 1);
    set!(Strlen, 1);
    set!(Strgetitem, 2);
    set!(GetarrayitemGcPureI, 2);
    set!(GetarrayitemGcPureR, 2);
    set!(GetarrayitemGcPureF, 2);
    set!(Unicodelen, 1);
    set!(Unicodegetitem, 2);
    set!(LoadFromGcTable, 1);
    set!(LoadEffectiveAddress, 4);
    // GC load
    set!(GcLoadI, 3);
    set!(GcLoadR, 3);
    set!(GcLoadF, 3);
    set!(GcLoadIndexedI, 5);
    set!(GcLoadIndexedR, 5);
    set!(GcLoadIndexedF, 5);
    // Array/field get
    set!(GetarrayitemGcI, 2);
    set!(GetarrayitemGcR, 2);
    set!(GetarrayitemGcF, 2);
    set!(GetarrayitemRawI, 2);
    set!(GetarrayitemRawF, 2);
    set!(RawLoadI, 2);
    set!(RawLoadF, 2);
    set!(VecLoadI, 4);
    set!(VecLoadF, 4);
    set!(GetinteriorfieldGcI, 2);
    set!(GetinteriorfieldGcR, 2);
    set!(GetinteriorfieldGcF, 2);
    set!(GetfieldGcI, 1);
    set!(GetfieldGcR, 1);
    set!(GetfieldGcF, 1);
    set!(GetfieldRawI, 1);
    set!(GetfieldRawR, 1);
    set!(GetfieldRawF, 1);
    set!(GetfieldGcPureI, 1);
    set!(GetfieldGcPureR, 1);
    set!(GetfieldGcPureF, 1);
    // Allocation
    set!(New, 0);
    set!(NewWithVtable, 0);
    set!(NewArray, 1);
    set!(NewArrayClear, 1);
    set!(Newstr, 1);
    set!(Newunicode, 1);
    // Misc no-side-effect
    set!(ForceToken, 0);
    set!(VirtualRefI, 2);
    set!(VirtualRefR, 2);
    set!(Strhash, 1);
    set!(Unicodehash, 1);
    // GC store
    set!(GcStore, 4);
    set!(GcStoreIndexed, 6);
    set!(IncrementDebugCounter, 1);
    // Array/field set
    set!(SetarrayitemGc, 3);
    set!(SetarrayitemRaw, 3);
    set!(RawStore, 3);
    set!(VecStore, 5);
    set!(SetinteriorfieldGc, 3);
    set!(SetinteriorfieldRaw, 3);
    set!(SetfieldGc, 2);
    set!(ZeroArray, 5);
    set!(SetfieldRaw, 2);
    set!(Strsetitem, 3);
    set!(Unicodesetitem, 3);
    // GC write barriers
    set!(CondCallGcWb, 1);
    set!(CondCallGcWbArray, 2);
    // Debug (variadic) - already None
    // Portal frames
    set!(EnterPortalFrame, 2);
    set!(LeavePortalFrame, 1);
    // Misc
    set!(ForceSpill, 1);
    set!(VirtualRefFinish, 2);
    set!(Copystrcontent, 5);
    set!(Copyunicodecontent, 5);
    set!(QuasiimmutField, 1);
    set!(AssertNotNone, 1);
    set!(RecordExactClass, 2);
    set!(RecordExactValueR, 2);
    set!(RecordExactValueI, 2);
    set!(Keepalive, 1);
    set!(SaveException, 0);
    set!(SaveExcClass, 0);
    set!(RestoreException, 2);
    // Calls: all variadic (None) - default
    set!(CheckMemoryError, 1);
    set!(CallMallocNursery, 1);
    set!(CallMallocNurseryVarsizeFrame, 1);
    // Overflow
    set!(IntAddOvf, 2);
    set!(IntSubOvf, 2);
    set!(IntMulOvf, 2);
    t
};

/// Whether the operation takes a descriptor.
static OPWITHDESCR: [bool; OPCODE_COUNT] = {
    let mut t = [false; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! set {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = true;)+
        };
    }
    set!(
        Jump,
        Finish,
        Label,
        // Guards
        GuardTrue,
        GuardFalse,
        VecGuardTrue,
        VecGuardFalse,
        GuardValue,
        GuardClass,
        GuardNonnull,
        GuardIsnull,
        GuardNonnullClass,
        GuardGcType,
        GuardIsObject,
        GuardSubclass,
        GuardNoException,
        GuardException,
        GuardNoOverflow,
        GuardOverflow,
        GuardNotForced,
        GuardNotForced2,
        GuardNotInvalidated,
        GuardFutureCondition,
        GuardAlwaysFails,
        // Array/field access
        ArraylenGc,
        GetarrayitemGcPureI,
        GetarrayitemGcPureR,
        GetarrayitemGcPureF,
        GetarrayitemGcI,
        GetarrayitemGcR,
        GetarrayitemGcF,
        GetarrayitemRawI,
        GetarrayitemRawF,
        RawLoadI,
        RawLoadF,
        VecLoadI,
        VecLoadF,
        GetinteriorfieldGcI,
        GetinteriorfieldGcR,
        GetinteriorfieldGcF,
        GetfieldGcI,
        GetfieldGcR,
        GetfieldGcF,
        GetfieldRawI,
        GetfieldRawR,
        GetfieldRawF,
        GetfieldGcPureI,
        GetfieldGcPureR,
        GetfieldGcPureF,
        // Allocation
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        // Stores
        GcStore,
        GcStoreIndexed,
        SetarrayitemGc,
        SetarrayitemRaw,
        RawStore,
        VecStore,
        SetinteriorfieldGc,
        SetinteriorfieldRaw,
        SetfieldGc,
        ZeroArray,
        SetfieldRaw,
        // GC barriers
        CondCallGcWb,
        CondCallGcWbArray,
        // Misc
        QuasiimmutField,
        // Calls
        CallI,
        CallR,
        CallF,
        CallN,
        CondCallN,
        CondCallValueI,
        CondCallValueR,
        CallAssemblerI,
        CallAssemblerR,
        CallAssemblerF,
        CallAssemblerN,
        CallMayForceI,
        CallMayForceR,
        CallMayForceF,
        CallMayForceN,
        CallLoopinvariantI,
        CallLoopinvariantR,
        CallLoopinvariantF,
        CallLoopinvariantN,
        CallReleaseGilI,
        CallReleaseGilF,
        CallReleaseGilN,
        CallPureI,
        CallPureR,
        CallPureF,
        CallPureN,
        CallMallocNurseryVarsize,
        RecordKnownResult
    );
    t
};

/// Whether the operation returns a boolean result.
static OPBOOL: [bool; OPCODE_COUNT] = {
    let mut t = [false; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! set {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = true;)+
        };
    }
    set!(
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        VecFloatEq,
        VecFloatNe,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq
    );
    t
};

/// Result type of each operation.
static OPRESTYPE: [Type; OPCODE_COUNT] = {
    let mut t = [Type::Void; OPCODE_COUNT];
    use OpCode::*;

    macro_rules! int {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Int;)+
        };
    }
    macro_rules! float {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Float;)+
        };
    }
    macro_rules! ref_ {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Ref;)+
        };
    }

    int!(
        IntAdd,
        IntSub,
        IntMul,
        UintMulHigh,
        IntAnd,
        IntOr,
        IntXor,
        IntRshift,
        IntLshift,
        UintRshift,
        IntSignext,
        CastFloatToInt,
        CastFloatToSinglefloat,
        ConvertFloatBytesToLonglong,
        // Vector int
        VecIntAdd,
        VecIntSub,
        VecIntMul,
        VecIntAnd,
        VecIntOr,
        VecIntXor,
        VecFloatEq,
        VecFloatNe,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq,
        VecIntSignext,
        VecCastFloatToSinglefloat,
        VecCastFloatToInt,
        // Comparisons (all return int)
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        IntNeg,
        IntInvert,
        IntForceGeZero,
        SameAsI,
        CastPtrToInt,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        ArraylenGc,
        Strlen,
        Strgetitem,
        GetarrayitemGcPureI,
        Unicodelen,
        Unicodegetitem,
        LoadEffectiveAddress,
        GcLoadI,
        GcLoadIndexedI,
        GetarrayitemGcI,
        GetarrayitemRawI,
        RawLoadI,
        GetinteriorfieldGcI,
        GetfieldGcI,
        GetfieldRawI,
        GetfieldGcPureI,
        Strhash,
        Unicodehash,
        CondCallValueI,
        CallI,
        CallPureI,
        CallMayForceI,
        CallAssemblerI,
        CallLoopinvariantI,
        CallReleaseGilI,
        SaveExcClass,
        RecordExactValueI,
        IntAddOvf,
        IntSubOvf,
        IntMulOvf
    );

    float!(
        FloatAdd,
        FloatSub,
        FloatMul,
        FloatTrueDiv,
        FloatNeg,
        FloatAbs,
        CastIntToFloat,
        CastSinglefloatToFloat,
        ConvertLonglongBytesToFloat,
        VecFloatAdd,
        VecFloatSub,
        VecFloatMul,
        VecFloatTrueDiv,
        VecFloatNeg,
        VecFloatAbs,
        VecFloatXor,
        VecCastSinglefloatToFloat,
        VecCastIntToFloat,
        SameAsF,
        GetarrayitemGcPureF,
        GcLoadF,
        GcLoadIndexedF,
        GetarrayitemGcF,
        GetarrayitemRawF,
        RawLoadF,
        GetinteriorfieldGcF,
        GetfieldGcF,
        GetfieldRawF,
        GetfieldGcPureF,
        CallF,
        CallPureF,
        CallMayForceF,
        CallAssemblerF,
        CallLoopinvariantF,
        CallReleaseGilF
    );

    ref_!(
        CastIntToPtr,
        SameAsR,
        NurseryPtrIncrement,
        GetarrayitemGcPureR,
        LoadFromGcTable,
        GcLoadR,
        GcLoadIndexedR,
        GetarrayitemGcR,
        GetinteriorfieldGcR,
        GetfieldGcR,
        GetfieldRawR,
        GetfieldGcPureR,
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        Newstr,
        Newunicode,
        ForceToken,
        VirtualRefR,
        GuardException,
        CondCallValueR,
        CallR,
        CallPureR,
        CallMayForceR,
        CallAssemblerR,
        CallLoopinvariantR,
        CallMallocNursery,
        CallMallocNurseryVarsize,
        CallMallocNurseryVarsizeFrame,
        SaveException
    );

    // VecI/VecF, VecUnpack*, VecPack*, VecExpand* can be either int or float
    // depending on usage. Default to int for I variants, float for F variants.
    int!(VecI, VecUnpackI, VecPackI, VecExpandI, VecLoadI);
    float!(VecF, VecUnpackF, VecPackF, VecExpandF, VecLoadF);
    int!(VirtualRefI);
    // Escape ops (testing)
    int!(EscapeI);
    ref_!(EscapeR);
    float!(EscapeF);

    t
};

/// Operation names for debugging.
static OPNAME: [&str; OPCODE_COUNT] = {
    let mut t = [""; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! name {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = stringify!($op);)+
        };
    }
    name!(
        Jump,
        Finish,
        Label,
        GuardTrue,
        GuardFalse,
        VecGuardTrue,
        VecGuardFalse,
        GuardValue,
        GuardClass,
        GuardNonnull,
        GuardIsnull,
        GuardNonnullClass,
        GuardGcType,
        GuardIsObject,
        GuardSubclass,
        GuardNoException,
        GuardException,
        GuardNoOverflow,
        GuardOverflow,
        GuardNotForced,
        GuardNotForced2,
        GuardNotInvalidated,
        GuardFutureCondition,
        GuardAlwaysFails,
        IntAdd,
        IntSub,
        IntMul,
        UintMulHigh,
        IntAnd,
        IntOr,
        IntXor,
        IntRshift,
        IntLshift,
        UintRshift,
        IntSignext,
        FloatAdd,
        FloatSub,
        FloatMul,
        FloatTrueDiv,
        FloatNeg,
        FloatAbs,
        CastFloatToInt,
        CastIntToFloat,
        CastFloatToSinglefloat,
        CastSinglefloatToFloat,
        ConvertFloatBytesToLonglong,
        ConvertLonglongBytesToFloat,
        VecIntAdd,
        VecIntSub,
        VecIntMul,
        VecIntAnd,
        VecIntOr,
        VecIntXor,
        VecFloatAdd,
        VecFloatSub,
        VecFloatMul,
        VecFloatTrueDiv,
        VecFloatNeg,
        VecFloatAbs,
        VecFloatEq,
        VecFloatNe,
        VecFloatXor,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq,
        VecIntSignext,
        VecCastFloatToSinglefloat,
        VecCastSinglefloatToFloat,
        VecCastFloatToInt,
        VecCastIntToFloat,
        VecI,
        VecF,
        VecUnpackI,
        VecUnpackF,
        VecPackI,
        VecPackF,
        VecExpandI,
        VecExpandF,
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        IntNeg,
        IntInvert,
        IntForceGeZero,
        SameAsI,
        SameAsR,
        SameAsF,
        CastPtrToInt,
        CastIntToPtr,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        NurseryPtrIncrement,
        ArraylenGc,
        Strlen,
        Strgetitem,
        GetarrayitemGcPureI,
        GetarrayitemGcPureR,
        GetarrayitemGcPureF,
        Unicodelen,
        Unicodegetitem,
        LoadFromGcTable,
        LoadEffectiveAddress,
        GcLoadI,
        GcLoadR,
        GcLoadF,
        GcLoadIndexedI,
        GcLoadIndexedR,
        GcLoadIndexedF,
        GetarrayitemGcI,
        GetarrayitemGcR,
        GetarrayitemGcF,
        GetarrayitemRawI,
        GetarrayitemRawF,
        RawLoadI,
        RawLoadF,
        VecLoadI,
        VecLoadF,
        GetinteriorfieldGcI,
        GetinteriorfieldGcR,
        GetinteriorfieldGcF,
        GetfieldGcI,
        GetfieldGcR,
        GetfieldGcF,
        GetfieldRawI,
        GetfieldRawR,
        GetfieldRawF,
        GetfieldGcPureI,
        GetfieldGcPureR,
        GetfieldGcPureF,
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        Newstr,
        Newunicode,
        ForceToken,
        VirtualRefI,
        VirtualRefR,
        Strhash,
        Unicodehash,
        GcStore,
        GcStoreIndexed,
        IncrementDebugCounter,
        SetarrayitemGc,
        SetarrayitemRaw,
        RawStore,
        VecStore,
        SetinteriorfieldGc,
        SetinteriorfieldRaw,
        SetfieldGc,
        ZeroArray,
        SetfieldRaw,
        Strsetitem,
        Unicodesetitem,
        CondCallGcWb,
        CondCallGcWbArray,
        DebugMergePoint,
        EnterPortalFrame,
        LeavePortalFrame,
        JitDebug,
        EscapeI,
        EscapeR,
        EscapeF,
        EscapeN,
        ForceSpill,
        VirtualRefFinish,
        Copystrcontent,
        Copyunicodecontent,
        QuasiimmutField,
        AssertNotNone,
        RecordExactClass,
        RecordExactValueR,
        RecordExactValueI,
        Keepalive,
        SaveException,
        SaveExcClass,
        RestoreException,
        CallI,
        CallR,
        CallF,
        CallN,
        CondCallN,
        CondCallValueI,
        CondCallValueR,
        CallAssemblerI,
        CallAssemblerR,
        CallAssemblerF,
        CallAssemblerN,
        CallMayForceI,
        CallMayForceR,
        CallMayForceF,
        CallMayForceN,
        CallLoopinvariantI,
        CallLoopinvariantR,
        CallLoopinvariantF,
        CallLoopinvariantN,
        CallReleaseGilI,
        CallReleaseGilF,
        CallReleaseGilN,
        CallPureI,
        CallPureR,
        CallPureF,
        CallPureN,
        CheckMemoryError,
        CallMallocNursery,
        CallMallocNurseryVarsize,
        CallMallocNurseryVarsizeFrame,
        RecordKnownResult,
        IntAddOvf,
        IntSubOvf,
        IntMulOvf
    );
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_classification() {
        assert!(OpCode::Jump.is_final());
        assert!(OpCode::Finish.is_final());
        assert!(!OpCode::Label.is_final());

        assert!(OpCode::GuardTrue.is_guard());
        assert!(OpCode::GuardAlwaysFails.is_guard());
        assert!(!OpCode::IntAdd.is_guard());

        assert!(OpCode::IntAdd.is_always_pure());
        assert!(OpCode::FloatMul.is_always_pure());
        assert!(!OpCode::SetfieldGc.is_always_pure());

        assert!(OpCode::IntAddOvf.is_ovf());
        assert!(!OpCode::IntAdd.is_ovf());

        assert!(OpCode::CallI.is_call());
        assert!(OpCode::CallPureN.is_call());
        assert!(!OpCode::IntAdd.is_call());
    }

    #[test]
    fn test_bool_inverse() {
        assert_eq!(OpCode::IntEq.bool_inverse(), Some(OpCode::IntNe));
        assert_eq!(OpCode::IntLt.bool_inverse(), Some(OpCode::IntGe));
        assert_eq!(OpCode::IntAdd.bool_inverse(), None);
    }

    #[test]
    fn test_bool_reflex() {
        assert_eq!(OpCode::IntLt.bool_reflex(), Some(OpCode::IntGt));
        assert_eq!(OpCode::IntEq.bool_reflex(), Some(OpCode::IntEq));
    }

    #[test]
    fn test_result_types() {
        assert_eq!(OpCode::IntAdd.result_type(), Type::Int);
        assert_eq!(OpCode::FloatAdd.result_type(), Type::Float);
        assert_eq!(OpCode::New.result_type(), Type::Ref);
        assert_eq!(OpCode::SetfieldGc.result_type(), Type::Void);
    }

    #[test]
    fn test_arity() {
        assert_eq!(OpCode::IntAdd.arity(), Some(2));
        assert_eq!(OpCode::FloatNeg.arity(), Some(1));
        assert_eq!(OpCode::New.arity(), Some(0));
        assert_eq!(OpCode::CallI.arity(), None); // variadic
    }

    #[test]
    fn test_ovf_to_non_ovf_alignment() {
        // Mirrors the assertion in resoperation.py setup():
        // INT_ADD_OVF - _OVF_FIRST == INT_ADD - _ALWAYS_PURE_FIRST
        let add_ovf_offset = OpCode::IntAddOvf as u16 - OVF_FIRST;
        let add_offset = OpCode::IntAdd as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(add_ovf_offset, add_offset);

        let sub_ovf_offset = OpCode::IntSubOvf as u16 - OVF_FIRST;
        let sub_offset = OpCode::IntSub as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(sub_ovf_offset, sub_offset);

        let mul_ovf_offset = OpCode::IntMulOvf as u16 - OVF_FIRST;
        let mul_offset = OpCode::IntMul as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(mul_ovf_offset, mul_offset);
    }
}
