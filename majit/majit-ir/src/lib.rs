pub mod descr;
pub mod effectinfo;
pub mod resoperation;
pub mod resumecode;
pub mod resumedata;
pub mod value;

// Re-export key types at crate root for convenience.
pub use descr::{
    AccumVectorInfo, ArrayDescr, ArrayFlag, CallDescr, DebugMergePointDescr, DebugMergePointInfo,
    Descr, DescrRef, FailDescr, FieldDescr, GcCache, InteriorFieldDescr, LLType, LoopTargetDescr,
    SimpleCallDescr, SimpleFailDescr, SimpleFieldDescr, SizeDescr, TargetArgLoc, VableExpansion,
    descr_identity, make_array_descr, make_field_descr, make_loop_target_descr,
    make_raw_malloc_calldescr, make_size_descr_full, make_size_descr_with_vtable,
    unpack_fielddescr,
};
pub use effectinfo::{
    CallInfoCollection, EffectInfo, ExtraEffect, OopSpecIndex, QuasiImmutAnalyzer,
    RandomEffectsAnalyzer, UnsupportedFieldExc, VirtualizableAnalyzer, consider_array,
    consider_struct, frozenset_or_none,
};
pub use resoperation::{
    ArrayDescrInfo, BoxEnv, FieldDescrInfo, GuardPendingFieldEntry, OPCODE_COUNT, Op, OpCode,
    OpRef, RdVirtualInfo, VectorizationInfo, VirtualFieldsInfo, format_trace,
};
pub use value::{
    Const, FAILARGS_LIMIT, GcRef, GreenKey, InputArg, JitDriverVar, Type, Value, VarKind,
};
