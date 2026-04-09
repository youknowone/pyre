pub mod descr;
pub mod resoperation;
pub mod resumecode;
pub mod resumedata;
pub mod value;

// Re-export key types at crate root for convenience.
pub use descr::{
    AccumVectorInfo, ArrayDescr, CallDescr, DebugMergePointDescr, DebugMergePointInfo, Descr,
    DescrRef, EffectInfo, ExtraEffect, FailDescr, FieldDescr, InteriorFieldDescr, LoopTargetDescr,
    OopSpecIndex, SimpleCallDescr, SimpleFailDescr, SimpleFieldDescr, SizeDescr, TargetArgLoc,
    descr_identity, make_array_descr, make_field_descr, make_size_descr_full,
    make_size_descr_with_vtable, unpack_fielddescr,
};
pub use resoperation::{
    BoxEnv, FieldDescrInfo, GuardPendingFieldEntry, OPCODE_COUNT, Op, OpCode, OpRef, RdVirtualInfo,
    VectorizationInfo, VirtualFieldsInfo, format_trace,
};
pub use value::{
    Const, FAILARGS_LIMIT, GcRef, GreenKey, InputArg, JitDriverVar, Type, Value, VarKind,
};
