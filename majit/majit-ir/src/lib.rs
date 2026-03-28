pub mod descr;
pub mod resoperation;
pub mod resumecode;
pub mod resumedata;
pub mod value;

// Re-export key types at crate root for convenience.
pub use descr::{
    ArrayDescr, CallDescr, DebugMergePointDescr, DebugMergePointInfo, Descr, DescrRef, EffectInfo,
    ExtraEffect, FailDescr, FieldDescr, InteriorFieldDescr, OopSpecIndex, SimpleFailDescr,
    SizeDescr, make_array_descr, make_field_descr, unpack_fielddescr,
};
pub use resoperation::{
    BoxEnv, GuardPendingFieldEntry, GuardVirtualEntry, OPCODE_COUNT, Op, OpCode, OpRef,
    VectorizationInfo, VirtualFieldsInfo, format_trace,
};
pub use value::{
    Const, FAILARGS_LIMIT, GcRef, GreenKey, InputArg, JitDriverVar, Type, Value, VarKind,
};
