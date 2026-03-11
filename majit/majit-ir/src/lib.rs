pub mod descr;
pub mod op;
pub mod value;

// Re-export key types at crate root for convenience.
pub use descr::{
    ArrayDescr, CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, FailDescr, FieldDescr,
    InteriorFieldDescr, OopSpecIndex, SizeDescr,
};
pub use op::{format_trace, Op, OpCode, OpRef, OPCODE_COUNT};
pub use value::{Const, GcRef, InputArg, Type, Value, FAILARGS_LIMIT};
