pub mod bitstring;
pub mod debug;
pub mod descr;
pub mod descr_registry;
pub mod effectinfo;
pub mod field_entry;
pub mod forwarding;
pub mod intbound;
pub mod op_descr;
pub mod op_info;
pub mod op_type_index;
pub mod operand;
pub mod optimize;
pub mod ptr_info;
pub mod rawbuffer;
pub mod resoperation;
pub mod resumecode;
pub mod resumedata;
pub mod value;
pub mod vec_map;
pub mod vec_set;

// Re-export key types at crate root for convenience.
pub use descr::{
    AccumInfo, ArrayDescr, ArrayFlag, CallDescr, DebugMergePointDescr, DebugMergePointInfo, Descr,
    DescrRef, FailDescr, FailDescrCell, FieldDescr, GcCache, InteriorFieldDescr, JitCodeDescr,
    LLType, LoopTargetDescr, LoopTokenDescr, SimpleCallDescr, SimpleFailDescr, SimpleFieldDescr,
    SizeDescr, SwitchDescr, TargetArgLoc, UnpackAtExitInfo, VableExpansion, descr_identity,
    make_array_descr, make_array_descr_signed, make_call_descr, make_field_descr,
    make_field_descr_full, make_loop_target_descr, make_malloc_array_calldescr,
    make_malloc_array_nonstandard_calldescr, make_malloc_big_fixedsize_calldescr,
    make_malloc_str_calldescr, make_malloc_unicode_calldescr, make_memcpy_calldescr,
    make_size_descr_full, make_size_descr_with_vtable, make_tid_field_descr,
    make_vtable_field_descr, memcpy_fn_addr, recover_fail_descr_cell, unpack_fielddescr,
};
pub use effectinfo::{
    CallInfoCollection, EffectInfo, ExtraEffect, OopSpecIndex, PyreHelperKind, UnsupportedFieldExc,
    consider_array, consider_struct, frozenset_or_none,
};
pub use op_type_index::OpTypeIndex;
pub use resoperation::{
    AbstractValue, ArrayDescrInfo, BoxEnv, FieldDescrInfo, GuardPendingFieldEntry, OPCODE_COUNT,
    Op, OpCode, OpRc, OpRef, RdVirtualInfo, VectorizationInfo, VirtualFieldsInfo, format_trace,
};
pub use value::{
    Const, FAILARGS_LIMIT, GcRef, GreenAsI64, GreenKey, GreenType, InputArg, InputArgRc,
    JitDriverVar, StrEqFn, StrHashFn, Type, Value, VarKind, green_type_to_ir, make_str_slot,
    pypyjit_greenkey_uhash, set_str_resolver, set_unicode_resolver,
};
pub use vec_map::{VecMap, VecMapExt};
pub use vec_set::VecSet;
