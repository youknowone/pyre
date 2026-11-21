use majit_ir::Type;

/// pypy/module/pypyjit/interp_jit.py PyPyJitDriver — jd0 portal driver layout.
pub const PYPYJIT_GREEN_VARS: [(&str, Type); 3] = [
    ("next_instr", Type::Int),
    ("is_being_profiled", Type::Int),
    ("pycode", Type::Ref),
];

pub const PYPYJIT_RED_VARS: [(&str, Type); 2] = [("frame", Type::Ref), ("ec", Type::Ref)];

/// Split-portal graph after `warmspot.py` `split_graph_and_record_jitdriver`.
pub const PYPYJIT_PORTAL_KEY: &str = "eval::eval_loop_jit_portal";

pub const PYPYJIT_VIRTUALIZABLE: &str = "frame";

pub const PYPYJIT_RED_TYPES: [&str; 2] = ["PyFrame", "ExecutionContext"];
