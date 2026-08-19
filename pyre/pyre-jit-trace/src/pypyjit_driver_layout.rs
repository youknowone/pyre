use majit_ir::Type;

/// pypy/module/pypyjit/interp_jit.py:66-69 — jd0 portal driver layout.
pub const PYPYJIT_GREEN_VARS: [(&str, Type); 3] = [
    ("next_instr", Type::Int),
    ("is_being_profiled", Type::Int),
    ("pycode", Type::Ref),
];

pub const PYPYJIT_RED_VARS: [(&str, Type); 2] = [("frame", Type::Ref), ("ec", Type::Ref)];

pub const PYPYJIT_VIRTUALIZABLE: &str = "frame";

pub const PYPYJIT_RED_TYPES: [&str; 2] = ["PyFrame", "ExecutionContext"];
