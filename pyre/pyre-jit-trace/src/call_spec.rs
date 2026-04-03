//! Canonical call/effect specification for `PyFrame`.
//!
//! This module is intentionally data-only so `build.rs` can share the same
//! translator-facing call/effect contract without re-encoding helper policy in
//! the analyzer.

pub const PYFRAME_CALL_OWNER_ROOT: &str = "PyFrame";

#[derive(Clone, Copy)]
pub enum CallEffectKind {
    Elidable,
    Residual,
}

#[derive(Clone, Copy)]
pub enum CallPatternRole {
    IntArithmetic,
    FloatArithmetic,
    LocalRead,
    LocalWrite,
    FunctionCall,
    TruthCheck,
    StackManip,
    NamespaceLoadLocal,
    NamespaceLoadGlobal,
    NamespaceStoreLocal,
    NamespaceStoreGlobal,
    RangeIterNext,
    IterCleanup,
    Return,
    BuildList,
    BuildTuple,
    UnpackSequence,
    SequenceSetitem,
    CollectionAppend,
}

#[derive(Clone, Copy)]
pub enum CallTargetSpec {
    Method {
        name: &'static str,
        receiver_root: &'static str,
    },
    FunctionPath(&'static [&'static str]),
}

#[derive(Clone, Copy)]
pub struct CallEffectSpec {
    pub target: CallTargetSpec,
    pub effect: CallEffectKind,
    pub role: Option<CallPatternRole>,
}

pub const PYFRAME_CALL_EFFECTS: &[CallEffectSpec] = &[
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["w_int_add"]),
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["w_int_sub"]),
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["w_int_mul"]),
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["crate", "math", "w_int_add"]),
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["crate", "math", "w_int_sub"]),
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["crate", "math", "w_int_mul"]),
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["w_float_add"]),
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::FloatArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["w_float_sub"]),
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::FloatArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "peek_at",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Elidable,
        role: Some(CallPatternRole::StackManip),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "push_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::LocalRead),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "pop_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::LocalWrite),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "call_callable",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::FunctionCall),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "call_function_ex",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::FunctionCall),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "call_kw",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::FunctionCall),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "truth_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::TruthCheck),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "bool_value_from_truth",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::TruthCheck),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "concrete_truth_as_bool",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::TruthCheck),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "to_bool",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::TruthCheck),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "swap_values",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::StackManip),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "copy_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::StackManip),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_global",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceLoadGlobal),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_global",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceStoreGlobal),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_name",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceLoadLocal),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_name_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceLoadLocal),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_from_dict_or_globals",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceLoadLocal),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_from_dict_or_deref",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceLoadLocal),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_name",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceStoreLocal),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_name_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceStoreLocal),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "iter_next_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::RangeIterNext),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "for_iter",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::RangeIterNext),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "end_for",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::IterCleanup),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "pop_iter",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::IterCleanup),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "return_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::Return),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "build_list",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::BuildList),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "build_tuple",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::BuildTuple),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unpack_sequence",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::UnpackSequence),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unpack_ex",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::UnpackSequence),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_subscr",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::SequenceSetitem),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "list_append",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::CollectionAppend),
    },
    // ── Trait handler methods (called by opcode_* free functions) ──
    // These have generic receivers (e.g. handler: &mut H) in the source,
    // but receiver matching handles lowercase names as wildcards.
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_local_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::LocalRead),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_local_checked_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::LocalRead),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_local_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::LocalWrite),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "binary_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "compare_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unary_negative_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unary_invert_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::IntArithmetic),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "set_next_instr",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "make_function",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::FunctionCall),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "load_attr",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "store_attr",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "build_map",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "build_set",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "ensure_iter_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::RangeIterNext),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "on_iter_exhausted",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::IterCleanup),
    },
];
