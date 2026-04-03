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
    ConstLoad,
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
        role: Some(CallPatternRole::StackManip),
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "pop_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::StackManip),
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
    // binary_value, compare_value, unary_negative_value, unary_invert_value
    // are generic object-space operations (space.add, space.neg, space.eq etc.).
    // Type specialization happens at JIT trace time, not at codewriter
    // classification time. These are NOT int-specific.
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "binary_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "compare_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unary_negative_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "unary_invert_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
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
    // ensure_iter_value is GET_ITER semantics (iterable → iterator),
    // not next/branch semantics. Residual, not RangeIterNext.
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "ensure_iter_value",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "on_iter_exhausted",
            receiver_root: PYFRAME_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::IterCleanup),
    },
    // ── opcode_* free functions (FunctionPath targets) ──
    // These are called by OpcodeStepExecutor default methods.
    // Adding them here lets the classifier match at the default method
    // graph level without needing to follow into the free function body.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_call"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::FunctionCall),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_return_value"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::Return),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_const"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::ConstLoad),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_small_int"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::ConstLoad),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_fast_checked"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::LocalRead),
    },
    // Multi-local superinstructions: these are CPython 3.12+ fused opcodes
    // with no PyPy equivalent. Cannot be represented as a single LocalRead
    // or LocalWrite since they perform two distinct local operations.
    // Classified as Residual — the JIT decomposes them at trace time.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_fast_pair_checked"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_fast_load_fast"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_fast"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::LocalWrite),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_fast_load_fast"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_fast_store_fast"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_name"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceStoreLocal),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_name"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceLoadLocal),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_global"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::NamespaceLoadGlobal),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_pop_top"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::StackManip),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_push_null"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::StackManip),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_copy_value"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::StackManip),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_swap"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::StackManip),
    },
    // Generic object-space operations — type specialization at trace time.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_binary_op"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_compare_op"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_unary_negative"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_unary_invert"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_unary_not"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::TruthCheck),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_jump_forward"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_jump_backward"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_pop_jump_if_false"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::TruthCheck),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_pop_jump_if_true"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::TruthCheck),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_build_list"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::BuildList),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_build_tuple"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::BuildTuple),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_build_map"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_subscr"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::SequenceSetitem),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_list_append"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::CollectionAppend),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_unpack_sequence"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::UnpackSequence),
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_load_attr"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_store_attr"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    // GET_ITER is space.iter() — converts iterable to iterator.
    // NOT next/branch semantics. Residual.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_get_iter"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_for_iter"]),
        effect: CallEffectKind::Residual,
        role: Some(CallPatternRole::RangeIterNext),
    },
    // MAKE_FUNCTION creates a function object from code+defaults,
    // it does NOT call a function. Residual, not FunctionCall.
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["opcode_make_function"]),
        effect: CallEffectKind::Residual,
        role: None,
    },
];
