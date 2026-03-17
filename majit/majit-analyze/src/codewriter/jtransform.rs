//! Semantic rewrite layer for trace patterns.
//!
//! This is majit's equivalent of RPython's `jtransform.py`. It sits between
//! pattern classification (`patterns.rs`) and code generation (`codewriter.rs`),
//! transforming high-level `TracePattern`s into concrete IR lowering recipes.
//!
//! RPython's jtransform rewrites each SSA operation into JIT-friendly form:
//! - `int_add_ovf` → overflow-checked add + guard
//! - `direct_call` → call kind classification (residual, elidable, may_force)
//! - `getarrayitem` → GC/raw dispatch + descriptor attachment
//! - `hint` → promote, virtual_ref, etc.
//!
//! majit's jtransform does the analogous work on `TracePattern`s:
//! - `UnboxIntBinop` → unbox + IntAddOvf + GuardNoOverflow + box
//! - `FunctionCall` → call_assembler vs residual decision
//! - `LocalRead` → getarrayitem_raw on frame locals
//! - `Residual` → residual call with effect info

use crate::patterns::TracePattern;
use serde::{Deserialize, Serialize};

/// A concrete IR lowering recipe produced by the jtransform layer.
///
/// Each recipe describes exactly what IR operations to emit for a
/// classified trace pattern. This is what the code generator consumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoweringRecipe {
    /// Unbox two operands → binary IR op → guard overflow → box result.
    IntBinopWithOverflow {
        ir_opcode: String,       // e.g., "IntAddOvf"
        guard: String,           // e.g., "GuardNoOverflow"
        unbox_fn: String,        // e.g., "trace_unbox_int"
        box_fn: String,          // e.g., "trace_box_int"
    },

    /// Unbox two operands → binary IR op (no overflow) → box result.
    IntBinopNoOverflow {
        ir_opcode: String,
        unbox_fn: String,
        box_fn: String,
    },

    /// Unbox two float operands → float IR op → box result.
    FloatBinop {
        ir_opcode: String,
        unbox_fn: String,
        box_fn: String,
    },

    /// Unbox two operands → comparison IR op → box bool result.
    IntCompare {
        ir_opcode: String,
        unbox_fn: String,
        box_fn: String,
    },

    /// Unbox single operand → unary IR op → box result.
    IntUnary {
        ir_opcode: String,
        unbox_fn: String,
        box_fn: String,
    },

    /// Read from frame locals array → getarrayitem_raw.
    LocalRead {
        array_field: String,     // e.g., "locals_w"
        index_source: String,    // e.g., "oparg"
    },

    /// Write to frame locals array → setarrayitem_raw.
    LocalWrite {
        array_field: String,
        index_source: String,
    },

    /// Load a constant from the code object's constant pool.
    ConstLoad {
        pool_field: String,      // e.g., "co_consts"
    },

    /// Guard on truth value → GuardTrue/GuardFalse.
    TruthCheck {
        unbox_fn: String,
    },

    /// Function call → call_assembler or residual call.
    FunctionCall {
        /// Whether to use call_assembler (true) or residual call (false).
        use_call_assembler: bool,
    },

    /// Iterator protocol: get next value, guard not exhausted.
    IterNext {
        step_fn: String,
    },

    /// Jump (unconditional).
    Jump,

    /// Conditional jump with guard.
    ConditionalJump {
        guard_kind: String,      // "GuardTrue" or "GuardFalse"
    },

    /// Return value from function.
    Return,

    /// Namespace access (load/store global/name).
    NamespaceAccess {
        is_load: bool,
        is_global: bool,
    },

    /// Collection construction → NEW_ARRAY + setitems.
    BuildCollection {
        kind: String,
    },

    /// Unpack sequence → multiple getarrayitems.
    UnpackSequence,

    /// Collection write → setarrayitem.
    SequenceSetitem,

    /// Collection append → call + possible resize.
    CollectionAppend,

    /// Stack manipulation (pop, swap, copy).
    StackManip,

    /// Iterator cleanup (end_for, pop_iter).
    IterCleanup,

    /// No-op instruction.
    Noop,

    /// Residual call — emit opaque CALL_I/CALL_R.
    ResidualCall {
        helper_name: String,
    },

    /// Abort — interpreter fallback.
    Abort,
}

/// Transform a `TracePattern` into a concrete `LoweringRecipe`.
///
/// This is the core of the jtransform layer. It decides HOW each
/// pattern should be lowered to IR, based on the pattern kind and
/// optional configuration.
pub fn transform_pattern(pattern: &TracePattern, config: &TransformConfig) -> LoweringRecipe {
    match pattern {
        TracePattern::UnboxIntBinop { op_name, has_overflow_guard } => {
            let ir_opcode = map_binop_name(op_name);
            if *has_overflow_guard {
                LoweringRecipe::IntBinopWithOverflow {
                    ir_opcode: format!("{ir_opcode}Ovf"),
                    guard: "GuardNoOverflow".into(),
                    unbox_fn: config.int_unbox_fn.clone(),
                    box_fn: config.int_box_fn.clone(),
                }
            } else {
                LoweringRecipe::IntBinopNoOverflow {
                    ir_opcode,
                    unbox_fn: config.int_unbox_fn.clone(),
                    box_fn: config.int_box_fn.clone(),
                }
            }
        }
        TracePattern::UnboxFloatBinop { op_name } => {
            LoweringRecipe::FloatBinop {
                ir_opcode: map_float_binop_name(op_name),
                unbox_fn: config.float_unbox_fn.clone(),
                box_fn: config.float_box_fn.clone(),
            }
        }
        TracePattern::UnboxIntCompare { op_name } => {
            LoweringRecipe::IntCompare {
                ir_opcode: map_compare_name(op_name),
                unbox_fn: config.int_unbox_fn.clone(),
                box_fn: config.bool_box_fn.clone(),
            }
        }
        TracePattern::UnboxIntUnary { op_name } => {
            LoweringRecipe::IntUnary {
                ir_opcode: map_unary_name(op_name),
                unbox_fn: config.int_unbox_fn.clone(),
                box_fn: config.int_box_fn.clone(),
            }
        }
        TracePattern::LocalRead => LoweringRecipe::LocalRead {
            array_field: config.locals_field.clone(),
            index_source: "oparg".into(),
        },
        TracePattern::LocalWrite => LoweringRecipe::LocalWrite {
            array_field: config.locals_field.clone(),
            index_source: "oparg".into(),
        },
        TracePattern::ConstLoad => LoweringRecipe::ConstLoad {
            pool_field: config.consts_field.clone(),
        },
        TracePattern::TruthCheck => LoweringRecipe::TruthCheck {
            unbox_fn: config.int_unbox_fn.clone(),
        },
        TracePattern::FunctionCall => LoweringRecipe::FunctionCall {
            use_call_assembler: config.use_call_assembler,
        },
        TracePattern::RangeIterNext => LoweringRecipe::IterNext {
            step_fn: "range_iter_next".into(),
        },
        TracePattern::SequenceGetitem => LoweringRecipe::ResidualCall {
            helper_name: "sequence_getitem".into(),
        },
        TracePattern::Jump => LoweringRecipe::Jump,
        TracePattern::ConditionalJump => LoweringRecipe::ConditionalJump {
            guard_kind: "GuardTrue".into(),
        },
        TracePattern::Return => LoweringRecipe::Return,
        TracePattern::NamespaceAccess { is_load, is_global } => {
            LoweringRecipe::NamespaceAccess {
                is_load: *is_load,
                is_global: *is_global,
            }
        }
        TracePattern::BuildCollection { kind } => LoweringRecipe::BuildCollection {
            kind: kind.clone(),
        },
        TracePattern::UnpackSequence => LoweringRecipe::UnpackSequence,
        TracePattern::SequenceSetitem => LoweringRecipe::SequenceSetitem,
        TracePattern::CollectionAppend => LoweringRecipe::CollectionAppend,
        TracePattern::StackManip => LoweringRecipe::StackManip,
        TracePattern::IterCleanup => LoweringRecipe::IterCleanup,
        TracePattern::Noop => LoweringRecipe::Noop,
        TracePattern::Residual { helper_name } => LoweringRecipe::ResidualCall {
            helper_name: helper_name.clone(),
        },
        TracePattern::Unknown => LoweringRecipe::Abort,
    }
}

/// Configuration for the jtransform layer.
///
/// Interpreter-specific names for unbox/box functions, field names, etc.
pub struct TransformConfig {
    pub int_unbox_fn: String,
    pub int_box_fn: String,
    pub float_unbox_fn: String,
    pub float_box_fn: String,
    pub bool_box_fn: String,
    pub locals_field: String,
    pub consts_field: String,
    pub use_call_assembler: bool,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            int_unbox_fn: "trace_unbox_int".into(),
            int_box_fn: "trace_box_int".into(),
            float_unbox_fn: "trace_unbox_float".into(),
            float_box_fn: "trace_box_float".into(),
            bool_box_fn: "trace_box_bool".into(),
            locals_field: "locals_w".into(),
            consts_field: "co_consts".into(),
            use_call_assembler: true,
        }
    }
}

/// Transform all opcodes in an analysis result.
pub fn transform_all(
    opcodes: &[crate::OpcodeArm],
    config: &TransformConfig,
) -> Vec<(String, LoweringRecipe)> {
    opcodes
        .iter()
        .map(|arm| {
            let recipe = arm
                .trace_pattern
                .as_ref()
                .map(|p| transform_pattern(p, config))
                .unwrap_or(LoweringRecipe::Abort);
            (arm.pattern.clone(), recipe)
        })
        .collect()
}

// ── Mapping helpers ──

fn map_binop_name(op_name: &str) -> String {
    match op_name {
        "dispatch" | "add" => "IntAdd".into(),
        "sub" => "IntSub".into(),
        "mul" => "IntMul".into(),
        "div" | "floor_div" => "IntFloorDiv".into(),
        "mod" | "modulo" => "IntMod".into(),
        other => format!("Int{}", capitalize(other)),
    }
}

fn map_float_binop_name(op_name: &str) -> String {
    match op_name {
        "dispatch" | "FloatAdd" | "add" => "FloatAdd".into(),
        "sub" => "FloatSub".into(),
        "mul" => "FloatMul".into(),
        "div" | "true_div" => "FloatTrueDiv".into(),
        other => format!("Float{}", capitalize(other)),
    }
}

fn map_compare_name(op_name: &str) -> String {
    match op_name {
        "dispatch" | "lt" => "IntLt".into(),
        "le" => "IntLe".into(),
        "eq" => "IntEq".into(),
        "ne" => "IntNe".into(),
        "gt" => "IntGt".into(),
        "ge" => "IntGe".into(),
        other => format!("Int{}", capitalize(other)),
    }
}

fn map_unary_name(op_name: &str) -> String {
    match op_name {
        "unary_negative" | "neg" => "IntNeg".into(),
        "unary_invert" | "invert" => "IntInvert".into(),
        other => format!("Int{}", capitalize(other)),
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_int_binop_with_overflow() {
        let pattern = TracePattern::UnboxIntBinop {
            op_name: "dispatch".into(),
            has_overflow_guard: true,
        };
        let config = TransformConfig::default();
        let recipe = transform_pattern(&pattern, &config);
        match recipe {
            LoweringRecipe::IntBinopWithOverflow { ir_opcode, guard, .. } => {
                assert_eq!(ir_opcode, "IntAddOvf");
                assert_eq!(guard, "GuardNoOverflow");
            }
            other => panic!("expected IntBinopWithOverflow, got {other:?}"),
        }
    }

    #[test]
    fn test_transform_local_read() {
        let pattern = TracePattern::LocalRead;
        let config = TransformConfig::default();
        let recipe = transform_pattern(&pattern, &config);
        match recipe {
            LoweringRecipe::LocalRead { array_field, .. } => {
                assert_eq!(array_field, "locals_w");
            }
            other => panic!("expected LocalRead, got {other:?}"),
        }
    }

    #[test]
    fn test_transform_residual() {
        let pattern = TracePattern::Residual {
            helper_name: "make_function".into(),
        };
        let config = TransformConfig::default();
        let recipe = transform_pattern(&pattern, &config);
        match recipe {
            LoweringRecipe::ResidualCall { helper_name } => {
                assert_eq!(helper_name, "make_function");
            }
            other => panic!("expected ResidualCall, got {other:?}"),
        }
    }

    #[test]
    fn test_transform_unknown_becomes_abort() {
        let pattern = TracePattern::Unknown;
        let config = TransformConfig::default();
        let recipe = transform_pattern(&pattern, &config);
        assert!(matches!(recipe, LoweringRecipe::Abort));
    }

    #[test]
    fn test_transform_all_produces_recipes_for_all_arms() {
        let arms = vec![
            crate::OpcodeArm {
                pattern: "BinaryOp".into(),
                handler_calls: vec![],
                resolved_calls: vec![],
                trace_pattern: Some(TracePattern::UnboxIntBinop {
                    op_name: "add".into(),
                    has_overflow_guard: false,
                }),
            },
            crate::OpcodeArm {
                pattern: "LoadFast".into(),
                handler_calls: vec![],
                resolved_calls: vec![],
                trace_pattern: Some(TracePattern::LocalRead),
            },
            crate::OpcodeArm {
                pattern: "Unknown".into(),
                handler_calls: vec![],
                resolved_calls: vec![],
                trace_pattern: None,
            },
        ];
        let config = TransformConfig::default();
        let recipes = transform_all(&arms, &config);
        assert_eq!(recipes.len(), 3);
        assert!(matches!(recipes[0].1, LoweringRecipe::IntBinopNoOverflow { .. }));
        assert!(matches!(recipes[1].1, LoweringRecipe::LocalRead { .. }));
        assert!(matches!(recipes[2].1, LoweringRecipe::Abort));
    }
}
