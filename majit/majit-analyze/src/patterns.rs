//! Trace pattern recognition.
//!
//! Identifies common interpreter patterns and maps them to IR templates:
//! - UnboxIntBinop: unbox two ints → binary op → box result
//! - UnboxFloatBinop: same for floats
//! - LocalRead/LocalWrite: frame local variable access
//! - FieldRead/FieldWrite: object field access
//! - TruthCheck: convert value to bool
//! - BoxInt/BoxFloat: allocate boxed numeric value

use serde::{Deserialize, Serialize};

/// Recognized trace patterns for automatic IR generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TracePattern {
    /// Unbox two int operands → binary IR op → box result.
    /// Example: a + b where a, b are W_IntObject
    UnboxIntBinop { op_name: String, has_overflow_guard: bool },

    /// Unbox two float operands → binary IR op → box result.
    UnboxFloatBinop { op_name: String },

    /// Unbox two int operands → comparison IR op → box bool result.
    UnboxIntCompare { op_name: String },

    /// Read from frame locals array.
    LocalRead,

    /// Write to frame locals array.
    LocalWrite,

    /// Push a constant onto the stack.
    ConstLoad,

    /// Convert a value to boolean (truth check).
    TruthCheck,

    /// Range iterator next value.
    RangeIterNext,

    /// Unary int operation (negate, invert).
    UnboxIntUnary { op_name: String },

    /// List/tuple subscript.
    SequenceGetitem,

    /// Function call (dispatch to CALL_ASSEMBLER/inline/residual).
    FunctionCall,

    /// Stack manipulation (swap, dup, rot).
    StackManip,

    /// Opaque — emit residual call.
    Residual { helper_name: String },

    /// Not yet classified.
    Unknown,
}

/// Classify an opcode from its resolved call chain.
pub fn classify_from_resolved(calls: &[crate::ResolvedCall]) -> Option<TracePattern> {
    for call in calls {
        // Check the handler method name and its body
        let name = &call.name;
        let body = &call.body_summary;

        match name.as_str() {
            "binary_op" => {
                // ArithmeticOpcodeHandler::binary_op → dispatches to w_binary_op etc.
                return Some(TracePattern::UnboxIntBinop {
                    op_name: "dispatch".into(),
                    has_overflow_guard: true,
                });
            }
            "compare_op" => {
                return Some(TracePattern::UnboxIntCompare {
                    op_name: "dispatch".into(),
                });
            }
            "unary_negative" | "unary_invert" => {
                return Some(TracePattern::UnboxIntUnary {
                    op_name: name.clone(),
                });
            }
            "unary_not" => {
                return Some(TracePattern::TruthCheck);
            }
            "load_fast" | "load_fast_checked" => return Some(TracePattern::LocalRead),
            "store_fast" | "store_fast_checked" => return Some(TracePattern::LocalWrite),
            "load_const" => return Some(TracePattern::ConstLoad),
            "call" => return Some(TracePattern::FunctionCall),
            "for_iter" => return Some(TracePattern::RangeIterNext),
            "get_iter" | "build_list" | "build_tuple" | "build_map"
            | "unpack_sequence" | "store_subscr" | "list_append" => {
                return Some(TracePattern::Residual {
                    helper_name: name.clone(),
                });
            }
            _ => {
                // Try body heuristics
                if let Some(p) = classify_method_body(body) {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Classify a method body summary into a trace pattern.
pub fn classify_method_body(body_summary: &str) -> Option<TracePattern> {
    // Heuristic pattern matching on the body text
    if body_summary.contains("w_int_add")
        || body_summary.contains("w_int_sub")
        || body_summary.contains("w_int_mul")
    {
        return Some(TracePattern::UnboxIntBinop {
            op_name: "IntAddOvf".into(),
            has_overflow_guard: true,
        });
    }

    if body_summary.contains("w_float_add") || body_summary.contains("w_float_sub") {
        return Some(TracePattern::UnboxFloatBinop {
            op_name: "FloatAdd".into(),
        });
    }

    if body_summary.contains("locals_w") && body_summary.contains("push") {
        return Some(TracePattern::LocalRead);
    }

    if body_summary.contains("locals_w") && body_summary.contains("pop") {
        return Some(TracePattern::LocalWrite);
    }

    if body_summary.contains("constants") {
        return Some(TracePattern::ConstLoad);
    }

    if body_summary.contains("truth") || body_summary.contains("bool") {
        return Some(TracePattern::TruthCheck);
    }

    None
}
