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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TracePattern {
    /// Unbox two int operands → binary IR op → box result.
    /// Example: a + b where a, b are W_IntObject
    UnboxIntBinop {
        op_name: String,
        has_overflow_guard: bool,
    },

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

    /// Stack manipulation (pop, swap, copy, dup, rot).
    StackManip,

    /// Unconditional jump (forward/backward).
    Jump,

    /// Conditional jump (pop_jump_if_true/false).
    ConditionalJump,

    /// Return value from function.
    Return,

    /// Namespace access (load/store name/global).
    NamespaceAccess { is_load: bool, is_global: bool },

    /// Iterator cleanup (end_for, pop_iter).
    IterCleanup,

    /// No-op instructions (nop, cache, resume, extended_arg).
    Noop,

    /// Collection construction (build_list, build_tuple, build_map).
    BuildCollection { kind: String },

    /// Unpack sequence into locals.
    UnpackSequence,

    /// Collection item write (store_subscr).
    SequenceSetitem,

    /// Collection append (list_append).
    CollectionAppend,

    /// Virtualizable field read (jtransform.py:760 getfield_vable).
    VableFieldRead {
        field_index: usize,
        field_type: String,
    },

    /// Virtualizable field write (jtransform.py setfield_vable).
    VableFieldWrite {
        field_index: usize,
        field_type: String,
    },

    /// Virtualizable array item read (jtransform.py getarrayitem_vable).
    VableArrayRead {
        array_index: usize,
        item_type: String,
    },

    /// Virtualizable array item write (jtransform.py setarrayitem_vable).
    VableArrayWrite {
        array_index: usize,
        item_type: String,
    },

    /// Virtualizable array length (jtransform.py arraylen_vable).
    VableArrayLen { array_index: usize },

    /// Opaque — emit residual call.
    Residual { helper_name: String },

    /// Not yet classified.
    Unknown,
}

// String-based classify_from_resolved, classify_method_body,
// classify_method_body_with_vable, and VirtualizableClassifyConfig
// have been removed. Use classify_from_graph() instead.

/// Classify from the opcode pattern text (fallback when call chain resolution fails).
///
/// Handles cases where the match arm body doesn't call a named handler method
/// (e.g., no-op arms that just return `Ok(StepResult::Continue)`).
pub fn classify_from_pattern(pattern: &str) -> Option<TracePattern> {
    // No-op instructions
    if pattern.contains("ExtendedArg")
        || pattern.contains("Resume")
        || pattern.contains("Nop")
        || pattern.contains("Cache")
        || pattern.contains("NotTaken")
    {
        return Some(TracePattern::Noop);
    }

    // Local variable access (superword instructions)
    if pattern.contains("LoadFastBorrowLoadFastBorrow") || pattern.contains("LoadFastLoadFast") {
        return Some(TracePattern::LocalRead);
    }
    if pattern.contains("StoreFastStoreFast") {
        return Some(TracePattern::LocalWrite);
    }
    if pattern.contains("StoreFastLoadFast") {
        return Some(TracePattern::LocalWrite);
    }

    // Namespace
    if pattern.contains("StoreName") || pattern.contains("StoreGlobal") {
        return Some(TracePattern::NamespaceAccess {
            is_load: false,
            is_global: pattern.contains("Global"),
        });
    }
    if pattern.contains("LoadGlobal") {
        return Some(TracePattern::NamespaceAccess {
            is_load: true,
            is_global: true,
        });
    }
    if pattern.contains("LoadName") {
        return Some(TracePattern::NamespaceAccess {
            is_load: true,
            is_global: false,
        });
    }

    // Stack
    if pattern.contains("PopTop")
        || pattern.contains("PushNull")
        || pattern.contains("Copy")
        || pattern.contains("Swap")
    {
        return Some(TracePattern::StackManip);
    }

    // Jumps
    if pattern.contains("JumpForward") || pattern.contains("JumpBackward") {
        return Some(TracePattern::Jump);
    }
    if pattern.contains("PopJumpIf") {
        return Some(TracePattern::ConditionalJump);
    }

    // Return
    if pattern.contains("ReturnValue") {
        return Some(TracePattern::Return);
    }

    // Iterator cleanup
    if pattern.contains("EndFor") || pattern.contains("PopIter") {
        return Some(TracePattern::IterCleanup);
    }

    // Constants
    if pattern.contains("LoadSmallInt") {
        return Some(TracePattern::ConstLoad);
    }

    // Collection construction
    if pattern.contains("BuildList") {
        return Some(TracePattern::BuildCollection {
            kind: "list".into(),
        });
    }
    if pattern.contains("BuildTuple") {
        return Some(TracePattern::BuildCollection {
            kind: "tuple".into(),
        });
    }
    if pattern.contains("BuildMap") {
        return Some(TracePattern::BuildCollection { kind: "map".into() });
    }

    // Unpack sequence
    if pattern.contains("UnpackSequence") {
        return Some(TracePattern::UnpackSequence);
    }

    // Collection item write
    if pattern.contains("StoreSubscr") {
        return Some(TracePattern::SequenceSetitem);
    }

    // Collection append
    if pattern.contains("ListAppend") {
        return Some(TracePattern::CollectionAppend);
    }

    // MakeFunction (true residual)
    if pattern.contains("MakeFunction") {
        return Some(TracePattern::Residual {
            helper_name: "make_function".into(),
        });
    }

    None
}

// ── Graph-based classification ──────────────────────────────────
//
// Replaces body_summary string heuristics with semantic graph analysis.
// RPython equivalent: annotator + jtransform working on the flow graph
// rather than string matching.

/// Classify a function from its semantic graph.
///
/// This is the graph-based replacement for `classify_method_body()`.
/// Instead of matching substrings in body_summary, it analyzes the
/// actual ops in the MajitGraph.
pub fn classify_from_graph(graph: &crate::MajitGraph) -> Option<TracePattern> {
    use crate::graph::OpKind;

    let entry = graph.block(graph.entry);
    let ops = &entry.ops;

    // Count op kinds
    let mut field_reads: Vec<String> = Vec::new();
    let mut field_writes: Vec<String> = Vec::new();
    let mut array_reads = 0usize;
    let mut array_writes = 0usize;
    let mut calls: Vec<String> = Vec::new();
    let mut has_guard = false;

    let mut has_vable_field = false;
    let mut has_vable_array = false;

    // Analyze ALL blocks (not just entry) for multi-block graphs
    for block in &graph.blocks {
        for op in &block.ops {
            match &op.kind {
                OpKind::FieldRead { field, .. } => field_reads.push(field.clone()),
                OpKind::FieldWrite { field, .. } => field_writes.push(field.clone()),
                OpKind::ArrayRead { .. } => array_reads += 1,
                OpKind::ArrayWrite { .. } => array_writes += 1,
                OpKind::Call { target, .. }
                | OpKind::CallElidable { target, .. }
                | OpKind::CallResidual { target, .. }
                | OpKind::CallMayForce { target, .. } => calls.push(target.clone()),
                OpKind::GuardTrue { .. } | OpKind::GuardFalse { .. } => has_guard = true,
                OpKind::VableFieldRead { .. } | OpKind::VableFieldWrite { .. } => {
                    has_vable_field = true;
                }
                OpKind::VableArrayRead { .. } | OpKind::VableArrayWrite { .. } => {
                    has_vable_array = true;
                }
                _ => {}
            }
        }
    }

    // Classify based on op patterns (RPython jtransform-level analysis)

    // Integer binary operations
    if calls.iter().any(|c| {
        c.contains("w_int_add") || c.contains("w_int_sub") || c.contains("w_int_mul")
    }) {
        return Some(TracePattern::UnboxIntBinop {
            op_name: "dispatch".into(),
            has_overflow_guard: true,
        });
    }

    // Float binary operations
    if calls
        .iter()
        .any(|c| c.contains("w_float_add") || c.contains("w_float_sub"))
    {
        return Some(TracePattern::UnboxFloatBinop {
            op_name: "FloatAdd".into(),
        });
    }

    // Local read: reads from locals_w array (field read + array read, or push call)
    if (field_reads.iter().any(|f| f == "locals_w") && array_reads > 0)
        || (array_reads > 0 && calls.iter().any(|c| c == "push" || c == "push_value"))
    {
        return Some(TracePattern::LocalRead);
    }

    // Local write: writes to locals_w array (field write + array write, or pop call)
    if (field_writes.iter().any(|f| f == "locals_w") && array_writes > 0)
        || field_writes.iter().any(|f| f == "locals_w")
        || (array_writes > 0 && calls.iter().any(|c| c == "pop" || c == "pop_value"))
    {
        return Some(TracePattern::LocalWrite);
    }

    // Constant load (reads from constants/co_consts)
    if field_reads
        .iter()
        .any(|f| f.contains("constants") || f.contains("co_consts"))
    {
        return Some(TracePattern::ConstLoad);
    }

    // Function call
    if calls
        .iter()
        .any(|c| c.contains("call_function") || c.contains("invoke"))
    {
        return Some(TracePattern::FunctionCall);
    }

    // Truth check
    if calls
        .iter()
        .any(|c| c.contains("truth") || c.contains("bool") || c.contains("is_true"))
    {
        return Some(TracePattern::TruthCheck);
    }

    // Stack manipulation (pop/swap/peek without array access)
    if calls
        .iter()
        .any(|c| c == "pop_value" || c == "swap_values" || c == "peek_at" || c == "copy_value")
        && array_reads == 0
        && array_writes == 0
    {
        return Some(TracePattern::StackManip);
    }

    // Namespace access (load/store name/global)
    if calls
        .iter()
        .any(|c| c.contains("load_name") || c.contains("store_name"))
    {
        return Some(TracePattern::NamespaceAccess {
            is_load: calls.iter().any(|c| c.contains("load")),
            is_global: calls.iter().any(|c| c.contains("global")),
        });
    }

    // Iterator
    if calls.iter().any(|c| c.contains("iter_next") || c.contains("for_iter")) {
        return Some(TracePattern::RangeIterNext);
    }
    if calls.iter().any(|c| c.contains("end_for") || c.contains("pop_iter")) {
        return Some(TracePattern::IterCleanup);
    }

    // Return (few ops ending in Return terminator)
    if matches!(
        &entry.terminator,
        crate::graph::Terminator::Return(Some(_))
    ) && ops.len() <= 5
    {
        if calls.iter().any(|c| c.contains("finish") || c.contains("return")) {
            return Some(TracePattern::Return);
        }
    }

    // Conditional jump (guard + pc/next_instr write)
    if has_guard
        && field_writes
            .iter()
            .any(|f| f.contains("next_instr") || f.contains("pc"))
    {
        return Some(TracePattern::ConditionalJump);
    }

    // Unconditional jump (pc write without guard)
    if !has_guard
        && field_writes
            .iter()
            .any(|f| f.contains("next_instr") || f.contains("pc"))
    {
        return Some(TracePattern::Jump);
    }

    // Build collection
    if calls.iter().any(|c| c.contains("build_list") || c.contains("new_list")) {
        return Some(TracePattern::BuildCollection { kind: "list".into() });
    }
    if calls.iter().any(|c| c.contains("build_tuple") || c.contains("new_tuple")) {
        return Some(TracePattern::BuildCollection { kind: "tuple".into() });
    }

    // Sequence operations
    if calls.iter().any(|c| c.contains("unpack")) {
        return Some(TracePattern::UnpackSequence);
    }
    if calls.iter().any(|c| c.contains("store_subscr") || c.contains("setitem")) {
        return Some(TracePattern::SequenceSetitem);
    }
    if calls.iter().any(|c| c.contains("list_append") || c.contains("append")) {
        return Some(TracePattern::CollectionAppend);
    }

    // Noop (empty or trivial ops only)
    if ops.iter().all(|op| matches!(&op.kind, OpKind::Input { .. } | OpKind::Unknown { .. }))
        && ops.len() <= 2
    {
        return Some(TracePattern::Noop);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MajitGraph, OpKind, Terminator, ValueType};

    #[test]
    fn classify_int_binop_from_graph() {
        let mut graph = MajitGraph::new("binary_op");
        let entry = graph.entry;
        graph.push_op(
            entry,
            OpKind::Call {
                target: "w_int_add".into(),
                args: vec![],
                result_ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(entry, Terminator::Return(None));

        let pattern = classify_from_graph(&graph);
        assert!(
            matches!(pattern, Some(TracePattern::UnboxIntBinop { .. })),
            "w_int_add call should classify as UnboxIntBinop, got {:?}",
            pattern
        );
    }

    #[test]
    fn classify_local_read_from_graph() {
        let mut graph = MajitGraph::new("load_fast");
        let entry = graph.entry;
        let base = graph.alloc_value();
        graph.push_op(
            entry,
            OpKind::FieldRead {
                base,
                field: "locals_w".into(),
                ty: ValueType::Unknown,
            },
            true,
        );
        let arr = graph.alloc_value();
        let idx = graph.alloc_value();
        graph.push_op(
            entry,
            OpKind::ArrayRead {
                base: arr,
                index: idx,
                item_ty: ValueType::Unknown,
            },
            true,
        );
        graph.set_terminator(entry, Terminator::Return(None));

        let pattern = classify_from_graph(&graph);
        assert_eq!(pattern, Some(TracePattern::LocalRead));
    }

    #[test]
    fn classify_truth_check_from_graph() {
        let mut graph = MajitGraph::new("unary_not");
        let entry = graph.entry;
        graph.push_op(
            entry,
            OpKind::Call {
                target: "is_true".into(),
                args: vec![],
                result_ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(entry, Terminator::Return(None));

        assert_eq!(classify_from_graph(&graph), Some(TracePattern::TruthCheck));
    }

    #[test]
    fn classify_noop_from_graph() {
        let mut graph = MajitGraph::new("nop");
        let entry = graph.entry;
        graph.push_op(
            entry,
            OpKind::Input {
                name: "self".into(),
                ty: ValueType::Unknown,
            },
            true,
        );
        graph.set_terminator(entry, Terminator::Return(None));

        assert_eq!(classify_from_graph(&graph), Some(TracePattern::Noop));
    }
}
