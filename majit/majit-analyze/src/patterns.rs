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
// classify_method_body_with_vable, VirtualizableClassifyConfig, and
// opcode-pattern-text fallback have been removed. Use classify_from_graph()
// and pipeline opcode dispatch instead.

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
    let mut field_reads: Vec<crate::graph::FieldDescriptor> = Vec::new();
    let mut field_writes: Vec<crate::graph::FieldDescriptor> = Vec::new();
    let mut array_reads = 0usize;
    let mut array_writes = 0usize;
    let mut call_descriptors: Vec<crate::call_match::CallDescriptor> = Vec::new();
    let mut has_guard = false;

    let mut first_vable_field_read: Option<(usize, crate::graph::ValueType)> = None;
    let mut first_vable_field_write: Option<(usize, crate::graph::ValueType)> = None;
    let mut first_vable_array_read: Option<(usize, crate::graph::ValueType)> = None;
    let mut first_vable_array_write: Option<(usize, crate::graph::ValueType)> = None;

    // Analyze ALL blocks (not just entry) for multi-block graphs
    for block in &graph.blocks {
        for op in &block.ops {
            match &op.kind {
                OpKind::FieldRead { field, .. } => field_reads.push(field.clone()),
                OpKind::FieldWrite { field, .. } => field_writes.push(field.clone()),
                OpKind::ArrayRead { .. } => array_reads += 1,
                OpKind::ArrayWrite { .. } => array_writes += 1,
                OpKind::CallElidable { descriptor, .. }
                | OpKind::CallResidual { descriptor, .. }
                | OpKind::CallMayForce { descriptor, .. } => {
                    call_descriptors.push(descriptor.clone());
                }
                OpKind::GuardTrue { .. } | OpKind::GuardFalse { .. } => has_guard = true,
                OpKind::BinOp { .. } | OpKind::UnaryOp { .. } => {}
                OpKind::VableFieldRead { field_index, ty } => {
                    first_vable_field_read.get_or_insert((*field_index, ty.clone()));
                }
                OpKind::VableFieldWrite {
                    field_index, ty, ..
                } => {
                    first_vable_field_write.get_or_insert((*field_index, ty.clone()));
                }
                OpKind::VableArrayRead {
                    array_index,
                    item_ty,
                    ..
                } => {
                    first_vable_array_read.get_or_insert((*array_index, item_ty.clone()));
                }
                OpKind::VableArrayWrite {
                    array_index,
                    item_ty,
                    ..
                } => {
                    first_vable_array_write.get_or_insert((*array_index, item_ty.clone()));
                }
                _ => {}
            }
        }
    }

    // Classify based on op patterns (RPython jtransform-level analysis)
    // Virtualizable-specialized ops come from the graph rewrite pass and
    // should take precedence over legacy local/heap classifications.
    if let Some((array_index, item_ty)) = first_vable_array_read {
        return Some(TracePattern::VableArrayRead {
            array_index,
            item_type: value_type_name(&item_ty).into(),
        });
    }
    if let Some((array_index, item_ty)) = first_vable_array_write {
        return Some(TracePattern::VableArrayWrite {
            array_index,
            item_type: value_type_name(&item_ty).into(),
        });
    }
    if let Some((field_index, field_ty)) = first_vable_field_read {
        return Some(TracePattern::VableFieldRead {
            field_index,
            field_type: value_type_name(&field_ty).into(),
        });
    }
    if let Some((field_index, field_ty)) = first_vable_field_write {
        return Some(TracePattern::VableFieldWrite {
            field_index,
            field_type: value_type_name(&field_ty).into(),
        });
    }

    // Integer binary operations
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_int_arithmetic_target(&descriptor.target))
    {
        return Some(TracePattern::UnboxIntBinop {
            op_name: "dispatch".into(),
            has_overflow_guard: true,
        });
    }

    // Float binary operations
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_float_arithmetic_target(&descriptor.target))
    {
        return Some(TracePattern::UnboxFloatBinop {
            op_name: "FloatAdd".into(),
        });
    }

    // Local read: reads from locals_w array (field read + array read, or push call)
    if (field_reads
        .iter()
        .any(crate::field_match::is_local_array_field)
        && array_reads > 0)
        || (array_reads > 0
            && call_descriptors
                .iter()
                .any(|descriptor| crate::call_match::is_local_read_target(&descriptor.target)))
    {
        return Some(TracePattern::LocalRead);
    }

    // Local write: writes to locals_w array (field write + array write, or pop call)
    if (field_writes
        .iter()
        .any(crate::field_match::is_local_array_field)
        && array_writes > 0)
        || field_writes
            .iter()
            .any(crate::field_match::is_local_array_field)
        || (array_writes > 0
            && call_descriptors
                .iter()
                .any(|descriptor| crate::call_match::is_local_write_target(&descriptor.target)))
    {
        return Some(TracePattern::LocalWrite);
    }

    // Constant load (reads from constants/co_consts)
    if field_reads
        .iter()
        .any(crate::field_match::is_constant_pool_field)
    {
        return Some(TracePattern::ConstLoad);
    }

    // Function call
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_function_invoke_target(&descriptor.target))
    {
        return Some(TracePattern::FunctionCall);
    }

    // Truth check
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_truth_check_target(&descriptor.target))
    {
        return Some(TracePattern::TruthCheck);
    }

    // Stack manipulation (pop/swap/peek without array access)
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_stack_manip_target(&descriptor.target))
        && array_reads == 0
        && array_writes == 0
    {
        return Some(TracePattern::StackManip);
    }

    // Namespace access (load/store name/global)
    let namespace_kinds: Vec<_> = call_descriptors
        .iter()
        .filter_map(|descriptor| crate::call_match::namespace_access_kind(&descriptor.target))
        .collect();
    if !namespace_kinds.is_empty() {
        return Some(TracePattern::NamespaceAccess {
            is_load: namespace_kinds.iter().any(|kind| {
                matches!(
                    kind,
                    crate::call_match::NamespaceAccessKind::LoadLocal
                        | crate::call_match::NamespaceAccessKind::LoadGlobal
                )
            }),
            is_global: namespace_kinds.iter().any(|kind| {
                matches!(
                    kind,
                    crate::call_match::NamespaceAccessKind::LoadGlobal
                        | crate::call_match::NamespaceAccessKind::StoreGlobal
                )
            }),
        });
    }

    // Iterator
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_range_iter_next_target(&descriptor.target))
    {
        return Some(TracePattern::RangeIterNext);
    }
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_iter_cleanup_target(&descriptor.target))
    {
        return Some(TracePattern::IterCleanup);
    }

    // Return (few ops ending in Return terminator)
    if matches!(&entry.terminator, crate::graph::Terminator::Return(Some(_))) && ops.len() <= 5 {
        if call_descriptors
            .iter()
            .any(|descriptor| crate::call_match::is_return_target(&descriptor.target))
        {
            return Some(TracePattern::Return);
        }
    }

    // Conditional jump (guard + pc/next_instr write)
    if has_guard
        && field_writes
            .iter()
            .any(crate::field_match::is_instruction_position_field)
    {
        return Some(TracePattern::ConditionalJump);
    }

    // Unconditional jump (pc write without guard)
    if !has_guard
        && field_writes
            .iter()
            .any(crate::field_match::is_instruction_position_field)
    {
        return Some(TracePattern::Jump);
    }

    // Build collection
    if let Some(kind) = call_descriptors
        .iter()
        .find_map(|descriptor| crate::call_match::build_collection_kind(&descriptor.target))
    {
        return Some(TracePattern::BuildCollection {
            kind: match kind {
                crate::call_match::BuildCollectionKind::List => "list",
                crate::call_match::BuildCollectionKind::Tuple => "tuple",
            }
            .into(),
        });
    }

    // Sequence operations
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_unpack_sequence_target(&descriptor.target))
    {
        return Some(TracePattern::UnpackSequence);
    }
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_sequence_setitem_target(&descriptor.target))
    {
        return Some(TracePattern::SequenceSetitem);
    }
    if call_descriptors
        .iter()
        .any(|descriptor| crate::call_match::is_collection_append_target(&descriptor.target))
    {
        return Some(TracePattern::CollectionAppend);
    }

    // Noop (empty or trivial ops only)
    if ops
        .iter()
        .all(|op| matches!(&op.kind, OpKind::Input { .. } | OpKind::Unknown { .. }))
    {
        return Some(TracePattern::Noop);
    }

    None
}

fn value_type_name(ty: &crate::graph::ValueType) -> &'static str {
    match ty {
        crate::graph::ValueType::Int => "int",
        crate::graph::ValueType::Ref => "ref",
        crate::graph::ValueType::Float => "float",
        crate::graph::ValueType::Void => "void",
        crate::graph::ValueType::State => "state",
        crate::graph::ValueType::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CallTarget, MajitGraph, OpKind, Terminator, ValueType};
    use crate::passes::{GraphTransformConfig, rewrite_graph};

    #[test]
    fn classify_int_binop_from_graph() {
        let mut graph = MajitGraph::new("binary_op");
        let entry = graph.entry;
        graph.push_op(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["w_int_add"]),
                args: vec![],
                result_ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(entry, Terminator::Return(None));

        let rewritten = rewrite_graph(&graph, &GraphTransformConfig::default());
        let pattern = classify_from_graph(&rewritten.graph);
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
                field: crate::graph::FieldDescriptor::new("locals_w", Some("Frame".into())),
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
    fn classify_pyframe_local_read_from_graph() {
        let mut graph = MajitGraph::new("load_fast");
        let entry = graph.entry;
        let base = graph.alloc_value();
        graph.push_op(
            entry,
            OpKind::FieldRead {
                base,
                field: crate::graph::FieldDescriptor::new(
                    "locals_cells_stack_w",
                    Some("PyFrame".into()),
                ),
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
    fn classify_rpython_last_instr_jump_from_graph() {
        let mut graph = MajitGraph::new("jump_absolute");
        let entry = graph.entry;
        let base = graph.alloc_value();
        let value = graph.alloc_value();
        graph.push_op(
            entry,
            OpKind::FieldWrite {
                base,
                field: crate::graph::FieldDescriptor::new("last_instr", Some("PyFrame".into())),
                value,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_terminator(entry, Terminator::Return(None));

        let pattern = classify_from_graph(&graph);
        assert_eq!(pattern, Some(TracePattern::Jump));
    }

    #[test]
    fn classify_truth_check_from_graph() {
        let mut graph = MajitGraph::new("unary_not");
        let entry = graph.entry;
        graph.push_op(
            entry,
            OpKind::Call {
                target: CallTarget::method("truth_value", Some("PyFrame".into())),
                args: vec![],
                result_ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(entry, Terminator::Return(None));

        let rewritten = rewrite_graph(
            &graph,
            &crate::test_support::pyre_pipeline_config().transform,
        );
        assert_eq!(
            classify_from_graph(&rewritten.graph),
            Some(TracePattern::TruthCheck)
        );
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

    #[test]
    fn classify_vable_array_read_from_graph() {
        let mut graph = MajitGraph::new("load_fast_vable");
        let entry = graph.entry;
        let idx = graph.alloc_value();
        graph.push_op(
            entry,
            OpKind::VableArrayRead {
                array_index: 0,
                elem_index: idx,
                item_ty: ValueType::Ref,
            },
            true,
        );
        graph.set_terminator(entry, Terminator::Return(None));

        assert_eq!(
            classify_from_graph(&graph),
            Some(TracePattern::VableArrayRead {
                array_index: 0,
                item_type: "ref".into(),
            })
        );
    }
}
