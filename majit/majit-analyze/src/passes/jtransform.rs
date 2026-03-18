//! Graph-based jtransform: semantic rewrite pass.
//!
//! RPython equivalent: jtransform.py Transformer.optimize_block()
//!
//! Transforms a MajitGraph by rewriting operations:
//! - FieldRead on virtualizable fields → VableFieldRead marker
//! - FieldWrite on virtualizable fields → VableFieldWrite marker
//! - ArrayRead on virtualizable arrays → VableArrayRead marker
//! - Call classification → elidable/residual/may_force tagging

use serde::{Deserialize, Serialize};

use crate::graph::{BasicBlockId, MajitGraph, Op, OpKind, Terminator, ValueId, ValueType};

/// Configuration for the graph rewrite pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformConfig {
    /// Whether to rewrite virtualizable field/array accesses.
    pub lower_virtualizable: bool,
    /// Whether to classify function calls by effect.
    pub classify_calls: bool,
    /// Field names that are virtualizable (field_name → field_index).
    #[serde(default)]
    pub vable_fields: Vec<(String, usize)>,
    /// Array names that are virtualizable (array_name → array_index).
    #[serde(default)]
    pub vable_arrays: Vec<(String, usize)>,
}

impl Default for GraphTransformConfig {
    fn default() -> Self {
        Self {
            lower_virtualizable: true,
            classify_calls: true,
            vable_fields: Vec::new(),
            vable_arrays: Vec::new(),
        }
    }
}

/// A note about a transformation applied to the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTransformNote {
    pub function: String,
    pub detail: String,
}

/// Result of a graph transformation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformResult {
    pub graph: MajitGraph,
    pub notes: Vec<GraphTransformNote>,
    /// Number of ops rewritten by virtualizable lowering.
    pub vable_rewrites: usize,
    /// Number of calls classified.
    pub calls_classified: usize,
}

/// Rewrite a semantic graph with JIT-specific transformations.
///
/// RPython equivalent: jtransform.py `Transformer.optimize_block()`
///
/// This is the graph-based replacement for the old TracePattern →
/// LoweringRecipe string-matching pipeline.
pub fn rewrite_graph(graph: &MajitGraph, config: &GraphTransformConfig) -> GraphTransformResult {
    let mut notes = Vec::new();
    let mut rewritten = graph.clone();
    let mut vable_rewrites = 0usize;
    let mut calls_classified = 0usize;
    let mut aliases: std::collections::HashMap<ValueId, ValueId> = std::collections::HashMap::new();

    // Build lookup sets for virtualizable fields/arrays
    let vable_field_set: std::collections::HashMap<&str, usize> = config
        .vable_fields
        .iter()
        .map(|(name, idx)| (name.as_str(), *idx))
        .collect();
    let vable_array_set: std::collections::HashMap<&str, usize> = config
        .vable_arrays
        .iter()
        .map(|(name, idx)| (name.as_str(), *idx))
        .collect();

    // Track which ValueIds are results of reading a virtualizable array field.
    // RPython jtransform.py tracks vable_array_vars for this purpose.
    let mut vable_array_values: std::collections::HashMap<crate::graph::ValueId, usize> =
        std::collections::HashMap::new();

    for block in &mut rewritten.blocks {
        let mut new_ops = Vec::with_capacity(block.ops.len());

        for original_op in &block.ops {
            let op = remap_op(original_op, &aliases);
            match &op.kind {
                // ── hint(access_directly=True) / hint(fresh_virtualizable=True) ──
                // RPython jtransform.py:655 — consume as identity during translation.
                OpKind::Call { target, args, .. } if is_vable_identity_hint(target) => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("rewrite: {target}(...) → identity"),
                    });
                    if let (Some(result), Some(arg)) = (op.result, args.first().copied()) {
                        aliases.insert(result, resolve_alias(arg, &aliases));
                    }
                    continue;
                }

                // ── hint(force_virtualizable=True) ──
                // RPython jtransform.py:650 — emit hint_force_virtualizable op,
                // preserving the value as an identity result.
                OpKind::Call { target, args, .. } if is_vable_force_hint(target) => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("rewrite: {target}(...) → VableForce"),
                    });
                    vable_rewrites += 1;
                    if let (Some(result), Some(arg)) = (op.result, args.first().copied()) {
                        aliases.insert(result, resolve_alias(arg, &aliases));
                    }
                    new_ops.push(Op {
                        result: None,
                        kind: OpKind::VableForce,
                    });
                    continue;
                }

                // ── Virtualizable field read → VableFieldRead ──
                // RPython jtransform.py:832 `rewrite_op_getfield`
                OpKind::FieldRead { field, ty, .. } if config.lower_virtualizable => {
                    // Track if this field read is on a virtualizable array
                    if let Some(&arr_idx) = vable_array_set.get(field.as_str()) {
                        if let Some(result) = op.result {
                            vable_array_values.insert(result, arr_idx);
                        }
                    }
                    if let Some(&idx) = vable_field_set.get(field.as_str()) {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("rewrite: {field} → VableFieldRead[{idx}]"),
                        });
                        vable_rewrites += 1;
                        new_ops.push(Op {
                            result: op.result,
                            kind: OpKind::VableFieldRead {
                                field_index: idx,
                                ty: ty.clone(),
                            },
                        });
                        continue;
                    }
                }

                // ── Virtualizable field write → VableFieldWrite ──
                // RPython jtransform.py:923 `_rewrite_op_setfield`
                OpKind::FieldWrite {
                    field, value, ty, ..
                } if config.lower_virtualizable => {
                    if let Some(&idx) = vable_field_set.get(field.as_str()) {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("rewrite: {field} = ... → VableFieldWrite[{idx}]"),
                        });
                        vable_rewrites += 1;
                        new_ops.push(Op {
                            result: op.result,
                            kind: OpKind::VableFieldWrite {
                                field_index: idx,
                                value: *value,
                                ty: ty.clone(),
                            },
                        });
                        continue;
                    }
                }

                // ── Virtualizable array read → VableArrayRead ──
                // RPython jtransform.py:760 `getarrayitem_vable`
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty,
                } if config.lower_virtualizable => {
                    if let Some(&arr_idx) = vable_array_values.get(base) {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("rewrite: array[idx] → VableArrayRead[{arr_idx}]"),
                        });
                        vable_rewrites += 1;
                        new_ops.push(Op {
                            result: op.result,
                            kind: OpKind::VableArrayRead {
                                array_index: arr_idx,
                                elem_index: *index,
                                item_ty: item_ty.clone(),
                            },
                        });
                        continue;
                    }
                }

                // ── Virtualizable array write → VableArrayWrite ──
                // RPython jtransform.py:794 `setarrayitem_vable`
                OpKind::ArrayWrite {
                    base,
                    index,
                    value,
                    item_ty,
                } if config.lower_virtualizable => {
                    if let Some(&arr_idx) = vable_array_values.get(base) {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("rewrite: array[idx] = v → VableArrayWrite[{arr_idx}]"),
                        });
                        vable_rewrites += 1;
                        new_ops.push(Op {
                            result: op.result,
                            kind: OpKind::VableArrayWrite {
                                array_index: arr_idx,
                                elem_index: *index,
                                value: *value,
                                item_ty: item_ty.clone(),
                            },
                        });
                        continue;
                    }
                }

                // ── Call classification → rewrite to typed call ──
                // RPython jtransform.py: classify calls by effect info
                OpKind::Call {
                    target,
                    args,
                    result_ty,
                } if config.classify_calls => {
                    let effect = classify_call_effect(target);
                    if effect != "unknown" {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("call {target} → {effect}"),
                        });
                        calls_classified += 1;
                        let rewritten_kind = match effect {
                            "elidable" => OpKind::CallElidable {
                                target: target.clone(),
                                args: args.clone(),
                                result_ty: result_ty.clone(),
                            },
                            "residual" => OpKind::CallResidual {
                                target: target.clone(),
                                args: args.clone(),
                                result_ty: result_ty.clone(),
                            },
                            "io" => OpKind::CallMayForce {
                                target: target.clone(),
                                args: args.clone(),
                                result_ty: result_ty.clone(),
                            },
                            _ => op.kind.clone(),
                        };
                        new_ops.push(Op {
                            result: op.result,
                            kind: rewritten_kind,
                        });
                        continue;
                    }
                }

                // ── Unknown ops ──
                OpKind::Unknown { summary } => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("unknown op: {}", truncate(summary, 60)),
                    });
                }

                _ => {}
            }
            new_ops.push(op);
        }

        block.ops = new_ops;

        block.terminator = remap_terminator(&block.terminator, &aliases);

        if let Terminator::Abort { reason } = &block.terminator {
            notes.push(GraphTransformNote {
                function: graph.name.clone(),
                detail: format!("abort: {reason}"),
            });
        }
    }

    GraphTransformResult {
        graph: rewritten,
        notes,
        vable_rewrites,
        calls_classified,
    }
}

fn resolve_alias(
    mut value: ValueId,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> ValueId {
    while let Some(next) = aliases.get(&value).copied() {
        if next == value {
            break;
        }
        value = next;
    }
    value
}

fn remap_value(value: ValueId, aliases: &std::collections::HashMap<ValueId, ValueId>) -> ValueId {
    resolve_alias(value, aliases)
}

fn remap_op(op: &Op, aliases: &std::collections::HashMap<ValueId, ValueId>) -> Op {
    let kind = match &op.kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::VableForce
        | OpKind::Unknown { .. } => op.kind.clone(),
        OpKind::FieldRead { base, field, ty } => OpKind::FieldRead {
            base: remap_value(*base, aliases),
            field: field.clone(),
            ty: ty.clone(),
        },
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } => OpKind::FieldWrite {
            base: remap_value(*base, aliases),
            field: field.clone(),
            value: remap_value(*value, aliases),
            ty: ty.clone(),
        },
        OpKind::ArrayRead {
            base,
            index,
            item_ty,
        } => OpKind::ArrayRead {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            item_ty: item_ty.clone(),
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
        } => OpKind::ArrayWrite {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
        },
        OpKind::Call {
            target,
            args,
            result_ty,
        } => OpKind::Call {
            target: target.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::GuardTrue { cond } => OpKind::GuardTrue {
            cond: remap_value(*cond, aliases),
        },
        OpKind::GuardFalse { cond } => OpKind::GuardFalse {
            cond: remap_value(*cond, aliases),
        },
        OpKind::VableFieldRead { .. } => op.kind.clone(),
        OpKind::VableFieldWrite {
            field_index,
            value,
            ty,
        } => OpKind::VableFieldWrite {
            field_index: *field_index,
            value: remap_value(*value, aliases),
            ty: ty.clone(),
        },
        OpKind::VableArrayRead {
            array_index,
            elem_index,
            item_ty,
        } => OpKind::VableArrayRead {
            array_index: *array_index,
            elem_index: remap_value(*elem_index, aliases),
            item_ty: item_ty.clone(),
        },
        OpKind::VableArrayWrite {
            array_index,
            elem_index,
            value,
            item_ty,
        } => OpKind::VableArrayWrite {
            array_index: *array_index,
            elem_index: remap_value(*elem_index, aliases),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
        },
        OpKind::BinOp {
            op,
            lhs,
            rhs,
            result_ty,
        } => OpKind::BinOp {
            op: op.clone(),
            lhs: remap_value(*lhs, aliases),
            rhs: remap_value(*rhs, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } => OpKind::UnaryOp {
            op: op.clone(),
            operand: remap_value(*operand, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::CallElidable {
            target,
            args,
            result_ty,
        } => OpKind::CallElidable {
            target: target.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::CallResidual {
            target,
            args,
            result_ty,
        } => OpKind::CallResidual {
            target: target.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::CallMayForce {
            target,
            args,
            result_ty,
        } => OpKind::CallMayForce {
            target: target.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
    };
    Op {
        result: op.result,
        kind,
    }
}

fn remap_terminator(
    term: &Terminator,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> Terminator {
    match term {
        Terminator::Goto { target, args } => Terminator::Goto {
            target: *target,
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
        },
        Terminator::Branch {
            cond,
            if_true,
            true_args,
            if_false,
            false_args,
        } => Terminator::Branch {
            cond: remap_value(*cond, aliases),
            if_true: *if_true,
            true_args: true_args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            if_false: *if_false,
            false_args: false_args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
        },
        Terminator::Return(val) => Terminator::Return(val.map(|v| remap_value(v, aliases))),
        Terminator::Abort { reason } => Terminator::Abort {
            reason: reason.clone(),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

fn is_vable_identity_hint(target: &str) -> bool {
    let last = target.split("::").last().unwrap_or(target);
    matches!(last, "hint_access_directly" | "hint_fresh_virtualizable")
}

fn is_vable_force_hint(target: &str) -> bool {
    target.split("::").last().unwrap_or(target) == "hint_force_virtualizable"
}

/// Classify a call's side-effect level.
///
/// RPython equivalent: jtransform.py effect classification
/// (EF_ELIDABLE, EF_FORCES_VIRTUAL, etc.)
fn classify_call_effect(target: &str) -> &'static str {
    // Known pure/elidable functions
    if target.contains("len")
        || target.contains("is_empty")
        || target.contains("get")
        || target.contains("peek")
    {
        return "elidable";
    }
    // Known may-force functions (can trigger GC/exceptions)
    if target.contains("push")
        || target.contains("pop")
        || target.contains("insert")
        || target.contains("remove")
    {
        return "residual";
    }
    // Known I/O
    if target.contains("print") || target.contains("write") || target.contains("read") {
        return "io";
    }
    "unknown"
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MajitGraph, OpKind, ValueId, ValueType};

    #[test]
    fn rewrite_graph_tags_vable_fields() {
        let mut graph = MajitGraph::new("test");
        let base = graph.alloc_value();
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base,
                field: "next_instr".into(),
                ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let config = GraphTransformConfig {
            vable_fields: vec![("next_instr".into(), 0)],
            ..Default::default()
        };
        let result = rewrite_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 1);
        // Should be rewritten to VableFieldRead
        let rewritten_op = &result.graph.block(graph.entry).ops[0];
        assert!(
            matches!(
                &rewritten_op.kind,
                OpKind::VableFieldRead { field_index: 0, .. }
            ),
            "expected VableFieldRead, got {:?}",
            rewritten_op.kind
        );
    }

    #[test]
    fn rewrite_graph_classifies_calls() {
        let mut graph = MajitGraph::new("test");
        graph.push_op(
            graph.entry,
            OpKind::Call {
                target: "vec_push".into(),
                args: vec![],
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let result = rewrite_graph(&graph, &GraphTransformConfig::default());
        assert_eq!(result.calls_classified, 1);
    }

    #[test]
    fn rewrite_graph_reports_unknowns() {
        let mut graph = MajitGraph::new("demo");
        graph.push_op(
            graph.entry,
            OpKind::Unknown {
                summary: "complex expression".into(),
            },
            false,
        );
        graph.set_terminator(
            graph.entry,
            Terminator::Abort {
                reason: "not implemented".into(),
            },
        );
        let result = rewrite_graph(&graph, &GraphTransformConfig::default());
        assert_eq!(result.notes.len(), 2); // unknown + abort
    }

    #[test]
    fn rewrite_graph_consumes_identity_virtualizable_hints() {
        let mut graph = MajitGraph::new("demo");
        let frame = graph.alloc_value();
        let hinted = graph.alloc_value();
        graph.block_mut(graph.entry).inputargs.push(frame);
        graph.push_op(
            graph.entry,
            OpKind::Call {
                target: "hint_access_directly".into(),
                args: vec![frame],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph.block_mut(graph.entry).ops.last_mut().unwrap().result = Some(hinted);
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base: hinted,
                field: "next_instr".into(),
                ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let result = rewrite_graph(
            &graph,
            &GraphTransformConfig {
                vable_fields: vec![("next_instr".into(), 0)],
                ..Default::default()
            },
        );

        assert_eq!(result.graph.block(graph.entry).ops.len(), 1);
        match &result.graph.block(graph.entry).ops[0].kind {
            OpKind::VableFieldRead { field_index, .. } => assert_eq!(*field_index, 0),
            other => panic!("expected VableFieldRead after hint suppression, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_rewrites_hint_force_virtualizable() {
        let mut graph = MajitGraph::new("demo");
        let frame = graph.alloc_value();
        let forced = graph.alloc_value();
        graph.block_mut(graph.entry).inputargs.push(frame);
        graph.push_op(
            graph.entry,
            OpKind::Call {
                target: "hint_force_virtualizable".into(),
                args: vec![frame],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph.block_mut(graph.entry).ops.last_mut().unwrap().result = Some(forced);
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base: forced,
                field: "next_instr".into(),
                ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let result = rewrite_graph(
            &graph,
            &GraphTransformConfig {
                vable_fields: vec![("next_instr".into(), 0)],
                ..Default::default()
            },
        );

        let ops = &result.graph.block(graph.entry).ops;
        assert!(matches!(ops[0].kind, OpKind::VableForce));
        assert!(matches!(
            ops[1].kind,
            OpKind::VableFieldRead { field_index: 0, .. }
        ));
    }
}
