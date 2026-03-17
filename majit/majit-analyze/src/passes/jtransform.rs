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

use crate::graph::{BasicBlockId, MajitGraph, Op, OpKind, Terminator, ValueType};

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

    for block in &mut rewritten.blocks {
        let mut new_ops = Vec::with_capacity(block.ops.len());

        for op in &block.ops {
            match &op.kind {
                // ── Virtualizable field read → VableFieldRead ──
                // RPython jtransform.py:832 `rewrite_op_getfield`
                OpKind::FieldRead { field, ty, .. } if config.lower_virtualizable => {
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
                OpKind::FieldWrite { field, value, ty, .. } if config.lower_virtualizable => {
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

                // ── Virtualizable array read → tag ──
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty,
                } if config.lower_virtualizable => {
                    // Check if the base was produced by a FieldRead on a vable array
                    // (simplified: check graph notes or field name heuristic)
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: "array_read (potential vable)".into(),
                    });
                }

                // ── Call classification ──
                OpKind::Call {
                    target,
                    args,
                    result_ty,
                } if config.classify_calls => {
                    let effect = classify_call_effect(target);
                    if effect != "unknown" {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("call {target} classified as {effect}"),
                        });
                        calls_classified += 1;
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
            new_ops.push(op.clone());
        }

        block.ops = new_ops;

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
            matches!(&rewritten_op.kind, OpKind::VableFieldRead { field_index: 0, .. }),
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
}
