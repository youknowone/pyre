//! Graph-based jtransform scaffold.
//!
//! Long-term, this should become the semantic rewrite layer that sits between
//! AST graph construction and final lowering.

use serde::{Deserialize, Serialize};

use crate::{MajitGraph, OpKind, Terminator};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformConfig {
    pub lower_virtualizable: bool,
    pub classify_calls: bool,
}

impl Default for GraphTransformConfig {
    fn default() -> Self {
        Self {
            lower_virtualizable: true,
            classify_calls: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTransformNote {
    pub function: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformResult {
    pub graph: MajitGraph,
    pub notes: Vec<GraphTransformNote>,
}

/// Rewrite a semantic graph with JIT-specific transformations.
///
/// RPython equivalent: jtransform.py `Transformer.optimize_block()`
///
/// Recognized patterns:
/// - FieldRead on virtualizable fields → notes "vable_field_read"
/// - FieldWrite on virtualizable fields → notes "vable_field_write"
/// - ArrayRead on virtualizable arrays → notes "vable_array_read"
/// - Call to known helpers → classifies as elidable/residual/may_force
/// - Unknown ops → reported as unresolved
pub fn rewrite_graph(graph: &MajitGraph, config: &GraphTransformConfig) -> GraphTransformResult {
    let mut notes = Vec::new();
    let mut rewritten = graph.clone();

    for block in &mut rewritten.blocks {
        for op in &mut block.ops {
            match &op.kind {
                OpKind::FieldRead { field, .. } if config.lower_virtualizable => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("vable_field_read: {field}"),
                    });
                }
                OpKind::FieldWrite { field, .. } if config.lower_virtualizable => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("vable_field_write: {field}"),
                    });
                }
                OpKind::ArrayRead { .. } if config.lower_virtualizable => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: "vable_array_read".into(),
                    });
                }
                OpKind::ArrayWrite { .. } if config.lower_virtualizable => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: "vable_array_write".into(),
                    });
                }
                OpKind::Call { target, .. } if config.classify_calls => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("call: {target}"),
                    });
                }
                OpKind::Unknown { summary } => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("unknown op: {summary}"),
                    });
                }
                _ => {}
            }
        }
        if let Terminator::Abort { reason } = &block.terminator {
            notes.push(GraphTransformNote {
                function: graph.name.clone(),
                detail: format!("abort terminator: {reason}"),
            });
        }
    }
    GraphTransformResult {
        graph: rewritten,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MajitGraph, OpKind};

    #[test]
    fn rewrite_graph_reports_abort_shells() {
        let mut graph = MajitGraph::new("demo");
        graph.notes.push("shell".into());
        graph.push_op(
            graph.entry,
            OpKind::Unknown {
                summary: "todo".into(),
            },
            false,
        );
        graph.set_terminator(
            graph.entry,
            crate::Terminator::Abort {
                reason: "todo".into(),
            },
        );
        let result = rewrite_graph(&graph, &GraphTransformConfig::default());
        assert_eq!(result.notes.len(), 2);
    }
}
