//! Flatten pass: CFG → linear instruction sequence.
//!
//! RPython equivalent: `jit/codewriter/flatten.py` flatten_graph().
//!
//! Converts a multi-block MajitGraph into a linear sequence of
//! FlatOps with Labels and Jumps. This is the last graph pass
//! before register allocation and JitCode assembly.

use serde::{Deserialize, Serialize};

use crate::graph::{BasicBlockId, MajitGraph, Op, OpKind, Terminator, ValueId, ValueType};

/// A label in the flattened instruction stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Label(pub usize);

/// A flattened instruction (post-CFG).
///
/// RPython equivalent: SSARepr instruction tuples from flatten.py.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlatOp {
    /// Label definition (target for jumps).
    Label(Label),
    /// Semantic op (from the graph).
    Op(Op),
    /// Unconditional jump to label.
    Jump(Label),
    /// Conditional jump: if cond is true, jump to label.
    JumpIfTrue { cond: ValueId, target: Label },
    /// Conditional jump: if cond is false, jump to label.
    JumpIfFalse { cond: ValueId, target: Label },
    /// Copy value (for Phi-node resolution: Link.args → target.inputargs).
    Move { dst: ValueId, src: ValueId },
}

/// Register kind for a value (RPython regalloc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegKind {
    Int,
    Ref,
    Float,
}

/// Result of the flatten pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlattenedFunction {
    pub name: String,
    pub ops: Vec<FlatOp>,
    /// Total number of values used (for register allocation).
    pub num_values: usize,
    /// Number of basic blocks in the source graph.
    pub num_blocks: usize,
    /// Value kinds inferred from the type resolution pass.
    #[serde(default)]
    pub value_kinds: std::collections::HashMap<ValueId, RegKind>,
}

/// Flatten a MajitGraph into a linear instruction sequence.
///
/// RPython equivalent: `flatten_graph()` from flatten.py.
///
/// Block ordering: entry first, then BFS order. Back-edges (loops)
/// become jumps to earlier labels.
pub fn flatten(graph: &MajitGraph) -> FlattenedFunction {
    let mut ops = Vec::new();
    let mut block_labels: std::collections::HashMap<BasicBlockId, Label> =
        std::collections::HashMap::new();
    let mut next_label = 0usize;

    // Assign labels to all blocks
    let order = block_order(graph);
    for &bid in &order {
        block_labels.insert(bid, Label(next_label));
        next_label += 1;
    }

    // Emit instructions in block order
    for &bid in &order {
        let block = graph.block(bid);
        let label = block_labels[&bid];

        // Label
        ops.push(FlatOp::Label(label));

        // Ops
        for op in &block.ops {
            ops.push(FlatOp::Op(op.clone()));
        }

        // Terminator → jumps + moves (Phi resolution)
        match &block.terminator {
            Terminator::Goto { target, args } => {
                // Emit moves for Phi args
                let target_block = graph.block(*target);
                for (dst, src) in target_block.inputargs.iter().zip(args.iter()) {
                    ops.push(FlatOp::Move {
                        dst: *dst,
                        src: *src,
                    });
                }
                ops.push(FlatOp::Jump(block_labels[target]));
            }
            Terminator::Branch {
                cond,
                if_true,
                true_args,
                if_false,
                false_args,
            } => {
                // Emit conditional jump to true branch
                // (false branch falls through or jumps)
                let true_label = block_labels[if_true];
                let false_label = block_labels[if_false];

                // Phi moves for true branch (before jump)
                // Note: in proper SSA, parallel moves need careful ordering.
                // Simplified: emit moves + jump for true, then false.
                if !true_args.is_empty() || !false_args.is_empty() {
                    // Complex case: need to handle Phi args for both branches
                    // For now, emit the branch directly
                    ops.push(FlatOp::JumpIfTrue {
                        cond: *cond,
                        target: true_label,
                    });
                    // False path moves
                    let false_block = graph.block(*if_false);
                    for (dst, src) in false_block.inputargs.iter().zip(false_args.iter()) {
                        ops.push(FlatOp::Move {
                            dst: *dst,
                            src: *src,
                        });
                    }
                    ops.push(FlatOp::Jump(false_label));
                } else {
                    ops.push(FlatOp::JumpIfTrue {
                        cond: *cond,
                        target: true_label,
                    });
                    ops.push(FlatOp::Jump(false_label));
                }
            }
            Terminator::Return(val) => {
                // Return is implicit at the end (no jump needed)
                // Could emit a FlatOp::Return if needed
            }
            Terminator::Abort { .. } | Terminator::Unreachable => {
                // Terminal — no jump
            }
        }
    }

    // Count total values
    let mut max_value = 0usize;
    for op in &ops {
        match op {
            FlatOp::Op(op) => {
                if let Some(ValueId(v)) = op.result {
                    max_value = max_value.max(v + 1);
                }
            }
            FlatOp::Move {
                dst: ValueId(d),
                src: ValueId(s),
            } => {
                max_value = max_value.max(*d + 1);
                max_value = max_value.max(*s + 1);
            }
            _ => {}
        }
    }

    FlattenedFunction {
        name: graph.name.clone(),
        ops,
        num_values: max_value,
        num_blocks: graph.blocks.len(),
        value_kinds: std::collections::HashMap::new(),
    }
}

/// Flatten with type information from rtype pass.
///
/// Like `flatten()` but populates `value_kinds` from the TypeResolutionState.
pub fn flatten_with_types(
    graph: &MajitGraph,
    types: &super::rtype::TypeResolutionState,
) -> FlattenedFunction {
    let mut result = flatten(graph);
    for (&vid, concrete) in &types.concrete_types {
        let kind = match concrete {
            super::rtype::ConcreteType::Signed => RegKind::Int,
            super::rtype::ConcreteType::GcRef => RegKind::Ref,
            super::rtype::ConcreteType::Float => RegKind::Float,
            _ => continue,
        };
        result.value_kinds.insert(vid, kind);
    }
    result
}

/// Compute block ordering (entry first, then BFS).
fn block_order(graph: &MajitGraph) -> Vec<BasicBlockId> {
    let mut order = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    queue.push_back(graph.entry);
    visited.insert(graph.entry);

    while let Some(bid) = queue.pop_front() {
        order.push(bid);
        let block = graph.block(bid);
        for succ in successors(&block.terminator) {
            if visited.insert(succ) {
                queue.push_back(succ);
            }
        }
    }

    // Add any unreachable blocks (shouldn't happen in well-formed graphs)
    for block in &graph.blocks {
        if !visited.contains(&block.id) {
            order.push(block.id);
        }
    }

    order
}

/// Get successor block IDs from a terminator.
fn successors(term: &Terminator) -> Vec<BasicBlockId> {
    match term {
        Terminator::Goto { target, .. } => vec![*target],
        Terminator::Branch {
            if_true, if_false, ..
        } => vec![*if_true, *if_false],
        Terminator::Return(_) | Terminator::Abort { .. } | Terminator::Unreachable => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MajitGraph, OpKind, Terminator, ValueType};

    #[test]
    fn flatten_single_block() {
        let mut graph = MajitGraph::new("simple");
        let entry = graph.entry;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v)));

        let flat = flatten(&graph);
        assert_eq!(flat.name, "simple");
        // Label + ConstInt op = 2 flat ops
        assert!(flat.ops.len() >= 2);
        assert!(matches!(flat.ops[0], FlatOp::Label(Label(0))));
    }

    #[test]
    fn flatten_if_else_produces_jumps() {
        let mut graph = MajitGraph::new("branch");
        let entry = graph.entry;
        let cond = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let then_block = graph.create_block();
        let else_block = graph.create_block();
        let merge = graph.create_block();

        graph.set_terminator(
            entry,
            Terminator::Branch {
                cond,
                if_true: then_block,
                true_args: vec![],
                if_false: else_block,
                false_args: vec![],
            },
        );
        graph.set_terminator(
            then_block,
            Terminator::Goto {
                target: merge,
                args: vec![],
            },
        );
        graph.set_terminator(
            else_block,
            Terminator::Goto {
                target: merge,
                args: vec![],
            },
        );
        graph.set_terminator(merge, Terminator::Return(None));

        let flat = flatten(&graph);
        // Should have labels + jumps
        let has_jump = flat
            .ops
            .iter()
            .any(|op| matches!(op, FlatOp::Jump(_) | FlatOp::JumpIfTrue { .. }));
        assert!(has_jump, "flattened if/else should have jumps");
        // Should have 4 labels (one per block)
        let label_count = flat
            .ops
            .iter()
            .filter(|op| matches!(op, FlatOp::Label(_)))
            .count();
        assert_eq!(label_count, 4, "should have 4 labels for 4 blocks");
    }

    #[test]
    fn flatten_while_loop_has_back_edge() {
        let mut graph = MajitGraph::new("loop");
        let entry = graph.entry;
        let header = graph.create_block();
        let body = graph.create_block();
        let exit = graph.create_block();

        graph.set_terminator(
            entry,
            Terminator::Goto {
                target: header,
                args: vec![],
            },
        );
        let cond = graph.push_op(header, OpKind::ConstInt(1), true).unwrap();
        graph.set_terminator(
            header,
            Terminator::Branch {
                cond,
                if_true: body,
                true_args: vec![],
                if_false: exit,
                false_args: vec![],
            },
        );
        graph.set_terminator(
            body,
            Terminator::Goto {
                target: header,
                args: vec![],
            },
        );
        graph.set_terminator(exit, Terminator::Return(None));

        let flat = flatten(&graph);
        // Body should jump back to header label
        let jumps: Vec<_> = flat
            .ops
            .iter()
            .filter(|op| matches!(op, FlatOp::Jump(_)))
            .collect();
        assert!(
            jumps.len() >= 2,
            "loop should have >=2 jumps (entry→header, body→header)"
        );
    }

    #[test]
    fn flatten_phi_produces_move_ops() {
        // When a Goto carries Link args to a target with inputargs,
        // flatten should emit Move ops for Phi resolution.
        let mut graph = MajitGraph::new("phi");
        let entry = graph.entry;
        let val = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();

        let (target, phi_args) = graph.create_block_with_args(1);
        let _phi = phi_args[0];

        graph.set_terminator(
            entry,
            Terminator::Goto {
                target,
                args: vec![val],
            },
        );
        graph.set_terminator(target, Terminator::Return(None));

        let flat = flatten(&graph);
        let moves: Vec<_> = flat
            .ops
            .iter()
            .filter(|op| matches!(op, FlatOp::Move { .. }))
            .collect();
        assert_eq!(moves.len(), 1, "should have 1 Move for Phi resolution");
    }
}
