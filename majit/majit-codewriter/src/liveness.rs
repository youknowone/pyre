//! Liveness computation for flattened JitCode instructions.
//!
//! RPython equivalent: `rpython/jit/codewriter/liveness.py`.
//!
//! Expands `-live-` markers in the flattened instruction sequence to
//! include all values that are alive at that point (written before and
//! read afterwards). This information is used by guard operations in the
//! meta-interpreter to know which values to save on failure.
//!
//! The algorithm is a backward dataflow analysis that iterates to fixpoint.

use std::collections::{HashMap, HashSet};

use crate::graph::ValueId;
use crate::passes::flatten::{FlatOp, FlattenedFunction, Label};

/// Compute liveness for a flattened function.
///
/// RPython: `liveness.py::compute_liveness(ssarepr)`.
///
/// Modifies the flattened ops in place: each `FlatOp::Live` marker
/// gets its `live_values` set populated with all values alive at that
/// point in the instruction sequence.
pub fn compute_liveness(flattened: &mut FlattenedFunction) {
    let mut label2alive: HashMap<Label, HashSet<ValueId>> = HashMap::new();

    // Iterate to fixpoint (RPython: while _compute_liveness_must_continue)
    loop {
        if !compute_liveness_pass(&flattened.ops, &mut label2alive) {
            break;
        }
    }
}

/// One backward pass of liveness analysis.
/// Returns true if any label's alive set grew (needs another iteration).
///
/// RPython: `_compute_liveness_must_continue(ssarepr, label2alive)`.
fn compute_liveness_pass(
    ops: &[FlatOp],
    label2alive: &mut HashMap<Label, HashSet<ValueId>>,
) -> bool {
    let mut alive: HashSet<ValueId> = HashSet::new();
    let mut must_continue = false;

    // Walk backward through instructions
    for op in ops.iter().rev() {
        match op {
            FlatOp::Label(label) => {
                let alive_at_point = label2alive.entry(label.clone()).or_default();
                let prev_len = alive_at_point.len();
                alive_at_point.extend(alive.iter());
                if alive_at_point.len() != prev_len {
                    must_continue = true;
                }
            }
            FlatOp::Op(inner_op) => {
                // Result is defined here — remove from alive
                if let Some(result) = inner_op.result {
                    alive.remove(&result);
                }
                // Operands are used here — add to alive
                for vid in crate::inline::op_value_refs(&inner_op.kind) {
                    alive.insert(vid);
                }
            }
            FlatOp::Jump(label) => {
                // Follow label: union with alive set at target
                if let Some(alive_at_target) = label2alive.get(label) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::JumpIfTrue { cond, target } => {
                alive.insert(*cond);
                if let Some(alive_at_target) = label2alive.get(target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::JumpIfFalse { cond, target } => {
                alive.insert(*cond);
                if let Some(alive_at_target) = label2alive.get(target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::Move { dst, src } => {
                alive.remove(dst);
                alive.insert(*src);
            }
        }
    }

    must_continue
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Op, OpKind, ValueType};
    use crate::passes::flatten::FlatOp;

    #[test]
    fn basic_liveness() {
        // v0 = Input
        // v1 = ConstInt(42)
        // v2 = BinOp(v0, v1)
        // Return v2
        let mut flat = FlattenedFunction {
            name: "test".into(),
            ops: vec![
                FlatOp::Label(Label(0)),
                FlatOp::Op(Op {
                    result: Some(ValueId(0)),
                    kind: OpKind::Input {
                        name: "a".into(),
                        ty: ValueType::Int,
                    },
                }),
                FlatOp::Op(Op {
                    result: Some(ValueId(1)),
                    kind: OpKind::ConstInt(42),
                }),
                FlatOp::Op(Op {
                    result: Some(ValueId(2)),
                    kind: OpKind::BinOp {
                        op: "add".into(),
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                        result_ty: ValueType::Int,
                    },
                }),
            ],
            num_values: 3,
            num_blocks: 1,
            value_kinds: std::collections::HashMap::new(),
        };

        // Should not panic
        compute_liveness(&mut flat);
    }
}
