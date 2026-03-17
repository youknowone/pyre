//! Annotation propagation pass.
//!
//! RPython equivalent: `annotator/annrpython.py` RPythonAnnotator.
//!
//! Propagates ValueType annotations through the graph by analyzing
//! each op's inputs and computing the output type. Iterates to
//! fixpoint when Block.inputargs (Phi nodes) need widening.

use crate::graph::{BasicBlockId, MajitGraph, Op, OpKind, ValueId, ValueType};
use std::collections::HashMap;

/// Annotation state: maps ValueId → inferred ValueType.
#[derive(Debug, Clone)]
pub struct AnnotationState {
    pub types: HashMap<ValueId, ValueType>,
}

impl AnnotationState {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
        }
    }

    pub fn get(&self, id: ValueId) -> &ValueType {
        self.types.get(&id).unwrap_or(&ValueType::Unknown)
    }

    pub fn set(&mut self, id: ValueId, ty: ValueType) {
        self.types.insert(id, ty);
    }
}

/// Run annotation propagation to fixpoint.
///
/// RPython equivalent: `RPythonAnnotator.complete()` — processes all
/// blocks until no annotation changes.
pub fn annotate(graph: &MajitGraph) -> AnnotationState {
    let mut state = AnnotationState::new();

    // Process all blocks (simple single-pass for acyclic; loops need fixpoint)
    let mut changed = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 20;

    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;

        for block in &graph.blocks {
            // Annotate inputargs from predecessor Link args
            // (simplified: inputargs keep their current annotation)

            for op in &block.ops {
                if let Some(result) = op.result {
                    let inferred = infer_op_type(&op.kind, &state);
                    let current = state.get(result).clone();
                    let merged = union_type(&current, &inferred);
                    if merged != current {
                        state.set(result, merged);
                        changed = true;
                    }
                }
            }
        }
    }

    state
}

/// Infer the output type of an operation from its inputs.
///
/// RPython equivalent: annotator dispatch (e.g., `annotate_int_add`
/// returns `SomeInteger()`).
fn infer_op_type(kind: &OpKind, state: &AnnotationState) -> ValueType {
    match kind {
        OpKind::Input { ty, .. } => ty.clone(),
        OpKind::ConstInt(_) => ValueType::Int,
        OpKind::FieldRead { ty, .. } => ty.clone(),
        OpKind::FieldWrite { .. } => ValueType::Void,
        OpKind::ArrayRead { item_ty, .. } => item_ty.clone(),
        OpKind::ArrayWrite { .. } => ValueType::Void,
        OpKind::Call { result_ty, target, args, .. } => {
            // Infer from target name heuristic
            if result_ty != &ValueType::Unknown {
                return result_ty.clone();
            }
            // Known operations that return Int
            if target.contains("+")
                || target.contains("-")
                || target.contains("*")
                || target.contains("len")
                || target.contains("size")
            {
                return ValueType::Int;
            }
            // If all args are Int and it's an arithmetic-like op, result is Int
            if !args.is_empty()
                && args.iter().all(|a| state.get(*a) == &ValueType::Int)
                && (target.contains("add")
                    || target.contains("sub")
                    || target.contains("mul"))
            {
                return ValueType::Int;
            }
            ValueType::Unknown
        }
        OpKind::GuardTrue { .. } | OpKind::GuardFalse { .. } => ValueType::Void,
        OpKind::VableFieldRead { ty, .. } => ty.clone(),
        OpKind::VableFieldWrite { .. } => ValueType::Void,
        OpKind::VableArrayRead { item_ty, .. } => item_ty.clone(),
        OpKind::VableArrayWrite { .. } => ValueType::Void,
        OpKind::VableForce => ValueType::Void,
        OpKind::Unknown { .. } => ValueType::Unknown,
    }
}

/// Merge two annotations (RPython `unionof()`).
///
/// Returns the wider type. Unknown absorbs everything (top of lattice).
fn union_type(a: &ValueType, b: &ValueType) -> ValueType {
    if a == b {
        return a.clone();
    }
    match (a, b) {
        (ValueType::Unknown, other) | (other, ValueType::Unknown) => {
            if other == &ValueType::Unknown {
                ValueType::Unknown
            } else {
                other.clone()
            }
        }
        _ => ValueType::Unknown, // Conflicting types → Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MajitGraph, OpKind, Terminator, ValueType};

    #[test]
    fn annotates_const_int() {
        let mut graph = MajitGraph::new("test");
        let entry = graph.entry;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v)));

        let state = annotate(&graph);
        assert_eq!(state.get(v), &ValueType::Int);
    }

    #[test]
    fn annotates_field_read_type() {
        let mut graph = MajitGraph::new("test");
        let entry = graph.entry;
        let base = graph.alloc_value();
        let v = graph
            .push_op(
                entry,
                OpKind::FieldRead {
                    base,
                    field: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v)));

        let state = annotate(&graph);
        assert_eq!(state.get(v), &ValueType::Int);
    }

    #[test]
    fn annotates_call_with_int_args() {
        let mut graph = MajitGraph::new("test");
        let entry = graph.entry;
        let a = graph
            .push_op(entry, OpKind::ConstInt(1), true)
            .unwrap();
        let b = graph
            .push_op(entry, OpKind::ConstInt(2), true)
            .unwrap();
        let result = graph
            .push_op(
                entry,
                OpKind::Call {
                    target: "+".into(),
                    args: vec![a, b],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(result)));

        let state = annotate(&graph);
        assert_eq!(state.get(a), &ValueType::Int);
        assert_eq!(state.get(b), &ValueType::Int);
        assert_eq!(state.get(result), &ValueType::Int);
    }
}
