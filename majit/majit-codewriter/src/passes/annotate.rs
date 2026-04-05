//! Annotation propagation pass.
//!
//! RPython equivalent: `annotator/annrpython.py` RPythonAnnotator.
//!
//! Propagates ValueType annotations through the graph by analyzing
//! each op's inputs and computing the output type. Iterates to
//! fixpoint when Block.inputargs (Phi nodes) need widening.

use crate::model::{
    BlockId, FunctionGraph, OpKind, SpaceOperation, Terminator, ValueId, ValueType,
};
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
pub fn annotate(graph: &FunctionGraph) -> AnnotationState {
    let mut state = AnnotationState::new();

    // Process all blocks (simple single-pass for acyclic; loops need fixpoint)
    let mut changed = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 20;

    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;

        for block in &graph.blocks {
            // Propagate annotations through ops in this block
            for op in &block.operations {
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

            // Cross-block propagation: Link args → target inputargs (RPython)
            // Terminator carries values to successor block's inputargs.
            match &block.terminator {
                Terminator::Goto { target, args } => {
                    let target_block = graph.block(*target);
                    for (dst, src) in target_block.inputargs.iter().zip(args.iter()) {
                        let src_ty = state.get(*src).clone();
                        let current = state.get(*dst).clone();
                        let merged = union_type(&current, &src_ty);
                        if merged != current {
                            state.set(*dst, merged);
                            changed = true;
                        }
                    }
                }
                Terminator::Branch {
                    if_true,
                    true_args,
                    if_false,
                    false_args,
                    ..
                } => {
                    let true_block = graph.block(*if_true);
                    for (dst, src) in true_block.inputargs.iter().zip(true_args.iter()) {
                        let src_ty = state.get(*src).clone();
                        let current = state.get(*dst).clone();
                        let merged = union_type(&current, &src_ty);
                        if merged != current {
                            state.set(*dst, merged);
                            changed = true;
                        }
                    }
                    let false_block = graph.block(*if_false);
                    for (dst, src) in false_block.inputargs.iter().zip(false_args.iter()) {
                        let src_ty = state.get(*src).clone();
                        let current = state.get(*dst).clone();
                        let merged = union_type(&current, &src_ty);
                        if merged != current {
                            state.set(*dst, merged);
                            changed = true;
                        }
                    }
                }
                _ => {}
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
        OpKind::InteriorFieldRead { item_ty, .. } => item_ty.clone(),
        OpKind::InteriorFieldWrite { .. } => ValueType::Void,
        OpKind::Call {
            result_ty,
            target,
            args,
            ..
        } => {
            if result_ty != &ValueType::Unknown {
                return result_ty.clone();
            }
            infer_call_result_type(target, args, state)
        }
        OpKind::GuardTrue { .. } | OpKind::GuardFalse { .. } => ValueType::Void,
        OpKind::VableFieldRead { ty, .. } => ty.clone(),
        OpKind::VableFieldWrite { .. } => ValueType::Void,
        OpKind::VableArrayRead { item_ty, .. } => item_ty.clone(),
        OpKind::VableArrayWrite { .. } => ValueType::Void,
        OpKind::BinOp { result_ty, .. } | OpKind::UnaryOp { result_ty, .. } => {
            if result_ty != &ValueType::Unknown {
                result_ty.clone()
            } else {
                ValueType::Int // Arithmetic defaults to Int
            }
        }
        OpKind::VableForce | OpKind::Live | OpKind::GuardValue { .. } => ValueType::Void,
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => kind_char_to_value_type(*result_kind),
        OpKind::Unknown { .. } => ValueType::Unknown,
    }
}

fn kind_char_to_value_type(kind: char) -> ValueType {
    match kind {
        'i' => ValueType::Int,
        'r' => ValueType::Ref,
        'f' => ValueType::Float,
        'v' => ValueType::Void,
        _ => ValueType::Unknown,
    }
}

fn infer_call_result_type(
    target: &crate::model::CallTarget,
    _args: &[ValueId],
    _state: &AnnotationState,
) -> ValueType {
    if crate::call::is_int_arithmetic_target(target) {
        return ValueType::Int;
    }
    ValueType::Unknown
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
    use crate::model::{CallTarget, FunctionGraph, OpKind, Terminator, ValueType};

    #[test]
    fn annotates_const_int() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v)));

        let state = annotate(&graph);
        assert_eq!(state.get(v), &ValueType::Int);
    }

    #[test]
    fn annotates_field_read_type() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let base = graph.alloc_value();
        let v = graph
            .push_op(
                entry,
                OpKind::FieldRead {
                    base,
                    field: crate::model::FieldDescriptor::new("x", None),
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
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let a = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let b = graph.push_op(entry, OpKind::ConstInt(2), true).unwrap();
        let result = graph
            .push_op(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["w_int_add"]),
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

    #[test]
    fn annotates_path_like_int_helper_call() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let a = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let b = graph.push_op(entry, OpKind::ConstInt(2), true).unwrap();
        let result = graph
            .push_op(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["crate", "math", "w_int_add"]),
                    args: vec![a, b],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(result)));

        let state = annotate(&graph);
        assert_eq!(state.get(result), &ValueType::Int);
    }

    #[test]
    fn propagates_across_blocks_via_phi() {
        // Test cross-block annotation propagation through Link args → inputargs
        let mut graph = FunctionGraph::new("phi_test");
        let entry = graph.startblock;

        // Entry: produce an Int value
        let val = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();

        // Create target block with one inputarg (Phi node)
        let (target, phi_args) = graph.create_block_with_args(1);
        let phi = phi_args[0];

        // Link: entry → target, passing val as the Phi arg
        graph.set_terminator(
            entry,
            Terminator::Goto {
                target,
                args: vec![val],
            },
        );
        graph.set_terminator(target, Terminator::Return(Some(phi)));

        let state = annotate(&graph);
        // Phi should inherit Int from val via Link propagation
        assert_eq!(
            state.get(phi),
            &ValueType::Int,
            "Phi node should receive Int annotation from Link args"
        );
    }
}
