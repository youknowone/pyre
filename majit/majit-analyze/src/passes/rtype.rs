//! Type resolution pass.
//!
//! RPython equivalent: `rtyper/rtyper.py` RPythonTyper.
//!
//! Transforms annotated ValueTypes into concrete low-level types
//! and specializes operations accordingly.

use std::collections::HashMap;

use crate::graph::{MajitGraph, OpKind, ValueId, ValueType};
use crate::passes::annotate::AnnotationState;

/// Concrete low-level type (RPython Repr.lowleveltype).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConcreteType {
    /// Signed integer (RPython Signed / i64)
    Signed,
    /// GC reference (RPython Ptr(GcStruct))
    GcRef,
    /// Float (RPython Float / f64)
    Float,
    /// Void (RPython Void)
    Void,
    /// Unknown / unresolved
    Unknown,
}

/// Type resolution state: maps ValueId → ConcreteType.
pub struct TypeResolutionState {
    pub concrete_types: HashMap<ValueId, ConcreteType>,
}

impl TypeResolutionState {
    pub fn get(&self, id: ValueId) -> &ConcreteType {
        self.concrete_types
            .get(&id)
            .unwrap_or(&ConcreteType::Unknown)
    }
}

/// Resolve annotations to concrete types.
///
/// RPython equivalent: `RPythonTyper.specialize_block()` — walks
/// each block and converts annotation → Repr → lowleveltype.
pub fn resolve_types(graph: &MajitGraph, annotations: &AnnotationState) -> TypeResolutionState {
    let mut state = TypeResolutionState {
        concrete_types: HashMap::new(),
    };

    for (&vid, vtype) in &annotations.types {
        let concrete = valuetype_to_concrete(vtype);
        state.concrete_types.insert(vid, concrete);
    }

    // Resolve from ops with explicit type info
    for block in &graph.blocks {
        // Resolve inputargs (Phi nodes) from annotations
        for &vid in &block.inputargs {
            if state.get(vid) == &ConcreteType::Unknown {
                let vtype = annotations.types.get(&vid).unwrap_or(&ValueType::Unknown);
                let concrete = valuetype_to_concrete(vtype);
                if concrete != ConcreteType::Unknown {
                    state.concrete_types.insert(vid, concrete);
                }
            }
        }
        for op in &block.ops {
            if let Some(result) = op.result {
                if state.get(result) == &ConcreteType::Unknown {
                    let inferred = infer_concrete_from_op(&op.kind);
                    if inferred != ConcreteType::Unknown {
                        state.concrete_types.insert(result, inferred);
                    }
                }
            }
        }
    }

    // Cross-block: propagate through Link args → target inputargs
    for block in &graph.blocks {
        match &block.terminator {
            crate::graph::Terminator::Goto { target, args } => {
                let target_block = graph.block(*target);
                for (dst, src) in target_block.inputargs.iter().zip(args.iter()) {
                    if state.get(*dst) == &ConcreteType::Unknown {
                        let src_ty = state.get(*src).clone();
                        if src_ty != ConcreteType::Unknown {
                            state.concrete_types.insert(*dst, src_ty);
                        }
                    }
                }
            }
            crate::graph::Terminator::Branch {
                if_true,
                true_args,
                if_false,
                false_args,
                ..
            } => {
                for (target, args) in [(*if_true, true_args), (*if_false, false_args)] {
                    let target_block = graph.block(target);
                    for (dst, src) in target_block.inputargs.iter().zip(args.iter()) {
                        if state.get(*dst) == &ConcreteType::Unknown {
                            let src_ty = state.get(*src).clone();
                            if src_ty != ConcreteType::Unknown {
                                state.concrete_types.insert(*dst, src_ty);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    state
}

fn valuetype_to_concrete(vt: &ValueType) -> ConcreteType {
    match vt {
        ValueType::Int => ConcreteType::Signed,
        ValueType::Ref => ConcreteType::GcRef,
        ValueType::Float => ConcreteType::Float,
        ValueType::Void => ConcreteType::Void,
        ValueType::State | ValueType::Unknown => ConcreteType::Unknown,
    }
}

fn infer_concrete_from_op(kind: &OpKind) -> ConcreteType {
    match kind {
        OpKind::ConstInt(_) => ConcreteType::Signed,
        OpKind::FieldRead { ty, .. } => valuetype_to_concrete(ty),
        OpKind::ArrayRead { item_ty, .. } => valuetype_to_concrete(item_ty),
        OpKind::Call { result_ty, .. }
        | OpKind::CallElidable { result_ty, .. }
        | OpKind::CallResidual { result_ty, .. }
        | OpKind::CallMayForce { result_ty, .. } => valuetype_to_concrete(result_ty),
        OpKind::BinOp { result_ty, .. } | OpKind::UnaryOp { result_ty, .. } => {
            let c = valuetype_to_concrete(result_ty);
            if c != ConcreteType::Unknown {
                c
            } else {
                ConcreteType::Signed
            }
        }
        _ => ConcreteType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MajitGraph, OpKind, Terminator, ValueType};
    use crate::passes::annotate;

    #[test]
    fn resolves_int_types() {
        let mut graph = MajitGraph::new("test");
        let entry = graph.entry;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v)));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(v), &ConcreteType::Signed);
    }

    #[test]
    fn resolves_ref_field() {
        let mut graph = MajitGraph::new("test");
        let entry = graph.entry;
        let base = graph.alloc_value();
        let v = graph
            .push_op(
                entry,
                OpKind::FieldRead {
                    base,
                    field: crate::graph::FieldDescriptor::new("obj", None),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v)));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(v), &ConcreteType::GcRef);
    }

    #[test]
    fn resolves_phi_through_link_args() {
        let mut graph = MajitGraph::new("phi");
        let entry = graph.entry;
        let val = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let (target, phi_args) = graph.create_block_with_args(1);
        let phi = phi_args[0];
        graph.set_terminator(
            entry,
            Terminator::Goto {
                target,
                args: vec![val],
            },
        );
        graph.set_terminator(target, Terminator::Return(Some(phi)));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(phi), &ConcreteType::Signed);
    }
}
