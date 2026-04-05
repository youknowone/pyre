//! Type resolution pass.
//!
//! RPython equivalent: `rtyper/rtyper.py` RPythonTyper.
//!
//! Transforms annotated ValueTypes into concrete low-level types
//! and specializes operations accordingly.

use std::collections::HashMap;

use crate::model::{FunctionGraph, OpKind, ValueId, ValueType};
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
pub fn resolve_types(graph: &FunctionGraph, annotations: &AnnotationState) -> TypeResolutionState {
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
        for op in &block.operations {
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
            crate::model::Terminator::Goto { target, args } => {
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
            crate::model::Terminator::Branch {
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

fn kind_char_to_concrete(kind: char) -> ConcreteType {
    match kind {
        'i' => ConcreteType::Signed,
        'r' => ConcreteType::GcRef,
        'f' => ConcreteType::Float,
        'v' => ConcreteType::Void,
        _ => ConcreteType::Unknown,
    }
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

/// Build value kind map from type resolution state.
///
/// RPython: `getkind(v.concretetype)` — in RPython, types live directly
/// on variables. In majit, we extract them from TypeResolutionState.
///
/// Used by both `perform_all_register_allocations()` (before flatten)
/// and `flatten_with_types()` (populates SSARepr.value_kinds).
pub fn build_value_kinds(
    types: &TypeResolutionState,
) -> HashMap<ValueId, crate::passes::flatten::RegKind> {
    use crate::passes::flatten::RegKind;
    types
        .concrete_types
        .iter()
        .filter_map(|(&vid, ct)| {
            let kind = match ct {
                ConcreteType::Signed => RegKind::Int,
                ConcreteType::GcRef => RegKind::Ref,
                ConcreteType::Float => RegKind::Float,
                _ => return None,
            };
            Some((vid, kind))
        })
        .collect()
}

fn infer_concrete_from_op(kind: &OpKind) -> ConcreteType {
    match kind {
        OpKind::ConstInt(_) => ConcreteType::Signed,
        OpKind::FieldRead { ty, .. } => valuetype_to_concrete(ty),
        OpKind::ArrayRead { item_ty, .. } => valuetype_to_concrete(item_ty),
        OpKind::InteriorFieldRead { item_ty, .. } => valuetype_to_concrete(item_ty),
        OpKind::Call { result_ty, .. } => valuetype_to_concrete(result_ty),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => kind_char_to_concrete(*result_kind),
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
    use crate::model::{FunctionGraph, OpKind, Terminator, ValueType};
    use crate::passes::annotate;

    #[test]
    fn resolves_int_types() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v)));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(v), &ConcreteType::Signed);
    }

    #[test]
    fn resolves_ref_field() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let base = graph.alloc_value();
        let v = graph
            .push_op(
                entry,
                OpKind::FieldRead {
                    base,
                    field: crate::model::FieldDescriptor::new("obj", None),
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
        let mut graph = FunctionGraph::new("phi");
        let entry = graph.startblock;
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
