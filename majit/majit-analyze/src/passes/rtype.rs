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

    // Additionally resolve from ops that have explicit type info
    for block in &graph.blocks {
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
        OpKind::Call { result_ty, .. } => valuetype_to_concrete(result_ty),
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
                    field: "obj".into(),
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
}
