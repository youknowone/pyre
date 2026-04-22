//! Type resolution pass.
//!
//! **LEGACY.** Flat `ConcreteType` enum with ad-hoc lowering.
//! Line-by-line port of `rtyper/rtyper.py:RPythonTyper` +
//! `rtyper/rmodel.py:Repr` hierarchy is landing at
//! `majit-rtyper/src/{rtyper,rmodel}.rs` (roadmap Phase 6). This file
//! is deleted at roadmap commit P8.11.
//!
//! Transforms annotated ValueTypes into concrete low-level types
//! and specializes operations accordingly.

use std::collections::HashMap;

use crate::flowspace::model::ConstValue;
use crate::model::{FunctionGraph, Link, LinkArg, OpKind, ValueId, ValueType};
use crate::translate_legacy::annotator::annrpython::AnnotationState;
use crate::translator::rtyper::rpbc::LLCallTable;

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
    pub concrete_calltables: HashMap<usize, (LLCallTable, usize)>,
}

impl TypeResolutionState {
    pub fn new() -> Self {
        TypeResolutionState {
            concrete_types: HashMap::new(),
            concrete_calltables: HashMap::new(),
        }
    }

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
    let mut state = TypeResolutionState::new();

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

    // Cross-block: propagate through Link args → target inputargs.
    // Keep the exception-link split explicit, mirroring upstream's
    // `_convert_link()` handling of `last_exception` /
    // `last_exc_value` before the per-arg conversion loop.
    for block in &graph.blocks {
        for link in &block.exits {
            if link_is_raise_like(link) {
                convert_raise_link(&mut state, graph, link);
            } else {
                convert_link(&mut state, graph, link);
            }
        }
    }

    state
}

fn const_value_to_concrete(value: &ConstValue) -> ConcreteType {
    match value {
        ConstValue::Int(_) | ConstValue::Bool(_) | ConstValue::SpecTag(_) => ConcreteType::Signed,
        ConstValue::Float(_) => ConcreteType::Float,
        ConstValue::Placeholder => ConcreteType::Unknown,
        ConstValue::Atom(_)
        | ConstValue::Dict(_)
        | ConstValue::Str(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Graphs(_)
        | ConstValue::None
        | ConstValue::Code(_)
        | ConstValue::LLPtr(_)
        | ConstValue::Function(_)
        | ConstValue::HostObject(_) => ConcreteType::GcRef,
    }
}

fn link_is_raise_like(link: &Link) -> bool {
    link.last_exception.is_some() && link.last_exc_value.is_some()
}

fn convert_link(state: &mut TypeResolutionState, graph: &FunctionGraph, link: &Link) {
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        maybe_seed_concrete_type(state, *dst, link_arg_concrete_type(state, src));
    }
}

fn convert_raise_link(state: &mut TypeResolutionState, graph: &FunctionGraph, link: &Link) {
    if let Some(LinkArg::Value(value)) = link.last_exception.as_ref() {
        maybe_seed_concrete_type(state, *value, ConcreteType::Signed);
    }
    if let Some(LinkArg::Value(value)) = link.last_exc_value.as_ref() {
        maybe_seed_concrete_type(state, *value, ConcreteType::GcRef);
    }

    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ConcreteType::Signed
        } else if Some(src) == link.last_exc_value.as_ref() {
            ConcreteType::GcRef
        } else {
            link_arg_concrete_type(state, src)
        };
        maybe_seed_concrete_type(state, *dst, src_ty);
    }
}

fn link_arg_concrete_type(state: &TypeResolutionState, src: &LinkArg) -> ConcreteType {
    match src {
        LinkArg::Value(src) => state.get(*src).clone(),
        LinkArg::Const(value) => const_value_to_concrete(value),
    }
}

fn maybe_seed_concrete_type(state: &mut TypeResolutionState, dst: ValueId, src_ty: ConcreteType) {
    if state.get(dst) == &ConcreteType::Unknown && src_ty != ConcreteType::Unknown {
        state.concrete_types.insert(dst, src_ty);
    }
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
pub fn build_value_kinds(types: &TypeResolutionState) -> HashMap<ValueId, crate::flatten::RegKind> {
    use crate::flatten::RegKind;
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
        // Vtable funcptr extraction returns an integer pointer (RPython
        // `op.args[0]` of `indirect_call` is `Ptr(FuncType)`).
        OpKind::VtableMethodPtr { .. } => ConcreteType::Signed,
        OpKind::IndirectCall { result_ty, .. } => valuetype_to_concrete(result_ty),
        _ => ConcreteType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType, exception_exitcase,
    };
    use crate::translate_legacy::annotator::annrpython as annotate;

    #[test]
    fn resolves_int_types() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_return(entry, Some(v));

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
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v));

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
        graph.set_goto(entry, target, vec![val]);
        graph.set_return(target, Some(phi));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(phi), &ConcreteType::Signed);
    }

    #[test]
    fn resolves_raise_link_exception_pair() {
        let mut graph = FunctionGraph::new("raise_link");
        let entry = graph.startblock;
        let (exc_block, etype, evalue) = graph.exceptblock_args();
        let last_exception = graph.alloc_value();
        let last_exc_value = graph.alloc_value();
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
                Link::new(
                    vec![last_exception, last_exc_value],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::from(last_exception)),
                    Some(LinkArg::from(last_exc_value)),
                ),
            ],
        );

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(last_exception), &ConcreteType::Signed);
        assert_eq!(types.get(last_exc_value), &ConcreteType::GcRef);
        assert_eq!(types.get(etype), &ConcreteType::Signed);
        assert_eq!(types.get(evalue), &ConcreteType::GcRef);
    }
}
