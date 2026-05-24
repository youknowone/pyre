//! Concrete-kind helpers — `ConcreteType` projection from
//! `LowLevelType`, `ValueType`, op-result kinds, plus
//! `apply_from_flowspace_variables` (rebinds graph slots to the
//! rtyper-typed Variables, propagating `Variable.concretetype` into
//! every alias).
//!
//! Type kinds flow through `Variable.concretetype`
//! (`rpython/flowspace/model.py:280 Variable.__slots__ = [..., "concretetype"]`;
//! `:355 Constant.__slots__ = ["concretetype"]`) — set inline by the
//! rtyper via `RPythonTyper.setconcretetype()`
//! (`rpython/rtyper/rtyper.py:258 v.concretetype = ...`).  Pyre
//! reproduces this through `FunctionGraph::set_concretetype_of_inline`
//! writes followed by `FunctionGraph::concretetype_of(&v)` reads
//! (which routes to the backing `Variable.concretetype` cell).  No
//! external slot table survives.

use std::collections::HashMap;

use crate::flowspace::model::Variable;
use crate::model::{FunctionGraph, OpKind, ValueType};

/// Re-export the canonical [`ConcreteType`] from [`crate::model`].
///
/// The kind enum used to live here as a side-table value type;
/// after the medium-term parity push it lives on each backing
/// `Variable.concretetype` cell stored in
/// [`FunctionGraph::value_variables`] (mirroring upstream
/// `Variable.concretetype` line-for-line).  The alias keeps existing
/// imports working while consumers migrate to reading
/// `FunctionGraph::concretetype_of(&v)`.
pub use crate::model::ConcreteType;

/// Rebind each per-slot Variable handle to the upstream-typed
/// Variable in `value_to_var`, so subsequent
/// `FunctionGraph::concretetype_of(&v)` reads route through the
/// rtyper's `Variable.concretetype` directly.
///
/// The codewriter reads kinds via `FunctionGraph::concretetype_of(&v)`,
/// which routes to each `Variable.concretetype` cell (set by the
/// `RPythonTyper`) and projects through [`crate::model::getkind`],
/// matching upstream's `getkind(v.concretetype)` access pattern.
///
/// Variables whose `concretetype` is still `None` (rtyper hasn't
/// processed them yet) leave the graph slot untouched —
/// equivalent to RPython's "no `.concretetype` attribute" window
/// before `setconcretetype` runs.  Pyre's
/// [`crate::model::ConcreteType::Unknown`] sentinel covers that
/// state.
pub fn apply_from_flowspace_variables(
    graph: &mut FunctionGraph,
    value_to_var: &crate::translator::rtyper::flowspace_adapter::SlotToVariable,
) {
    for (idx, var) in value_to_var.iter() {
        // Honour the docstring contract above: a source `Variable`
        // whose `concretetype` is still `None` represents the pre-
        // `setconcretetype` window in RPython, where the graph slot
        // must remain untouched.  `bind_variable_at` is defensive
        // about this (it only copies a `Some` concretetype onto the
        // placeholder), but invoking it with an untyped source still
        // registers a spurious `variable_to_vid[var.id()] -> slot`
        // entry that subsequent `slot_of(&var)` lookups would
        // resolve unexpectedly.  Skip the call outright so the
        // docstring claim holds bit-for-bit.
        if var.concretetype().is_none() {
            continue;
        }
        // `bind_variable_at` merges the rtyper Variable's
        // `concretetype` onto the existing placeholder in
        // `value_variables[slot]`, preserving Variable identity
        // across every graph slot that holds the placeholder
        // (Block.inputargs, op operands, Link.args, exitswitch,
        // last_exception, last_exc_value).  Mirrors upstream
        // `v.concretetype = T` attribute aliasing.
        graph.bind_variable_at(*idx, var.clone());
    }
}

/// `ValueType` → `ConcreteType` projection used by both
/// `resolve_types` (legacy graph walk) and `authoritative_result_types`
/// (post-jtransform op-result projection).
///
/// `Bool` collapses to `Signed` because RPython `BoolRepr.lowleveltype
/// = Bool` lifts to LL `Signed` for the codewriter; the legacy resolver
/// followed the same collapse and the post-jtransform projection
/// matches it.
pub(crate) fn valuetype_to_concrete(vt: &ValueType) -> ConcreteType {
    match vt {
        // `Unsigned` shares the `Signed` ConcreteType — the codewriter
        // / regalloc do not distinguish signedness (`getkind(Unsigned)
        // == 'int'`); only the rtyper picks `IntegerRepr.lowleveltype
        // = Unsigned` based on `SomeInteger.unsigned`.
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => ConcreteType::Signed,
        ValueType::Ref => ConcreteType::GcRef,
        ValueType::Float => ConcreteType::Float,
        ValueType::Void => ConcreteType::Void,
        ValueType::State | ValueType::Unknown => ConcreteType::Unknown,
    }
}

/// `result_kind: char` → `ConcreteType` projection used by jtransform
/// call families (`CallElidable` / `CallResidual` / `CallMayForce` /
/// `InlineCall` / `RecursiveCall`).
pub(crate) fn kind_char_to_concrete(kind: char) -> ConcreteType {
    match kind {
        'i' => ConcreteType::Signed,
        'r' => ConcreteType::GcRef,
        'f' => ConcreteType::Float,
        'v' => ConcreteType::Void,
        _ => ConcreteType::Unknown,
    }
}

fn concrete_if_known(concrete: ConcreteType) -> Option<ConcreteType> {
    if concrete == ConcreteType::Unknown {
        None
    } else {
        Some(concrete)
    }
}

/// Per-op `ConcreteType` declared by the rewritten graph's op-result
/// fields (`result_ty` / `result_kind`).  Authoritative for op-result
/// kinds because the rewriter declares them at lowering time, so this
/// projection wins over `original` operand inferences in
/// [`merge_synth_kinds`]'s precedence chain.
pub(crate) fn authoritative_result_type_from_op(kind: &OpKind) -> Option<ConcreteType> {
    match kind {
        OpKind::ConstInt(_) => Some(ConcreteType::Signed),
        OpKind::ConstBool(_) => Some(ConcreteType::Signed),
        OpKind::ConstFloat(_) => Some(ConcreteType::Float),
        OpKind::Input { ty, .. } => concrete_if_known(valuetype_to_concrete(ty)),
        OpKind::FieldRead { ty, .. } | OpKind::VableFieldRead { ty, .. } => {
            concrete_if_known(valuetype_to_concrete(ty))
        }
        OpKind::ArrayRead { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. } => {
            concrete_if_known(valuetype_to_concrete(item_ty))
        }
        OpKind::Call { result_ty, .. }
        | OpKind::IndirectCall { result_ty, .. }
        | OpKind::BinOp { result_ty, .. }
        | OpKind::UnaryOp { result_ty, .. } => concrete_if_known(valuetype_to_concrete(result_ty)),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => {
            concrete_if_known(kind_char_to_concrete(*result_kind))
        }
        OpKind::VtableMethodPtr { .. } => Some(ConcreteType::Signed),
        _ => None,
    }
}

/// Walk the rewritten graph and collect every op-result that carries an
/// authoritative `ConcreteType` (per-op declaration), keyed on the backing
/// [`Variable`].  Feeds [`merge_synth_kinds`]'s `post_result` lane.
pub(crate) fn authoritative_result_types(graph: &FunctionGraph) -> HashMap<Variable, ConcreteType> {
    let mut result = HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(var) = op.result.as_ref() else {
                continue;
            };
            if let Some(concrete) = authoritative_result_type_from_op(&op.kind) {
                result.insert(var.clone(), concrete);
            }
        }
    }
    result
}

// `build_value_kinds` retired — the regalloc / flatten / assemble
// pipeline now reads kinds straight off the Variable's
// `.concretetype` cell (the upstream-orthodox source).
// Per-Variable `RegKind` projections happen at the use site via
// `regalloc::perform_register_allocation`'s internal
// `concretetype_to_regkind`, matching RPython's
// `getkind(v.concretetype)` access pattern bit for bit.
