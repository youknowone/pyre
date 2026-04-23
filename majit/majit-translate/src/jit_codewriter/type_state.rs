//! ValueId → ConcreteType side table for the jit_codewriter IR.
//!
//! PRE-EXISTING-ADAPTATION. RPython stores `.concretetype` inline on each
//! `Variable` after `RPythonTyper.specialize()` rewrites the graph, so no
//! side table exists upstream. Pyre's current jit_codewriter consumes a
//! `crate::model::FunctionGraph` (value-id-based, not variable-based),
//! so the post-rtyper kind information lives in this separate table.
//!
//! `build_value_kinds` (pure `ConcreteType → RegKind` projection) lives
//! here beside the data type. The graph-walking algorithm `resolve_types`
//! remains in `translate_legacy/rtyper/rtyper.rs` until the real rtyper
//! (`translator/rtyper/`) produces per-Variable concretetypes end-to-end
//! and replaces it.

use std::collections::HashMap;

use crate::model::ValueId;

/// Concrete low-level type. RPython `Repr.lowleveltype` collapsed to the
/// four kinds the jit_codewriter needs (Signed / GcRef / Float / Void)
/// plus `Unknown` for pre-resolution slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConcreteType {
    /// Signed integer (RPython `Signed` / i64).
    Signed,
    /// GC reference (RPython `Ptr(GcStruct)`).
    GcRef,
    /// Float (RPython `Float` / f64).
    Float,
    /// Void (RPython `Void`).
    Void,
    /// Unknown / unresolved.
    Unknown,
}

/// Type resolution state: `ValueId → ConcreteType`.
pub struct TypeResolutionState {
    pub concrete_types: HashMap<ValueId, ConcreteType>,
}

impl TypeResolutionState {
    pub fn new() -> Self {
        TypeResolutionState {
            concrete_types: HashMap::new(),
        }
    }

    pub fn get(&self, id: ValueId) -> &ConcreteType {
        self.concrete_types
            .get(&id)
            .unwrap_or(&ConcreteType::Unknown)
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
