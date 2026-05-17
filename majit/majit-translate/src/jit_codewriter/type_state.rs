//! `ValueId → ConcreteType` slot table — pyre structural adapter.
//!
//! **NOT** RPython-orthodox.  RPython attaches `concretetype` to each
//! `Variable` / `Constant` object as an inline slot
//! (`rpython/flowspace/model.py:280 Variable.__slots__ = [..., "concretetype"]`;
//! `:355 Constant.__slots__ = ["concretetype"]`) and the rtyper writes
//! it in place via `RPythonTyper.setconcretetype()`
//! (`rpython/rtyper/rtyper.py:258  v.concretetype = self.bindingrepr(v).lowleveltype`).
//! Reading a Variable's type upstream is a plain attribute access on
//! the object itself; there is no external lookup table and no
//! iterator over a type-store.
//!
//! Pyre's `FunctionGraph` is `ValueId(usize)`-based: a ValueId is a
//! detached index, not an object that can carry an inline slot.  Two
//! genuine ports of RPython's per-Variable inline shape are possible
//! and remain multi-session future work:
//!
//!   1. Carry the `concretetype` inline at each defining site —
//!      `SpaceOperation.result_concretetype` and
//!      `Block.inputarg_concretetypes` (parallel to
//!      `Block.inputargs`).  Reads walk the graph to the defining
//!      site.  Requires modifying every `SpaceOperation` / `Block`
//!      construction site across the front-end, rtyper, and
//!      jit_codewriter.
//!
//!   2. Replace `ValueId(usize)` with a `Variable`-like handle that
//!      owns its `concretetype` slot, mirroring
//!      `flowspace.model.Variable` more directly.
//!
//! Until then this file holds the structural adapter that bridges the
//! gap: a `ValueId`-indexed `Vec<ConcreteType>` populated by the
//! rtyper projection at `crate::translator::rtyper::cutover` ~1266
//! (`for (&vid, var) in &value_to_var { state.set(vid, lowleveltype_to_concrete(...)) }`)
//! and read by jtransform / codewriter / assembler.  The dense Vec
//! preserves the "every ValueId has a slot" invariant — closer to
//! RPython's per-object slot than a sparse `HashMap` would be, since
//! HashMap models "Variables that *happen* to carry a type binding"
//! whereas RPython's `__slots__` makes the slot a property of the
//! object itself.  The storage type is an implementation detail
//! callers must not depend on (access goes through `get` / `try_get`
//! / `contains` / `set` / `iter`); the visible shape is "every
//! ValueId resolves to one ConcreteType, with `Unknown` as the
//! pre-rtyper sentinel".
//!
//! Resource-behaviour note: pyre's `FunctionGraph::set_next_value`
//! (`model.rs:2114`) can advance the allocator cursor past unminted
//! ValueIds — `Transformer::allocate_synthetic_value` does this at
//! `jtransform.rs:452`, and front-end test fixtures seed the cursor
//! to `100`.  The Vec pays Unknown-padding cost across such skip
//! ranges; this is a Rust-IR-adapter property of pyre's cursor API,
//! not a parity argument for or against the Vec shape.  Sizing this
//! out either way requires the structural port (option 1 or 2 above),
//! at which point the side-table goes away entirely.
//!
//! Related structural adapters worth keeping in view alongside this
//! one: `crate::translator::rtyper::rclass::lower_vtable_method_ptr`
//! (`rclass.rs` ~75) emits `OpKind::VtableMethodPtr` to materialise a
//! callable funcptr ValueId before `OpKind::IndirectCall`; RPython's
//! `ClassRepr.getclsfield` (`rpython/rtyper/rclass.py:371`) emits
//! vtable field-access low-level ops directly, so the
//! `VtableMethodPtr` shape is itself a Rust-IR bridge rather than an
//! orthodox port.  Retiring the adapter list together is the
//! eventual shape of full structural parity.
//!
//! The `build_value_kinds` (pure `ConcreteType → RegKind` projection)
//! and `merge_synth_kinds` (post-jtransform 4-source merge) helpers
//! live here beside the data type — these are pyre-only divergences
//! from RPython that survive the rtyper cutover
//! (`~/.claude/plans/0-warm-raccoon.md` Slice 3).  The graph-walking
//! algorithm `resolve_types` remains in
//! `translator/rtyper/legacy_resolve.rs` until the real rtyper
//! (`translator/rtyper/`) produces per-Variable concretetypes
//! end-to-end and replaces it.

use std::collections::HashMap;

use crate::model::{FunctionGraph, OpKind, ValueId, ValueType};

/// Re-export the canonical [`ConcreteType`] from [`crate::model`].
///
/// The kind enum used to live here as a side-table value type;
/// after the medium-term parity push it lives on each backing
/// `Variable.concretetype` cell stored in
/// [`FunctionGraph::value_variables`] (mirroring upstream
/// `Variable.concretetype` line-for-line).  The alias keeps existing
/// imports working while consumers migrate to reading
/// `graph.concretetype(v)`.
pub use crate::model::ConcreteType;

/// Returned by [`TypeResolutionState::get`] for slots that have not
/// been populated.  Pyre's adapter returns this `Unknown` sentinel
/// instead of failing the way RPython does on unset
/// `var.concretetype` (`AttributeError`) — merge / resolver code
/// paths poll bulk ValueIds blindly and need a total function rather
/// than the upstream's selective access pattern.  Acknowledging the
/// divergence: pre-`setconcretetype` Variables upstream are *invalid
/// to read*; pyre's slot table treats absence as a regular value.
const UNKNOWN: ConcreteType = ConcreteType::Unknown;

/// `ValueId → ConcreteType` slot table — structural adapter, see the
/// file header for the orthodox shape this stands in for.
///
/// Storage is a `Vec<ConcreteType>` indexed by `ValueId.0`, so every
/// ValueId in `[0, next_value)` has exactly one slot — the same
/// "every Variable has a slot" invariant RPython gets from
/// `__slots__`, just routed through a side table because pyre's
/// `ValueId(usize)` is not an object.  Public surface is method
/// shaped (`get` / `set` / `try_get` / `contains` / `iter`) to mirror
/// `var.concretetype` / `setattr` / `hasattr` patterns; the Vec is
/// an implementation detail.
///
/// **Long-term role** — this struct is the build-time scratch buffer
/// used by jtransform / legacy-resolve while they compute per-value
/// kinds.  The authoritative store after the rtyper handoff is each
/// backing `Variable.concretetype` cell on
/// [`FunctionGraph::value_variables`]: [`apply_to_graph`] writes
/// every non-Unknown slot through `graph.set_concretetype`, after
/// which downstream consumers read kinds via `graph.concretetype(v)`.
///
/// Collapsed to the four-way `Signed` / `GcRef` / `Float` / `Void` axis
/// used by the JIT codewriter, per `rpython/jit/metainterp/history.py:45-71
/// getkind`.
#[derive(Debug, Default, Clone)]
pub struct TypeResolutionState {
    slots: Vec<ConcreteType>,
}

impl TypeResolutionState {
    pub fn new() -> Self {
        TypeResolutionState { slots: Vec::new() }
    }

    /// Reserve dense slot storage for ValueIds in `[0, capacity)`.
    pub fn with_capacity(capacity: usize) -> Self {
        TypeResolutionState {
            slots: vec![ConcreteType::Unknown; capacity],
        }
    }

    /// Lookup the concretetype for `id`.  Returns `&Unknown` for
    /// slots that have not been populated (out-of-range or explicit
    /// Unknown).  Divergence from upstream `Variable.concretetype`
    /// noted on [`UNKNOWN`].
    pub fn get(&self, id: ValueId) -> &ConcreteType {
        self.slots.get(id.0).unwrap_or(&UNKNOWN)
    }

    /// Lookup with `Option` semantics — returns `None` for slots that
    /// have not been populated (and also for slots explicitly set to
    /// `Unknown`).  Load-bearing at jtransform's `get_value_type`
    /// where an unset slot must propagate as `None`.
    pub fn try_get(&self, id: ValueId) -> Option<&ConcreteType> {
        match self.slots.get(id.0) {
            None | Some(ConcreteType::Unknown) => None,
            Some(ct) => Some(ct),
        }
    }

    /// True iff `id` has an explicitly-populated, non-`Unknown` slot.
    /// Closest pyre analogue to RPython's `hasattr(var,
    /// 'concretetype')`, modulo the `Unknown`-as-sentinel divergence.
    pub fn contains(&self, id: ValueId) -> bool {
        matches!(self.slots.get(id.0), Some(ct) if *ct != ConcreteType::Unknown)
    }

    /// Bind `id`'s concretetype.  Auto-extends the slot table with
    /// `Unknown` padding when `id.0` is past the current length so
    /// the "every ValueId has a slot" invariant holds.  Idempotent;
    /// later writes win, mirroring RPython's `var.concretetype =
    /// lltype` re-assignment.
    pub fn set(&mut self, id: ValueId, ct: ConcreteType) {
        if id.0 >= self.slots.len() {
            self.slots.resize(id.0 + 1, ConcreteType::Unknown);
        }
        self.slots[id.0] = ct;
    }

    /// Read the per-`ValueId` slice directly — exposed so callers
    /// that just need an indexable view (e.g. liveness /
    /// assembler kind lookups) can avoid going through the
    /// HashMap-shaped `get` accessor.  The slice is indexed by
    /// `ValueId.0`; an out-of-range index denotes "not yet
    /// populated", same as `Unknown`.
    pub fn as_slice(&self) -> &[ConcreteType] {
        &self.slots
    }

    /// Iterate populated slots in ascending `ValueId` order.  Stable
    /// ordering is load-bearing for `cutover::compare_real_against_legacy`
    /// (`cutover.rs:408`) whose "first divergence" message must be
    /// deterministic across runs.  Skips both unpopulated and
    /// explicit-`Unknown` slots (the latter is the sentinel for the
    /// pre-`setconcretetype` state).
    pub fn iter(&self) -> impl Iterator<Item = (ValueId, &ConcreteType)> + '_ {
        self.slots.iter().enumerate().filter_map(|(idx, ct)| {
            if *ct == ConcreteType::Unknown {
                None
            } else {
                Some((ValueId(idx), ct))
            }
        })
    }
}

/// Bulk-write every entry of a transitional [`TypeResolutionState`]
/// into each `ValueId`'s backing Variable via
/// `graph.set_concretetype` (which writes through to
/// `Variable.concretetype`).  After this call
/// `graph.concretetype(v) == types.get(v)` for every `v` covered by
/// the transitional table; values absent from `types` keep their
/// existing kind (`Unknown` for fresh allocations).
///
/// Mirrors RPython's "rtyper finishes, every Variable now has
/// `.concretetype`" handoff — pyre's scratch table is the staging
/// area and `apply_to_graph` is the commit.
pub fn apply_to_graph(types: &TypeResolutionState, graph: &mut FunctionGraph) {
    // `iter()` already filters out `Unknown` slots so the graph
    // keeps its existing per-value kind for unpopulated entries —
    // the rtyper-finishes handoff only stamps positively-classified
    // slots, mirroring upstream's `setconcretetype` only writing
    // when a concretetype actually resolved.
    for (v, ty) in types.iter() {
        graph.set_concretetype(v, ty.clone());
    }
}

/// Rebind each ValueId's backing
/// [`crate::flowspace::model::Variable`] to the upstream-typed
/// Variable in `value_to_var`, so subsequent
/// `graph.concretetype(v)` reads route through the rtyper's
/// `Variable.concretetype` directly.
///
/// **Long-term parity path** — this is the path the codewriter
/// will use once every value has a backing flowspace `Variable`:
/// the kind comes from `Variable.concretetype` (set by the
/// `RPythonTyper`) projected through [`crate::model::getkind`],
/// matching upstream's
/// `getkind(v.concretetype)` access pattern bit for bit.  No
/// transitional [`TypeResolutionState`] needed.
///
/// Variables whose `concretetype` is still `None` (rtyper hasn't
/// processed them yet) leave the graph slot untouched —
/// equivalent to RPython's "no `.concretetype` attribute" window
/// before `setconcretetype` runs.  Pyre's
/// [`crate::model::ConcreteType::Unknown`] sentinel covers that
/// state.
pub fn apply_from_flowspace_variables(
    graph: &mut FunctionGraph,
    value_to_var: &crate::translator::rtyper::flowspace_adapter::ValueIdToVariable,
) {
    for (vid, var) in value_to_var.iter() {
        // Honour the docstring contract above: a source `Variable`
        // whose `concretetype` is still `None` represents the pre-
        // `setconcretetype` window in RPython, where the graph slot
        // must remain untouched.  `bind_variable` is defensive about
        // this (it only copies a `Some` concretetype onto the
        // placeholder), but invoking it with an untyped source still
        // registers a spurious `variable_to_vid[var.id()] -> vid`
        // entry that subsequent `value_id_of(&var)` lookups would
        // resolve unexpectedly.  Skip the call outright so the
        // docstring claim holds bit-for-bit.
        if var.concretetype().is_none() {
            continue;
        }
        // `bind_variable` merges the rtyper Variable's `concretetype`
        // onto the existing placeholder in `value_variables[vid]`,
        // preserving Variable identity across every graph slot that
        // holds the placeholder (Block.inputargs, op operands,
        // Link.args, exitswitch, last_exception, last_exc_value).
        // Mirrors upstream `v.concretetype = T` attribute aliasing.
        graph.bind_variable(*vid, var.clone());
    }
}

/// Post-jtransform 4-source merge of `ConcreteType` views into a
/// single authoritative `TypeResolutionState`.
///
/// **Why this exists** — pyre divergence, not RPython parity.
///
/// RPython has no merge step: every `Variable` carries `.concretetype`
/// inline, and jtransform-created operations preserve or assign that
/// type on the result variable. Pyre's legacy jit_codewriter graph
/// uses a `ValueId -> ConcreteType` side table, so the codewriter has
/// up to four partial sources after jtransform that must be reconciled
/// into a single `TypeResolutionState` before regalloc / flatten /
/// assemble:
///
/// 1. `original` — the pre-jtransform rtyper output. Carries backward
///    inferences that may disappear after jtransform removed the
///    consumer op that caused them.
/// 2. `post_resolve` — a fresh `resolve_types(rewritten_graph,
///    annotations)` walk over the post-jtransform graph. Picks up
///    anything visible on the rewritten shape.
/// 3. `post_result` — the per-op `result_ty` / `result_kind` declared
///    on each rewritten op. Authoritative for op-result kinds (since
///    the rewriter declares them), so it wins over `original`'s
///    operand inferences.
///
/// Precedence: `post_result` > `post_resolve` > `original`.  `original`
/// only fills in slots that `post_result` doesn't claim, matching
/// RPython's "single authoritative `op.result.concretetype`" invariant:
/// a stale pre-rewrite operand kind never overrides a freshly-declared
/// post-rewrite result kind.
///
/// jtransform-stamped values are not a separate layer: they were
/// written straight to each backing `Variable.concretetype` cell via
/// `graph.set_concretetype` during the transform pass (mirroring
/// RPython's inline `v.concretetype = T` writes), so they are already
/// observable via `post_resolve` (which reads them out of the graph)
/// and via `post_result` (declared on the rewritten op).
pub fn merge_synth_kinds(
    original: &TypeResolutionState,
    post_resolve: TypeResolutionState,
    post_result: HashMap<ValueId, ConcreteType>,
) -> TypeResolutionState {
    let mut merged = post_resolve;

    // (1) `original` operand inferences fill unresolved slots, but
    // never override `post_result`.  `iter` already skips `Unknown`
    // sentinel slots.
    for (value, kind) in original.iter() {
        if !post_result.contains_key(&value) {
            merged.set(value, kind.clone());
        }
    }
    // (2) Op-result kinds win over operand inferences.
    for (value, kind) in post_result {
        merged.set(value, kind);
    }

    merged
}

/// Direct-to-graph variant of [`merge_synth_kinds`].
///
/// Same precedence stack (`post_result > post_resolve > original`) but
/// writes the per-value result straight through to each backing
/// `Variable.concretetype` cell via `graph.set_concretetype` instead of
/// building a transitional [`TypeResolutionState`].  Production callers
/// go through this entry so the graph IS the merge target — no
/// intermediate side table to thread downstream.  Skips Unknown writes
/// so the canonical-exceptblock stamp performed elsewhere
/// (`augment_canonical_exceptblock_on_graph`) is not clobbered.
pub fn merge_synth_kinds_into_graph(
    graph: &mut crate::model::FunctionGraph,
    original: &TypeResolutionState,
    post_resolve: &TypeResolutionState,
    post_result: &HashMap<ValueId, ConcreteType>,
) {
    use crate::model::ConcreteType;
    // Precedence per the signature docstring is
    // `post_result > post_resolve > original`, so `post_resolve` is the
    // base layer and `post_result` writes last.  Apply order matches
    // precedence inverted — write the lowest-precedence layer first,
    // then upper layers override:
    //
    //   (0) `post_resolve` — base layer; subsequent writes may
    //       override.  Higher precedence than `original` because
    //       `original` skips entries that `post_result` will
    //       overwrite anyway, and otherwise carries
    //       operand-inference kinds.
    for (value, kind) in post_resolve.iter() {
        if !matches!(kind, ConcreteType::Unknown) {
            graph.set_concretetype(value, kind.clone());
        }
    }
    //   (1) `original` operand inferences fill unresolved slots, but
    //       skip `post_result`-claimed values so step (2) wins
    //       unambiguously.
    for (value, kind) in original.iter() {
        if !post_result.contains_key(&value) && !matches!(kind, ConcreteType::Unknown) {
            graph.set_concretetype(value, kind.clone());
        }
    }
    //   (2) Op-result kinds — highest precedence; last write wins.
    for (value, kind) in post_result {
        if !matches!(kind, ConcreteType::Unknown) {
            graph.set_concretetype(*value, kind.clone());
        }
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
/// authoritative `ConcreteType` (per-op declaration).  Feeds
/// [`merge_synth_kinds`]'s `post_result` lane.
pub(crate) fn authoritative_result_types(graph: &FunctionGraph) -> HashMap<ValueId, ConcreteType> {
    let mut result = HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(value) = op.result else {
                continue;
            };
            if let Some(concrete) = authoritative_result_type_from_op(&op.kind) {
                result.insert(value, concrete);
            }
        }
    }
    result
}

// `build_value_kinds` retired — the regalloc / flatten / assemble
// pipeline now reads kinds straight off `graph.concretetype(v)`
// (which routes to each `ValueId`'s backing
// `Variable.concretetype` cell, the upstream-orthodox source).
// Per-`ValueId` `RegKind` projections
// happen at the use site via `regalloc::perform_register_allocation`'s
// internal `concretetype_to_regkind`, matching RPython's
// `getkind(v.concretetype)` access pattern bit for bit.

#[cfg(test)]
mod tests {
    use super::*;

    fn state_from(pairs: &[(ValueId, ConcreteType)]) -> TypeResolutionState {
        let mut s = TypeResolutionState::new();
        for (v, k) in pairs {
            s.set(*v, k.clone());
        }
        s
    }

    fn map_from(pairs: &[(ValueId, ConcreteType)]) -> HashMap<ValueId, ConcreteType> {
        pairs.iter().cloned().collect()
    }

    #[test]
    fn merge_synth_kinds_post_resolve_starts_as_base() {
        // No original / post_result overrides — the merged state is
        // just `post_resolve`.
        let post_resolve = state_from(&[
            (ValueId(1), ConcreteType::Signed),
            (ValueId(2), ConcreteType::Float),
        ]);
        let original = TypeResolutionState::new();
        let post_result: HashMap<ValueId, ConcreteType> = HashMap::new();

        let merged = merge_synth_kinds(&original, post_resolve, post_result);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::Signed);
        assert_eq!(merged.get(ValueId(2)), &ConcreteType::Float);
    }

    #[test]
    fn merge_synth_kinds_original_fills_unresolved_slots() {
        // post_resolve missing a value; original supplies it (since
        // post_result does not claim it).
        let post_resolve = TypeResolutionState::new();
        let original = state_from(&[(ValueId(7), ConcreteType::Signed)]);
        let post_result: HashMap<ValueId, ConcreteType> = HashMap::new();

        let merged = merge_synth_kinds(&original, post_resolve, post_result);
        assert_eq!(merged.get(ValueId(7)), &ConcreteType::Signed);
    }

    #[test]
    fn merge_synth_kinds_original_unknown_does_not_propagate() {
        // Unknown is a placeholder, not an inference.  `iter` skips it
        // so the merged state never sees the entry.
        let post_resolve = TypeResolutionState::new();
        let mut original = TypeResolutionState::new();
        original.set(ValueId(7), ConcreteType::Unknown);
        let post_result: HashMap<ValueId, ConcreteType> = HashMap::new();

        let merged = merge_synth_kinds(&original, post_resolve, post_result);
        assert!(
            !merged.contains(ValueId(7)),
            "Unknown originals must not seed the merged state"
        );
    }

    #[test]
    fn merge_synth_kinds_post_result_overrides_original() {
        // original infers Signed; post_result declares Float.
        // post_result wins (rewriter ground truth).
        let post_resolve = state_from(&[(ValueId(1), ConcreteType::Signed)]);
        let original = state_from(&[(ValueId(1), ConcreteType::Signed)]);
        let post_result = map_from(&[(ValueId(1), ConcreteType::Float)]);

        let merged = merge_synth_kinds(&original, post_resolve, post_result);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::Float);
    }

    #[test]
    fn merge_synth_kinds_original_skipped_when_post_result_claims_value() {
        // Even non-Unknown original is skipped if post_result claims
        // the slot — the rewriter's authoritative declaration takes
        // precedence (mirrors RPython's single-source-of-truth shape).
        let post_resolve = TypeResolutionState::new();
        let original = state_from(&[(ValueId(1), ConcreteType::GcRef)]);
        let post_result = map_from(&[(ValueId(1), ConcreteType::Signed)]);

        let merged = merge_synth_kinds(&original, post_resolve, post_result);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::Signed);
    }

    #[test]
    fn merge_synth_kinds_full_precedence_chain() {
        // Three ValueIds, each contributed by a different source —
        // confirm each source's lane reaches the merged state.
        let post_resolve = state_from(&[(ValueId(1), ConcreteType::Float)]);
        let original = state_from(&[(ValueId(2), ConcreteType::Signed)]);
        let post_result = map_from(&[(ValueId(3), ConcreteType::GcRef)]);

        let merged = merge_synth_kinds(&original, post_resolve, post_result);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::Float);
        assert_eq!(merged.get(ValueId(2)), &ConcreteType::Signed);
        assert_eq!(merged.get(ValueId(3)), &ConcreteType::GcRef);
    }

    #[test]
    fn iter_yields_ascending_value_id_order() {
        // `cutover::compare_real_against_legacy` (cutover.rs:408)
        // returns the first divergence as its diff message; with a
        // HashMap-backed store the "first" entry would be hash-order
        // and the message would jitter across runs.  Confirm we walk
        // slots in ascending ValueId order so the message stays
        // deterministic.
        let state = state_from(&[
            (ValueId(5), ConcreteType::Float),
            (ValueId(2), ConcreteType::Signed),
            (ValueId(9), ConcreteType::GcRef),
        ]);
        let collected: Vec<_> = state.iter().map(|(vid, _)| vid).collect();
        assert_eq!(collected, vec![ValueId(2), ValueId(5), ValueId(9)]);
    }
}
