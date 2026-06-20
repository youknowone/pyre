//! `specialize_legacy_graph` cutover wrapper.
//!
//! Drives the real `RPythonTyper::specialize` against a pyre
//! `model::FunctionGraph` by way of the
//! [`crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace`]
//! adapter, then projects each per-`Variable` `LowLevelType` back to a
//! `ConcreteType` keyed by the legacy `Variable` identity.
//!
//! ## Why this file is in `translator/rtyper/`
//!
//! The legacy algorithms live alongside this file as
//! `translator/rtyper/legacy_{annotator,resolve,pipeline}.rs`.  This
//! cutover module is the bridge between them: the dual-gate
//! comparison and the production type-state path both call into the
//! entry points defined here.  The `flowspace_adapter`
//! sibling continues bridging pyre's surface-DSL
//! `model::FunctionGraph` to the RPython `flowspace::FunctionGraph`
//! shape the rtyper consumes, until pyre's
//! `parse → front → SemanticProgram` chain learns to emit
//! `flowspace::FunctionGraph` directly.
//!
//! ## Scope
//!
//! - Build a `FlowspaceAdapterOutput` via the adapter.
//! - Construct a fresh `RPythonAnnotator`; bypass `build_types`
//!   (pyre's surface DSL has no `HostObject` to feed it) by populating
//!   `annotator.annotated` and `annotator.all_blocks` directly with
//!   the adapter's blocks. The annotation shells are
//!   already attached to each `Variable.annotation`, which is what
//!   `RPythonTyper.bindingrepr` reads.
//! - Construct an `RPythonTyper` and call `specialize(true)` —
//!   `dont_simplify_again=true` because pyre's legacy graph is already
//!   in simplified SSA shape; running the simplify pass would attempt
//!   to call into bookkeeper machinery that requires
//!   `RPythonAnnotator.translator.entry_point_graph`, which we have
//!   not seeded.
//! - Walk the `value_to_var` map and project each `Variable.concretetype`
//!   to `ConcreteType` (Signed/Float/GcRef/Void/Unknown).
//!
//! ## Dual-gate readiness
//!
//! Today this function still fails on graphs containing followups
//! that are not yet ported (`Call`, `FieldRead`, `ArrayRead`, ...).
//! The anchor corpus will surface which
//! followup is the next priority. `Ref`-typed operands now route
//! through `valuetype_to_someshell(Ref) → SomeInstance(classdef=None)`
//! (`jit_codewriter/annotation_state.rs:69`), so the rtyper picks
//! `getinstancerepr(rtyper, None, Gc) → InstanceRepr::new_rootinstance
//! → Ptr(GcStruct(OBJECT))` and the projection collapses to
//! `ConcreteType::GcRef` matching the legacy resolver — the previous
//! `SomeObject` blocker is closed.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::flowspace::argument::Signature;
use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, GraphFunc, Hlvalue, Link, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::jit_codewriter::type_state::ConcreteType;
use crate::model::FunctionGraph as LegacyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::flowspace_adapter::{FlowspaceAdapterOutput, LegacyToTyped};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::pyre_call_registry::{
    FunctionPathKey, PyreCallRegistry, PyreFunctionEntry,
};

/// Project a post-`specialize` `LowLevelType` back to the legacy
/// `ConcreteType` bucket the codewriter consumes (Signed / Float /
/// GcRef / Void).
///
/// `Result`-routing wrapper around [`crate::model::getkind`] for
/// the `specialize_legacy_graph` callback channel.
///
/// `getkind` itself panics on the `NotImplementedError` cases
/// (longlonglong / longfloat / non-pointer aggregates / InteriorPtr
/// on a Variable's concretetype) — the upstream `history.py:62,70
/// raise NotImplementedError` shape.  Pyre's legacy specialize
/// callers want those failures routed through `Result<…, TyperError>`
/// instead so a single unported rtype path doesn't unwind the whole
/// `transform_graph_to_jitcode` driver; this wrapper catches the
/// `getkind: …not supported…` payload only and re-raises everything
/// else (so an assertion or logic bug inside `getkind` is NOT
/// silently rebranded as a missing-rtype error in the dual gate).
pub(crate) fn lowleveltype_to_concrete(ll: &LowLevelType) -> Result<ConcreteType, TyperError> {
    let ll_clone = ll.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::model::getkind(&ll_clone)
    }));
    match result {
        Ok(kind) => Ok(kind),
        Err(payload) => {
            // `crate::model::getkind` raises its NotImplementedError-
            // equivalent via `panic!("getkind: type … not supported …")`.
            // Match that exact prefix + substring; resume_unwind anything
            // else so we don't disguise real bugs as missing rtype ops.
            let msg = payload
                .downcast_ref::<&'static str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            if msg.starts_with("getkind:") && msg.contains("not supported") {
                Err(TyperError::missing_rtype_operation(format!(
                    "lowleveltype_to_concrete: type {ll:?} not supported \
                     (history.py:70 raise NotImplementedError)"
                )))
            } else {
                std::panic::resume_unwind(payload)
            }
        }
    }
}

/// Seed every cached callee PyGraph's blocks into the annotator's
/// `annotated`/`all_blocks` so the rtyper's specialize walk reaches
/// them.
///
// There is no callee-block pre-seed.  The orthodox path
// `pycall -> recursivecall -> addpendingblock` from
// description.py:283-305 / annrpython.py:315-336 is reachable
// through the `simple_call_SomeObject` registration; the
// program-wide `compute_at_fixpoint` loop discovers callees from
// subject call sites without any pre-seed.

/// RAII guard that snapshots every live `Variable.annotation` cell
/// reachable from `iter_variables()` and restores it on `Drop`,
/// isolating the dual-gate baseline's `legacy_annotator::annotate`
/// writes from any subsequent reader on the same graph.  See the
/// `dual_gate_check` doc for the failure mode this prevents.
struct LegacyAnnotationGuard {
    snapshot: Vec<(
        crate::flowspace::model::Variable,
        Option<Rc<crate::annotator::model::SomeValue>>,
    )>,
}

impl LegacyAnnotationGuard {
    fn snapshot(graph: &LegacyGraph) -> Self {
        let snapshot = graph
            .iter_variables()
            .into_iter()
            .map(|var| {
                let ann = var.annotation.borrow().clone();
                (var, ann)
            })
            .collect();
        Self { snapshot }
    }
}

impl Drop for LegacyAnnotationGuard {
    fn drop(&mut self) {
        for (var, ann) in self.snapshot.drain(..) {
            *var.annotation.borrow_mut() = ann;
        }
    }
}

/// Run `specialize_legacy_graph` and diff the real path against the
/// legacy walker's `Variable.concretetype` view.
///
/// Returns `Err(message)` when:
///
/// - the real path errors out (typer error from an unported `OpKind`
///   arm), OR
/// - a legacy-known `Variable` is missing / `Unknown` / different in
///   the real path, OR
/// - the real path produced a definite kind for a `Variable` the legacy
///   resolver did not resolve.
///
/// Returns `Ok(())` only when the legacy `legacy_graph.concretetype`
/// view and the real-path `Variable.concretetype` projection agree on
/// every definite kind.
///
/// Today this entry survives only as a test helper: production
/// callers go through [`dual_gate_check_with_registry`], which dropped
/// per-graph divergence comparison once legacy was narrowed to the
/// Skip arm.  The legacy-baseline diff stays here so
/// anchor tests can keep validating the LL→Concrete projection.
///
/// Both `dual_gate_check` and `dual_gate_check_with_registry` wrap
/// the baseline in a [`LegacyAnnotationGuard`] so the dual-gate
/// comparison is side-effect-free on `legacy_graph.variable.annotation`.
/// Without the guard the baseline's `legacy_annotator::annotate` would
/// publish the wider legacy lift onto the graph's annotation cells
/// (annotate-write contract); a subsequent real-path pass over the same
/// `legacy_graph` would then see the residue at
/// `flowspace_adapter::seed_variable` (which copies
/// `legacy.variable(vid).annotation` onto the fresh flowspace
/// Variable) and trip `flowin`'s `setbinding: new value does not
/// contain old` monotonicity assertion when the wider legacy seed
/// later gets narrowed.  The guard's `Drop` restores the pre-baseline
/// `.annotation` slot contents on every exit path (success, panic,
/// or early `Err` return).
#[cfg(test)]
pub(crate) fn dual_gate_check(legacy_graph: &LegacyGraph) -> Result<(), String> {
    // The real path goes through `RPythonTyper::specialize`, which
    // asserts internal invariants (e.g. `genop`'s "wrong level!"
    // contract that every operand carry `concretetype`). Followups
    // occasionally surface those asserts on graphs whose
    // shape exposes an unported pyre-front idiom — those panics are
    // diagnostic for "next blocker", not crashes the dual-gate
    // should propagate. Catch the unwind so the gate uniformly
    // returns a stringified error.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        specialize_legacy_graph(legacy_graph)
    }));
    let (value_to_var, constants) = match result {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => return Err(format!("real path failed: {e}")),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<unrecognised panic payload>".to_string()
            };
            return Err(format!("real path panicked: {msg}"));
        }
    };

    // Defensive baseline diff against the legacy walker.  Mirrors the
    // `_with_registry` variant (cutover.rs:370-373): runs after the
    // real path's flowin so `Variable.annotation` is
    // not pre-populated with the legacy walker's wider lift before
    // `seed_variable` runs (orthodox `_setbinding` monotonicity).
    // `legacy_resolve::resolve_types` writes `graph.concretetype` from
    // the post-publish `graph.variable.annotation` cells; the
    // comparison loop below reads `graph.concretetype` directly.
    //
    // The annotation guard snapshots every live
    // `Variable.annotation` cell before the
    // baseline and restores the snapshot on `Drop` (end of this
    // function's scope, including early returns and panic unwinds).
    // Without it
    // the wider legacy lift would persist as residue on the graph
    // (`annotate` writes directly to
    // `Variable.annotation`); a subsequent dual-gate pass over the
    // same `legacy_graph` would then trip the orthodox
    // `_setbinding` monotonicity check in flowin.
    let _annotation_guard = LegacyAnnotationGuard::snapshot(legacy_graph);
    let baseline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::legacy_annotator::annotate(legacy_graph);
        super::legacy_resolve::resolve_types(legacy_graph);
    }));
    if let Err(payload) = baseline {
        let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<unrecognised panic payload>".to_string()
        };
        return Err(format!(
            "dual-gate baseline panicked (legacy walker crashed before \
             comparison could run): {msg}"
        ));
    }

    // Diff every legacy-resolved value against the real path. Once this
    // gate is used to prove cutover parity, real-path `Unknown` for a
    // legacy-known value (and the reverse asymmetry — a real kind for a
    // value the legacy walker left Unknown) is a coverage bug, not
    // success.  The legacy walker's kinds are read off each Variable's
    // `concretetype` cell, which `resolve_types` populated through
    // `FunctionGraph::set_concretetype_of_inline`.
    let real_state = project_value_to_var(&value_to_var, &constants);
    let divergences = collect_divergences(&real_state, legacy_graph);

    if divergences.is_empty() {
        Ok(())
    } else {
        Err(divergences.join("; "))
    }
}

/// Outcome of a dual-gate run.  Distinguishes a real-path success
/// (`Match` — production consumes `real_state` authoritatively) from
/// a known-unported feature (`Skip(reason)` — production falls back
/// to the legacy walker for the affected graph).
///
/// `Match` carries the real path's `LegacyToTyped` map (each
/// `Variable.concretetype` cell set by `RPythonTyper::specialize`) so
/// production callers consume it directly.  The real path is the
/// authoritative producer; legacy is the fallback for
/// Skip-classified graphs.  The legacy-baseline diff inside
/// `dual_gate_check_with_registry` still runs while the transition is
/// active; `Match` means the real path succeeded and matched the legacy
/// baseline when that baseline could be produced.  Test-time anchor
/// invariants also run [`dual_gate_check`] for legacy-baseline
/// regression checks against hand-built fixtures.  PyPy
/// `codewriter.py:33` consumes the rtyper-produced graph directly,
/// with no dual-gate equivalent; pyre's `Skip` arm is transitional
/// scaffolding that retires once every category in
/// `is_known_unported`'s table is implemented.
#[derive(Debug)]
pub(crate) enum DualGateOutcome {
    /// Real path completed without panicking and (when the legacy
    /// walker also succeeded as defensive baseline) every legacy-
    /// known `Variable` carried the same `ConcreteType` in the real
    /// path's projection.  Production consumes `real_state`
    /// authoritatively.  Per-`Variable` diff against the legacy
    /// baseline runs whenever the legacy walker can produce a
    /// state — divergence routes the graph through
    /// `Skip("dual-gate divergence: ...")` so the codewriter falls
    /// back to the legacy walker output for the affected graph.  PyPy
    /// `codewriter.py:33` consumes the rtyper-produced graph
    /// directly with no comparison stage; the legacy-baseline diff
    /// is pyre-only scaffolding that retires once the legacy walker
    /// itself retires.
    Match {
        /// `legacy Variable → typed flowspace::Variable`
        /// (`LegacyToTyped = HashMap<Variable, Variable>`) mapping built
        /// by the flowspace adapter.  Each typed Variable carries the
        /// `RPythonTyper`-set `concretetype` inline (`flowspace/
        /// model.py:280`), so codewriter callers copy that lltype onto
        /// the matching legacy Variable via
        /// [`crate::jit_codewriter::type_state::apply_from_flowspace_variables`];
        /// `FunctionGraph::concretetype_of(&v)` then reads the legacy
        /// Variable's `concretetype` cell directly.
        real_value_to_var: LegacyToTyped,
    },
    /// Real path failed on a known-unported feature — the gate
    /// cannot validate this graph yet but the failure is *not* a
    /// ConcreteType divergence.  Categories include:
    ///
    /// - `OpKind::Call::FunctionPath { segments }` not in the
    ///   registry (cross-crate / primitive paths the production
    ///   walker doesn't reach yet).
    /// - `undefined operand` from cross-block locals
    ///   threading not yet covered.
    /// - `unimplemented operation` from a not-yet-ported rtyper op
    ///   (e.g. `direct_call` for graphs the rpbc port doesn't
    ///   cover).
    /// - `checkgraph` / flowspace consistency panics from the
    ///   adapter's downstream stages.
    Skip(String),
}

/// Production dual-gate — registry-aware entry.
///
/// Drives the real path via [`specialize_legacy_graph_with_registry_returning_value_to_var`]
/// against a pre-populated `PyreCallRegistry` so graphs with
/// `OpKind::Call::FunctionPath` callsites resolve through the upstream
/// `Constant(<function>) -> getdesc -> FunctionDesc` chain
/// (`bookkeeper.py:353-409`).  Production callers
/// (codewriter.rs `transform_graph_to_jitcode`) build the registry
/// once per `CallControl` (program-wide) and reuse across every
/// dual-gated graph.
///
/// Returns:
///
/// - `Ok(DualGateOutcome::Match)` when the real path succeeds.  The
///   real path's `LegacyToTyped` map (with each
///   `Variable.concretetype` cell populated) is the authoritative
///   source for production consumption.
/// - `Ok(DualGateOutcome::Skip(reason))` when the real path failed
///   on a known-unported feature (registry miss / adapter
///   invariant break / unimplemented rtyper op).  Callers fall back
///   to legacy walker output for the affected graph.
/// - `Err(message)` when the real path failed for an unrecognised
///   reason — surfaces upstream so the codewriter can re-classify
///   the panic if the message later turns out to be in the
///   known-unported table.
///
/// Defensive per-`Variable` diff against the legacy walker baseline
/// runs whenever both paths succeed.  Divergence is reported as
/// `Skip("dual-gate divergence: ...")` so the codewriter falls back
/// to the legacy walker output.  Anchor tests use [`dual_gate_check`]
/// directly for hand-built fixtures.
pub(crate) fn dual_gate_check_with_registry(
    legacy_graph: &LegacyGraph,
    call_registry: &PyreCallRegistry,
    lift_sources: &crate::jit_codewriter::call::GraphStore,
) -> Result<DualGateOutcome, String> {
    // Same panic-catch contract as `dual_gate_check` — the rtyper's
    // internal `genop`/`level` asserts surface as diagnostic panics
    // for unported pyre-front idioms; the gate uniformly returns a
    // stringified error so the env-flag wrapper can decide whether
    // to panic, log, or skip.
    //
    // Snapshot the session's `fixed_graphs` keys so the failure arms
    // below can identify the shared callee graphs this subject fixed
    // (and partially LL-rewrote) before it failed — see
    // `unpoison_failed_subject_callees`.
    let fixed_at_entry: HashSet<crate::flowspace::model::GraphKey> = call_registry
        .session_if_started()
        .map(|(ann, _)| ann.fixed_graphs.borrow().keys().cloned().collect())
        .unwrap_or_default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        specialize_legacy_graph_with_registry_returning_value_to_var(legacy_graph, call_registry)
    }));
    let (real_value_to_var, real_constants) = match result {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => {
            unpoison_failed_subject_callees(call_registry, &fixed_at_entry, lift_sources);
            let msg = format!("{e}");
            if is_known_unported(&msg) {
                return Ok(DualGateOutcome::Skip(msg));
            }
            return Err(format!("real path failed: {msg}"));
        }
        Err(payload) => {
            unpoison_failed_subject_callees(call_registry, &fixed_at_entry, lift_sources);
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<unrecognised panic payload>".to_string()
            };
            if is_known_unported(&msg) {
                return Ok(DualGateOutcome::Skip(msg));
            }
            return Err(format!("real path panicked: {msg}"));
        }
    };
    let _annotation_guard = LegacyAnnotationGuard::snapshot(legacy_graph);
    let baseline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::legacy_annotator::annotate(legacy_graph);
        super::legacy_resolve::resolve_types(legacy_graph);
    }));
    if let Err(payload) = baseline {
        let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<unrecognised panic payload>".to_string()
        };
        return Err(format!(
            "dual-gate baseline panicked (legacy walker crashed before \
             comparison could run — comparison cannot validate the real \
             path's result): {msg}"
        ));
    }
    if let Some(divergence) =
        compare_real_against_legacy(&real_value_to_var, &real_constants, legacy_graph)
    {
        return Ok(DualGateOutcome::Skip(format!(
            "dual-gate divergence: {divergence}"
        )));
    }
    Ok(DualGateOutcome::Match { real_value_to_var })
}

/// Repair shared callee state poisoned by a failed subject scope.
///
/// `specialize_more_blocks` mutates callee graphs in place: it
/// registers each graph in `annotator.fixed_graphs` at its first
/// specialized block (rtyper.py:268) and rewrites block operations to
/// LL form. When the subject then fails, `AddedBlocksGuard` evicts the
/// scope's blocks and clears their annotations — but the callee's
/// `FunctionDesc.cache` (and any calltable row built during the scope)
/// still holds the fixed + LL-rewritten + annotation-cleared graph.
/// Every later subject calling the same callee then dies at
/// `addpendingblock`'s fixed-graph safety check (annrpython.py:181
/// reads `arg.annotation`), or at `flowin`'s "unimplemented operation"
/// on the LL ops.
///
/// Upstream has no counterpart: RPython's `specialize()` runs once per
/// Translator and any failure is fatal. Pyre's per-subject dual-gate
/// continues past failed subjects, so a failed scope must leave shared
/// callee state as if the subject never ran. Each graph newly fixed
/// during the scope is swapped for a fresh re-lift from its
/// `GraphStore` source, and the rtyper/annotator records keyed to the
/// stale graph are dropped:
///
/// - `annotator.fixed_graphs` row (graph identity)
/// - `rtyper.already_seen` rows for the stale graph's blocks
/// - `translator.graphs` entry (appended on `cachedgraph` hit)
/// - `FunctionDesc.cache` values (patched to the fresh `PyGraph`)
/// - calltable rows holding the same `Rc<PyGraph>` (patched in place
///   so `total_calltable_size` and `get_concrete_calltable`'s size
///   invariant hold; the family's cached concrete calltable is
///   dropped so it rebuilds against the fresh graph)
fn unpoison_failed_subject_callees(
    call_registry: &PyreCallRegistry,
    fixed_at_entry: &HashSet<crate::flowspace::model::GraphKey>,
    lift_sources: &crate::jit_codewriter::call::GraphStore,
) {
    let Some((annotator, rtyper)) = call_registry.session_if_started() else {
        return;
    };
    let newly_fixed: Vec<(
        crate::flowspace::model::GraphKey,
        crate::flowspace::model::GraphRef,
    )> = annotator
        .fixed_graphs
        .borrow()
        .iter()
        .filter(|(k, _)| !fixed_at_entry.contains(*k))
        .map(|(k, g)| (k.clone(), g.clone()))
        .collect();
    for (gkey, stale) in newly_fixed {
        annotator.fixed_graphs.borrow_mut().shift_remove(&gkey);
        for block in stale.borrow().iterblocks() {
            rtyper
                .already_seen
                .borrow_mut()
                .remove(&crate::flowspace::model::BlockKey::of(&block));
        }
        annotator
            .translator
            .graphs
            .borrow_mut()
            .retain(|g| !Rc::ptr_eq(g, &stale));
        let Some((key, entry)) = call_registry.find_entry_with_cached_graph(&stale) else {
            // The subject graph itself (lifted through the adapter, not
            // a registry callee) — the shared-map cleanup above is all
            // it needs.
            continue;
        };
        let fresh = call_registry
            .keys_for_entry(&key)
            .iter()
            .find_map(|k| {
                lift_sources.get(&crate::parse::CallPath::from_segments(
                    k.segments().iter().cloned(),
                ))
            })
            .and_then(|source| {
                lift_callee_to_pygraph(source, signature_for_graph(source), call_registry).ok()
            });
        match fresh {
            Some(fresh) => {
                let fd = entry.function_desc.borrow();
                for pg in fd.cache.borrow_mut().values_mut() {
                    if Rc::ptr_eq(&pg.graph, &stale) {
                        *pg = fresh.clone();
                    }
                }
                if let Ok(family) = fd.base.getcallfamily() {
                    for table in family.borrow_mut().calltables.values_mut() {
                        for row in table.iter_mut() {
                            for pg in row.values_mut() {
                                if Rc::ptr_eq(&pg.graph, &stale) {
                                    *pg = fresh.clone();
                                }
                            }
                        }
                    }
                    rtyper
                        .concrete_calltables
                        .borrow_mut()
                        .remove(&(Rc::as_ptr(&family) as usize));
                }
            }
            None => {
                // No re-lift source — drop the poisoned cache rows and
                // record the condition so later callers surface a lift
                // error (Skip-classified) instead of the fixed-graph
                // safety panic.
                entry
                    .function_desc
                    .borrow()
                    .cache
                    .borrow_mut()
                    .retain(|_, pg| !Rc::ptr_eq(&pg.graph, &stale));
                entry.record_lift_error(
                    "callee graph poisoned by a failed subject scope; no re-lift source"
                        .to_string(),
                );
            }
        }
    }
}

/// Project the real path's resolved kind per legacy `Variable`, keyed
/// by Variable identity.  `value_to_var` already pairs each legacy
/// Variable with its rtyper-typed twin, and `constant_concretetypes`
/// carries the ground-truth lltype for const-define results, so no slot
/// projection is needed.  Errors from unsupported lltypes are treated as
/// missing entries — `specialize_legacy_graph` already propagates such a
/// failure before the dual-gate runs, so reaching here implies every
/// lltype is projectable.
///
/// Used by both [`dual_gate_check_with_registry`] (via
/// [`compare_real_against_legacy`]) and the [`dual_gate_check`]
/// anchor-test helper to diff the real path against the legacy walker.
fn project_value_to_var(
    value_to_var: &LegacyToTyped,
    constant_concretetypes: &HashMap<Variable, LowLevelType>,
) -> HashMap<Variable, ConcreteType> {
    let mut real_state: HashMap<Variable, ConcreteType> = HashMap::new();
    for (legacy_var, typed_var) in value_to_var {
        if let Some(lltype) = typed_var.concretetype().as_ref() {
            if let Ok(kind) = lowleveltype_to_concrete(lltype) {
                real_state.insert(legacy_var.clone(), kind);
            }
        }
    }
    // `Constant.concretetype` is the ground truth for constant operands;
    // the const-define result Variable overrides the value_to_var entry.
    for (legacy_var, lltype) in constant_concretetypes {
        if let Ok(kind) = lowleveltype_to_concrete(lltype) {
            real_state.insert(legacy_var.clone(), kind);
        }
    }
    real_state
}

/// Variables defined (inputarg or op result) in the reachable block
/// closure (DFS from `startblock` over `Block.exits`).  The real path
/// converts exactly this closure (flowspace `iterblocks()` parity —
/// an unreachable block cannot exist upstream), while legacy
/// `resolve_types` types every lowered block — including pruned
/// `SwitchInt` abort-default arms and the orphan `set_raise`
/// `[etype, evalue]` Link.args they carry.  A variable defined only
/// outside the closure is legacy-only by construction; diffing it
/// against the real path is a guaranteed false `real=Unknown`.
fn reachable_defined_vars(graph: &LegacyGraph) -> std::collections::HashSet<Variable> {
    // Id-keyed lookup, not the dense `blocks[id.0]` projection — block ids
    // need not be index-aligned (`flowspace_adapter::reachable_block_ids`).
    let by_id: std::collections::HashMap<crate::model::BlockId, &crate::model::Block> =
        graph.blocks.iter().map(|b| (b.id, b)).collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![graph.startblock];
    let mut vars = std::collections::HashSet::new();
    while let Some(bid) = stack.pop() {
        if !seen.insert(bid) {
            continue;
        }
        let Some(block) = by_id.get(&bid) else {
            continue;
        };
        for ia in &block.inputargs {
            vars.insert(ia.clone());
        }
        for op in &block.operations {
            if let Some(r) = &op.result {
                vars.insert(r.clone());
            }
        }
        for link in &block.exits {
            stack.push(link.target);
        }
    }
    vars
}

/// Diff the real path's per-Variable kinds against the legacy walker's
/// kinds committed onto the same legacy Variables.  Walks
/// [`FunctionGraph::iter_variables`] for a deterministic, slot-free
/// traversal restricted to [`reachable_defined_vars`]; `value_to_var`
/// aligns the two paths by Variable identity, so the legacy kind is
/// read straight off each Variable's `concretetype` cell.  Returns
/// every divergence; the caller decides first-only versus all.
fn collect_divergences(
    real_state: &HashMap<Variable, ConcreteType>,
    legacy_graph: &LegacyGraph,
) -> Vec<String> {
    let reachable_vars = reachable_defined_vars(legacy_graph);
    let mut divergences = Vec::new();
    for (pos, var) in legacy_graph.iter_variables().iter().enumerate() {
        // `iterblocks()` parity: the real path annotates only the
        // startblock-reachable closure, so a value defined solely in an
        // unreachable legacy block has no real-path kind.  Comparing it
        // reads real=Unknown and reports a false divergence; skip it
        // exactly as the real path's `remove_dead_blocks` prune does.
        if !reachable_vars.contains(var) {
            continue;
        }
        let legacy_kind = LegacyGraph::concretetype_of(var);
        let real_present = real_state.get(var).copied();
        let real_kind = real_present.unwrap_or(ConcreteType::Unknown);
        // A kind the legacy walker resolved must match the real path;
        // conversely the real path must not resolve a kind for a value
        // the legacy walker left Unknown.
        let diverges = if legacy_kind != ConcreteType::Unknown {
            real_kind != legacy_kind
        } else {
            real_present.is_some()
        };
        if diverges {
            let label = legacy_graph
                .value_name_for(var)
                .unwrap_or_else(|| format!("v{pos}"));
            divergences.push(format!(
                "{label}: legacy={legacy_kind:?}, real={real_kind:?}"
            ));
        }
    }
    divergences
}

fn compare_real_against_legacy(
    value_to_var: &LegacyToTyped,
    constants: &HashMap<Variable, LowLevelType>,
    legacy_graph: &LegacyGraph,
) -> Option<String> {
    let real_state = project_value_to_var(value_to_var, constants);
    collect_divergences(&real_state, legacy_graph)
        .into_iter()
        .next()
}

/// Return true when `msg` matches one of the known-unported
/// categories the dual-gate currently treats as `Skip` rather than
/// `Err`.  Each category is its own future port; the predicate is
/// the central place to enumerate what's still unimplemented.
///
/// As ports land, entries here move out (the port closes the
/// category) and the surface of "unknown failure" shrinks.  When
/// the predicate matches nothing, the gate is at full coverage.
///
/// Each remaining entry is blocked on a specific unimplemented
/// feature; retiring an entry early is unsafe until that feature
/// lands:
///
/// | Substring                                  | Unimplemented feature                                                        |
/// |--------------------------------------------|------------------------------------------------------------------------------|
/// | `not registered in PyreCallRegistry`       | Extern Rust helper registry walker.                                          |
/// | `translate_op: undefined operand`          | Adapter producer correctness.                                                |
/// | `unimplemented operation`                  | Per-opname rtyper handlers.                                                  |
/// | `variable used before definition`          | Cross-block locals threading.                                                |
/// | `MissingRTypeAttribute`                    | Typed-Ref → SomeInstance(ClassDef).                                          |
/// | `KeyError: no binding for arg`             | Rtyper bindingrepr gap on annotation-less Variable (e.g. `current_sp`).      |
/// | `compute_at_fixpoint failed`               | PBC dispatch / call-family coverage (per-call).                              |
/// | `post-rtyper jtransform variant`           | Per-variant emit-site retracing (rpbc / rclass / front-end).  Includes `OpKind::Abort` (pyre-only marker; retire every `Expr::ForLoop` / `stop_unsupported` / `continue_with_unknown` emit-site at the front-end). |
/// | `adapter cross-block body Input`           | Final emit-site retirement.                                                  |
/// | `noneify() not supported`                  | Front-end typed null-pointer lowering (`Option<*T>` / null → `SomePtr`, not `SomeNone`); the raise itself is parity-correct. |
///
/// PyPy `bookkeeper.py:108-127` propagates fixpoint failures uncaught;
/// pyre's dual-gate Skip defers exactly the categories enumerated
/// above.  Once every category is implemented, every match returns
/// `false`, the legacy walker fallback at
/// `codewriter.rs::dual_gate_type_state` becomes dead code, and the
/// predicate retires entirely.
pub(crate) fn is_known_unported(msg: &str) -> bool {
    msg.contains("not registered in PyreCallRegistry")
        || msg.contains("translate_op: undefined operand")
        || msg.contains("unimplemented operation")
        || msg.contains("variable ")
            && msg.contains(" used before definition")
        // `normalize_unary_op_name: pyre UnaryOp` and
        // `normalize_binop_name: pyre BinOp` are not Skip-classified:
        // the `not` / `deref` / `same_as` / `invert` surfaces are
        // desugared by the MIR front-end (rustc lowers `!`/`*` and the
        // `&&`/`||` short-circuits into MIR branches/calls before
        // lowering), and the bitwise `and` / `or` labels normalize to
        // the flowspace `and_` / `or_` registrations.  Synthetic
        // graphs that inject unported ops (anchor tests in
        // `cutover.rs::tests::anchor_unary_*_surfaces_*`) call
        // `specialize_legacy_graph` directly and never reach
        // `is_known_unported`, so the absence of these substring
        // matches does not affect them.  Any production reach surfaces
        // as a dual-gate divergence panic — the parity-correct outcome.
        // Field / method dispatch on a `SomeInstance(classdef=None)`
        // — pyre's `Ref` ValueType currently lifts to a classdef-less
        // SomeInstance, so `find_attribute`
        // (`rclass.py:556+find_attribute_or_None`) cannot route the
        // dispatch.  `InstanceRepr::rtype_getattr`
        // (`rclass.py:838-857`) routes through `getclsfield`, which
        // surfaces the upstream-orthodox `MissingRTypeAttribute(attr)`
        // when find_attribute returns None.  The `"no method ... on
        // Instance("` substring (`rmodel.rs:828` default
        // `rtype_getattr` find_method failure path) is not classified
        // here: every SomeInstance dispatch goes through the
        // InstanceRepr override, so the default fn never fires for
        // `Instance(...)`-shaped operands.  The MissingRTypeAttribute
        // entry stays until typed-Ref ClassDef projection lands and
        // field/method dispatch starts succeeding.
        || msg.contains("MissingRTypeAttribute")
        // Variable's `.annotation` slot empty at `bindingrepr`
        // lookup time — `ValueType::Unknown` has no
        // annotation-stage shell and `valuetype_to_someshell` returns
        // `None` for it (intentionally fail-loud for "annotation gap"
        // so producers know which slot missed seeding, see
        // `seed_variable` at `flowspace_adapter.rs:96-115`).  Closing
        // the gap means tightening the front-end / annotator
        // producers so every slot has a non-`Unknown` annotation; the
        // gate skips until then.  The mergeinputargs-side flavour of
        // this gap (raising subjects' `follow_raise_link` reaching an
        // unannotated exceptblock seed) was resolved 2026-05-31 via
        // `setbinding(Impossible)` pre-registration in
        // `specialize_legacy_graph_with_registry_returning_value_to_var`;
        // the remaining producer is the rtyper-side `bindingrepr`
        // gap on synthesized `current_sp`-like graphs.
        || msg.contains("KeyError: no binding for arg")
        // `cast_ptr_to_int` reached the typer with a non-`Ptr`
        // operand — the front-end could not lower a real pointer for
        // the source of an `expr as i64` cast.  The canonical hitter
        // is `stack_check::current_sp` (`&probe as *const usize as
        // usize`): taking the address of a stack local is a
        // target-specific raw-SP read that `front::mir` lowers with
        // the local's *value* (`Int(0)`) rather than its address, so
        // `rtype_cast_ptr_to_int`'s late `InstanceRepr→PtrRepr` swap
        // fallback finds `Signed` where it needs `Ptr(...)`.  The
        // legacy walker handles these graphs; skip until the
        // address-of-local lowering lands a typed pointer operand.
        || msg.contains("rtype_cast_ptr_to_int: operand concretetype must be Ptr")
        // Unported per-annotation Repr families.  `rmodel.rs`
        // `rtyper_makerepr` fail-louds with this message shape for the
        // SomeValue kinds whose upstream Repr port has not landed
        // (rlist.py ListRepr, rdict.py DictRepr, rrange.py iterator
        // reprs, rbytearray.py, robject.py, rproperty.py).  Reached
        // once a graph annotates a list/dict-typed value past the
        // exc-edge and classdef walls; the legacy walker keeps
        // handling these graphs until each Repr is ported.
        || msg.contains("rtyper_makerepr — port")
        // TODO(annotator-fixpoint-fail-loud) — STRICT-PARITY REGRESSION
        // vs main / PyPy.  `bookkeeper.py:108-127` propagates fixpoint
        // exceptions uncaught and `annrpython.py:643` lets
        // `AnnotatorError` reach the caller; absorbing the four
        // patterns below is a deviation that hides real
        // annotator/rtyper parity gaps as "known unported".  Direct
        // removal breaks `pyre-jit-trace` `build.rs` at every reachable
        // hitter
        // (`make_green_key`, `Frame::load_fast`, `PyFrame::locals_w_mut`,
        // `<default methods of IterOpcodeHandler>::record_for_iter_guard`,
        // `pyjitpl_step::Cannot find attribute`); production cannot
        // compile until each underlying real-path gap is closed
        // (classdef-less SomeInstance dispatch + `PyreCallRegistry::
        // ensure_session` coverage — every analyser body routes through
        // per-touch `arg_at` rather than an eager-prefix concrete
        // walk).  Until each gap-hitter is
        // ported, this Skip stays as documented divergence and the
        // codewriter falls back to the legacy walker for production.
        || msg.contains("compute_at_fixpoint failed")
        || msg.contains("complete_pending_blocks failed")
        || msg.contains("Cannot find attribute ")
        || msg.contains("AnnotatorError:")
        // `rtyper.py:810-823 convertvar` — no conversion path between
        // the two reprs.  The production hitter is the generic-ADT
        // payload attribute (`core.result.Result.Ok.__pos_0` etc.):
        // pyre collapses every `Result<T, E>` instantiation onto ONE
        // classdef, so the attribute's merged annotation unions
        // unrelated payload classes (unit-tuple placeholder `Adt`,
        // `core.option.Option.None`, …) and `pairtype(InstanceRepr,
        // InstanceRepr).convert_from_to` (rclass.py:1035-1055)
        // correctly finds no common base.  Upstream never faces this
        // shape — RPython has no generics, so each class attribute
        // carries a single annotated type.  Skip until
        // per-instantiation classdef specialization lands.
        || msg.contains("don't know how to convert from")
        // Annotation-stage flavour of the convertvar entry above
        // (`annrpython.py:432 mergeinputargs` → `UnionError`): two
        // `SomeInstance`s with no common base meet at a block merge
        // because pyre collapses every generic-ADT instantiation onto
        // one classdef, so unrelated payload classes union at the phi.
        // Upstream propagates the UnionError uncaught; the dual gate
        // skips to the legacy walker until per-instantiation classdef
        // specialization lands.
        || msg.contains("cannot unify instances with no common base class")
        // `model.rs:3061` subset-gap marker — the union pair has no
        // ported `pair(s1, s2).union()` handler.  Upstream degenerates
        // such pairs to `SomeObject` (`annmodel` pair(SomeObject,
        // SomeObject).union, binaryop.py:64-72) instead of erroring,
        // so any hit here is by construction an unported handler, not
        // a divergence.  Skip to the legacy walker until the pair is
        // ported.
        || msg.contains("no upstream pair(s1, s2).union() handler in current subset")
        // `InstanceRepr.getfield` walking the `rbase` chain before the
        // repr's deferred `setup()` ran (upstream drains pending
        // setups via `call_all_setups`, rtyper.py:198, after every
        // specialized block; pyre's per-subject dual-gate can reach a
        // getfield on a freshly minted inner repr first).  Skip until
        // the setup-ordering port lands.
        || msg.contains("rbase missing")
        // `rpbc` calltable row lookup miss at rtype time — the
        // method-PBC came from the struct-root class-dict seeding
        // (`seed_struct_root_method_members`), whose `simple_call`
        // sites do not yet register call shapes into the CallFamily
        // the way `bookkeeper.pbc_call` does for ordinary descs
        // (pbc_call → getcallfamily row, bookkeeper.py:553-571).
        // Skip until the seeded-method family registration lands.
        || msg.contains("calltable row not found in CallFamily")
        // A non-instance constant (e.g. a string literal) reaching
        // `InstanceRepr.convert_const` — the receiver field was typed
        // by the untyped FORCE_ATTRIBUTES shell (classdef-less
        // SomeInstance) because no registry-row projection covered
        // it, so a String-valued field rtypes as an instance (hitter:
        // PyError.msg via the type_error raise stubs).  Skip until
        // the typed-Ref field projection covers the remaining rows.
        || msg.contains("InstanceRepr.convert_const: expected HostObject or None")
        // `rmodel.py:311 rtype_is_` — an `is` (pointer-identity)
        // comparison where one side is not a pointer repr.  The
        // production hitter is `py_type_check`'s `(*obj).ob_type ==
        // tp` chain when `tp` flowed from a `&STATIC` host address:
        // `HostStaticAddrs` annotates the static's address as a
        // constant `SomeInteger`, so the comparison pairs
        // `InstanceRepr` with `IntegerRepr`.  Upstream never faces
        // this — a prebuilt instance constant stays `SomeInstance`.
        // Skip until host static addresses annotate as typed
        // instance pointers.
        || msg.contains("is of instances of the non-pointers")
        // `dyn Trait` dispatch still enters the real-rtyper flowspace
        // adapter as pyre's pre-rtyper `CallTarget::Indirect` shape in
        // some registry-prefill paths.  The production codewriter's
        // jtransform path already runs `rpbc::lower_indirect_calls`
        // before consuming these graphs; the real-rtyper cutover path
        // cannot yet express the matching rclass/rpbc rewrite before
        // adapter input without leaking post-rtyper `VtableMethodPtr` /
        // `IndirectCall` ops into flowspace.  Treat this as a
        // known-unported real-path gap so the dual gate falls back to
        // the legacy type walker instead of panicking during registry
        // population.
        || (msg.contains("Call with CallTarget::Indirect") && msg.contains("rclass"))
        // `annrpython.py:432 mergeinputargs` — an inputarg Variable
        // has no annotation. Happens when cross-block locals
        // threading misses a name in the predecessor link; the
        // annotator cannot merge `None` annotations.
        || msg.contains("inputarg lacks annotation")
        // `rtyper.py:815 convertvar` TyperError — no pairtype
        // convert_from_to edge between the two reprs.  The live hitter
        // is generic-enum payload conflation: the front registers ONE
        // flat class per enum (`Result.Ok`) with first-writer-wins
        // field rows, so two monomorphic instantiations (`Result<Tuple,
        // _>` vs `Result<Option<_>, _>`) share one `__pos_0` attr and
        // the second write needs an unrelated-classdef conversion
        // (`Option.None` → `Tuple`) that
        // `pair_instance_instance_convert_from_to` correctly answers
        // NotImplemented for (rclass.py:1054-1055).  Skip until
        // per-instantiation variant classes (generic payload
        // monomorphization) land.
        || msg.contains("don't know how to convert from")
        // Unported `rtyper_makerepr` arms (`rmodel.rs:2895-2965`
        // SomeList / SomeDict / SomeIterator / SomeByteArray /
        // SomeObject) surface bare `MissingRTypeOperation` messages of
        // the form "<Some*>.rtyper_makerepr — port rpython/rtyper/…"
        // WITHOUT the trait helper's "unimplemented operation:" prefix
        // (`rmodel.rs:817-822`), so the substring above does not
        // absorb them.  Container reprs (rlist.py ListRepr, rdict.py
        // DictRepr, rrange.py iterator reprs, …) are not ported yet;
        // the legacy walker handles such graphs until each repr lands.
        || msg.contains("rtyper_makerepr — port rpython/rtyper/")
        // `ClassRepr` / `InstanceRepr` accessors that require a
        // completed `_setup_repr` (`rbase` populated, rclass.py:273-274)
        // reached through a path that never ran `Repr::setup()` on the
        // repr.  Upstream's `rtyper.getrepr` always queues `setup()`
        // before handing the repr out; pyre's host-struct-seeded
        // receivers (`*mut PyObject` params with a registry classdef)
        // can reach class-field reads through reprs minted outside
        // that queue.  Setup-ordering port gap — fall back to the
        // legacy walker until the getrepr/setup chain covers these
        // roots.
        || msg.contains("rbase missing — call setup() first")
        // `BrokenReprTyperError` (`rmodel.py:42-44`, ported at
        // `rmodel.rs:449-456`): a Repr whose `setup()` failed earlier
        // is re-requested and refuses to half-initialize.  In the
        // per-graph census a Skipped subject can leave a shared
        // session `ClassRepr` in the BROKEN state; the next graph
        // touching the same repr then surfaces this message.  It is a
        // secondary echo of the root failure (itself Skip-classified
        // above), not an independent cause — absorb it so one broken
        // repr does not fatal every later graph in the session.
        || msg.contains("cannot setup already failed Repr")
        // `AnnotatorError: immutablevalue(HostObject` is not
        // Skip-classified for `SyntheticTransparentCtor` (Ok/Err/Some/
        // None).  The adapter emits `HostObject::new_class(name, [])`
        // (`flowspace_adapter.rs:821 SyntheticTransparentCtor` arm),
        // routing through the existing `is_class()` arm in
        // [`crate::annotator::bookkeeper::Bookkeeper::immutablevalue_hostobject`]
        // (`bookkeeper.py:315-316` parity).  No production emit-site of
        // `HostObject::new_opaque` remains in `front/`/`translator/rtyper/`,
        // so any future `immutablevalue(HostObject` failure is a real
        // parity bug — let it surface as a dual-gate divergence panic.
        // TODO(post-rtyper-jtransform-variant-leak): retire this Skip
        // entry per upstream parity — `jit/codewriter/jtransform.py`
        // raises straight through on unexpected opnames.  The most
        // frequent reach is `OpKind::Abort` emitted by the surface-Rust
        // DSL front-end's `lower_expr` `stop_unsupported` /
        // `continue_with_unknown` helpers
        // when the surface DSL hits an unsupported expression — pyre
        // source like `execute_opcode_step` / `eval_loop_jit` carry such
        // placeholders.  Retiring this entry needs each Abort
        // emit-site at the front-end retired (closure
        // body, complex match arms, unsupported literals), then
        // every other post-rtyper variant (IndirectCall / ResidualCall
        // / Vable* / JitMergePoint / LoopHeader / InlineCall /
        // RecursiveCall) emitted from `rpbc.rs` / `rclass.rs` ahead
        // of the rtyper.  Until then the Skip absorbs the placeholder
        // leakage; without it `tests::test_codegen_output` /
        // `tests::test_recognition_report` and the analyse-pyre-source
        // pipeline panic at the first reachable Abort.
        || msg.contains("post-rtyper jtransform variant")
        // Cross-block body `Input` whose `name` was not threaded
        // through `Link.args` / target `inputargs` by the predecessor.
        // RPython flowspace has no body-`Input` op — every cross-block
        // local reference goes via `flowcontext.py:872-884 LOAD_FAST`
        // (which writes into `self.locals_w`) and the target block's
        // pre-allocated `inputargs[]`.  Pyre's body-`Input` emission
        // (the `OpKind::Input` ops `front::mir` pushes into each block)
        // is itself a TODO; when
        // cross-block locals threading misses a shape, the adapter
        // fails loud with this message instead of silently
        // fabricating a fresh Variable (which would hide an
        // SSA / alias-shape divergence from PyPy's flowspace).  Skip
        // until either cross-block locals threading covers every
        // shape, or the front-end's
        // body-`Input` emission is replaced by a Link.args / inputargs
        // threading pass that mirrors RPython.
        || msg.contains("adapter cross-block body Input")
        // `SomeValue::noneify()` reached a `SomePtr` operand — a block
        // merge unioned a `_ptr` with `NoneType`.  The raise itself is
        // parity-correct: RPython's default `noneify` raises `UnionError`
        // (model.py:121-122) and `SomePtr` defines no override, while
        // `pair(SomePtr, SomeObject).union` raises too
        // (llannotation.py:119-120) — so this Skip catches a *correct*
        // raise, it does not paper over a wrong annotation rule.  The
        // only divergence is upstream of the merge: RPython spells a null
        // pointer as `lltype.nullptr(T)`, which annotates as a
        // (null-valued) `SomePtr`, so `pair(SomePtr, SomePtr).union`
        // (llannotation.py:94-98) keeps it in the pointer lattice and
        // `noneify` is never reached.  Pyre's `front::mir` still lowers
        // some Rust null/`Option<*T>` shapes to a `SomeNone` rather than
        // a typed null `SomePtr` (the same null-pointer-typing gap
        // tracked by the `LLAddress(Null)` exceptblock work), so the
        // merge surfaces `_ptr ∪ NoneType`.  CONVERGENCE: lower a typed
        // null pointer at the `front::mir` producer; removing this Skip
        // without that fix only hard-breaks on the parity-correct raise.
        // Until then the legacy walker keeps handling the graphs.
        || msg.contains("noneify() not supported")
    // There is no `normalize_unary_op_name: pyre UnaryOp` Skip entry:
    // the 13 typed numeric / ptr / Unsigned casts route through
    // `simple_call(<host_callable>, v)` — reaching
    // `Repr.rtype_int/float/bool` or `BUILTIN_TYPER[lltype.cast_*]`
    // directly.  Only the `same_as` identity / source-type-unknown
    // fallback remains on the `OpKind::UnaryOp` route, dispatched by
    // `RPythonTyper::translate_operation` to `rbuiltin::rtype_same_as`
    // (verbatim port of `rtyper.py:478-481`).  `normalize_unary_op_name`
    // accepts `same_as` straight through; any residual fail-loud
    // surfaces as a dual-gate divergence panic — the parity-correct
    // outcome.
}

/// Populate a `PyreCallRegistry` from a
/// `HashMap<CallPath, FunctionGraph>` (the shape pyre's
/// `CallControl::function_graphs()` returns).
///
/// Mirrors [`populate_call_registry_from_program`] but keyed off
/// `CallPath` (parse layer's path type, used by `transform_graph_to_jitcode`'s
/// callcontrol map) instead of `SemanticProgram.functions`.  Every
/// callee referenced from any `OpKind::Call::FunctionPath` callsite
/// in the program lives in this map under the same path the front
/// end emitted when it canonicalised each call target,
/// so registry lookups during the lift step always hit.
///
/// Two-pass shape (same as the SemanticProgram walker — see that
/// helper's docstring for the recursivecall parity argument):
///
/// 1. `get_or_register` every entry's signature so HostObjects exist
///    before any callee body lifts.
/// 2. `lift_callee_to_pygraph` + `prefill_default_cache` per entry
///    so `cachedgraph` (`description.rs:1037-1039`) hits at the
///    rtyper's `direct_call`.
pub(crate) fn populate_call_registry_from_call_graphs(
    function_graphs: &crate::jit_codewriter::call::GraphStore,
    registry: &PyreCallRegistry,
) -> Result<(), TyperError> {
    // Dedupe by canonical path — RPython `Bookkeeper.getdesc(pyobj)`
    // (`bookkeeper.py:353-409`) returns the *same* FunctionDesc for
    // any reference to the same callable, keyed by `Constant(pyobj)`
    // identity.  Pyre's `function_graphs: HashMap<CallPath, FunctionGraph>`
    // can carry the same callable under multiple keys (`lib.rs`
    // registers free functions under both their canonical path AND a
    // `crate::`-prefixed alias for callsite matching); without dedupe,
    // each path key would create a separate HostObject / FunctionDesc
    // pair, violating upstream's "create once, share thereafter"
    // contract.
    //
    // The dedupe key strips a leading crate-prefix segment so the
    // five alias-explosion shapes registered by `lib.rs:438-490` for
    // each free function (`[mod, foo]`, `[crate, mod, foo]`,
    // `[pyre_interpreter, mod, foo]`, `[pyre_object, mod, foo]`,
    // `[pyre_jit, mod, foo]`) collapse to a single canonical entry
    // `[mod, foo]`.  Stripping only `crate` (the previous heuristic)
    // left the three external crate-name aliases as distinct canonical
    // entries with distinct `Rc<PyreFunctionEntry>` HostObjects, and
    // `lookup_with_leaf_match`'s multi-match convergence check
    // (`all_same` on `host_object` Arc identity) then failed —
    // callsites like `["crate", "w_str_new"]` (a `use crate::w_str_new`
    // bare callsite from another module) found four matches with four
    // distinct hosts and rejected the cluster.  Stripping all four
    // well-known prefixes preserves the rule that genuinely distinct
    // functions in different modules (`a::foo` vs `b::foo`) stay
    // separate (`a` / `b` aren't in the stripped set), and the
    // bare-name dedupe trap (using `FunctionGraph::name`) is still
    // avoided because we key on the post-strip segment sequence, not
    // on the bare leaf.
    fn canonical_dedup_key(path: &crate::parse::CallPath) -> Vec<String> {
        const CRATE_PREFIXES: &[&str] = &["crate", "pyre_interpreter", "pyre_object", "pyre_jit"];
        let mut segs: Vec<String> = path.segments.iter().cloned().collect();
        if segs
            .first()
            .map(|s| CRATE_PREFIXES.contains(&s.as_str()))
            .unwrap_or(false)
        {
            segs.remove(0);
        }
        segs
    }
    let mut pending: Vec<(FunctionPathKey, &LegacyGraph, Rc<PyreFunctionEntry>)> =
        Vec::with_capacity(function_graphs.len());
    // `by_canonical_path` tracks the canonical `FunctionPathKey` of
    // the first-encountered alias for each canonical-stripped key.
    // Subsequent aliases of the same callable register an alias row
    // pointing at that canonical key — the registry's `entries` map
    // gets exactly one row per distinct callable, with `aliases`
    // carrying the indirection.
    let mut by_canonical_path: HashMap<Vec<String>, FunctionPathKey> = HashMap::new();
    for (path, graph) in function_graphs.iter() {
        let key = FunctionPathKey::from_segments(path.segments.iter().cloned());
        let canonical_strip = canonical_dedup_key(path);
        let entry = if let Some(canonical_key) = by_canonical_path.get(&canonical_strip) {
            if canonical_key != &key {
                registry.alias(key.clone(), canonical_key);
            }
            registry
                .lookup(canonical_key)
                .expect("canonical entry registered")
        } else {
            let signature = signature_for_graph(graph);
            let entry = registry.get_or_register(key.clone(), signature);
            by_canonical_path.insert(canonical_strip, key.clone());
            entry
        };
        pending.push((key, graph, entry));
    }
    // Pass 2 — prefill the default-cache once per *unique* registry
    // entry.  Aliases already point at the same `Rc<PyreFunctionEntry>`
    // so their `prefill_default_cache` would be redundant; identify
    // unique entries by `Rc::as_ptr` so the dedupe survives any
    // alternate dedup-key shape.  Each unique entry is lifted exactly
    // once (`Rc::as_ptr` identity dedup); aliases share the same
    // pre-filled `FunctionDesc.cache` because they hold the same
    // `Rc<PyreFunctionEntry>`.
    //
    // Per-callee failure isolation matches RPython
    // `bookkeeper.py:353-409 getdesc(pyobj)` semantics: each
    // `FunctionDesc` is built independently via `newfuncdesc`, and a
    // failure on one callable does NOT abort the bookkeeper's `descs`
    // population for other callables.  Upstream's per-`Constant(pyobj)`
    // builder is invoked lazily; pyre's eager pre-pass approximates
    // that lazy shape by isolating per-callee failures here — but
    // unlike a previous draft that *swallowed* the error outright,
    // the lift error is now stashed on the entry via
    // `PyreFunctionEntry::record_lift_error` so the next
    // `cachedgraph` consumer surfaces the actual producer-side
    // failure instead of falling through to `buildflowgraph`'s
    // generic "missing code object" message
    // (`translator/translator.rs:439`).  This keeps the lazy-failure
    // *point of observation* aligned with upstream
    // `description.py:228` while preserving pyre's eager prefill
    // shape for the success path.  Without per-entry error capture a
    // single bad leaf (e.g. `function_write_barrier`'s unregistered
    // `try_gc_write_barrier` callee) would mask its own diagnosis
    // behind every later use-site's generic fallback — a divergence
    // from upstream's per-callable failure model where the original
    // exception propagates.
    let mut lifted: HashSet<*const PyreFunctionEntry> =
        HashSet::with_capacity(by_canonical_path.len());
    // Each lift failure pushes into the translator's
    // `_pyre_lift_errors` map keyed by the entry's `HostObject`
    // identity, making the recorded error visible to
    // `buildflowgraph` consumers downstream.  Resolved lazily so
    // test fixtures whose `Bookkeeper` runs without an attached
    // `RPythonAnnotator` (`bookkeeper.annotator()` panics with
    // "backlink absent or dropped") only pay the lookup on the
    // failure path; success paths remain unaffected.
    for (_key, graph, entry) in &pending {
        let entry_ptr = Rc::as_ptr(entry);
        if !lifted.insert(entry_ptr) {
            continue;
        }
        // `@jit.dont_look_inside` (`rlib/jit.py:142`) callees: the JIT
        // pipeline never builds a jitcode for the body
        // (`policy.py:48-84 look_inside_graph` returns False, so
        // `codewriter.py find_all_graphs` excludes the graph and the
        // callsite residualizes).  Pyre's annotator exists solely to
        // feed that jitcode pipeline, so lifting the body here is
        // upstream-dead work — and bodies marked opaque are typically
        // unliftable host plumbing (`OnceLock` init walks etc.) whose
        // lift failure would poison every caller.  Prefill the same
        // signature-only stub pygraph `register_unsafe_fn_stubs` uses
        // (the `ExtRegistryEntry.compute_result_annotation` shape) so
        // callers annotate the declared unit/bool result and the
        // codewriter emits the residual call via the fn's registered
        // C ABI address (`pyre/jit_fnaddr.rs`).  Non-scalar returns
        // fall through to the normal lift — no current marker needs
        // them, and the stub builder's coverage is unaudited for that
        // case.
        if graph.hints.iter().any(|h| h == "dont_look_inside") {
            let return_lltype = match graph.return_type.as_deref() {
                None | Some("()") => Some(LowLevelType::Void),
                Some("bool") => Some(LowLevelType::Bool),
                Some("i64") => Some(LowLevelType::Signed),
                _ => None,
            };
            if let Some(return_lltype) = return_lltype
                && let Some(stub) = build_stub_pygraph_for_unsafe_fn(
                    graph.name.clone(),
                    signature_for_graph(graph),
                    return_lltype,
                )
            {
                entry.prefill_default_cache(stub);
                continue;
            }
        }
        match lift_callee_to_pygraph(graph, signature_for_graph(graph), registry) {
            Ok(pygraph) => entry.prefill_default_cache(pygraph),
            Err(e) => {
                let message = format!("{e}");
                entry.record_lift_error(message.clone());
                if let Some(annotator) = registry.bookkeeper().try_annotator() {
                    annotator
                        .translator
                        ._pyre_lift_errors
                        .borrow_mut()
                        .insert(entry.host_object.clone(), message);
                }
            }
        }
    }
    // Pass 3 — make impl methods visible in their owner's class dict.
    // RPython's ClassDesc reads methods straight off the class object
    // (`classdesc.py:808-817 find_source_for` → `cls.__dict__[name]`):
    // a Python class object already carries its functions as members
    // when annotation starts.  Pyre's struct-root class objects are
    // minted with no members (`Bookkeeper::intern_class_by_qualname`),
    // so a receiver method call — `CallTarget::Method` lowers to
    // `getattr(recv, name)` + `simple_call`
    // (`flowspace_adapter.rs:1357`) — found no attribute source and
    // blocked.  Registering each `[owner, method]` registry entry's
    // user-function HostObject as a class member completes the
    // analogue: `find_source_for` imports it into the classdict as a
    // Constant, and `s_get_value` binds it under the classdef
    // (`bind_callables_under` → MethodDesc) whose `cachedgraph` hits
    // the pass-2 prefill.  Owner names are gated on the struct-field
    // registry (module and trait leaves are never type roots), and a
    // same-named instance field wins — a function source for a field
    // attribute would union-conflict in `generalize_attr`.
    let bk = registry.bookkeeper();
    for (key, _graph, entry) in &pending {
        // `CallPath::for_impl_method` splits the module-qualified owner
        // ("pyframe::PyFrame" → [pyframe, PyFrame, method]), so the
        // owner is the second-to-last segment.  Free-function paths
        // ([module, foo]) put a module there, and modules are never
        // type roots — the struct-field-registry gate filters them.
        let segs = key.segments();
        let [.., owner, method] = segs else {
            continue;
        };
        if !bk.is_pyre_struct_root(owner) || bk.pyre_struct_root_has_field(owner, method) {
            continue;
        }
        let class_host = bk.intern_class_by_qualname(owner);
        if class_host.class_get(method).is_none() {
            class_host.class_set(
                method.clone(),
                crate::flowspace::model::ConstValue::HostObject(entry.host_object.clone()),
            );
        }
    }
    Ok(())
}

// There is no program-wide `build_program_annotator` /
// `build_program_rtyper` / `ensure_program_specialize` pre-pass and no
// `ProgramSpecializeState` flag.  Per-session
// `specialize_legacy_graph_with_registry_returning_value_to_var`
// runs the
// orthodox flow directly: lift subject graph -> seed subject blocks
// -> compute_at_fixpoint (drives pycall->recursivecall to discover
// callees) -> RPythonTyper::specialize.

/// Derive a parameter `Signature` directly from a `FunctionGraph`'s
/// startblock inputargs.  Same shape as
/// [`signature_for`] but takes a graph rather than a
/// `SemanticFunction`, since `CallControl::function_graphs()`
/// carries graphs without their lowered SemanticFunction wrapper.
fn signature_for_graph(graph: &LegacyGraph) -> Signature {
    let startblock = graph.block(graph.startblock);
    let argnames: Vec<String> = startblock
        .inputargs
        .iter()
        .enumerate()
        .map(|(idx, var)| {
            graph
                .value_name_for(var)
                .unwrap_or_else(|| format!("arg{idx}"))
        })
        .collect();
    Signature::new(argnames, None, None)
}

/// Lift a pyre `model::FunctionGraph` (a callee that may appear on a
/// `OpKind::Call::FunctionPath` callsite of some other graph) into a
/// `Rc<PyGraph>` suitable for pre-filling the callee's
/// `FunctionDesc.cache` (`description.rs:794`).
///
/// `cachedgraph` (`description.rs:1037-1039`) returns the cached
/// `Rc<PyGraph>` as soon as the lookup key matches, skipping the
/// `buildgraph` path that delegates to
/// `translator.buildflowgraph(pyobj, false)` — that delegation
/// requires a real Python `__code__` body which pyre's surface-DSL
/// callees do not have.  Pyre instead lifts the callee through the
/// adapter and wraps the resulting flowspace `FunctionGraph` in a
/// `PyGraph` here.
///
/// `signature` is the callee's authoritative parameter `Signature`
/// (the same one stored on the `FunctionDesc` in `PyreCallRegistry`).
/// `defaults` is empty — pyre's surface DSL does not yet expose
/// per-parameter default values; when it does, this helper grows a
/// `defaults: Vec<Constant>` parameter mirroring upstream
/// `func.__defaults__`.
///
/// `nested_registry` carries the program-wide `PyreCallRegistry`.
/// For leaf callees (no `OpKind::Call` ops in the body) the registry
/// is unused; nested callees recursively consult it as the adapter
/// processes the callee's own `OpKind::Call` ops.
pub(crate) fn lift_callee_to_pygraph(
    callee_graph: &LegacyGraph,
    signature: Signature,
    nested_registry: &PyreCallRegistry,
) -> Result<Rc<PyGraph>, TyperError> {
    // The adapter also returns `value_to_var` and `constant_concretetypes`
    // side maps, but they are not consumed here.
    // RPython parity: `Variable.concretetype` and `Constant.concretetype`
    // already carry the per-variable / per-constant LL type after
    // specialise; downstream readers must consult those fields directly
    // (`history.py:204` `same_constant`, `model.py:438` `Variable.
    // concretetype`).  The by-slot side map was a pyre-only divergence.
    //
    // Test fixtures that hand-roll minimal SSA shapes must seed
    // each `Variable.annotation` directly via
    // `legacy_annotator::setbinding(&var, ValueType::…)` before calling
    // this so `seed_variable` has type info to attach.
    let FlowspaceAdapterOutput { graph, .. } =
        crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace(
            callee_graph,
            nested_registry,
        )?;
    // Pyre's synthetic `GraphFunc` mirrors `description.py:193-203
    // FunctionDesc.__init__` test fixtures — empty Dict globals,
    // name from the legacy graph.  No HostCode body — the cache
    // pre-fill ensures `cachedgraph` never asks for one.
    let func = GraphFunc::new(
        callee_graph.name.clone(),
        Constant::new(ConstValue::Dict(HashMap::new())),
    );
    let pygraph = Rc::new(PyGraph {
        graph,
        func,
        signature: RefCell::new(signature),
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
    });
    Ok(pygraph)
}

/// Synthesize a minimal flowed `PyGraph` for a
/// callee whose real body cannot be lowered through the adapter
/// (`unsafe fn` in `pyre-object` / `pyre-interpreter`).  The stub has
/// a single Link from `startblock` to `returnblock` carrying a
/// pre-annotated `Variable` so the annotator's `flowin` projects the
/// callsite result to the declared return type without touching the
/// (non-modellable) body.
///
/// `name` becomes the synthetic graph's `FunctionGraph.name`.
/// `signature.argnames` are turned into named `Variable`s on the
/// startblock; `return_lltype` is projected through
/// [`crate::translator::rtyper::llannotation::lltype_to_annotation`]
/// to a `SomeValue` shell (no `const_box`) attached to a fresh
/// `Variable` carried on the Link to the returnblock.  Returns
/// `None` when the return type is a container
/// (`Func` / `Struct` / `Array` / `Opaque` / `ForwardReference` /
/// `FixedSizeArray`) or `Address` (no `SomeAddress` port yet) —
/// the caller skips registration for that fn and the original
/// "not registered in PyreCallRegistry" Skip path covers it.
///
/// **Why pre-annotated Variable rather than `Constant`.** A
/// `Constant(default_value)` in the return Link routes through
/// `Bookkeeper::immutableconstant`, which lifts e.g.
/// `ConstValue::Bool(false)` to `SomeBool { const_box = Some(...) }`.
/// That constant slot leaks into the rtyper as a fold-eligible
/// "known false" annotation and can mis-specialise downstream code
/// that observes the callsite result.  Upstream `ExtRegistryEntry.
/// compute_result_annotation` (`extregistry.py:33`) returns a
/// `SomeXXX()` shell with no `const`; the pre-annotated Variable
/// here carries exactly that shape via `binding(arg)` reading
/// `v.annotation` directly (`annrpython.py:282-287`).
///
/// Mirrors how `description.py:193-203` test fixtures build a
/// `FunctionDesc` with a minimal `PyGraph` body — the upstream
/// equivalent is `Translator._prebuilt_graphs[entry_point] = pygraph`
/// where `pygraph` is hand-constructed without going through
/// `build_flow`.  Pure constructor — no annotator / rtyper side
/// effects.
pub(crate) fn build_stub_pygraph_for_unsafe_fn(
    name: String,
    signature: Signature,
    return_lltype: LowLevelType,
) -> Option<Rc<PyGraph>> {
    let return_someval = default_someshell_for_lltype(&return_lltype)?;
    let inputargs: Vec<Hlvalue> = signature
        .argnames
        .iter()
        .map(|n| Hlvalue::Variable(Variable::named(n)))
        .collect();
    let startblock = Block::shared(inputargs);
    let func = GraphFunc::new(
        name.clone(),
        Constant::new(ConstValue::Dict(HashMap::new())),
    );
    let mut graph_inner = crate::flowspace::model::FunctionGraph::new(name, startblock.clone());
    graph_inner.func = Some(func.clone());
    let return_var = Variable::named("__unsafe_stub_result");
    *return_var.annotation.borrow_mut() = Some(Rc::new(return_someval));
    let return_hlvalue = Hlvalue::Variable(return_var);
    let link = Rc::new(RefCell::new(Link::new(
        vec![return_hlvalue],
        Some(graph_inner.returnblock.clone()),
        None,
    )));
    startblock.closeblock(vec![link]);
    Some(Rc::new(PyGraph {
        graph: Rc::new(RefCell::new(graph_inner)),
        func,
        signature: RefCell::new(signature),
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
    }))
}

/// Project a `LowLevelType` to a `SomeValue` shell suitable for
/// pre-annotating the stub-graph return Variable
/// ([`build_stub_pygraph_for_unsafe_fn`]).
///
/// Delegates to
/// [`crate::translator::rtyper::llannotation::lltype_to_annotation`]
/// for every primitive lltype `lltype_to_annotation` itself handles
/// (Void / Bool / Float family / Char / UniChar / integer family /
/// `Ptr(_)` / `InteriorPtr(_)`), so this helper inherits the
/// upstream `lltype_to_annotation` (`llannotation.py:172-185`) shape
/// — every returned `SomeXXX` has no `const_box` set, matching
/// `ExtRegistryEntry.compute_result_annotation` semantics.
///
/// Returns `None` for container types
/// (`Func` / `Struct` / `Array` / `FixedSizeArray` / `Opaque` /
/// `ForwardReference`) that `lltype_to_annotation` rejects, and for
/// `Address` (upstream `SomeAddress`; not yet ported to the pyre
/// `SomeValue` enum — see `model.rs:21` TODO).  The caller
/// treats `None` as "skip this fn"; the unported path then surfaces
/// the original "not registered" Skip.
pub(crate) fn default_someshell_for_lltype(
    lltype: &LowLevelType,
) -> Option<crate::annotator::model::SomeValue> {
    if lltype.is_container_type() {
        return None;
    }
    match lltype {
        // SomeAddress is not yet present in the SomeValue enum
        // (model.rs:21 TODO); skip until ported.
        LowLevelType::Address => None,
        _ => Some(crate::translator::rtyper::llannotation::lltype_to_annotation(lltype.clone())),
    }
}

/// Register a batch of unsafe-fn stub
/// `(segments, signature, return_lltype)` specs into `registry`.
/// Each entry is wrapped through [`build_stub_pygraph_for_unsafe_fn`]
/// + [`PyreCallRegistry::register_callee`], so subsequent
/// `flowspace_adapter::translate_op` lookups via
/// `call_registry.lookup_with_leaf_match` find a registered entry and
/// the dual gate no longer Skips with "not registered in
/// PyreCallRegistry" for these paths.
///
/// `specs` is typically the output of
/// `front::mir::collect_unsafe_fn_stubs_from_llbc` (the Charon/LLBC-
/// sourced stub-spec list).  Per-fn failures (stub-pygraph builder returns
/// `None` for compound lltypes, or registry already has the same key
/// at a conflicting signature) propagate as silent skips — the
/// upstream "not registered" Skip path then absorbs that specific fn
/// while the rest of the batch lands.
///
/// Mirrors `populate_call_registry_from_call_graphs`'s
/// "register and prefill" contract (`cutover.rs:856-875`) but feeds
/// from the LLBC-sourced stub-spec list (`collect_unsafe_fn_stubs_from_llbc`)
/// instead of pyre's
/// `function_graphs: HashMap<CallPath, LegacyGraph>` (which excludes
/// unsafe fns by validate_signature rejection).
///
/// **Annotator-only carrier — never reaches the codewriter.**
/// The stub graph carries a single default-Constant return link
/// suitable for `RPythonAnnotator` return-type inference via
/// `cachedgraph` (see `pyre_call_registry::prefill_default_cache`).
/// Because `CallControl::function_graphs` is populated exclusively
/// by lowered safe-fn bodies (unsafe fns never produce a flow graph —
/// they only get a metadata-only stub key), the unsafe stub key
/// is never present in `function_graphs`.  `CallControl::
/// find_all_graphs` walks `function_graphs.keys()` only and resolves
/// each call target via `target_to_path_and_graph`
/// (`jit_codewriter/call.rs:2601`) which returns `None` for any
/// path absent from `function_graphs` — so an unsafe-stub target
/// triggers `continue` and is never added to `candidate_graphs`,
/// never reaches `transform_graph_to_jitcode`, and never compiles
/// into executable JITCode.  The actual unsafe-fn body executes
/// through the residual-call / direct-call fnaddr lowering that
/// the codewriter emits for the call op whose target resolves
/// to the host-evaluator entry point.  The default-Constant return is
/// safe by virtue of the `function_graphs` gate, not by any
/// `look_inside_graph` policy on the stub itself.
pub(crate) fn register_unsafe_fn_stubs(
    registry: &PyreCallRegistry,
    specs: &[(Vec<String>, Signature, LowLevelType)],
) {
    for (segments, signature, return_lltype) in specs {
        let Some(stub_pygraph) = build_stub_pygraph_for_unsafe_fn(
            segments.last().cloned().unwrap_or_default(),
            signature.clone(),
            return_lltype.clone(),
        ) else {
            continue;
        };
        let key = FunctionPathKey::from_segments(segments.iter().cloned());
        if registry.lookup(&key).is_some() {
            // Already registered via the function_graphs pass (e.g. a
            // safe fn with the same path).  Don't overwrite.
            continue;
        }
        registry.register_callee(key, signature.clone(), stub_pygraph);
    }
}

/// Restricted entry: only graphs without `OpKind::Call::FunctionPath`
/// ops resolve through this path.  An empty `PyreCallRegistry` is
/// constructed internally and shared with the annotator; any
/// `simple_call` op the adapter emits would surface a fail-loud
/// `TyperError` at `flowspace_adapter::translate_op`'s
/// `FunctionPath` arm because there is no `(HostObject, FunctionDesc)`
/// pair to resolve the callable to.
///
/// The upstream
/// `Constant(<function>) -> getdesc -> FunctionDesc` chain
/// (`bookkeeper.py:353-409`) is unreachable from this entry because
/// pyre's surface DSL has no Python callable object to resolve;
/// production callers that need free function calls must build a
/// pre-populated registry (one entry per reachable `FunctionPath`,
/// with `register_callee(key, signature, lifted_pygraph)`) and call
/// [`specialize_legacy_graph_with_registry_returning_value_to_var`] directly.  Until the
/// production walker that traverses `SemanticProgram.functions`
/// lands, this entry remains in-place for anchor tests and
/// dual-gate validation against graphs that have no `Call` ops.
pub fn specialize_legacy_graph(
    legacy: &LegacyGraph,
) -> Result<(LegacyToTyped, HashMap<Variable, LowLevelType>), TyperError> {
    let registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(Rc::new(
        crate::annotator::bookkeeper::Bookkeeper::new(),
    ));
    specialize_legacy_graph_with_registry_returning_value_to_var(legacy, &registry)
}

/// `specialize_legacy_graph_with_registry` extended return shape that
/// also surfaces the [`LegacyToTyped`] map and the per-`Variable`
/// `Constant.concretetype` table produced by the flowspace adapter.
/// Codewriter callers consume `value_to_var` directly: the rtyper's
/// typed `Variable.concretetype` writes are copied onto the matching
/// legacy Variables by
/// [`crate::jit_codewriter::type_state::apply_from_flowspace_variables`].
/// `constants` feeds [`project_value_to_var`] for the
/// dual-gate baseline comparison.
///
/// Test fixtures that hand-roll minimal SSA shapes without
/// production-shape `OpKind::Input { ty }` ops must seed
/// `legacy.variable(vid).annotation` directly via
/// `legacy_annotator::setbinding(&var, ValueType::...)` before calling
/// this so `seed_variable` has type info to attach.
pub fn specialize_legacy_graph_with_registry_returning_value_to_var(
    legacy: &LegacyGraph,
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
) -> Result<(LegacyToTyped, HashMap<Variable, LowLevelType>), TyperError> {
    let (_graph, value_to_var, constant_concretetypes) =
        drive_subject(legacy, call_registry, true)?;
    Ok((value_to_var, constant_concretetypes))
}

/// Annotate one legacy subject against the shared session annotator/rtyper,
/// optionally rtyping it in the same call.
///
/// `do_rtype = true` is the fused per-graph dual-gate path: annotate-half +
/// `specialize_more_blocks` + lltype validation, committing the subject's
/// blocks after rtype (the original behaviour, byte-identical).
///
/// `do_rtype = false` is the two-phase Phase-A annotate-only path: it runs the
/// annotate-half and commits the subject's blocks into the shared session
/// WITHOUT rtyping, so a later whole-program Phase-B `specialize_more_blocks`
/// rtypes every annotated block at once — after all callers of a shared callee
/// (e.g. `type_error`) have unioned its arg to a general `SomeString`, which is
/// what removes the `fixed_graphs` late-stage-modify Skip. Returns the lifted
/// flowspace `GraphRef` so the two-phase cache can record its `GraphKey`.
fn drive_subject(
    legacy: &LegacyGraph,
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
    do_rtype: bool,
) -> Result<
    (
        crate::flowspace::model::GraphRef,
        LegacyToTyped,
        HashMap<Variable, LowLevelType>,
    ),
    TyperError,
> {
    // RPython parity path.
    //
    // Upstream `RPythonTyper.specialize` runs ONCE per `Translator`,
    // not per graph.  Pyre's per-graph dual-gate enters this function
    // once per graph (one per `transform_graph_to_jitcode`); the
    // upstream "specialize-once" semantics are reproduced by sharing
    // one annotator + rtyper across subjects through
    // `PyreCallRegistry::ensure_session` and rtyping only the newly
    // added blocks each entry.  Already-cached PyGraphs hold
    // post-specialize LL ops; re-seeding them into the per-session
    // annotator would let `specialize_more_blocks` walk the LL ops a
    // second time and trip on `unimplemented operation: 'int_add'` /
    // `'direct_call'`.
    // ── Step 1 — adapter ─────────────────────────────────────────────
    let FlowspaceAdapterOutput {
        graph,
        value_to_var,
        constant_concretetypes,
        block_map: _,
    } = crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace(
        legacy,
        call_registry,
    )?;

    // ── Step 2 — annotator surface ────────────────────────────────
    //
    // The annotator + rtyper are
    // session-shared through `PyreCallRegistry::ensure_session`,
    // mirroring RPython's "one annotator + one rtyper per Translator"
    // (`translator.py:69-83`).  Each per-graph subject seeds its
    // blocks into the shared annotator; subsequent
    // `specialize_more_blocks` rtype only the newly-added blocks.
    let (annotator, rtyper) = call_registry.ensure_session()?;
    // Scope an `added_blocks` tracker over this subject's drive of the
    // shared annotator so the post-drain blocked-annotation guard
    // (`raise_if_subject_blocked` below) can see exactly the blocks
    // seeded for this subject and roll them back if any stays in the
    // `None` (False) sentinel state. Mirrors `complete_helpers`'s
    // `saved = self.added_blocks; self.added_blocks = {}` prologue
    // (annrpython.py:113-114); the guard restores the previous tracker
    // when this function returns.
    let _subject_block_scope = annotator.enter_added_blocks_scope();
    // Queue `graph.startblock` onto the orthodox `addpendingblock`
    // queue, mirroring how callees enter through
    // `pycall -> recursivecall -> addpendingblock`
    // (`description.py:283-305`, `annrpython.py:315-336`).
    // `addpendingblock` (`annrpython.rs:1245-1302`) writes
    // `Variable.annotation` for each inputarg via
    // `bindinputargs.setbinding`, inserts `all_blocks[startblock]`
    // and `annotated[startblock] = None`, then
    // `schedulependingblock` queues the block.
    // `complete_pending_blocks` below drains the queue:
    // `processblock` flips `annotated[block] = Some(graph)`,
    // `flowin` walks every op and recurses into successors via
    // `process_link -> addpendingblock(target, inputs_s)`
    // (`annrpython.rs:1502/1690`).  Transitive blocks register
    // themselves into `all_blocks`/`annotated` through the same
    // bindinputargs path on first arrival.
    let subject_inputcells =
        crate::translator::rtyper::flowspace_adapter::derive_subject_inputcells(
            legacy,
            Some(call_registry.bookkeeper()),
        )?;
    let (startblock, exceptblock) = {
        let g = graph.borrow();
        (g.startblock.clone(), g.exceptblock.clone())
    };
    let subject_inputcells_opt: Vec<Option<crate::annotator::model::SomeValue>> =
        subject_inputcells.into_iter().map(Some).collect();
    annotator.addpendingblock(&graph, &startblock, &subject_inputcells_opt);
    // Pre-register the eagerly-created `exceptblock`
    // (`flowspace/model.py:21-25` parity) — non-raising subjects
    // never reach it through a `Link`, but `specialize_more_blocks`
    // must walk it so its inputargs receive an exception-typed
    // `concretetype`.  The rtyper's `setup_block_entry`
    // exception-block branch (`rtyper.rs:1915-1942`) writes
    // `Variable.concretetype = ExceptionData.lltype_of_exception_*`
    // without reading `Variable.annotation`, so the block needs no
    // `flowin`; it only needs to appear in `annotator.annotated`.
    //
    // Additionally, seed each inputarg with a `setbinding`-level
    // annotation (`Impossible` when none is already present) so that
    // a raising subject's `follow_raise_link` — which reaches the
    // same exceptblock via `addpendingblock` with `seen_before=true`
    // and routes to `mergeinputargs` — does not panic on the unbound
    // `seed_variable(legacy_v)` inputargs from
    // `flowspace_adapter.rs:1839`.  `setbinding` widens via the
    // lattice's `contains` check, so an `Integer`-already-annotated
    // slot stays `Integer`; a `None`-annotated slot becomes
    // `Impossible`, which any later `mergeinputargs` widens to the
    // real cell.  The `annotated[block] = Some(graph)` bare insert
    // (the next block) keeps the pre-fix `specialize_block` walk
    // semantic intact — the block is treated as already annotated
    // and never enters `complete_pending_blocks`.
    {
        let blk = exceptblock.borrow();
        let inputargs: Vec<crate::flowspace::model::Hlvalue> = blk.inputargs.clone();
        drop(blk);
        for a in &inputargs {
            if let crate::flowspace::model::Hlvalue::Variable(v) = a {
                let needs_seed = v.annotation.borrow().is_none();
                if needs_seed {
                    let mut tmp = v.clone();
                    annotator.setbinding(&mut tmp, crate::annotator::model::SomeValue::Impossible);
                }
            }
        }
    }
    {
        let bkey = crate::flowspace::model::BlockKey::of(&exceptblock);
        annotator
            .annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(graph.clone()));
        annotator
            .all_blocks
            .borrow_mut()
            .insert(bkey, exceptblock.clone());
    }

    // Callee blocks are seeded naturally with no explicit pre-seed:
    // `@op.simple_call.register(SomeObject)` (unaryop.py:114-118
    // parity) makes flowin dispatch the first
    // `simple_call(host_object_const, args)` op through `s_func.call(
    // argspec)` -> `SomePBC.call` -> `Bookkeeper.pbc_call` ->
    // `FunctionDesc.pycall` -> `annotator.recursivecall`, which calls
    // `addpendingblock(graph, startblock, inputcells)` for the
    // callee.

    // Addpendingblock conversion — drain the pending
    // queue so `flowin` walks every block reachable from
    // `graph.startblock` and writes `Variable.annotation` for each
    // op result.  Mirrors upstream `RPythonAnnotator.complete()`
    // (`annrpython.py:226-232`), which loops `complete_pending_blocks`
    // → policy hook → exit-on-empty.  Without this drain, the
    // `addpendingblock(startblock, inputcells)` queued just above
    // stays in `genpendingblocks`, `annotated[block]` remains the
    // `None` sentinel, and `specialize_block` (`rtyper.rs:1656`)
    // panics on "annotator.annotated[block] is False sentinel".
    annotator
        .complete_pending_blocks()
        .map_err(|err| TyperError::message(format!("complete_pending_blocks failed: {err}")))?;
    // Orthodox `complete()` blocked-annotation guard
    // (annrpython.py:243-255): a block that stayed in the `None` (False)
    // sentinel state after the fixpoint drain is permanently
    // `BlockedInference`. Upstream `complete()` raises here, before
    // `simplify`/`compute_at_fixpoint`. The dual-gate calls
    // `complete_pending_blocks` (the bare drain) instead of
    // `complete()`, so the guard is replicated here; the error reuses
    // the "complete_pending_blocks failed" prefix already recognised by
    // `is_known_unported`, routing the subject to the legacy walker.
    annotator
        .raise_if_subject_blocked()
        .map_err(|err| TyperError::message(format!("complete_pending_blocks failed: {err}")))?;

    // Populate per-callsite call-family / calltable state
    // by walking the seeded blocks' call_ops.  `compute_at_fixpoint`
    // (`bookkeeper.py:108-118`, pyre `bookkeeper.rs:627-648`) drains
    // `annotator.call_sites()` through `consider_call_site`
    // (`bookkeeper.py:152-166`, pyre `bookkeeper.rs:675`); each
    // `simple_call(callable_const, *args)` op resolves the callable
    // to a `SomePBC` (via `immutablevalue_hostobject` for the
    // pre-registered `HostObject::UserFunction`), then records the
    // call site in `bookkeeper.pbc_maximal_call_families` so the
    // rtyper's `FunctionRepr.call(hop)` (`rpbc.py:199`) finds the
    // matching call-family row at `find_row` time.
    //
    // The subject just seeded into this annotator is brand-new and its
    // `simple_call` ops have not been processed yet; this drain
    // processes the subject's (and any newly seeded callees')
    // call_sites so their rows land in `pbc_maximal_call_families`
    // before specialize looks them up.
    //
    // Errors propagate verbatim — upstream
    // `bookkeeper.py:108-118` runs the call-site walk without a
    // `try`/`except`, so a failed `consider_call_site` terminates
    // `simplify` and unwinds out of the annotator driver.  Pyre's
    // port surfaces the same condition through `?`-propagation
    // here rather than swallowing it at `bookkeeper.rs:627-648`.
    call_registry
        .bookkeeper()
        .compute_at_fixpoint()
        .map_err(|err| TyperError::message(format!("compute_at_fixpoint failed: {err}")))?;

    // ── Step 3 — incremental rtyper drive ──────────────────────────
    //
    // Lifecycle mapping vs upstream `RPythonTyper.specialize(self,
    // dont_simplify_again=False)` (rtyper.py:178-189):
    //
    //   1. `if not dont_simplify_again: self.annotator.simplify()`
    //      — pyre per-subject flow does not run annotator-wide
    //        simplify; the subject was lifted via the adapter and
    //        its block list is already in canonical shape.
    //   2. `self.exceptiondata.finish(self)` — hoisted into
    //      `PyreCallRegistry::ensure_session` so it runs **once** at
    //      session start (idempotent — `getclassrepr` is cached on
    //      `rtyper.instance_reprs`).
    //   3. `self.already_seen = {}` — pyre keeps `already_seen`
    //      across per-subject calls so previously-rtyped subjects
    //      are not re-walked.
    //   4. `self.specialize_more_blocks()` — invoked here per
    //      subject; walks annotator.annotated\\already_seen.
    //   5. `self.exceptiondata.make_helpers(self)` — currently
    //      no-op (R4 / `MixLevelHelperAnnotator` not ported), see
    //      module header on `translator/rtyper/rclass.rs`.
    //   6. second `specialize_more_blocks()` for helpers — redundant
    //      while make_helpers is a no-op.
    //
    // Step 4 is the only per-subject step; steps 2/5/6 are
    // session-global one-shots, step 1/3 are pyre-incremental
    // adaptations.  Once R4 lands, `make_helpers` migrates to
    // `ensure_session` after `finish_exceptiondata` (single
    // session-prologue), keeping the per-subject body unchanged.
    //
    // In the two-phase path (`do_rtype = false`) the rtype-half is
    // deferred to the whole-program Phase-B `specialize_more_blocks`
    // (see `run_two_phase_prepass`); this call only annotates, leaving
    // the subject's blocks in `annotated` for that single later pass.
    if do_rtype {
        rtyper.specialize_more_blocks()?;

        // ── Step 4 — validate per-slot lltype projection ──────────────
        //
        // No eager per-slot side table is built here.  Each callable that
        // needs a kind view derives one on demand from `value_to_var` +
        // `constant_concretetypes` via
        // [`project_value_to_var`] — same `Variable.concretetype`
        // / `Constant.concretetype` ground truth, just routed at the
        // consumer instead of eagerly materialised here.  Run a
        // validation pass so unsupported lltypes still surface as a
        // [`TyperError`] at this boundary (a fail-loud `?` propagation).
        for var in value_to_var.values() {
            let cell = var.concretetype.borrow();
            if let Some(lltype) = cell.as_ref() {
                lowleveltype_to_concrete(lltype)?;
            }
        }
        for lltype in constant_concretetypes.values() {
            lowleveltype_to_concrete(lltype)?;
        }
    }

    // The subject specialized cleanly — keep its blocks in the shared
    // annotator (they are now `already_seen`). Every earlier `?` exit
    // above leaves `_subject_block_scope` uncommitted, so a Skipped
    // subject's partially-annotated blocks are evicted on drop and never
    // poison a later subject's session-global `specialize_more_blocks`.
    _subject_block_scope.commit();

    // Inert opname-reachability gauge over the rtyped flowspace graph this
    // subject specialized cleanly — the dual-gate Match path the gauge's
    // placement note targets.  Gated on `PYRE_JTRANSFORM_SHADOW` so the
    // default build neither borrows nor walks the graph; the gauge never
    // mutates it.
    if crate::jit_codewriter::jtransform_shadow::is_enabled() {
        crate::jit_codewriter::jtransform_shadow::report_if_enabled(&graph.borrow());
    }

    Ok((graph, value_to_var, constant_concretetypes))
}

/// Whole-program two-phase rtyper prepass (`PYRE_TWO_PHASE_RTYPE`).
///
/// Mirrors upstream `translator.annotate()` → `rtyper.specialize()`
/// (`driver.py:325/345`): Phase A annotates EVERY graph in the portal closure
/// to fixpoint — sharing the session annotator so all callers of a shared
/// callee (e.g. `type_error`) union its arg to a general `SomeString` — THEN
/// Phase B rtypes every annotated block in one pass. The per-graph publish
/// ([`dual_gate_outcome_from_cache`]) reads the cached result instead of
/// re-running the fused per-graph path, so no callee is rtyped before all its
/// callers are annotated, removing the `fixed_graphs` late-stage-modify Skip.
///
/// Populates `call_registry.two_phase()`. The prepass is fully panic-contained
/// and ALWAYS sets `prepass_done = true`: `make_jitcodes` runs in a spawned
/// worker thread, so any escaping panic (the annrpython late-stage `panic!`, a
/// `call_all_setups` failure, an unported repr setup) would unwind the thread
/// and fail the build. The publish path ([`dual_gate_outcome_from_cache`]) is
/// session-independent — it reads cached `Variable`s and re-runs the legacy
/// walker, never the flowspace session — so a partial cache from an aborted
/// prepass still yields Match for the graphs that completed and a safe
/// legacy-walker Skip for the rest (zero regression, never the poisoned session).
pub fn run_two_phase_prepass(
    call_registry: &PyreCallRegistry,
    candidate_graphs: &HashSet<crate::parse::CallPath>,
    function_graphs: &crate::jit_codewriter::call::GraphStore,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_two_phase_prepass_inner(call_registry, candidate_graphs, function_graphs)
    }));
    call_registry.two_phase().prepass_done = true;
}

fn run_two_phase_prepass_inner(
    call_registry: &PyreCallRegistry,
    candidate_graphs: &HashSet<crate::parse::CallPath>,
    function_graphs: &crate::jit_codewriter::call::GraphStore,
) {
    // Deterministic order (R3): candidate_graphs is a HashSet; iterating it
    // directly would make classdef numbering (and thus Match/Skip
    // classification) vary run-to-run.
    let mut paths: Vec<&crate::parse::CallPath> = candidate_graphs
        .iter()
        .filter(|p| function_graphs.get(p).is_some())
        .collect();
    paths.sort_by_key(|p| p.canonical_key());

    // ── Phase A — annotate-all over the portal closure ───────────────
    for path in &paths {
        let Some(legacy) = function_graphs.get(path) else {
            continue;
        };
        let fixed_at_entry: HashSet<crate::flowspace::model::GraphKey> = call_registry
            .session_if_started()
            .map(|(ann, _)| ann.fixed_graphs.borrow().keys().cloned().collect())
            .unwrap_or_default();
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drive_subject(legacy, call_registry, /* do_rtype = */ false)
        }));
        match attempt {
            Ok(Ok((graph, value_to_var, constant_concretetypes))) => {
                let key = path.canonical_key();
                call_registry.two_phase().subjects.insert(
                    key,
                    crate::translator::rtyper::pyre_call_registry::TwoPhaseSubject {
                        graph_key: crate::flowspace::model::GraphKey::of(&graph),
                        value_to_var,
                        constant_concretetypes,
                    },
                );
            }
            ref other @ (Ok(Err(_)) | Err(_)) => {
                // De-aggregating census (PYRE_RTYPER_VERBOSE): the dual-gate
                // otherwise lumps every Phase-A failure under one opaque
                // "subject not annotated/rtyped" Skip. Surface the per-graph
                // reason so the onion can be triaged.
                if std::env::var("PYRE_RTYPER_VERBOSE").is_ok() {
                    let reason = match other {
                        Ok(Err(e)) => format!("annotate Err: {e:?}"),
                        Err(p) => {
                            let msg = p
                                .downcast_ref::<String>()
                                .cloned()
                                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                                .unwrap_or_else(|| "<non-string panic>".to_string());
                            format!("annotate PANIC: {msg}")
                        }
                        Ok(Ok(_)) => unreachable!(),
                    };
                    eprintln!("[PREPASS phaseA fail] {:?}: {reason}", path);
                }
                // Annotate-half failed (or panicked): repair shared-callee state
                // and leave the graph uncached so publish Skips it to the legacy
                // walker. unpoison is itself contained — a panic here must not
                // abort the remaining Phase A graphs.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    unpoison_failed_subject_callees(
                        call_registry,
                        &fixed_at_entry,
                        function_graphs,
                    );
                }));
            }
        }
    }

    // ── compute_at_fixpoint runs per-subject inside drive_subject's
    // annotate-half (cutover compute_at_fixpoint call); it is monotonic so
    // per-subject invocation reaches the same fixpoint (R2: the once-at-end
    // upstream shape — annotator.complete() — is a deferred optimisation).

    // ── Phase B — rtype-all with per-graph isolation ─────────────────
    // Writes its rtype_skipped set into the cache incrementally so partial
    // progress survives even if this call is cut short.
    run_phase_b_rtype_isolated(call_registry, function_graphs);
}

/// Phase B: rtype every annotated block in one whole-program pass, tolerant of
/// per-graph rtype failure (the migration scaffold — upstream `specialize()` is
/// single-pass-fatal, rtyper.py:177-296). A block's `specialize_block` failure
/// records its owning graph into the cache's `rtype_skipped` set; that graph's
/// remaining blocks are excluded from later rounds (the publish then Skips it to
/// the legacy walker) and its half-fixed callees are repaired via
/// `unpoison_failed_subject_callees`. The skip set is written into the cache
/// incrementally so partial progress survives a cut-short call.
fn run_phase_b_rtype_isolated(
    call_registry: &PyreCallRegistry,
    lift_sources: &crate::jit_codewriter::call::GraphStore,
) {
    use crate::flowspace::model::{BlockRef, GraphKey, GraphRef};
    let Ok((annotator, rtyper)) = call_registry.ensure_session() else {
        return;
    };
    loop {
        // Skip-tolerant repr setup: a single unported repr (e.g. DictRepr)
        // must not abort rtyping of every other annotated graph. The graph
        // that needs the skipped repr fails its own specialize_block below and
        // is Skip-classified (parity fallback to the legacy walker).
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rtyper.call_all_setups_skip_tolerant()
        }));
        // Snapshot the skip set (cache borrow drops before specialize_block).
        let skipped: HashSet<GraphKey> = call_registry.two_phase().rtype_skipped.clone();
        // annotated\already_seen, excluding blocks of already-skipped graphs.
        let pending: Vec<(BlockRef, Option<GraphRef>)> = {
            let annotated = annotator.annotated.borrow();
            let all_blocks = annotator.all_blocks.borrow();
            let already_seen = rtyper.already_seen.borrow();
            annotated
                .iter()
                .filter(|(bkey, _)| !already_seen.contains_key(*bkey))
                .filter_map(|(bkey, gopt)| {
                    if let Some(g) = gopt {
                        if skipped.contains(&GraphKey::of(g)) {
                            return None;
                        }
                    }
                    all_blocks.get(bkey).map(|b| (b.clone(), gopt.clone()))
                })
                .collect()
        };
        if pending.is_empty() {
            break;
        }
        for (block, gopt) in pending {
            if let Some(g) = &gopt {
                if skipped.contains(&GraphKey::of(g)) {
                    continue;
                }
            }
            let fixed_at_entry: HashSet<GraphKey> =
                annotator.fixed_graphs.borrow().keys().cloned().collect();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rtyper.specialize_block(&block)
            }));
            match res {
                Ok(Ok(())) => rtyper.mark_already_seen(&block),
                ref other @ (Ok(Err(_)) | Err(_)) => {
                    if std::env::var("PYRE_RTYPER_VERBOSE").is_ok() {
                        let reason = match other {
                            Ok(Err(e)) => format!("rtype Err: {e:?}"),
                            Err(p) => {
                                let msg = p
                                    .downcast_ref::<String>()
                                    .cloned()
                                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                                    .unwrap_or_else(|| "<non-string panic>".to_string());
                                format!("rtype PANIC: {msg}")
                            }
                            Ok(Ok(())) => unreachable!(),
                        };
                        eprintln!(
                            "[PREPASS phaseB fail] {:?}: {reason}",
                            gopt.as_ref().map(GraphKey::of)
                        );
                    }
                    match &gopt {
                        Some(g) => {
                            // Skip this graph; the pending recompute excludes
                            // its blocks, guaranteeing progress (one graph
                            // removed per failure).
                            call_registry
                                .two_phase()
                                .rtype_skipped
                                .insert(GraphKey::of(g));
                        }
                        None => {
                            // No graph to attribute (None-sentinel block);
                            // mark it seen so the loop still progresses.
                            rtyper.mark_already_seen(&block);
                        }
                    }
                    // Repair half-fixed callees; contained so an unpoison panic
                    // can't escape Phase B.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        unpoison_failed_subject_callees(
                            call_registry,
                            &fixed_at_entry,
                            lift_sources,
                        );
                    }));
                    break;
                }
            }
        }
    }
    // MixLevelHelperAnnotator drain (rtyper.py:238-241) is a no-op while R4 is
    // unported — specialize_block never queues helper graphs today. When R4
    // lands, take()+finish() rtyper.annmixlevel here.
}

/// Two-phase publish: derive the [`DualGateOutcome`] for `legacy` from the
/// prepass cache instead of re-running the real path. Mirrors
/// [`dual_gate_check_with_registry`]'s legacy-baseline comparison (the
/// migration safety oracle) but consumes the cached real types.
pub(crate) fn dual_gate_outcome_from_cache(
    legacy: &LegacyGraph,
    call_registry: &PyreCallRegistry,
    diag_key: &str,
) -> Result<DualGateOutcome, String> {
    // Clone the cached real types out so the cache borrow drops before the
    // legacy baseline runs (the baseline never touches the cache).
    let cached = {
        let tp = call_registry.two_phase();
        match tp.subjects.get(diag_key) {
            Some(subj) if !tp.rtype_skipped.contains(&subj.graph_key) => Some((
                subj.value_to_var.clone(),
                subj.constant_concretetypes.clone(),
            )),
            _ => None,
        }
    };
    let Some((value_to_var, constants)) = cached else {
        return Ok(DualGateOutcome::Skip(
            "two-phase: subject not annotated/rtyped in prepass".to_string(),
        ));
    };

    // Validate the cached lltypes project to concrete kinds (the Step-4 check
    // deferred from `drive_subject`, which did not rtype in Phase A).
    for var in value_to_var.values() {
        let cell = var.concretetype.borrow();
        if let Some(lltype) = cell.as_ref() {
            if lowleveltype_to_concrete(lltype).is_err() {
                return Ok(DualGateOutcome::Skip(
                    "two-phase: cached lltype does not project".to_string(),
                ));
            }
        }
    }
    for lltype in constants.values() {
        if lowleveltype_to_concrete(lltype).is_err() {
            return Ok(DualGateOutcome::Skip(
                "two-phase: cached const lltype does not project".to_string(),
            ));
        }
    }

    // Legacy-baseline comparison oracle (the `dual_gate_check_with_registry`
    // pattern): the real path is trusted only when it agrees with the proven
    // legacy walker at `ConcreteType` granularity.
    let _annotation_guard = LegacyAnnotationGuard::snapshot(legacy);
    let baseline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::legacy_annotator::annotate(legacy);
        super::legacy_resolve::resolve_types(legacy);
    }));
    if let Err(payload) = baseline {
        let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<unrecognised panic payload>".to_string()
        };
        return Err(format!(
            "two-phase baseline panicked (legacy walker crashed before comparison): {msg}"
        ));
    }
    if let Some(divergence) = compare_real_against_legacy(&value_to_var, &constants, legacy) {
        return Ok(DualGateOutcome::Skip(format!(
            "two-phase divergence: {divergence}"
        )));
    }
    Ok(DualGateOutcome::Match {
        real_value_to_var: value_to_var,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Block, BlockId, LinkArg, ValueType};
    use crate::translator::rtyper::legacy_annotator::setbinding;

    fn link_to_returnblock(args: Vec<LinkArg>, returnblock_id: BlockId) -> crate::model::Link {
        crate::model::Link::new_mixed(args, returnblock_id, None)
    }

    /// Mint `n` fresh values on `graph`, returned indexed `0..n` so
    /// fixtures can refer to inputargs / operands positionally.
    fn mint_vars(graph: &mut LegacyGraph, n: usize) -> Vec<crate::flowspace::model::Variable> {
        (0..n).map(|_| graph.alloc_value_var()).collect()
    }

    /// Read the concrete kind for a legacy graph `Variable` from the
    /// post-rtyper `LegacyToTyped` map's `Variable.concretetype` cell.
    /// The map is keyed by the legacy Variable's identity.
    fn kind_of_in(
        value_to_var: &LegacyToTyped,
        var: &crate::flowspace::model::Variable,
    ) -> ConcreteType {
        let typed = value_to_var
            .get(var)
            .unwrap_or_else(|| panic!("value missing from value_to_var"));
        let ll = typed
            .concretetype()
            .unwrap_or_else(|| panic!("Variable.concretetype not populated"));
        lowleveltype_to_concrete(&ll)
            .unwrap_or_else(|_| panic!("lltype does not project to ConcreteType"))
    }

    /// Project positional indices to held Variables for
    /// `Block { inputargs, .. }` literals.
    fn block_inputargs(
        vars: &[crate::flowspace::model::Variable],
        vids: &[usize],
    ) -> Vec<crate::flowspace::model::Variable> {
        vids.iter().map(|&i| vars[i].clone()).collect()
    }

    #[test]
    fn lowleveltype_to_concrete_signed_family_collapses_to_signed() {
        for ll in [
            LowLevelType::Signed,
            LowLevelType::Unsigned,
            LowLevelType::Bool,
            LowLevelType::Char,
            LowLevelType::UniChar,
            LowLevelType::SingleFloat,
            LowLevelType::Address,
        ] {
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                ConcreteType::Signed,
                "{ll:?} must project to Signed"
            );
        }
    }

    #[test]
    fn known_unported_classifies_indirect_call_adapter_invariant() {
        let msg = "translate_op: Call with CallTarget::Indirect at result=Some(v4) \
                   must be lowered to VtableMethodPtr + IndirectCall by rclass.rs before \
                   reaching the flowspace adapter";
        assert!(
            is_known_unported(msg),
            "registry population must skip/fallback on the current indirect-call cutover gap"
        );
    }

    #[test]
    fn known_unported_classifies_noneify_on_ptr_merge() {
        // A `_ptr ∪ NoneType` block merge reaches `SomeValue::noneify()`
        // on a `SomePtr` operand — a parity-correct raise (default
        // `noneify` raises, `SomePtr` has no override) the dual gate
        // defers to legacy until `front::mir` lowers a typed null pointer.
        let msg = "RPython noneify() not supported for this annotation";
        assert!(
            is_known_unported(msg),
            "noneify-on-ptr merge must skip/fallback on the typed-null-pointer-lowering gap"
        );
    }

    #[test]
    fn lowleveltype_to_concrete_float_family_collapses_to_float() {
        for ll in [LowLevelType::Float] {
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                ConcreteType::Float,
                "{ll:?} must project to Float"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_longlong_follows_target_word_size() {
        // history.py:55-58 — `if rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed):
        //   if supports_longlong and TYPE is not lltype.LongFloat: return 'float'`
        let expected = if std::mem::size_of::<i64>() > std::mem::size_of::<isize>() {
            ConcreteType::Float
        } else {
            ConcreteType::Signed
        };
        for ll in [LowLevelType::SignedLongLong, LowLevelType::UnsignedLongLong] {
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                expected,
                "{ll:?} must match history.getkind's sizeof(TYPE) > sizeof(Signed) branch"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_signedlonglonglong_returns_missing_rtype_err() {
        let err = lowleveltype_to_concrete(&LowLevelType::SignedLongLongLong)
            .expect_err("SignedLongLongLong has no history.getkind mapping");
        assert!(err.is_missing_rtype_operation());
    }

    #[test]
    fn lowleveltype_to_concrete_unsignedlonglonglong_returns_missing_rtype_err() {
        let err = lowleveltype_to_concrete(&LowLevelType::UnsignedLongLongLong)
            .expect_err("UnsignedLongLongLong has no history.getkind mapping");
        assert!(err.is_missing_rtype_operation());
    }

    #[test]
    fn lowleveltype_to_concrete_longfloat_returns_missing_rtype_err() {
        let err = lowleveltype_to_concrete(&LowLevelType::LongFloat)
            .expect_err("LongFloat has no history.getkind mapping");
        assert!(err.is_missing_rtype_operation());
    }

    #[test]
    fn lowleveltype_to_concrete_void_passes_through() {
        assert_eq!(
            lowleveltype_to_concrete(&LowLevelType::Void).expect("supported lltype"),
            ConcreteType::Void
        );
    }

    /// Process-local guard serialising specialize-driving tests.
    /// The rtyper's lattice has process-global singletons that mutate
    /// during `setup()`; cargo's parallel test runner can interleave
    /// two specialize-driving tests so one observes the other's
    /// `Setupstate::InProgress` and panics with "recursive invocation
    /// of Repr setup()".  Holding this `Mutex` serialises the runs.
    fn anchor_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn specialize_legacy_graph_minimal_int_identity_resolves_signed() {
        let _lock = anchor_lock();
        // Smallest validation: identity-return graph carrying a single
        // Int-typed inputarg. Dual-gate's first anchor — proves
        // the adapter + annotator-surface seeding + specialize +
        // projection chain works end-to-end on a graph the rtyper can
        // resolve without any unported OpKind variants.
        let mut graph = LegacyGraph::new("identity_int");
        let vars = mint_vars(&mut graph, 2);
        let inputargs = block_inputargs(&vars, &[1]);
        let v1_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v1_var.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        setbinding(&v1_var, ValueType::Int);
        let (value_to_var, _constants) =
            specialize_legacy_graph(&graph).expect("identity int graph must specialize");

        assert_eq!(
            kind_of_in(&value_to_var, &vars[1]),
            ConcreteType::Signed,
            "Int-typed inputarg must specialize to Signed via SomeInteger → IntegerRepr"
        );
    }

    #[test]
    fn specialize_legacy_graph_minimal_float_identity_resolves_float() {
        let _lock = anchor_lock();
        let mut graph = LegacyGraph::new("identity_float");
        let vars = mint_vars(&mut graph, 2);
        let inputargs = block_inputargs(&vars, &[1]);
        let v1_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v1_var.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        setbinding(&v1_var, ValueType::Float);
        let (value_to_var, _constants) =
            specialize_legacy_graph(&graph).expect("identity float graph must specialize");

        assert_eq!(
            kind_of_in(&value_to_var, &vars[1]),
            ConcreteType::Float,
            "Float-typed inputarg must specialize to Float via SomeFloat → FloatRepr"
        );
    }

    #[test]
    fn specialize_legacy_graph_ref_typed_inputarg_resolves_to_gcref() {
        let _lock = anchor_lock();
        // `valuetype_to_someshell(Ref)` lifts to
        // `SomeInstance(classdef=None)` instead of the illegal
        // `SomeObject` placeholder (`model.py:51-69` `SomeObject` is
        // abstract).  The rtyper routes through `getinstancerepr(rtyper,
        // None, Gc)` -> `InstanceRepr::new_rootinstance` ->
        // `Ptr(GcStruct(OBJECT))` and `lowleveltype_to_concrete`
        // collapses any GC pointer to `ConcreteType::GcRef`, matching
        // legacy `resolve_types(Ref) -> GcRef`.
        let mut graph = LegacyGraph::new("identity_ref");
        let vars = mint_vars(&mut graph, 2);
        let inputargs = block_inputargs(&vars, &[1]);
        let v1_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v1_var.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        setbinding(&v1_var, ValueType::Ref(None));
        let (value_to_var, _constants) = specialize_legacy_graph(&graph)
            .expect("Ref-typed inputarg must specialize via SomeInstance(classdef=None)");
        assert_eq!(
            kind_of_in(&value_to_var, &vars[1]),
            ConcreteType::GcRef,
            "Ref-typed inputarg must project to GcRef matching legacy"
        );
    }

    #[test]
    fn cachedgraph_hit_registers_callee_graph_into_translator_graphs() {
        let _lock = anchor_lock();
        // Keystone linkage: after a real caller -> callee specialize,
        // the lifted callee `FunctionGraph` must live in the session
        // `translator.graphs`.  Upstream reaches that state because
        // `buildgraph -> buildflowgraph` appends every graph
        // (`translator.py:61`); pyre's lazy-lift prefills
        // `FunctionDesc.cache` via `lift_callee_to_pygraph` +
        // `prefill_default_cache`, bypassing `buildflowgraph`, so the
        // graph would otherwise never enter `translator.graphs`.
        // `FunctionDesc.cachedgraph`'s hit path restores the invariant
        // (description.rs:1058-1082).  This pins that the registration
        // actually fires through the shared-bookkeeper session and that
        // `funcobj.graph` (= the cached PyGraph's `graph`) is the exact
        // `Rc` now resolvable by the flowspace effect analyzers
        // (`canraise::RaiseAnalyzer` resolving `direct_call` through
        // `translator.graphs`, graphanalyze.rs) instead of falling to
        // `top_result()`.
        let mut graph = LegacyGraph::new("call_resolved");
        let vars = mint_vars(&mut graph, 3);
        let inputargs = block_inputargs(&vars, &[1]);
        let v1_var = vars[1].clone();
        let v2_var = vars[2].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![crate::model::SpaceOperation {
                result: Some(v2_var.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: vec!["foo".into()],
                    },
                    args: vec![v1_var.clone()],
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v2_var.clone())],
                graph.returnblock,
            )],
            dead: false,
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            dead: false,
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let bookkeeper = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let registry =
            crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(bookkeeper);

        // Leaf callee `fn foo(x: i64) -> i64 { x }` lifted through a
        // child registry; the entry's `FunctionDesc` (whose
        // `cachedgraph` is hit during the caller's specialize) is owned
        // by the session `registry` via `register_callee`, so its
        // `base.bookkeeper` is the bookkeeper `ensure_session` attaches
        // the annotator to.
        let mut callee_graph = LegacyGraph::new("foo");
        let callee_vars = mint_vars(&mut callee_graph, 11);
        let foo_inputargs = block_inputargs(&callee_vars, &[10]);
        let foo_v10_var = callee_vars[10].clone();
        let foo_start = Block {
            id: callee_graph.startblock,
            inputargs: foo_inputargs,
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(foo_v10_var.clone())],
                callee_graph.returnblock,
            )],
            dead: false,
            framestate: None,
        };
        let foo_return = Block {
            id: callee_graph.returnblock,
            inputargs: block_inputargs(&callee_vars, &[10]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            dead: false,
            framestate: None,
        };
        callee_graph.blocks = vec![foo_start, foo_return];
        let leaf_registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(
            std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new()),
        );
        setbinding(&foo_v10_var, ValueType::Int);
        let pygraph = lift_callee_to_pygraph(
            &callee_graph,
            crate::flowspace::argument::Signature::new(vec!["x".to_string()], None, None),
            &leaf_registry,
        )
        .expect("leaf callee must lift to PyGraph");
        // Capture the callee's flowspace graph `Rc` before the pygraph
        // is moved into `register_callee`; this is the identity that
        // `cachedgraph` clones into `translator.graphs` and that
        // `funcobj.graph` points at after rtyping.
        let callee_graph_ref = pygraph.graph.clone();
        registry.register_callee(
            crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments(["foo"]),
            crate::flowspace::argument::Signature::new(vec!["x".to_string()], None, None),
            pygraph,
        );

        setbinding(&v1_var, ValueType::Int);
        setbinding(&v2_var, ValueType::Int);
        specialize_legacy_graph_with_registry_returning_value_to_var(&graph, &registry)
            .expect("cache pre-fill must let the leaf Call resolve end-to-end");

        // `ensure_session` ran inside the specialize and wired the
        // annotator backlink onto the registry's shared bookkeeper
        // (`set_annotator`), so the diagnostic accessor must now
        // upgrade.
        let annotator = registry
            .bookkeeper()
            .try_annotator()
            .expect("specialize must have attached the session annotator");
        let graphs = annotator.translator.graphs.borrow();
        assert!(
            graphs
                .iter()
                .any(|g| std::rc::Rc::ptr_eq(g, &callee_graph_ref)),
            "the lifted callee graph must be registered into translator.graphs by the \
             cachedgraph hit path, so RaiseAnalyzer can resolve funcobj.graph through \
             translator.graphs instead of falling to top_result()"
        );
    }
    // There is no program-wide annotator pre-pass to test: per-session
    // annotator construction inside
    // `specialize_legacy_graph_with_registry_returning_value_to_var`
    // is the only production lifecycle.

    // ─── unsafe-fn stub helpers ───
    #[test]
    fn default_someshell_for_lltype_integer_family_yields_someinteger_no_const() {
        use crate::annotator::model::SomeValue;
        for ll in [
            LowLevelType::Signed,
            LowLevelType::Unsigned,
            LowLevelType::SignedLongLong,
            LowLevelType::UnsignedLongLong,
        ] {
            let s = default_someshell_for_lltype(&ll)
                .unwrap_or_else(|| panic!("{ll:?} must project to a SomeValue"));
            match s {
                SomeValue::Integer(ref si) => assert!(
                    si.base.const_box.is_none(),
                    "{ll:?}: SomeInteger must carry no const_box"
                ),
                other => panic!("{ll:?}: expected SomeInteger, got {other:?}"),
            }
        }
    }

    #[test]
    fn default_someshell_for_lltype_bool_yields_somebool_no_const() {
        use crate::annotator::model::SomeValue;
        let s = default_someshell_for_lltype(&LowLevelType::Bool)
            .expect("Bool must project to SomeBool");
        match s {
            SomeValue::Bool(b) => assert!(
                b.base.const_box.is_none(),
                "SomeBool must carry no const_box (ExtRegistryEntry parity)"
            ),
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn default_someshell_for_lltype_void_yields_somenone() {
        use crate::annotator::model::SomeValue;
        let s = default_someshell_for_lltype(&LowLevelType::Void)
            .expect("Void must project to SomeNone");
        assert!(matches!(s, SomeValue::None_(_)), "got {s:?}");
    }

    #[test]
    fn default_someshell_for_lltype_float_family_yields_somefloat_no_const() {
        use crate::annotator::model::SomeValue;
        for ll in [
            LowLevelType::Float,
            LowLevelType::SingleFloat,
            LowLevelType::LongFloat,
        ] {
            let s = default_someshell_for_lltype(&ll)
                .unwrap_or_else(|| panic!("{ll:?} must project to a SomeValue"));
            let const_present = match &s {
                SomeValue::Float(f) => f.base.const_box.is_some(),
                SomeValue::SingleFloat(f) => f.base.const_box.is_some(),
                SomeValue::LongFloat(f) => f.base.const_box.is_some(),
                other => panic!("{ll:?}: expected float family, got {other:?}"),
            };
            assert!(!const_present, "{ll:?}: float shell must have no const_box");
        }
    }

    #[test]
    fn default_someshell_for_lltype_address_yields_none() {
        // SomeAddress not yet ported (model.rs:21 TODO).
        assert!(default_someshell_for_lltype(&LowLevelType::Address).is_none());
    }

    #[test]
    fn build_stub_pygraph_carries_signature_and_links_to_returnblock() {
        use crate::annotator::model::SomeValue;
        let sig = Signature::new(vec!["obj".to_string()], None, None);
        let pygraph = build_stub_pygraph_for_unsafe_fn(
            "is_none".to_string(),
            sig.clone(),
            LowLevelType::Bool,
        )
        .expect("Bool return must produce a stub pygraph");
        assert_eq!(*pygraph.signature.borrow(), sig);
        let graph = pygraph.graph.borrow();
        assert_eq!(graph.name, "is_none");
        let start = graph.startblock.borrow();
        assert_eq!(
            start.inputargs.len(),
            1,
            "argname must map to a single inputarg"
        );
        assert_eq!(
            start.exits.len(),
            1,
            "stub must Link directly to returnblock"
        );
        let link = start.exits[0].borrow();
        assert_eq!(
            link.args.len(),
            1,
            "stub Link carries exactly the pre-annotated return Variable"
        );
        // The link's target must be the graph's returnblock.
        let link_target = link.target.as_ref().expect("Link must target a block");
        assert!(
            std::rc::Rc::ptr_eq(link_target, &graph.returnblock),
            "stub Link must target the graph's returnblock"
        );
        // The return arg must be a Variable carrying a const-free
        // SomeBool annotation (ExtRegistryEntry parity).
        let ret_arg = link.args[0]
            .as_ref()
            .expect("stub return arg must be Some(Hlvalue::Variable)");
        match ret_arg {
            Hlvalue::Variable(v) => {
                let ann = v.annotation.borrow();
                let s = ann.as_ref().expect("return Variable must be pre-annotated");
                match &**s {
                    SomeValue::Bool(b) => assert!(
                        b.base.const_box.is_none(),
                        "stub return SomeBool must not leak a const_box"
                    ),
                    other => panic!("expected SomeBool annotation, got {other:?}"),
                }
            }
            other => panic!("stub return must be Hlvalue::Variable, got {other:?}"),
        }
    }

    #[test]
    fn register_unsafe_fn_stubs_registers_each_spec_and_skips_compound_returns() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::pyre_call_registry::PyreCallRegistry;
        let bk = std::rc::Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let specs = vec![
            (
                vec!["pyobject".to_string(), "is_none".to_string()],
                Signature::new(vec!["obj".to_string()], None, None),
                LowLevelType::Bool,
            ),
            (
                vec!["pyobject".to_string(), "is_int".to_string()],
                Signature::new(vec!["obj".to_string()], None, None),
                LowLevelType::Bool,
            ),
            // Compound lltype — the stub-pygraph builder returns None
            // and register_unsafe_fn_stubs must skip.
            (
                vec!["pyobject".to_string(), "unsupported".to_string()],
                Signature::new(vec!["x".to_string()], None, None),
                LowLevelType::Struct(Box::new(
                    crate::translator::rtyper::lltypesystem::lltype::StructType::new("S", vec![]),
                )),
            ),
        ];
        register_unsafe_fn_stubs(&registry, &specs);
        assert_eq!(
            registry.len(),
            2,
            "compound-return spec must be skipped, others registered"
        );
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyobject".to_string(),
                    "is_none".to_string()
                ]))
                .is_some()
        );
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyobject".to_string(),
                    "is_int".to_string()
                ]))
                .is_some()
        );
    }

    /// Pin the layering invariant
    /// that the synthetic stub-pygraph is annotator-only.  The
    /// `register_unsafe_fn_stubs` helper writes the stub into the
    /// `PyreCallRegistry` cache but does NOT mutate any
    /// `CallControl::function_graphs` map; the codewriter's BFS walks
    /// `function_graphs.keys()` and resolves each call op's target via
    /// `target_to_path_and_graph` which requires the target to be
    /// present in `function_graphs`.  This test mirrors the
    /// `codewriter.rs:192-195` production call shape (registry +
    /// `callcontrol.unsafe_fn_stubs`) and asserts the path is reachable
    /// via the registry but not via `CallControl::function_graphs`.
    #[test]
    fn register_unsafe_fn_stubs_does_not_populate_callcontrol_function_graphs() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::jit_codewriter::call::CallControl;
        use crate::translator::rtyper::pyre_call_registry::PyreCallRegistry;
        let mut callcontrol = CallControl::new();
        callcontrol.unsafe_fn_stubs = vec![(
            vec!["pyobject".to_string(), "is_none".to_string()],
            Signature::new(vec!["obj".to_string()], None, None),
            LowLevelType::Bool,
        )];
        let bk = std::rc::Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        register_unsafe_fn_stubs(&registry, &callcontrol.unsafe_fn_stubs);
        // Registry side: the stub is present for `cachedgraph` lookups.
        let key = FunctionPathKey::from_segments(["pyobject".to_string(), "is_none".to_string()]);
        assert!(
            registry.lookup(&key).is_some(),
            "registry must carry the unsafe stub for annotator lookup"
        );
        // CallControl side: function_graphs is untouched, so
        // `find_all_graphs_for_tests` (which BFS-walks
        // `function_graphs.keys()`) cannot route to the stub.
        let cp = crate::parse::CallPath::from_segments(["pyobject", "is_none"]);
        assert!(
            !callcontrol.function_graphs().contains_key(&cp),
            "CallControl::function_graphs must NOT carry the unsafe stub path",
        );
        callcontrol.find_all_graphs_for_tests();
        assert!(
            !callcontrol.is_candidate(&cp),
            "find_all_graphs must not pick up an unsafe-stub-only path",
        );
    }

    #[test]
    fn register_unsafe_fn_stubs_does_not_overwrite_existing_entries() {
        // A path already registered via function_graphs (e.g. a safe fn
        // wearing the same segments) must NOT be overwritten by a stub.
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::pyre_call_registry::PyreCallRegistry;
        let bk = std::rc::Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let key = FunctionPathKey::from_segments(["pyobject".to_string(), "shared".to_string()]);
        let signature = Signature::new(vec!["x".to_string()], None, None);
        let existing = registry.get_or_register(key.clone(), signature.clone());
        let existing_host_id = existing.host_object.identity_id();
        let specs = vec![(
            vec!["pyobject".to_string(), "shared".to_string()],
            signature,
            LowLevelType::Bool,
        )];
        register_unsafe_fn_stubs(&registry, &specs);
        assert_eq!(registry.len(), 1, "no new entry expected");
        let after = registry.lookup(&key).expect("entry still present");
        assert_eq!(
            after.host_object.identity_id(),
            existing_host_id,
            "register_unsafe_fn_stubs must not replace an existing entry"
        );
    }

    #[test]
    fn build_stub_pygraph_returns_none_for_compound_lltype() {
        // Compound lltypes have no representable default; the helper
        // must return `None` so the caller skips registration.
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let func_ll = LowLevelType::Func(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::FuncType {
                args: vec![],
                result: LowLevelType::Void,
            },
        ));
        let result = build_stub_pygraph_for_unsafe_fn("synth".to_string(), sig, func_ll);
        assert!(result.is_none(), "Func lltype must surface as None");
    }
}
