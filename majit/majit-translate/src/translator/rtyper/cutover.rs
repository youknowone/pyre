//! `specialize_legacy_graph` cutover wrapper.
//!
//! Drives the real `RPythonTyper::specialize` against a pyre
//! `model::FunctionGraph` by way of the
//! [`crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace`]
//! adapter, then projects each per-`Variable` `LowLevelType` back to a
//! `ConcreteType` keyed by the original pyre slot (`usize`).
//!
//! ## Why this file is in `translator/rtyper/`
//!
//! The legacy algorithms live alongside this file as
//! `translator/rtyper/legacy_{annotator,resolve,pipeline}.rs`.  This
//! cutover module is the long-lived bridge: the dual-gate, the
//! default flip, and the prod migration all
//! call into the entry points defined here.  The `flowspace_adapter`
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
#[cfg(test)]
use crate::front;
use crate::jit_codewriter::type_state::ConcreteType;
use crate::model::FunctionGraph as LegacyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::flowspace_adapter::{FlowspaceAdapterOutput, SlotToVariable};
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
// `seed_callee_blocks` retired at Step 3 (2026-05-07).  The orthodox
// path `pycall -> recursivecall -> addpendingblock` from
// description.py:283-305 / annrpython.py:315-336 is now reachable
// through Step 2's `simple_call_SomeObject` registration; the
// program-wide `compute_at_fixpoint` loop discovers callees from
// subject call sites without any pre-seed.  The prior pre-seed was
// a workaround for the missing dispatch.

/// RAII guard that snapshots every live `legacy_graph.variable_at(slot).
/// annotation` cell and restores it on `Drop`, isolating the
/// dual-gate baseline's `legacy_annotator::annotate` writes from any
/// subsequent reader on the same graph.  See the `dual_gate_check`
/// doc for the failure mode this prevents.
struct LegacyAnnotationGuard {
    snapshot: Vec<(
        crate::flowspace::model::Variable,
        Option<Rc<crate::annotator::model::SomeValue>>,
    )>,
}

impl LegacyAnnotationGuard {
    fn snapshot(graph: &LegacyGraph) -> Self {
        let snapshot = graph
            .iter_variable_slots()
            .map(|(_, var)| (var.clone(), var.annotation.borrow().clone()))
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

/// Run `specialize_legacy_graph` and diff against `legacy_state`.
///
/// Returns `Err(message)` when:
///
/// - the real path errors out (typer error from an unported `OpKind`
///   arm), OR
/// - a legacy-known slot is missing / `Unknown` / different in
///   the real path, OR
/// - the real path produced a definite kind for a slot the legacy
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
    let real_state = match result {
        Ok(Ok((value_to_var, constants))) => project_value_to_var_to_map(&value_to_var, &constants),
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
    // real path's flowin so `graph.variable_at(vid.0).annotation` is
    // not pre-populated with the legacy walker's wider lift before
    // `seed_variable` runs (orthodox `_setbinding` monotonicity).
    // `legacy_resolve::resolve_types` writes `graph.concretetype` from
    // the post-publish `graph.variable.annotation` cells; the
    // comparison loop below reads `graph.concretetype` directly.
    //
    // The annotation guard snapshots every live
    // `legacy_graph.variable_at(vid.0).annotation` cell before the
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

    let mut divergences: Vec<String> = Vec::new();

    // Diff every legacy-defined value. Once this gate is used to prove
    // cutover parity, real-path `Unknown` for a legacy-known value is a
    // coverage bug, not success.
    //
    // The legacy walker's kinds are read from `legacy_graph.concretetype`:
    // `resolve_types` dual-writes every populated slot through
    // `FunctionGraph::set_concretetype_of_inline`, so the graph cells
    // carry the same view the retired `legacy_state` parameter used to
    // surface.
    let legacy_snapshot = legacy_graph.concretetype_snapshot();
    for (idx, legacy_kind) in legacy_snapshot.iter().enumerate() {
        if *legacy_kind == ConcreteType::Unknown {
            continue;
        }
        let real_kind = real_state.get(&idx).unwrap_or(&ConcreteType::Unknown);
        if real_kind != legacy_kind {
            divergences.push(format!(
                "slot {}: legacy={:?}, real={:?}",
                idx, legacy_kind, real_kind
            ));
        }
    }
    // Asymmetry direction: real should not produce a definite kind for
    // a slot the legacy resolver never resolved.
    for (idx, real_kind) in &real_state {
        let legacy_kind = legacy_graph.concretetype_at(*idx);
        if legacy_kind == ConcreteType::Unknown {
            divergences.push(format!(
                "slot {}: legacy={:?}, real={:?}",
                idx, legacy_kind, real_kind
            ));
        }
    }

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
/// `Match` carries the real path's `SlotToVariable` map (each
/// `Variable.concretetype` cell set by `RPythonTyper::specialize`) so
/// production callers consume it directly.  The real path is the
/// authoritative producer; legacy is the fallback for
/// Skip-classified graphs.  The per-graph divergence comparison
/// against legacy in
/// production; the legacy-baseline diff inside
/// `dual_gate_check_with_registry` itself was retired afterwards —
/// `Match` now means the real path succeeded, full stop.  Test-time
/// anchor invariants still run [`dual_gate_check`] for legacy-baseline
/// regression checks against hand-built fixtures.  PyPy
/// `codewriter.py:33` consumes the rtyper-produced graph directly,
/// with no dual-gate equivalent; pyre's `Skip` arm is the transitional
/// scaffolding that retires once every owning epic in
/// `is_known_unported`'s table converges.
#[derive(Debug)]
pub(crate) enum DualGateOutcome {
    /// Real path completed without panicking and (when the legacy
    /// walker also succeeded as defensive baseline) every legacy-
    /// known slot carried the same `ConcreteType` in the real
    /// path's projection.  Production consumes `real_state`
    /// authoritatively.  Per-slot diff against the legacy
    /// baseline runs whenever the legacy walker can produce a
    /// state — divergence routes the graph through
    /// `Skip("dual-gate divergence: ...")` so the codewriter falls
    /// back to the legacy walker output for the affected graph.  PyPy
    /// `codewriter.py:33` consumes the rtyper-produced graph
    /// directly with no comparison stage; the legacy-baseline diff
    /// is pyre-only scaffolding that retires once the legacy walker
    /// itself retires.
    Match {
        /// `slot → flowspace::Variable` (`SlotToVariable =
        /// HashMap<usize, Variable>`) mapping built by the
        /// flowspace adapter.  Each Variable carries the
        /// `RPythonTyper`-set `concretetype` inline (`flowspace/
        /// model.py:280`), so codewriter callers rebind each slot to
        /// the upstream-typed Variable via
        /// [`crate::jit_codewriter::type_state::apply_from_flowspace_variables`];
        /// `FunctionGraph::concretetype_of(&v)` then reads its
        /// `concretetype` cell directly.
        real_value_to_var: SlotToVariable,
    },
    /// Real path failed on a known-unported feature — the gate
    /// cannot validate this graph yet but the failure is *not* a
    /// ConcreteType divergence.  Categories include:
    ///
    /// - `OpKind::Call::FunctionPath { segments }` not in the
    ///   registry (cross-crate / primitive paths the production
    ///   walker doesn't reach yet).
    /// - `undefined operand slot` from cross-block locals
    ///   threading (Cat 3.2 deferred).
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
///   real path's `SlotToVariable` map (with each
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
) -> Result<DualGateOutcome, String> {
    // Same panic-catch contract as `dual_gate_check` — the rtyper's
    // internal `genop`/`level` asserts surface as diagnostic panics
    // for unported pyre-front idioms; the gate uniformly returns a
    // stringified error so the env-flag wrapper can decide whether
    // to panic, log, or skip.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        specialize_legacy_graph_with_registry_returning_value_to_var(legacy_graph, call_registry)
    }));
    let (real_value_to_var, real_constants) = match result {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => {
            let msg = format!("{e}");
            if is_known_unported(&msg) {
                return Ok(DualGateOutcome::Skip(msg));
            }
            return Err(format!("real path failed: {msg}"));
        }
        Err(payload) => {
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

/// Local projection of `value_to_var` Variables' concretetype cells
/// into a `BTreeMap<usize, ConcreteType>` keyed by the dense slot
/// index.  Errors from unsupported lltypes are silently treated as
/// missing entries — `specialize_legacy_graph` already propagates the
/// failure to its caller before reaching the dual-gate, so reaching
/// here implies every Variable's lltype is projectable.  BTreeMap
/// iteration is ascending-slot-index so anchor-test "first
/// divergence" messages stay deterministic across runs.
///
/// Survives as a [`dual_gate_check`] test helper after the production
/// `dual_gate_check_with_registry` baseline strip — anchor tests still
/// diff the real path against the legacy walker for hand-built
fn project_value_to_var_to_map(
    value_to_var: &SlotToVariable,
    constant_concretetypes: &HashMap<usize, LowLevelType>,
) -> std::collections::BTreeMap<usize, ConcreteType> {
    let mut real_state = std::collections::BTreeMap::new();
    for (&idx, var) in value_to_var {
        if let Some(lltype) = var.concretetype().as_ref() {
            if let Ok(kind) = lowleveltype_to_concrete(lltype) {
                real_state.insert(idx, kind);
            }
        }
    }
    // `Constant.concretetype` is the ground truth for constant operands
    // — read it from the adapter's per-slot map rather than attempting
    // to reconstruct from the reduced legacy `ValueType` view.
    for (&idx, lltype) in constant_concretetypes {
        if let Ok(kind) = lowleveltype_to_concrete(lltype) {
            real_state.insert(idx, kind);
        }
    }
    real_state
}

fn compare_real_against_legacy(
    value_to_var: &SlotToVariable,
    constants: &HashMap<usize, LowLevelType>,
    legacy_graph: &LegacyGraph,
) -> Option<String> {
    let real_state = project_value_to_var_to_map(value_to_var, constants);
    let legacy_snapshot = legacy_graph.concretetype_snapshot();
    for (idx, legacy_kind) in legacy_snapshot.iter().enumerate() {
        if *legacy_kind == ConcreteType::Unknown {
            continue;
        }
        let real_kind = real_state.get(&idx).unwrap_or(&ConcreteType::Unknown);
        if real_kind != legacy_kind {
            return Some(format!(
                "slot {}: legacy={:?}, real={:?}",
                idx, legacy_kind, real_kind
            ));
        }
    }
    for (idx, real_kind) in &real_state {
        let legacy_kind = legacy_graph.concretetype_at(*idx);
        if legacy_kind == ConcreteType::Unknown {
            return Some(format!(
                "slot {}: legacy={:?}, real={:?}",
                idx, legacy_kind, real_kind
            ));
        }
    }
    None
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
/// **TODO** (audit 2.D, 2026-05-05): each remaining entry is blocked
/// on a specific porting epic; retiring an entry early is unsafe
/// until the owning work converges:
///
/// | Substring                                  | Owning work / followup                                                      |
/// |--------------------------------------------|------------------------------------------------------------------------------|
/// | `not registered in PyreCallRegistry`       | Extern Rust helper registry walker (blocked).                                |
/// | `undefined operand slot`                   | Adapter producer correctness audit.                                          |
/// | `unimplemented operation`                  | Per-opname rtyper handler ports.                                             |
/// | `variable used before definition`          | Cross-block locals threading.                                                |
/// | `MissingRTypeAttribute`                    | Typed-Ref → SomeInstance(ClassDef).                                          |
/// | `KeyError: no binding for arg`             | Annotator gap audit (per-call coverage).                                     |
/// | `compute_at_fixpoint failed`               | PBC dispatch / call-family coverage (per-call).                              |
/// | `post-rtyper jtransform variant`           | Per-variant emit-site retracing (rpbc / rclass / front-end).  Includes `OpKind::Abort` (pyre-only marker; retire every `Expr::ForLoop` / `stop_unsupported` / `continue_with_unknown` emit-site at the front-end). |
/// | `adapter cross-block body Input`           | Final emit-site retirement.                                                  |
///
/// PyPy `bookkeeper.py:108-127` propagates fixpoint failures uncaught;
/// pyre's dual-gate Skip is the cutover-time scaffolding that defers
/// exactly the categories enumerated above.  Once every owning epic
/// converges, every match returns `false`, the legacy walker fallback
/// at `codewriter.rs::dual_gate_type_state` becomes dead code, and
/// the predicate retires entirely.
pub(crate) fn is_known_unported(msg: &str) -> bool {
    msg.contains("not registered in PyreCallRegistry")
        || msg.contains("undefined operand slot")
        || msg.contains("unimplemented operation")
        || msg.contains("variable ")
            && msg.contains(" used before definition")
        // (Retired 2026-05-05.) `normalize_unary_op_name: pyre UnaryOp`
        // and `normalize_binop_name: pyre BinOp` were Skip-classified
        // when production still emitted `not` / `deref` / `same_as`
        // and `and` / `or`.  Both surfaces are now desugared upstream
        // — `UnOp::Not`/`Deref` at `front/ast.rs::Expr::Unary` +
        // `flowspace/rust_source/build_flow.rs::lower_unary_not`,
        // `&&`/`||` at the matching `Expr::Binary` arms.  `invert`
        // landed in `normalize_unary_op_name` 2026-05-05.  Synthetic
        // graphs that inject these ops (anchor tests in
        // `cutover.rs::tests::anchor_unary_*_surfaces_*`) call
        // `specialize_legacy_graph` directly and never reach
        // `is_known_unported`, so removing these substring matches
        // does not affect them.  Any production reach surfaces as a
        // dual-gate divergence panic — the parity-correct outcome.
        // Field / method dispatch on a `SomeInstance(classdef=None)`
        // — pyre's `Ref` ValueType currently lifts to a classdef-less
        // SomeInstance (Cat 3.1 placeholder), so `find_attribute`
        // (`rclass.py:556+find_attribute_or_None`) cannot route the
        // dispatch.  Cat 3.1.A's `InstanceRepr::rtype_getattr` port
        // (`rclass.py:838-857`) routes through `getclsfield`, which
        // surfaces the upstream-orthodox `MissingRTypeAttribute(attr)`
        // when find_attribute returns None.  The companion
        // `"no method ... on Instance("` substring (`rmodel.rs:828`
        // default `rtype_getattr` find_method failure path) was
        // retired 2026-05-06: post-Cat-3.1.A every SomeInstance
        // dispatch goes through the InstanceRepr override, so the
        // default fn never fires for `Instance(...)`-shaped operands.
        // The MissingRTypeAttribute entry stays until typed-Ref
        // ClassDef projection lands and field/method dispatch starts
        // succeeding.
        || msg.contains("MissingRTypeAttribute")
        // Variable's `.annotation` slot empty at `bindingrepr`
        // lookup time — `ValueType::Unknown` has no
        // annotation-stage shell and `valuetype_to_someshell` returns
        // `None` for it (intentionally fail-loud for "annotation gap"
        // so producers know which slot missed seeding, see
        // `seed_variable` at `flowspace_adapter.rs:96-115`).  Closing
        // the gap means
        // tightening the front-end / annotator producers so every
        // slot has a non-`Unknown` annotation; the gate skips
        // until then.
        || msg.contains("KeyError: no binding for arg")
        // TODO(annotator-fixpoint-fail-loud) — STRICT-PARITY REGRESSION
        // vs main / PyPy.  `bookkeeper.py:108-127` propagates fixpoint
        // exceptions uncaught and `annrpython.py:643` lets
        // `AnnotatorError` reach the caller; absorbing the four
        // patterns below is a deviation that hides real
        // annotator/rtyper parity gaps as "known unported".  Direct
        // removal verified 2026-05-12 to break `pyre-jit-trace`
        // `build.rs` at every reachable hitter
        // (`make_green_key`, `Frame::load_fast`, `PyFrame::locals_w_mut`,
        // `<default methods of IterOpcodeHandler>::record_for_iter_guard`,
        // `pyjitpl_step::Cannot find attribute`); production cannot
        // compile until each underlying real-path gap is closed
        // (classdef-less SomeInstance dispatch + `PyreCallRegistry::
        // ensure_session` coverage — the builtin-analyser binding-
        // failure reorder converged 2026-05-12 by routing every
        // analyser body through per-touch `arg_at` instead of the
        // retired `args_s_concrete_or_panic` eager prefix).  Multi-session epic — until each gap-hitter is
        // ported, this Skip stays as documented divergence and the
        // codewriter falls back to the legacy walker for production.
        || msg.contains("compute_at_fixpoint failed")
        || msg.contains("complete_pending_blocks failed")
        || msg.contains("Cannot find attribute ")
        || msg.contains("AnnotatorError:")
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
        // (Retired 2026-05-06.) `AnnotatorError: immutablevalue(HostObject`
        // was Skip-classified for `SyntheticTransparentCtor` (Ok/Err/Some/
        // None) — the adapter wrapped the ctor name as
        // `HostObject::new_opaque(name)` which has no `is_class()` /
        // `is_user_function()` classification, so `immutablevalue` fell
        // through to "Don't know how to represent".  The adapter now
        // emits `HostObject::new_class(name, [])` instead
        // (`flowspace_adapter.rs:821 SyntheticTransparentCtor` arm),
        // routing through the existing `is_class()` arm in
        // [`crate::annotator::bookkeeper::Bookkeeper::immutablevalue_hostobject`]
        // (`bookkeeper.py:315-316` parity).  No production emit-site of
        // `HostObject::new_opaque` remains in `front/`/`translator/rtyper/`,
        // so any future `immutablevalue(HostObject` failure is a real
        // parity bug — let it surface as a dual-gate divergence panic.
        // TODO(post-rtyper-jtransform-variant-leak): retire this Skip
        // entry per upstream parity — `jit/codewriter/jtransform.py`
        // raises straight through on unexpected opnames.  Today the
        // most frequent reach is `OpKind::Abort` emitted by
        // `front/ast.rs::stop_unsupported` / `continue_with_unknown`
        // when the surface DSL hits an unsupported expression — pyre
        // source like `execute_opcode_step` / `eval_loop_jit` /
        // `build_blackhole_frames_from_deadframe` all carry such
        // placeholders today.  Convergence path: retire each Abort
        // emit-site at the front-end (per-variant epic — closure
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
        // is itself a TODO (`front/ast.rs:2127-2156`); when Cat 2.1
        // cross-block locals threading misses a shape, the adapter
        // now fails loud with this message instead of silently
        // fabricating a fresh Variable (which would hide an
        // SSA / alias-shape divergence from PyPy's flowspace).  Skip
        // until either Cat 2.1 covers every shape, or the front-end's
        // body-`Input` emission is replaced by a Link.args / inputargs
        // threading pass that mirrors RPython.
        || msg.contains("adapter cross-block body Input")
    // `normalize_unary_op_name: pyre UnaryOp` Skip entry retired
    // 2026-05-06 + 2026-05-19: the 13 typed numeric / ptr / Unsigned
    // casts now route through `simple_call(<host_callable>, v)` —
    // reaching `Repr.rtype_int/float/bool` or
    // `BUILTIN_TYPER[lltype.cast_*]` directly.  Only the `same_as`
    // identity / source-type-unknown fallback remains on the
    // `OpKind::UnaryOp` route, dispatched by
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
/// end emitted (`canonical_call_target`, `front/ast.rs:3497-3528`),
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
    function_graphs: &std::collections::HashMap<crate::parse::CallPath, LegacyGraph>,
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
    for (path, graph) in function_graphs {
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
    Ok(())
}

// build_program_annotator / build_program_rtyper / ensure_program_
// specialize were retired at Step 4 first slice (2026-05-07).  They
// existed solely to drive a program-wide `compute_at_fixpoint` +
// `RPythonTyper::specialize` over a pre-seeded annotator (Step 3
// retired the seed; the empty annotator made the pass a no-op).
// The matching `ProgramSpecializeState` flag and its accessors were
// retired alongside.  Per-session
// `specialize_legacy_graph_with_registry_returning_value_to_var`
// now runs the
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
                .slot_of(var)
                .and_then(|slot| graph.value_name_at(slot))
                .map(str::to_string)
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
    // side maps, but every legacy consumer of those was a dead-write
    // path into `PyreCallRegistry` (Issue 2.5 retirement, 2026-05-07).
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

/// Z2.5 Path C slice 2 — synthesize a minimal flowed `PyGraph` for a
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
/// `SomeValue` enum — see `model.rs:21` TODO).  The slice 3c caller
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

/// Z2.5 Path C slice 3c — register a batch of unsafe-fn stub
/// `(segments, signature, return_lltype)` specs into `registry`.
/// Each entry is wrapped through [`build_stub_pygraph_for_unsafe_fn`]
/// + [`PyreCallRegistry::register_callee`], so subsequent
/// `flowspace_adapter::translate_op` lookups via
/// `call_registry.lookup_with_leaf_match` find a registered entry and
/// the dual gate no longer Skips with "not registered in
/// PyreCallRegistry" for these paths.
///
/// `specs` is typically the output of
/// `register::extract_unsafe_fn_stubs(file, prefix)` run over every
/// parsed source file.  Per-fn failures (stub-pygraph builder returns
/// `None` for compound lltypes, or registry already has the same key
/// at a conflicting signature) propagate as silent skips — the
/// upstream "not registered" Skip path then absorbs that specific fn
/// while the rest of the batch lands.
///
/// Mirrors `populate_call_registry_from_call_graphs`'s
/// "register and prefill" contract (`cutover.rs:856-875`) but feeds
/// from the syn-AST stub-spec list instead of pyre's
/// `function_graphs: HashMap<CallPath, LegacyGraph>` (which excludes
/// unsafe fns by validate_signature rejection).
///
/// **Annotator-only carrier — never reaches the codewriter.**
/// The stub graph carries a single default-Constant return link
/// suitable for `RPythonAnnotator` return-type inference via
/// `cachedgraph` (see `pyre_call_registry::prefill_default_cache`).
/// Because `CallControl::function_graphs` is populated exclusively
/// by safe-fn `build_flow` output (unsafe fns are rejected at
/// `flowspace/rust_source/build_flow.rs:215`), the unsafe stub key
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
/// to the host-evaluator entry point.  Reviewer 2026-05-24 round
/// item #5 audited this layering; the default-Constant return is
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

/// Derive the canonical `FunctionPathKey` for a `SemanticFunction`.
///
/// Mirrors the front-end's `canonical_call_target`
/// (`front/ast.rs:3497-3528`) inverted: what an `Expr::Path`
/// callsite emits is what we register the callee under.
///
/// Both `func.name` (e.g. `"a::helper"` after `build_graphs_from_items`
/// prepends module prefix at `front/ast.rs:630-632`) and
/// `func.self_ty_root` (e.g. `"a::Foo"` after `qualify_type_name` at
/// `:638`) carry `::`-joined module-qualified strings.  Each
/// `::`-separated component is one `FunctionPathKey` segment — the
/// callsite at `front/ast.rs:3827-3835` produces `["a", "helper"]`
/// (multi-segment, no `::` in any segment), so the registered key
/// must split each `::` boundary too.  Without the split, registry
/// entry `["a::helper"]` (single segment containing `::`) would never
/// match callsite lookup `["a", "helper"]`.
///
/// PyPy's `Bookkeeper.getdesc(<function object>)` (`bookkeeper.py:353-409`)
/// keys on Python function-object identity rather than path strings,
/// so the upstream chain has no segment-shape divergence to worry
/// about.  Pyre's segment-key approach is a stand-in for that
/// identity until host callable identity is available; aligning the
/// segment shape both sides of the registry is a prerequisite for
/// that stand-in to function correctly.
#[cfg(test)]
fn function_path_key_for(func: &front::SemanticFunction) -> FunctionPathKey {
    let mut segments: Vec<String> = Vec::new();
    if let Some(t) = &func.self_ty_root {
        segments.extend(t.split("::").map(str::to_string));
    }
    segments.extend(func.name.split("::").map(str::to_string));
    FunctionPathKey::from_segments(segments)
}

/// Derive the parameter `Signature` for a `SemanticFunction` from
/// the startblock's input arg names.
///
/// Mirrors upstream `bookkeeper.py:418
/// cpython_code_signature(pyfunc.__code__)`: the parameter names
/// come from the function's declared parameter list.  Pyre carries
/// these as `value_name(inputarg)` on the startblock; missing names
/// fall back to `arg{N}` to keep the `FunctionDesc.signature.argnames`
/// length matched to the actual parameter count.
#[cfg(test)]
fn signature_for(func: &front::SemanticFunction) -> Signature {
    let graph = &func.graph;
    let startblock = graph.block(graph.startblock);
    let argnames: Vec<String> = startblock
        .inputargs
        .iter()
        .enumerate()
        .map(|(idx, var)| {
            graph
                .slot_of(var)
                .and_then(|slot| graph.value_name_at(slot))
                .map(str::to_string)
                .unwrap_or_else(|| format!("arg{idx}"))
        })
        .collect();
    Signature::new(argnames, None, None)
}

/// Walk a `SemanticProgram` and pre-register every
/// reachable `SemanticFunction` in the call registry.
///
/// Closes Issue 2.1 from the post-A.4 audit: production callers
/// with `OpKind::Call::FunctionPath` ops cannot proceed through the
/// real-rtyper path without the registry pre-populated, because
/// pyre's surface DSL has no Python callable object for the
/// upstream `Constant(<function>) -> getdesc -> FunctionDesc` chain
/// (`bookkeeper.py:353-409`).  This walker registers each
/// `SemanticFunction` so any `simple_call` the adapter emits later
/// finds a matching entry.
///
/// Two passes are required because the per-function lift
/// (`lift_callee_to_pygraph`) calls
/// `function_graph_to_flowspace`, which runs the
/// `OpKind::Call::FunctionPath` arm in `translate_op` and expects
/// every callee path the body references to already be registered.
/// Pass 1 installs `(HostObject, FunctionDesc)` for every program
/// function (signature only, no PyGraph cache).  Pass 2 lifts each
/// function's body and prefills its `FunctionDesc.cache` so the
/// rtyper's `cachedgraph` (`description.rs:1037-1039`) hits at
/// `direct_call` time.
///
/// Annotation seeding lives inside [`function_graph_to_flowspace`]
/// inside `function_graph_to_flowspace`; this walker plumbs only the program + registry.
#[cfg(test)]
pub(crate) fn populate_call_registry_from_program(
    program: &front::SemanticProgram,
    registry: &PyreCallRegistry,
) -> Result<(), TyperError> {
    // Pass 1 — register every function with its signature; the
    // FunctionDesc.cache stays cold until Pass 2.
    let mut pending: Vec<(
        FunctionPathKey,
        &front::SemanticFunction,
        Rc<PyreFunctionEntry>,
    )> = Vec::with_capacity(program.functions.len());
    for func in &program.functions {
        let key = function_path_key_for(func);
        let signature = signature_for(func);
        let entry = registry.get_or_register(key.clone(), signature);
        pending.push((key, func, entry));
    }
    // Pass 2 — lift each function's body using the now-populated
    // registry, then prefill its FunctionDesc.cache.  Lift order is
    // independent of function topology: every callsite's
    // `simple_call` resolves through `registry.lookup` against the
    // Pass-1-installed HostObjects, regardless of whether the
    // callee's own body has been lifted yet.
    for (_key, func, entry) in pending {
        let pygraph = lift_callee_to_pygraph(&func.graph, signature_for(func), registry)?;
        entry.prefill_default_cache(pygraph);
    }
    Ok(())
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
) -> Result<(SlotToVariable, HashMap<usize, LowLevelType>), TyperError> {
    let registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(Rc::new(
        crate::annotator::bookkeeper::Bookkeeper::new(),
    ));
    specialize_legacy_graph_with_registry_returning_value_to_var(legacy, &registry)
}

/// `specialize_legacy_graph_with_registry` extended return shape that
/// also surfaces the [`SlotToVariable`] map and the per-slot
/// `Constant.concretetype` table produced by the flowspace adapter.
/// Codewriter callers consume `value_to_var` directly (the rtyper's
/// `Variable.concretetype` writes propagate via
/// `FunctionGraph::set_concretetype_of_inline` at the dual-gate `Match` arm);
/// `constants` feeds [`project_value_to_var_to_state`] for the
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
) -> Result<(SlotToVariable, HashMap<usize, LowLevelType>), TyperError> {
    // RPython parity path.
    //
    // Upstream `RPythonTyper.specialize` runs ONCE per `Translator`,
    // not per graph.  Pyre's per-graph dual-gate enters this function
    // once per graph (one per `transform_graph_to_jitcode`), so the
    // upstream "specialize-once" semantics are reproduced by lazily
    // driving the program-wide pass on first entry and gating
    // `seed_callee_blocks` on the resulting state.  After the
    // program-wide pass succeeds, the cached PyGraphs hold
    // post-specialize LL ops; re-seeding them into the per-session
    // annotator would let `specialize_more_blocks` walk the LL ops a
    // second time and trip on `unimplemented operation: 'int_add'` /
    // `'direct_call'` — see memory
    // `skip_unimplemented_int_add_diagnosis_2026_05_06`.
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
    // Step 4 second slice (2026-05-07): the annotator + rtyper are
    // now session-shared through `PyreCallRegistry::ensure_session`,
    // mirroring RPython's "one annotator + one rtyper per Translator"
    // (`translator.py:69-83`).  Each per-graph subject seeds its
    // blocks into the shared annotator; subsequent
    // `specialize_more_blocks` rtype only the newly-added blocks.
    let (annotator, rtyper) = call_registry.ensure_session()?;
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
    {
        let bkey = crate::flowspace::model::BlockKey::of(&exceptblock);
        annotator
            .annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(graph.clone()));
        annotator.all_blocks.borrow_mut().insert(bkey, exceptblock);
    }

    // Step 3 (2026-05-07): the prior per-session `seed_callee_blocks`
    // call was a workaround for the missing `simple_call_SomeObject`
    // registration — without that registration, flowin trips on the
    // first `simple_call(host_object_const, args)` op with
    // "no unary spec for SimpleCall(PBC)".  Pre-seeding callee blocks
    // bypassed the trip but skipped the orthodox `pycall ->
    // recursivecall -> addpendingblock` chain entirely.
    //
    // Step 2 added `@op.simple_call.register(SomeObject)` (unaryop.py:
    // 114-118 parity) so flowin now dispatches through `s_func.call(
    // argspec)` -> `SomePBC.call` -> `Bookkeeper.pbc_call` ->
    // `FunctionDesc.pycall` -> `annotator.recursivecall`, which calls
    // `addpendingblock(graph, startblock, inputcells)` for the
    // callee.  The orthodox path now seeds callee blocks naturally;
    // the explicit pre-seed is no longer needed.

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
    // The drain runs in BOTH Done and Failed branches:
    //
    // - Done: program-wide drained the cached graphs' callsites, but
    //   the subject we are about to specialize is brand-new (its
    //   freshly-lifted blocks were just seeded into this annotator)
    //   and its `simple_call` ops have not yet been processed.  The
    //   subject's call_sites entries on the bookkeeper are processed
    //   here so its callsites' rows land in
    //   `pbc_maximal_call_families` before specialize tries to look
    //   them up.
    // - Failed / Pending-skipped: legacy entry — drains every
    //   reachable callsite (subject + seeded callees) for the first
    //   time.
    //
    // Errors propagate verbatim (Issue 3.1) — upstream
    // `bookkeeper.py:108-118` runs the call-site walk without a
    // `try`/`except`, so a failed `consider_call_site` terminates
    // `simplify` and unwinds out of the annotator driver.  Pyre's
    // port surfaces the same condition through `?`-propagation
    // here, replacing the prior `let _ = ...` swallow at
    // `bookkeeper.rs:627-648` that masked PBC dispatch failures.
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
    rtyper.specialize_more_blocks()?;

    // ── Step 4 — validate per-slot lltype projection ──────────────
    //
    // The eager side-table build was retired and the struct itself
    // deleted.  Each callable that needs
    // a kind view derives one on demand from `value_to_var` +
    // `constant_concretetypes` via
    // [`project_value_to_var_to_state`] — same `Variable.concretetype`
    // / `Constant.concretetype` ground truth, just routed at the
    // consumer instead of eagerly materialised here.  Run a
    // validation pass so unsupported lltypes still surface as a
    // [`TyperError`] at this boundary (mirroring the earlier
    // fail-loud `?` propagation in the state-build loop).
    for var in value_to_var.values() {
        let cell = var.concretetype.borrow();
        if let Some(lltype) = cell.as_ref() {
            lowleveltype_to_concrete(lltype)?;
        }
    }
    for lltype in constant_concretetypes.values() {
        lowleveltype_to_concrete(lltype)?;
    }

    Ok((value_to_var, constant_concretetypes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Block, BlockId, LinkArg, ValueType};
    use crate::translator::rtyper::legacy_annotator::setbinding;

    fn link_to_returnblock(args: Vec<LinkArg>, returnblock_id: BlockId) -> crate::model::Link {
        crate::model::Link::new_mixed(args, returnblock_id, None)
    }

    /// Test helper — read the concrete kind for the slot at `idx`
    /// directly from the post-rtyper `SlotToVariable` map's
    /// `Variable.concretetype` cell (`flowspace/model.py:280`).
    /// Replaces the retired `TypeResolutionState::get(vid)` path.
    fn kind_of_in(value_to_var: &SlotToVariable, idx: usize) -> ConcreteType {
        let var = value_to_var
            .get(&idx)
            .unwrap_or_else(|| panic!("slot {idx} missing from value_to_var"));
        let ll = var
            .concretetype()
            .unwrap_or_else(|| panic!("Variable.concretetype for slot {idx} not populated"));
        lowleveltype_to_concrete(&ll)
            .unwrap_or_else(|_| panic!("lltype for slot {idx} does not project to ConcreteType"))
    }

    /// Test helper — project slot indices to their backing Variables
    /// on the graph so a `Block { inputargs: ..., .. }` struct literal
    /// can carry the upstream-orthodox `Vec<Variable>` shape.
    /// Auto-grows the graph via `set_next_value` when an index past
    /// the canonical 3 slots is referenced so each has a backing
    /// Variable registered in `variable_to_vid`.
    fn block_inputargs(
        graph: &mut LegacyGraph,
        vids: &[usize],
    ) -> Vec<crate::flowspace::model::Variable> {
        if let Some(max) = vids.iter().copied().max() {
            if max >= graph.next_value() {
                graph.set_next_value(max + 1);
            }
        }
        vids.iter().map(|v| graph.must_variable_at(*v)).collect()
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
        let msg = "translate_op: Call with CallTarget::Indirect at result=Some(slot 4) \
                   must be lowered to VtableMethodPtr + IndirectCall by rclass.rs before \
                   reaching the flowspace adapter";
        assert!(
            is_known_unported(msg),
            "registry population must skip/fallback on the current indirect-call cutover gap"
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

    #[test]
    fn specialize_legacy_graph_minimal_int_identity_resolves_signed() {
        let _lock = anchor_lock();
        // Smallest validation: identity-return graph carrying a single
        // Int-typed inputarg. Dual-gate's first anchor — proves
        // the adapter + annotator-surface seeding + specialize +
        // projection chain works end-to-end on a graph the rtyper can
        // resolve without any unported OpKind variants.
        let mut graph = LegacyGraph::new("identity_int");
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
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
            inputargs: block_inputargs(&mut graph, &[1]),
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
            kind_of_in(&value_to_var, 1),
            ConcreteType::Signed,
            "Int-typed inputarg must specialize to Signed via SomeInteger → IntegerRepr"
        );
    }

    #[test]
    fn specialize_legacy_graph_minimal_float_identity_resolves_float() {
        let _lock = anchor_lock();
        let mut graph = LegacyGraph::new("identity_float");
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
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
            inputargs: block_inputargs(&mut graph, &[1]),
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
            kind_of_in(&value_to_var, 1),
            ConcreteType::Float,
            "Float-typed inputarg must specialize to Float via SomeFloat → FloatRepr"
        );
    }

    #[test]
    fn specialize_legacy_graph_ref_typed_inputarg_resolves_to_gcref() {
        let _lock = anchor_lock();
        // Cat 3.1 fix: `valuetype_to_someshell(Ref)` now lifts to
        // `SomeInstance(classdef=None)` instead of the illegal
        // `SomeObject` placeholder (`model.py:51-69` `SomeObject` is
        // abstract).  The rtyper routes through `getinstancerepr(rtyper,
        // None, Gc)` -> `InstanceRepr::new_rootinstance` ->
        // `Ptr(GcStruct(OBJECT))` and `lowleveltype_to_concrete`
        // collapses any GC pointer to `ConcreteType::GcRef`, matching
        // legacy `resolve_types(Ref) -> GcRef`.
        let mut graph = LegacyGraph::new("identity_ref");
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
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
            inputargs: block_inputargs(&mut graph, &[1]),
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
            kind_of_in(&value_to_var, 1),
            ConcreteType::GcRef,
            "Ref-typed inputarg must project to GcRef matching legacy"
        );
    }

    // ───── dual-gate anchor tests ─────────────────────────────────
    //
    // Each anchor parses a small Rust DSL fragment, runs pyre's
    // `parse → front → SemanticProgram → FunctionGraph` chain, then
    // calls the legacy resolver and `dual_gate_check`. The expected
    // outcome — Ok or Err with a named OpKind / blocker — is asserted
    // so future followups (BinOp, FieldRead, ...) flip the assertion
    // automatically when their arm lands.
    //
    // Anchor selection covers (a) the trivially-resolvable identity
    // cases that already pass today, and (b) representative shapes
    // from the existing `pipeline_e2e_*` suite that surface the next
    // priority OpKind ports.

    /// Process-local guard serialising every anchor test that drives
    /// `RPythonTyper::specialize` end-to-end.
    ///
    /// The rtyper's lattice has process-global singletons (`bool_repr`,
    /// `none_repr`, the Repr setup-state machine) that mutate during
    /// `setup()`. Cargo's default parallel test runner can interleave
    /// two specialize-driving tests so one observes the other's
    /// `Setupstate::InProgress` and panics with "recursive invocation
    /// of Repr setup()" (`rmodel.rs:427`).
    ///
    /// Holding this `Mutex` for the entire specialize-driving section
    /// of each anchor serialises the runs without forcing
    /// `--test-threads=1` on the whole crate. If the rtyper's setup
    /// machinery is later reworked to be re-entrant, this lock can go
    /// away.
    fn anchor_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn build_anchor_graph(source: &str, fn_name: &str) -> LegacyGraph {
        let parsed = crate::parse::parse_source(source);
        let program = crate::front::build_semantic_program(&parsed)
            .expect("anchor source must build a semantic program");
        program
            .functions
            .iter()
            .find(|f| f.name == fn_name)
            .unwrap_or_else(|| {
                panic!(
                    "anchor source must define `{fn_name}`; got functions: {:?}",
                    program
                        .functions
                        .iter()
                        .map(|f| &f.name)
                        .collect::<Vec<_>>()
                )
            })
            .graph
            .clone()
    }

    fn run_legacy_resolve(_graph: &LegacyGraph) {
        // No-op since `dual_gate_check` runs its own baseline (the
        // legacy walker after the real path) — pre-warming
        // `graph.concretetype` here would write the wider legacy lift
        // onto `graph.variable.annotation` BEFORE the real path's
        // `seed_variable` runs and trip the orthodox `_setbinding`
        // monotonicity check.  Kept as a no-op adapter so the call
        // sites below can be retired without a separate sweep.
    }

    #[test]
    fn anchor_int_identity_dual_gate_agrees() {
        let _lock = anchor_lock();
        let graph = build_anchor_graph("fn id(x: i64) -> i64 { x }\n", "id");
        run_legacy_resolve(&graph);
        // Trivial identity — both paths must produce the same kinds.
        // Currently legacy resolves the inputarg as Signed; real path
        // produces the same via SomeInteger → IntegerRepr → Signed.
        let result = dual_gate_check(&graph);
        assert!(
            result.is_ok(),
            "Int identity must agree under dual-gate, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_float_identity_dual_gate_agrees() {
        let _lock = anchor_lock();
        let graph = build_anchor_graph("fn id(x: f64) -> f64 { x }\n", "id");
        run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph);
        assert!(
            result.is_ok(),
            "Float identity must agree under dual-gate, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_int_addition_dual_gate_agrees() {
        let _lock = anchor_lock();
        // From `pipeline_e2e_simple_function`: `fn add(a, b) { a + b }`
        // produces `OpKind::BinOp{op:"add", ...}`. The BinOp
        // followup ports the arm as a pre-rtyper opname pass-through
        // (`add` → flowspace `SpaceOperation("add", ...)`); the real
        // rtyper then rewrites `add` → `int_add` for `Signed` operands
        // via `pair_int_int`, agreeing with the legacy resolver.
        let graph = build_anchor_graph("fn add(a: i64, b: i64) -> i64 { a + b }\n", "add");
        run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph);
        assert!(
            result.is_ok(),
            "Int addition must agree under dual-gate post-BinOp port, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_load_fast_same_block_dedups_path_reads() {
        let _lock = anchor_lock();
        // Cat 3.2 first slice: a body `Expr::Path` whose name resolves
        // to a same-block local definition reuses the existing
        // slot instead of emitting a fresh `OpKind::Input`.
        // RPython `flowspace/flowcontext.py:835 LOAD_FAST` reads the
        // existing locals-stack entry; pyre's analogue forwards the
        // bound slot.
        //
        // The graph for `fn id(x: &Foo) -> &Foo { x }` lifts the
        // parameter as a single `OpKind::Input { name: "x", .. }` in
        // the entry block.  The body's `x` reference is in the same
        // block as the parameter binding, so the lookup returns the
        // existing slot and the graph carries exactly one
        // `OpKind::Input` named `"x"` — not two.
        let graph = build_anchor_graph(
            r#"
struct Foo {}
fn id(x: &Foo) -> &Foo { x }
"#,
            "id",
        );
        let input_x_count: usize = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    crate::model::OpKind::Input { name, .. } if name == "x"
                )
            })
            .count();
        assert_eq!(
            input_x_count, 1,
            "Cat 3.2 same-block dedup must collapse the body Path read \
             into the parameter's existing slot — expected exactly \
             one OpKind::Input{{name:\"x\"}}, got {input_x_count}"
        );
    }

    #[test]
    fn anchor_cat1_ptr_eq_lowers_to_binop_eq() {
        let _lock = anchor_lock();
        // Cat 1 anchor: `std::ptr::eq(a, b)` is pointer identity; it
        // previously lowered to an unregistered FunctionPath
        // ["std","ptr","eq"] → the "not registered in PyreCallRegistry"
        // Skip.  The front-end now emits BinOp { op: "eq" } on the two
        // Ref operands, which jtransform rewrites to ptr_eq
        // (jtransform.rs:849), so the cat1 signature must be gone.
        let graph = build_anchor_graph(
            r#"
struct W {
    x: i64,
}
fn same(a: &W, b: &W) -> bool {
    std::ptr::eq(a, b)
}
"#,
            "same",
        );
        run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph);
        let err = match &result {
            Ok(()) => String::new(),
            Err(e) => e.clone(),
        };
        assert!(
            !err.contains("not registered in PyreCallRegistry"),
            "`std::ptr::eq` must lower to BinOp eq (→ ptr_eq) instead of \
             an unregistered FunctionPath call; got: {err}"
        );
    }

    #[test]
    fn anchor_cat1_numeric_from_routes_through_cast() {
        let _lock = anchor_lock();
        // Cat 1 anchor: `i64::from(x)` (the function-call spelling of an
        // `as` widening) previously lowered to an OpKind::Call whose
        // target is FunctionPath ["i64","from"], which misses
        // PyreCallRegistry and every HOST_ENV fallback → the "not
        // registered in PyreCallRegistry" Skip.  The front-end now
        // recognizes `<prim>::from` and routes it through the same
        // coercion chain as `x as T` (here Unsigned→Int via
        // rarithmetic.intmask, which is registered), so the cat1
        // signature must be gone.
        let graph = build_anchor_graph(
            r#"
fn widen(x: u32) -> i64 {
    i64::from(x)
}
"#,
            "widen",
        );
        run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph);
        let err = match &result {
            Ok(()) => String::new(),
            Err(e) => e.clone(),
        };
        assert!(
            !err.contains("not registered in PyreCallRegistry"),
            "`<prim>::from` must route through the cast chain (intmask) \
             instead of an unregistered FunctionPath call; got: {err}"
        );
    }

    #[test]
    fn anchor_cat1_bare_imported_call_resolves_via_use_import() {
        let _lock = anchor_lock();
        // Cat 1 anchor: a bare imported free-function call binds in the
        // caller's lexical scope (LOAD_GLOBAL), so it must lower to the
        // import target's path — not the caller's own module, and not a
        // bare name.  Before the fix, `canonical_call_target` qualified a
        // bare name only with the caller's `module_prefix` (or left it
        // bare), so `helper(x)` missed the free function registered under
        // its callee-home key.  Now the use-import resolves the bare name
        // to `["objmod","helper"]`.
        let graph = build_anchor_graph(
            r#"
mod objmod {
    pub fn helper(x: i64) -> i64 { x }
}
use objmod::helper;
fn caller(x: i64) -> i64 { helper(x) }
"#,
            "caller",
        );
        let call_targets: Vec<Vec<String>> = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter_map(|op| match &op.kind {
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect();
        assert!(
            call_targets
                .iter()
                .any(|s| s.as_slice() == ["objmod".to_string(), "helper".to_string()]),
            "bare imported `helper()` must lower to the import path \
             [\"objmod\",\"helper\"]; got call targets: {call_targets:?}"
        );
    }

    #[test]
    fn anchor_cat14_try_canraise_continuation_reads_param() {
        let _lock = anchor_lock();
        // Cat 14 anchor: a `?` on a call (`step()?`) closes its block as
        // canraise (the call is the recorded raising op), opening a
        // continuation.  The continuation reads `x` cross-block.  Before
        // the canraise `?`-source stamped its framestate,
        // `lazy_install_local_at_current_block_var` bailed at
        // `pred_block.framestate.as_ref()?`, dropping `x` to a naked
        // body-`Input` the adapter rejects as "cross-block body Input".
        // The stamp threads `x` via the predecessor link, so the cat14
        // signature must be gone.  The graph still fails on a SEPARATE
        // downstream blocker (`step` is not registered in the anchor's
        // call registry — Epic A), so a full dual-gate green is not
        // reachable for an isolated `?` fixture; we assert only that the
        // cat14 threading gap is closed.
        let graph = build_anchor_graph(
            r#"
fn step() -> Result<i64, i64> { Ok(0) }
fn pop_and_store(x: i64) -> Result<i64, i64> {
    step()?;
    Ok(x)
}
"#,
            "pop_and_store",
        );
        run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph);
        let err = match &result {
            Ok(()) => String::new(),
            Err(e) => e.clone(),
        };
        assert!(
            !err.contains("cross-block body Input"),
            "Cat 14 canraise `?`-source framestate stamp must thread the \
             cross-block `x` read so no naked body-`Input` reaches the \
             adapter; got cat14 error: {err}"
        );
    }

    #[test]
    fn anchor_ref_identity_dual_gate_agrees() {
        let _lock = anchor_lock();
        // Cat 3.1 anchor: a typed-`Ref` identity graph must agree under
        // dual-gate now that `valuetype_to_someshell(Ref)` projects to
        // `SomeInstance(classdef=None)` instead of the illegal
        // `SomeObject` placeholder.  The rtyper routes through
        // `getinstancerepr(rtyper, None, Gc)` ->
        // `InstanceRepr::new_rootinstance` -> `Ptr(GcStruct(OBJECT))`,
        // which `lowleveltype_to_concrete` collapses to GcRef matching
        // legacy `resolve_types(Ref) -> GcRef`.
        let graph = build_anchor_graph(
            r#"
struct Foo {}
fn id(x: &Foo) -> &Foo { x }
"#,
            "id",
        );
        run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph);
        assert!(
            result.is_ok(),
            "Ref-typed identity graph must agree under dual-gate via \
             SomeInstance(classdef=None) -> GcRef, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_int_negation_dual_gate_agrees() {
        let _lock = anchor_lock();
        // Cat 3.3 follow-on: `OpKind::UnaryOp{op:"neg",..}` ports as
        // a pre-rtyper opname pass-through (`neg` is registered
        // upstream by `operation.py:466 add_operator('neg', 1, ...)`)
        // and the rtyper rewrites `neg` -> `int_neg` for `Signed`
        // operands via the unary `pair_int_*` dispatch, agreeing with
        // the legacy resolver.
        let graph = build_anchor_graph("fn negate(x: i64) -> i64 { -x }\n", "negate");
        run_legacy_resolve(&graph);
        let result = dual_gate_check(&graph);
        assert!(
            result.is_ok(),
            "Int negation must agree under dual-gate post-UnaryOp port, got: {:?}",
            result
        );
    }

    #[test]
    fn anchor_unary_not_surfaces_failloud_no_flowspace_peer() {
        let _lock = anchor_lock();
        // RPython `UNARY_NOT` (`flowspace/flowcontext.py:531-538`)
        // expands to `op.bool(x).eval(self);
        // const(not self.guessbool(w_bool))` — emit `bool`, branch
        // via `guessbool`, then push the negated boolean constant.
        // Rust `!int` (bitwise) corresponds to UNARY_INVERT →
        // `op.invert` (`operation.py:474`).  Pyre's surface DSL
        // lacks a `Bool` ValueType so the bool-vs-invert dispatch
        // cannot be made faithfully here, and mapping `not` to a
        // bare `bool` op alone loses the negation.  Fail-loud
        // preserves pyre IR's `!x` semantics (consumed verbatim
        // downstream) until the frontend desugars `!cond` to
        // `bool` + branch and bitwise `!int` to `invert`.
        let mut graph = LegacyGraph::new("not_int");
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
        let (_, _, v2_var) = graph.exceptblock_arg_vars();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![crate::model::SpaceOperation {
                result: Some(v2_var.clone()),
                kind: crate::model::OpKind::UnaryOp {
                    op: "not".into(),
                    operand: v1_var,
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v2_var)],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&mut graph, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let err = specialize_legacy_graph(&graph)
            .expect_err("UnaryOp `not` must surface a TyperError until frontend desugaring lands");
        let msg = format!("{err}");
        assert!(
            msg.contains("normalize_unary_op_name") && msg.contains("not"),
            "fail-loud must name the unsupported opname, got: {msg}"
        );
    }

    #[test]
    fn anchor_unary_deref_surfaces_failloud_no_flowspace_peer() {
        let _lock = anchor_lock();
        // `OpKind::UnaryOp{op:"deref",..}` (Rust `*x`) has no
        // RPython peer — the operand table at
        // `flowspace/operation.py:465-474` registers no `deref`,
        // and pyre carries no global invariant proving Rust `*x` is
        // type/reference-transparent.  Aliasing the result to the
        // operand could silently fold a real value load.  Until
        // the frontend either removes `deref` ops or proves the
        // invariant, the adapter must surface fail-loud.
        let mut graph = LegacyGraph::new("deref_int");
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
        let (_, _, v2_var) = graph.exceptblock_arg_vars();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![crate::model::SpaceOperation {
                result: Some(v2_var.clone()),
                kind: crate::model::OpKind::UnaryOp {
                    op: "deref".into(),
                    operand: v1_var,
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v2_var)],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&mut graph, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let err = specialize_legacy_graph(&graph)
            .expect_err("UnaryOp `deref` must surface a TyperError until frontend invariant lands");
        let msg = format!("{err}");
        assert!(
            msg.contains("normalize_unary_op_name") && msg.contains("deref"),
            "fail-loud must name the unsupported opname, got: {msg}"
        );
    }

    #[test]
    fn anchor_field_read_surfaces_followup() {
        let _lock = anchor_lock();
        // From `pipeline_e2e_with_virtualizable`. Frame.next_instr is a
        // FieldRead. Not yet implemented — fail-loud expected.
        let graph = build_anchor_graph(
            r#"
struct Frame { next_instr: usize, locals_w: Vec<i64> }
impl Frame {
    fn load_fast(&mut self) -> i64 {
        let idx = self.next_instr;
        self.locals_w[idx]
    }
}
"#,
            "load_fast",
        );
        run_legacy_resolve(&graph);
        let err = dual_gate_check(&graph).expect_err("FieldRead surfaces an unported followup");
        assert!(
            err.contains("followup")
                || err.contains("FieldRead")
                || err.contains("real path failed"),
            "anchor must surface a named unported followup pending, got: {err}"
        );
    }

    #[test]
    fn anchor_control_flow_fib_surfaces_followup() {
        let _lock = anchor_lock();
        // fib has if/else, comparison, and arithmetic.  Earlier this
        // anchor was an expected-failure case because the per-block
        // `Input` op shape and `setup_block_entry` produced a
        // "wrong level!" assertion in the rtyper.  The framestate-
        // based cross-block local merge replaces the body-`Input`
        // emission with explicit phi inputargs, so the dual-gate now
        // agrees end-to-end; flip the assertion to a positive check.
        let graph = build_anchor_graph(
            r#"
fn fib(n: i64) -> i64 {
    if n <= 1 { return n; }
    let a = n - 1;
    let b = n - 2;
    a + b
}
"#,
            "fib",
        );
        run_legacy_resolve(&graph);
        dual_gate_check(&graph)
            .expect("fib must dual-gate-agree under the framestate cross-block merge");
    }

    #[test]
    fn specialize_legacy_graph_guard_value_skips_translation_keeps_kind() {
        let _lock = anchor_lock();
        // Cat 3.3 first slice: pyre JIT trace markers
        // (`GuardTrue` / `GuardFalse` / `GuardValue`) have no peer in
        // RPython flowspace's high-level operator set
        // (`operation.py:475-510`).  The adapter now skips them
        // (`Ok(Vec::new())`); the operand they read is defined
        // elsewhere and the absence of a result keeps the SSA chain
        // intact.  Specialize must succeed and project the Int operand
        // to `Signed`.
        let mut graph = LegacyGraph::new("guard_passthrough");
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![crate::model::SpaceOperation {
                result: None,
                kind: crate::model::OpKind::GuardValue {
                    value: v1_var.clone(),
                    kind_char: 'i',
                },
            }],
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
            inputargs: block_inputargs(&mut graph, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        setbinding(&v1_var, ValueType::Int);
        let (value_to_var, _constants) = specialize_legacy_graph(&graph)
            .expect("GuardValue must be a no-op for the rtyper adapter");
        assert_eq!(
            kind_of_in(&value_to_var, 1),
            ConcreteType::Signed,
            "Int operand passing through GuardValue must specialize to Signed"
        );
    }

    #[test]
    fn specialize_legacy_graph_unported_opkind_propagates_failloud() {
        let _lock = anchor_lock();
        // Graph carrying a still-fail-loud OpKind (Call::Indirect —
        // requires rclass.rs lowering) must surface the variant's
        // fail-loud message — confirms the adapter's TyperError flows
        // through the full specialize pipeline.
        let mut graph = LegacyGraph::new("unported_call");
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
        let (_, _, v2_var) = graph.exceptblock_arg_vars();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![crate::model::SpaceOperation {
                result: Some(v2_var.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::Indirect {
                        trait_root: "MyTrait".into(),
                        method_name: "do_it".into(),
                    },
                    args: vec![v1_var],
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v2_var)],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&mut graph, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let err = specialize_legacy_graph(&graph)
            .expect_err("unported OpKind must surface as TyperError");
        let msg = format!("{err}");
        // The empty-registry default entry surfaces an unregistered
        // fail-loud naming the segments and producer slice (A.4) that
        // should populate the registry; this graph uses Indirect to
        // exercise the rclass-rewrite invariant break instead.
        assert!(
            msg.contains("Indirect") && msg.contains("rclass"),
            "fail-loud must propagate the variant + rclass tag, got: {msg}"
        );
    }

    #[test]
    fn anchor_call_function_path_registered_emits_simple_call() {
        let _lock = anchor_lock();
        // When the registry is pre-populated with the callee's
        // FunctionPath, the adapter emits flowspace
        // `simple_call(callable_const, *args)` where `callable_const`
        // wraps the registry entry's HostObject.  The rtyper's
        // `bookkeeper.getdesc(host)` short-circuits at the cache
        // returning the registry's pre-built FunctionDesc.
        //
        // This anchor verifies the plumbing reaches at least the
        // rtyper's simple_call dispatch — whether the rtyper succeeds
        // or surfaces a downstream `pair_simple_call` followup
        // depends on FunctionDesc.cache wiring (graph build via the
        // bookkeeper's `specialize` pass).  The anchor accepts either
        // outcome and locks the failure mode so future slices can
        // flip the assertion.
        let mut graph = LegacyGraph::new("call_resolved");
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
        let (_, _, v2_var) = graph.exceptblock_arg_vars();
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
            inputargs: block_inputargs(&mut graph, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            dead: false,
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        // Pre-populate the registry with the callee's FunctionPath +
        // Signature + lifted leaf PyGraph in a single
        // `register_callee` call (Issue 2.2 atomic API).  Without
        // the cache pre-fill, `cachedgraph`
        // (description.rs:1037-1039) would miss and `pair_simple_call`
        // would invoke `buildgraph` on the synthetic `GraphFunc`,
        // which has no `HostCode` and would fail at
        // `translator.buildflowgraph`.
        let bookkeeper = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let registry =
            crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(bookkeeper);

        // Build a leaf callee `fn foo(x: i64) -> i64 { x }` —
        // identity returns the inputarg, no nested Calls so a
        // child `PyreCallRegistry` can stay empty during the lift.
        let mut callee_graph = LegacyGraph::new("foo");
        let foo_inputargs = block_inputargs(&mut callee_graph, &[10]);
        let foo_v10_var = foo_inputargs[0].clone();
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
            inputargs: block_inputargs(&mut callee_graph, &[10]),
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
        registry.register_callee(
            crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments(["foo"]),
            crate::flowspace::argument::Signature::new(vec!["x".to_string()], None, None),
            pygraph,
        );

        setbinding(&v1_var, ValueType::Int);
        setbinding(&v2_var, ValueType::Int);
        let (value_to_var, _constants) =
            specialize_legacy_graph_with_registry_returning_value_to_var(&graph, &registry)
                .expect("cache pre-fill must let the leaf Call resolve end-to-end");
        // The cache pre-fill closes the loop:
        //   1. Adapter emits `simple_call(host_obj_const, *args)`.
        //   2. `prefill_default_cache` fills
        //      `FunctionDesc.cache[GraphCacheKey::None]` with the
        //      lifted leaf `PyGraph`, so `cachedgraph`
        //      (description.rs:1037-1039) returns the cached
        //      graph instead of calling `buildflowgraph` on the
        //      synthetic `GraphFunc`.
        //   3. `compute_at_fixpoint` populates
        //      `bookkeeper.pbc_maximal_call_families` /
        //      `calltable` from the seeded blocks' `simple_call` ops.
        //   4. `FunctionRepr.call(hop)` (rpbc.py:199) finds the
        //      call-family row, the rtyper completes
        //      `direct_call`, and the result Variable's
        //      `concretetype` is set to `Signed`
        //      (`int_is_true → Bool` style projection).
        assert_eq!(
            kind_of_in(&value_to_var, 2),
            ConcreteType::Signed,
            "leaf Call (fn foo(x: i64) -> i64 {{ x }}) result must project to Signed \
             through the rtyper's full simple_call → FunctionRepr.call chain"
        );
        assert_eq!(
            kind_of_in(&value_to_var, 1),
            ConcreteType::Signed,
            "Int inputarg must project to Signed (independent of the \
             call resolution path)"
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
        let inputargs = block_inputargs(&mut graph, &[1]);
        let v1_var = inputargs[0].clone();
        let (_, _, v2_var) = graph.exceptblock_arg_vars();
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
            inputargs: block_inputargs(&mut graph, &[2]),
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
        let foo_inputargs = block_inputargs(&mut callee_graph, &[10]);
        let foo_v10_var = foo_inputargs[0].clone();
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
            inputargs: block_inputargs(&mut callee_graph, &[10]),
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

    #[test]
    fn populate_call_graphs_mirrors_value_to_var_to_alias_keys() {
        let _lock = anchor_lock();
        // When two `CallPath` keys collapse to the same
        // canonical `PyreFunctionEntry` (the dedupe drops a leading
        // `crate::` segment), `populate_call_registry_from_call_graphs`
        // must mirror the lifted `value_to_var` and
        // `constant_concretetypes` snapshots to BOTH keys, so the
        // per-graph dual-gate's lookup by the subject's exact path
        // succeeds regardless of which alias of the canonical entry
        // got registered first.
        let graph = build_anchor_graph("fn helper(x: i64) -> i64 { x }\n", "helper");
        let canonical_path = crate::parse::CallPath::from_segments(["helper"]);
        let alias_path = crate::parse::CallPath::from_segments(["crate", "helper"]);
        let mut function_graphs: std::collections::HashMap<crate::parse::CallPath, LegacyGraph> =
            std::collections::HashMap::new();
        function_graphs.insert(canonical_path.clone(), graph.clone());
        function_graphs.insert(alias_path.clone(), graph);

        let registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(
            std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new()),
        );
        populate_call_registry_from_call_graphs(&function_graphs, &registry)
            .expect("populate must succeed for a simple identity function");

        // Both keys resolve to the same `Rc<PyreFunctionEntry>` (alias
        // dedupe contract).
        let canonical_key =
            crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments([
                "helper",
            ]);
        let alias_key =
            crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments([
                "crate", "helper",
            ]);
        let canonical_entry = registry
            .lookup(&canonical_key)
            .expect("canonical registered");
        let alias_entry = registry.lookup(&alias_key).expect("alias registered");
        assert!(
            std::rc::Rc::ptr_eq(&canonical_entry, &alias_entry),
            "alias must point at the same Rc<PyreFunctionEntry> as the canonical key"
        );

        // Issue 2.5 (2026-05-07): the canonical/alias mirror used to be
        // verified against per-`FunctionPathKey` `value_to_var` /
        // `constant_concretetypes` snapshot side maps.  Those side maps
        // were dead-write — no production reader — and were retired.
        // The shared-Rc identity assertion above now subsumes the alias
        // mirror invariant: any future reader of `Variable.concretetype`
        // walks the single canonical `PyGraph.graph` reachable through
        // either alias's `function_desc.cache`, so there is nothing to
        // mirror per-key.
    }

    #[test]
    fn anchor_populate_call_registry_from_program_registers_every_function() {
        let _lock = anchor_lock();
        // `populate_call_registry_from_program` must
        // register every `SemanticFunction` in `program.functions`
        // (Pass 1) and prefill each FunctionDesc.cache with a lifted
        // PyGraph (Pass 2).  After the walker runs, a caller graph
        // referencing any of those functions through
        // `OpKind::Call::FunctionPath` resolves end-to-end through
        // `specialize_legacy_graph_with_registry_returning_value_to_var`.
        let parsed = crate::parse::parse_source(
            r#"
            fn helper(x: i64) -> i64 { x }
            fn caller(y: i64) -> i64 {
                helper(y)
            }
        "#,
        );
        let program = crate::front::build_semantic_program(&parsed).expect("source must lower");

        let registry = crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(
            std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new()),
        );
        populate_call_registry_from_program(&program, &registry)
            .expect("walker must register every program function without error");

        // Both functions registered.
        assert_eq!(
            registry.len(),
            program.functions.len(),
            "registry must hold one entry per SemanticFunction in the program"
        );
        let helper_entry = registry
            .lookup(
                &crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments([
                    "helper",
                ]),
            )
            .expect("helper must be registered under its function name");
        let caller_entry = registry
            .lookup(
                &crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments([
                    "caller",
                ]),
            )
            .expect("caller must be registered too — walker covers every function");

        // Both FunctionDescs carry their lifted PyGraph.
        assert!(
            helper_entry
                .function_desc
                .borrow()
                .cache
                .borrow()
                .contains_key(&crate::annotator::description::GraphCacheKey::None),
            "helper's FunctionDesc.cache must be prefilled with the lifted PyGraph"
        );
        assert!(
            caller_entry
                .function_desc
                .borrow()
                .cache
                .borrow()
                .contains_key(&crate::annotator::description::GraphCacheKey::None),
            "caller's FunctionDesc.cache must be prefilled too"
        );
        assert_eq!(
            helper_entry.function_desc.borrow().signature.argnames,
            vec!["x".to_string()],
            "helper's signature must come from its declared parameter list"
        );
        assert_eq!(
            caller_entry.function_desc.borrow().signature.argnames,
            vec!["y".to_string()],
            "caller's signature must come from its declared parameter list"
        );
    }

    #[test]
    fn anchor_load_fast_repeated_cross_block_read_dedups_within_block() {
        let _lock = anchor_lock();
        // Cat 3.2 same-block dedup invariant — orthogonal to Cat 2.1
        // the lazy cross-block install.  RPython parity:
        // `flowspace/flowcontext.py:835 LOAD_FAST` reads the existing
        // locals slot rather than introducing a fresh Variable on
        // every read, so `x + x` inside one block must collapse to a
        // single `OpKind::Input { name: "x" }` followed by a single
        // BinOp consuming that op's result twice.
        //
        // Previously the post-`if` block emitted a naked
        // `OpKind::Input{name:"x"}` directly (no inputarg, no
        // `Link.args` thread-back) and the dedup invariant held
        // trivially because there was only one cross-block reading
        // block.  Now lazy thread-back installs an
        // `OpKind::Input{name:"x"}` per block on the predecessor
        // chain (entry parameter + each empty pass-through merge +
        // the actual reading block), and the dedup invariant becomes
        // a per-block assertion: no single block emits more than one
        // `Input{name:"x"}`.
        let graph = build_anchor_graph(
            r#"
fn cross_block(x: i64, cond: bool) -> i64 {
    if cond { return 0; }
    x + x
}
"#,
            "cross_block",
        );
        for block in &graph.blocks {
            let count: usize = block
                .operations
                .iter()
                .filter(|op| {
                    matches!(
                        &op.kind,
                        crate::model::OpKind::Input { name, .. } if name == "x"
                    )
                })
                .count();
            assert!(
                count <= 1,
                "Cat 3.2 same-block dedup: B{} emitted {} `Input{{name:\"x\"}}` \
                 ops but `x + x` must collapse to one within a single block",
                block.id.0,
                count,
            );
        }
        let total_input_x: usize = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    crate::model::OpKind::Input { name, .. } if name == "x"
                )
            })
            .count();
        assert!(
            total_input_x >= 1,
            "Cat 3.2 cross-block read: at least the entry block must define \
             `OpKind::Input{{name:\"x\"}}` for the function parameter",
        );
    }

    #[test]
    fn anchor_cat_2_1_both_open_arm_rebind_post_merge_read_phi_threads() {
        let _lock = anchor_lock();
        // Cat 2.1 Stage C — both-open-arm + per-arm rebind merge.
        // RPython `flowspace/flowcontext.py` parity: at the merge
        // point of `if cond { x = 1 } else { x = 2 }; x`, the post-
        // merge `LOAD_FAST x` resolves to a fresh phi inputarg in
        // the merge block whose `Link.args` from each arm carry the
        // arm-rebound slot.
        //
        // Stage A1-A3 + B1-B2 + C1 minimal land the foundations: per-
        // block `framestate`, first-bind positional slot order, union
        // / getoutputargs, ctx restore between arms, relaxed lazy
        // installer fence, None-kill of one-arm-only locals.  The
        // lazy installer handles the cross-block read of
        // the rebound `x` at the post-merge use site, allocating a
        // single merge-block inputarg + threading each arm's rebound
        // slot back through that arm's `Link.args`.
        //
        // This anchor pins the both-open-arm + rebind shape end-to-
        // end via `dual_gate_check`.  Per-block `Link.args` arity =
        // target `inputargs` arity is checked separately to
        // surface any predecessor / target mismatch the lazy install
        // might leave behind.
        let graph = build_anchor_graph(
            r#"
fn rebind_both_arms(cond: bool) -> i64 {
    let x: i64;
    if cond { x = 1; } else { x = 2; }
    x + x
}
"#,
            "rebind_both_arms",
        );
        run_legacy_resolve(&graph);
        dual_gate_check(&graph).expect(
            "Cat 2.1 Stage C — both-open-arm rebind merge resolves via lazy Link.args + phi inputarg",
        );
        for block in &graph.blocks {
            for link in &block.exits {
                let target_arity = graph.block(link.target).inputargs.len();
                assert_eq!(
                    link.args.len(),
                    target_arity,
                    "Stage C invariant: Link from B{} to B{} arity {} != target inputargs {}",
                    block.id.0,
                    link.target.0,
                    link.args.len(),
                    target_arity,
                );
            }
        }
    }

    #[test]
    fn anchor_cat_2_1_cross_block_reads_pass_dual_gate_after_link_args_thread() {
        let _lock = anchor_lock();
        // Cross-block locals threading via lazy
        // `Link.args` / target-`inputarg` install at the actual
        // cross-block read site.  RPython
        // `flowspace/flowcontext.py:835 LOAD_FAST` parity.
        //
        // `front/ast.rs::lower_expr`'s `Expr::If` arm is wired
        // through `lazy_install_local_at_current_block`: when a
        // post-`if` block performs a `LOAD_FAST` of a local whose
        // definition lives in a dominating block, the read site
        // allocates a fresh `OpKind::Input` in the merge block,
        // promotes it to `inputargs`, and walks back to every
        // predecessor edge whose `set_branch` / `set_goto` recorded a
        // `LocalsFrameState` snapshot, appending the predecessor-side
        // slot to that link's `args` so `len(link.args) ==
        // len(target.inputargs)` per `flowspace/model.py:114
        // Link.__init__`.
        //
        // This shape (single-open-arm-no-rebind: `if cond { return; }
        // x + x`) is the simplest case — only one predecessor, no
        // rebinding inside the open arm.  Further work extends the
        // recording to the both-open-arm + match + loop + break /
        // continue join sites.
        //
        // The graph-shape checks below pin the invariants — the merge
        // block has its
        // own inputargs and the predecessor's `link.args` arity now
        // matches it.
        let graph = build_anchor_graph(
            r#"
fn cross_block(x: i64, cond: bool) -> i64 {
    if cond { return 0; }
    x + x
}
"#,
            "cross_block",
        );
        run_legacy_resolve(&graph);
        dual_gate_check(&graph).expect("cross-block locals close via lazy Link.args threading");

        // Per-block invariant: every link from `block` to its target
        // carries `args` whose arity matches the target's inputargs.
        // Equivalent to `flowspace/model.py:114 Link.__init__` +
        // `:checkgraph`.
        for block in &graph.blocks {
            for link in &block.exits {
                let target_arity = graph.block(link.target).inputargs.len();
                assert_eq!(
                    link.args.len(),
                    target_arity,
                    "invariant: Link from B{} to B{} arity {} != target inputargs {}",
                    block.id.0,
                    link.target.0,
                    link.args.len(),
                    target_arity,
                );
            }
        }
    }

    #[test]
    fn anchor_cat_2_1_loop_invariant_live_var_threads_to_backedge() {
        let _lock = anchor_lock();
        // A variable live across a loop back-edge but not rebound or
        // read inside the body's forward path (a loop-invariant the
        // body only passes through, or — in the nested case — an outer
        // variable the inner loop merges) must still be threaded
        // through every body block so the loop-header phi is supplied
        // from a slot defined at the closing predecessor.  RPython's
        // fixed-size `locals_w` frame carries every live slot through
        // each block (`flowspace/framestate.py:92 getoutputargs`); the
        // pyre front-end re-threads dropped names via the lazy
        // installer at the back-edge close
        // (`link_arg_vars_from_ctx`) and at the pre-loop→header edge
        // (`allocate_loop_header_phis`).  Without it the back-edge /
        // pre-loop link references an out-of-scope slot and the rtyper
        // adapter rejects it as "undefined operand slot ... Link.args".
        let cases: &[(&str, &str)] = &[
            // single `while` + `break`: `cond` / `n` are live across the
            // back-edge but only read in the header / the break guard.
            (
                "loop_break_invariant",
                "fn loop_break_invariant(n: i64, cond: bool) -> i64 { \
                 let mut i: i64 = 0; \
                 while i < n { if cond { break; } i = i + 1; } i }\n",
            ),
            // nested `while`: `count` is merged by the inner loop but the
            // outer body block that bridges to the inner loop never reads
            // it directly, so the outer-header→body edge would otherwise
            // drop it before the inner-header phi needs it.
            (
                "nested_loop_outer_var",
                "fn nested_loop_outer_var(n: i64) -> i64 { \
                 let mut count: i64 = 0; let mut i: i64 = 0; \
                 while i < n { let mut j: i64 = 0; \
                 while j < n { count = count + 1; j = j + 1; } \
                 i = i + 1; } count }\n",
            ),
        ];
        for (name, src) in cases {
            let graph = build_anchor_graph(src, name);
            dual_gate_check(&graph).unwrap_or_else(|e| {
                panic!("{name} must thread loop-live vars to the back-edge: {e}")
            });
            // Every link's `args` arity must equal the target's
            // inputargs arity (`flowspace/model.py:114 Link.__init__`).
            for block in &graph.blocks {
                for link in &block.exits {
                    let target_arity = graph.block(link.target).inputargs.len();
                    assert_eq!(
                        link.args.len(),
                        target_arity,
                        "{name}: Link B{} -> B{} arity {} != target inputargs {}",
                        block.id.0,
                        link.target.0,
                        link.args.len(),
                        target_arity,
                    );
                }
            }
        }
    }

    /// `&&` / `||` short-circuit forks carry every pre-fork local on
    /// both arm Links (`[lhs_raw, ...locals]` shortcut, `[...locals]`
    /// rhs).  A local that was bound in a dominator but never read in
    /// the fork block points at the dominator's Variable; pyre threads
    /// locals lazily, so that slot is not yet a block-local inputarg of
    /// the fork block and threading it onto an arm Link trips the
    /// flowspace adapter's "undefined operand slot" invariant.  The
    /// `&&`-chain shapes below stack two/three such forks so the second
    /// and third conditions read locals (`a`, `b`) that the first fork's
    /// merge dropped — exercising both the predecessor-recursion ctx
    /// rebind leak (the carried `a` slot) and the never-read dominator
    /// local (`b`).  All must reach Match via the real dual-gate path.
    #[test]
    fn anchor_cat_2_1_shortcircuit_chain_threads_pre_fork_locals() {
        let _lock = anchor_lock();
        let cases: &[(&str, &str)] = &[
            (
                "m1",
                "fn m1(x: i64, y: i64) -> i64 { let a: i64 = x + 1; let b: i64 = y + 1; if a > 0 && b > 0 { return a; } a + b }\n",
            ),
            (
                "m2",
                "fn m2(x: i64, y: i64) -> i64 { let a: i64 = x + 1; let b: i64 = y + 1; if a > 0 && b > 0 { return a; } if a < 0 && b < 0 { return b; } a + b }\n",
            ),
            (
                "m3",
                "fn m3(x: i64, y: i64) -> i64 { let a: i64 = x + 1; let b: i64 = y + 1; if a > 0 && b > 0 { return a; } if a < 0 && b < 0 { return b; } if a > b && b > a { return 0; } a + b }\n",
            ),
        ];
        for (name, src) in cases {
            let g = build_anchor_graph(src, name);
            assert!(
                dual_gate_check(&g).is_ok(),
                "{name}: &&-chain pre-fork local threading must reach Match, got {:?}",
                dual_gate_check(&g),
            );
        }
    }

    // `build_program_annotator_starts_empty_after_step3` retired at
    // Step 4 first slice (2026-05-07).  The function being tested
    // (`build_program_annotator`) was retired together with the
    // dead-pass `build_program_rtyper` and `ensure_program_specialize`
    // helpers, since after Step 3 they ran a no-op against an empty
    // annotator.  Per-session annotator construction inside
    // `specialize_legacy_graph_with_registry_returning_value_to_var`
    // is the only remaining production lifecycle.

    // ─── Z2.5 Path C slice 2 helpers ───
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
            // Compound lltype — slice 2's default_constvalue_for_lltype
            // returns None and register_unsafe_fn_stubs must skip.
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

    /// Reviewer round 2026-05-24 item #5 — pin the layering invariant
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
        // must return `None` so slice 3's caller skips registration.
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
