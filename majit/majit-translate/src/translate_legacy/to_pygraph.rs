//! P8.3a — `FunctionGraph` (legacy) → `PyGraph` (real) adapter scaffolding.
//!
//! This file is a **PRE-EXISTING-ADAPTATION** with no upstream RPython
//! counterpart. RPython's pipeline never has a "legacy graph model" to
//! convert from — the annotator builds its `FunctionGraph` (from
//! `rpython/flowspace/model.py`) directly, and the rtyper consumes it
//! in place. Pyre carries two graph models in parallel during the
//! Phase 8 cutover (`crate::model::FunctionGraph` for the legacy
//! `translate_legacy/` pipeline, and
//! `crate::flowspace::model::FunctionGraph` wrapped as
//! [`PyGraph`](crate::flowspace::pygraph::PyGraph) for the real
//! `translator/rtyper/` pipeline) — this adapter exists solely to
//! bridge the gap until [`P8.8`](`.claude/plans/rtyper-cutover.md`)
//! deletes the legacy graph and its callers.
//!
//! ## Convergence path
//!
//! Per `.claude/plans/rtyper-cutover.md`:
//!
//! - **P8.3a (this file):** scaffolding only — function signature plus
//!   stub body that surfaces a clear "not yet implemented" error so
//!   downstream wiring can compile.
//! - **P8.4:** dual-path `PYRE_RTYPER=1` gate in
//!   `jit_codewriter/codewriter.rs` calls this adapter, runs both the
//!   legacy resolver and `translator::rtyper::RPythonTyper::specialize`,
//!   and diffs the per-value `ConcreteType ↔ LowLevelType` mapping.
//! - **P8.5:** flip default to the real path.
//! - **P8.8:** delete this adapter (and the legacy graph model) once
//!   `jit_codewriter/*` consumes `PyGraph` directly.
//!
//! Because this is a transitional surface, the file is intentionally
//! thin: any logic added here that has no clear path back to deletion
//! at P8.8 is a sign the cutover plan needs revisiting before more
//! code lands.

use crate::flowspace::pygraph::PyGraph;
use crate::jit_codewriter::annotation_state::AnnotationState;
use crate::model::FunctionGraph;
use crate::translator::rtyper::error::TyperError;

/// One-way conversion from the legacy `crate::model::FunctionGraph` +
/// `AnnotationState` pair into a real `PyGraph` whose blocks carry
/// `Hlvalue` operands and per-value `SomeValue` annotations on its
/// `Variable`s.
///
/// Returns [`TyperError::message`] until the body is filled in by the
/// P8.3a follow-up commits. The signature is stable: callers (P8.4
/// dual-gate logic in `jit_codewriter/codewriter.rs`) can wire against
/// it now and rely on the Result to short-circuit cleanly while the
/// adapter is still a stub.
///
/// # Out of scope here
///
/// - Per-op rewriting (`crate::model::OpKind` → `SpaceOperation` over
///   `Hlvalue`). Owned by P8.3a's body fill-in commits.
/// - `AnnotationState` → annotator state translation. P8.4 will wire
///   the real annotator output once Phase 5 progresses; for now this
///   adapter can lift `AnnotationState`'s legacy `ValueType` per value
///   into [`SomeValue`](crate::annotator::model::SomeValue) shells
///   (`Signed` / `Float` / `GcRef`-shaped) so `RPythonTyper`'s
///   `bindingrepr` lookup has something to bind to.
/// - `Block` topology rewriting (legacy's index-based `BlockId` to the
///   real model's `Rc<RefCell<Block>>` graph).
///
/// All three are landed in dedicated follow-up commits per
/// `rtyper-cutover.md`'s P8.3a row.
pub fn function_graph_to_pygraph(
    _legacy: &FunctionGraph,
    _annotations: &AnnotationState,
) -> Result<PyGraph, TyperError> {
    Err(TyperError::message(
        "function_graph_to_pygraph: P8.3a scaffolding only; \
         body not yet implemented (see .claude/plans/rtyper-cutover.md)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: the stub returns the documented "not yet implemented"
    /// error rather than panicking, so callers wiring against the
    /// signature in P8.4 can rely on `.is_err()` short-circuiting.
    #[test]
    fn function_graph_to_pygraph_stub_surfaces_typer_error() {
        let legacy = FunctionGraph::new("smoke");
        let annotations = AnnotationState::new();
        let err = function_graph_to_pygraph(&legacy, &annotations)
            .expect_err("scaffolding stub must error, not succeed");
        let rendered = format!("{err}");
        assert!(
            rendered.contains("P8.3a"),
            "stub error should cite the cutover phase tag, got: {rendered}"
        );
    }
}
