//! Phase F: static registry of `Arc<JitCode>`s produced from
//! pyre-interpreter handler graphs.
//!
//! ## Positioning (TODO)
//!
//! Per parity rule #1 (`CLAUDE.md` majit ↔ RPython Parity Rules): this
//! module has **no RPython counterpart**. Upstream's
//! `rpython/jit/codewriter/codewriter.py:74 make_jitcodes` is handed
//! `translator.graphs` — the rtyper has already materialised every graph
//! in process memory by the time codewriter runs. pyre cannot inherit
//! that assumption: Rust handler sources are on disk in a sibling crate
//! (`pyre/pyre-interpreter/`) and must become `FunctionGraph`s before
//! the codewriter can touch them.
//!
//! The adapter lives at the same logical boundary as the `front/` module
//! (Rust `syn::ItemFn` → `FunctionGraph`). Keeping the adapter here keeps
//! the parity layer (`jit_codewriter/`) untouched: no new opnames, no new
//! `OpKind`, no new jitcode-keying schemas. The pipeline this module
//! drives is exactly the canonical
//! `analyze_multiple_pipeline` (`crate::analyze_multiple_pipeline`) —
//! i.e. the same entry point `rpython/jit/codewriter/codewriter.py:33
//! transform_func_to_jitcode` is wrapped by in the tests.
//!
//! ## What this module provides
//!
//! `with_all_jitcodes(|reg| …)` → closure access to the per-thread
//! pyre-interpreter JitCode registry keyed by `CallPath`. First call on
//! a thread performs the full pipeline via `analyze_multiple_pipeline`;
//! subsequent calls are O(1) reads of a `thread_local!` `OnceCell`.
//! The closure form (rather than `&'static`) avoids forcing
//! `AllJitCodes: Sync`, which would in turn force the interior cells of
//! `Variable` to become thread-safe wrappers — a deviation from
//! RPython's single-thread annotator invariant.
//!
//! `AllJitCodes` itself lives on the parity layer at
//! `crate::codewriter::AllJitCodes` and is re-exported here as a
//! convenience for downstream consumers that already import
//! `crate::generated`.
//!
//! ## Audience: TEST FIXTURE (not production)
//!
//! As of 2026-04-25, `all_jitcodes()` is consumed **only** by
//! majit-translate's own integration tests
//! (`test_phase_f_all_jitcodes.rs`,
//! `test_make_jitcodes_produces_graph_keyed_output.rs`,
//! `tests` mod below). The pyre runtime never imports this function.
//!
//! Production builds run a parallel pipeline at `pyre-jit-trace/build.rs`
//! that calls `analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings`
//! with `pyre_interpreter::jit_trace_fnaddrs()` and then bincode-embeds
//! the resulting `pipeline.jitcodes` into the `pyre-jit-trace` binary
//! (`$OUT_DIR/opcode_jitcodes.bin`). That separate path is what supplies
//! real `JitCode.fnaddr` values to the metainterp / blackhole runtime;
//! `all_jitcodes()` here intentionally retains the **symbolic fnaddr
//! fallback** because exercising the pipeline without `&fnaddr_bindings`
//! is what the unit tests need to assert (graph discovery, alloc-order
//! invariants, by-path keying).
//!
//! The historical description ("route generated::all_jitcodes
//! through fnaddr_bindings instead of symbolic fallback") was written
//! before the build-time bincode path landed in `pyre-jit-trace/build.rs`;
//! production-side fnaddr resolution is therefore already done by that
//! script, and `generated::all_jitcodes` keeps its symbolic-fallback
//! contract as the deliberate test surface. **Do not rewire this module
//! to thread `&fnaddr_bindings` — doing so would either (1) introduce a
//! `majit-translate` ⇒ `pyre-interpreter` dependency that breaks the
//! `majit/* ⊥ pyre/*` invariant, or (2) require parameterising the
//! `OnceCell` (forfeiting memoisation). Both are wrong shapes for what
//! is, in practice, a test fixture.**
//!
//! ## Why this wraps the full pipeline
//!
//! An earlier draft of this module ran its own narrow pipeline
//! (opcode_* free functions + trait impl methods only, with empty
//! `StructFieldRegistry` / `fn_return_types` / `known_struct_names`).
//! That shape is **narrower than `translator.graphs`** upstream assumes
//! and drops structural context:
//!
//! - Inherent impl methods (e.g. `PyFrame::push`, `PyFrame::pop`) are
//!   registered via `extract_inherent_impl_methods` in
//!   `analyze_pipeline_from_parsed`; without them, direct_call targets
//!   like `self.pop()` cannot resolve to a concrete graph.
//! - `struct_fields` / `fn_return_types` / `known_struct_names` carry
//!   array-type identity that `extract_trait_impls` consults; an empty
//!   context silently collapses those identities and the rtyped graph
//!   becomes syntax-only.
//!
//! Re-using `analyze_multiple_pipeline` eliminates both gaps: the same
//! full-context registry the canonical analyzer consumes becomes this
//! module's input.
//!
//! ## What this module does NOT introduce
//!
//! - NOT a new key schema. The canonical key is `CallPath` (matching
//!   `CallControl.jitcodes`, which is `rpython/jit/codewriter/call.py:87
//!   self.jitcodes` keyed by graph identity).
//! - NOT a variant-keyed map. The plan's Phase E acceptance check
//!   (`rg "HashMap<Instruction" majit/majit-translate/src/` = 0) holds.
//! - NOT a new opname family. Every handler is transformed through the
//!   existing `CodeWriter::transform_graph_to_jitcode` without per-arm
//!   special cases.
//!
//! ## Source embedding
//!
//! pyre-interpreter source is pulled in with `include_str!`. `build.rs`
//! additionally emits `cargo:rerun-if-changed=...` on the same paths so
//! cargo rebuilds this crate when either source file changes.

use std::cell::OnceCell;

pub use crate::codewriter::AllJitCodes;

/// TODO: Rust source → FunctionGraph bridge. RPython's
/// rtyper has already produced `translator.graphs` before codewriter runs;
/// pyre lacks that pre-processing and must embed the source here.
///
/// This is the pyre-side equivalent of upstream's "reachable graph set"
/// consumed by `rpython/jit/codewriter/codewriter.py:74 make_jitcodes`.
/// The manifest must cover every Rust source file that defines a
/// function reachable by `direct_call` from a handler graph. pyre's
/// `analyze_multiple_pipeline` resolves cross-file `direct_call`s
/// against the union of `function_graphs` from every source in this
/// list; a callee defined in a file absent from the manifest would be
/// emitted as a residual call (or panic during drain) even though
/// upstream treats it as inlinable graph.
///
/// Current roots:
/// - `pyre-interpreter/src/pyopcode.rs` — freestanding `opcode_*`
///   handlers.
/// - `pyre-interpreter/src/eval.rs` — `PyFrame` trait impls
///   (LocalOpcodeHandler / SharedOpcodeHandler / ControlOpcodeHandler
///   / …).
/// - `pyre-interpreter/src/pyframe.rs` — inherent `impl PyFrame`
///   helpers (push / pop / peek / check_exc_match).
/// - `pyre-interpreter/src/shared_opcode.rs` — freestanding
///   `opcode_make_function`, `opcode_call`,
///   `opcode_build_{list,tuple,map}`, `opcode_store_subscr`,
///   `opcode_list_append`, `opcode_unpack_sequence`, `opcode_load_attr`,
///   `opcode_store_attr`. These are imported at `pyopcode.rs:6` and
///   called directly from default trait methods (pyopcode.rs:821).
///   Before their inclusion, `analyze_multiple_pipeline` would report
///   them as unresolved `direct_call` targets.
/// - `pyre-jit/src/eval.rs` — Phase D0 portal runner `eval_loop_jit`
///   (pyre analogue of upstream `warmspot.py::portal_runner`) and
///   its resume/allocation helpers (`allocate_struct`,
///   `allocate_with_vtable`). Seeding this root lets
///   `find_all_graphs(portal, policy)` find the portal graph for
///   `setup_jitdriver`; opcode handlers become BFS callees, not
///   entry points.
///
/// `build.rs` carries a parallel `cargo:rerun-if-changed=...` entry for
/// every string in this manifest; keep the two lists in lock-step.
/// `(source_text, crate-stripped module_path)` pairs.  The module path
/// matches the form `pyre-jit-trace/build.rs::module_path_from_source_file`
/// emits, so analyzer-side `struct_origins[bare_name] = module_path` and
/// `qualify_type_name_with_imports` produce the same canonical spelling
/// the runtime + production analyser pipeline produce.  Empty module
/// path (test fixtures that bypass module wiring) is reserved for
/// `parse_source`; here every entry carries its real module path.
const PYRE_JIT_GRAPH_SOURCES: &[(&str, &str)] = &[
    (
        include_str!("../../../pyre/pyre-interpreter/src/pyopcode.rs"),
        "pyopcode",
    ),
    (
        include_str!("../../../pyre/pyre-interpreter/src/eval.rs"),
        "eval",
    ),
    (
        include_str!("../../../pyre/pyre-interpreter/src/pyframe.rs"),
        "pyframe",
    ),
    (
        include_str!("../../../pyre/pyre-interpreter/src/shared_opcode.rs"),
        "shared_opcode",
    ),
    (include_str!("../../../pyre/pyre-jit/src/eval.rs"), "eval"),
];

thread_local! {
    /// Per-thread cache for the pyre-interpreter JitCode registry.
    ///
    /// The registry is `thread_local!` rather than a process-wide
    /// `static OnceLock<…>` because `AllJitCodes` transitively holds
    /// `Variable` graphs whose interior `RefCell` / `Cell` cells are
    /// !Sync — matching RPython's single-thread annotator invariant.
    /// All callers below this layer are single-thread test fixtures.
    static ALL_JITCODES: OnceCell<AllJitCodes> = const { OnceCell::new() };
}

/// Access the per-thread pyre-interpreter JitCode registry through a
/// closure.
///
/// First call on a given thread performs the full pipeline (see
/// [`build`]). Subsequent calls are O(1). A panic inside `build`
/// poisons the cell on that thread and every subsequent caller on the
/// same thread will panic too — by design, since a malformed handler
/// graph is a hard parity violation that should surface loudly.
///
/// The closure form (rather than `-> &'static AllJitCodes`) avoids the
/// `T: Sync` requirement that a process-wide static would impose;
/// `AllJitCodes` carries `Variable` graphs whose interior-mutability
/// cells are intentionally !Sync to match RPython's single-thread
/// annotator invariant.
pub fn with_all_jitcodes<R>(f: impl FnOnce(&AllJitCodes) -> R) -> R {
    ALL_JITCODES.with(|cell| f(cell.get_or_init(build)))
}

fn build() -> AllJitCodes {
    // Full canonical pipeline — the same entry point the
    // `test_analyze_pipeline_runs_canonical_graph_path` integration test
    // exercises. Builds a `SemanticProgram` from the embedded sources
    // listed in `PYRE_JIT_GRAPH_SOURCES`, runs `analyze_program`,
    // collects trait impls + inherent impl methods with full
    // struct-field / return-type / known-struct context, wires up
    // jitdriver / portal / oopspec metadata, then calls
    // `grab_initial_jitcodes` + `drain_pending_graphs` through
    // `build_canonical_opcode_dispatch`. The output mirrors RPython
    // `call.py:87 self.jitcodes` (dict) + `call.py:88 self.all_jitcodes`
    // (list).
    //
    // KNOWN DEVIATION: this path uses the
    // symbolic `JitCode.fnaddr` fallback at
    // `crate::call::symbolic_fnaddr_for_path`, NOT upstream's real
    // `getfunctionptr(graph)` surface (`rpython/jit/codewriter/
    // call.py:181-187`).
    //
    // The blocker is wider than "need a binding table":
    //
    // - Many graphs in `PYRE_JIT_GRAPH_SOURCES` are generic source-level
    //   functions, e.g. `pyopcode.rs` / `shared_opcode.rs`
    //   `opcode_*<H: ...>` helpers and trait default methods on
    //   `OpcodeStepExecutor`. A source graph like
    //   `opcode_load_const<H>` has no single concrete Rust fnaddr until a
    //   monomorphization is chosen (`<PyFrame as ...>`,
    //   trace-recorder handler, blackhole handler, ...).
    // - The binding-aware public entry points
    //   (`analyze_multiple_pipeline_with_*_fnaddr_bindings`) work for
    //   nongeneric helper surfaces whose concrete fnaddrs are known to a
    //   caller. `generated::with_all_jitcodes` is different: it caches one
    //   per-thread, monomorphization-neutral registry in a thread-local
    //   `OnceCell`, so it cannot pick one concrete instantiation without
    //   changing the meaning of the graph set it exposes.
    //
    // In short, this registry is parity-accurate for graph discovery and
    // JitCode bodies, but intentionally not for `fnaddr`. The symbolic
    // fallback is therefore part of the current API contract and is locked
    // down by the unit tests below.
    let sources: Vec<&str> = PYRE_JIT_GRAPH_SOURCES.iter().map(|(s, _)| *s).collect();
    let module_paths: Vec<&str> = PYRE_JIT_GRAPH_SOURCES.iter().map(|(_, m)| *m).collect();
    let result = crate::analyze_multiple_pipeline_with_modules(
        &sources,
        &module_paths,
        &crate::AnalyzeConfig::default(),
        None,
        &|_, _| None,
        &[],
    );
    AllJitCodes {
        by_path: result.jitcodes_by_path,
        in_order: result.jitcodes,
    }
}

#[cfg(test)]
mod tests {
    use super::with_all_jitcodes;
    use crate::call::symbolic_fnaddr_for_path;
    use crate::parse::CallPath;

    #[test]
    fn generic_handler_graphs_keep_symbolic_fnaddr_surface() {
        with_all_jitcodes(|reg| {
            // `execute_opcode_step` is consumed by
            // `build_canonical_opcode_dispatch` (`lib.rs:965-1027`): each match
            // arm body is registered as a synthetic graph keyed under
            // `["__opcode_dispatch__", "<selector>#<arm_id>"]`. The wrapper
            // function itself is not separately registered; the dispatch arms
            // are the "generic handler graphs" whose fnaddr surface this test
            // pins. Pick one arm per representative opcode family (load /
            // build / pop) to lock the symbolic-fnaddr contract.
            for path in [
                CallPath::from_segments(["__opcode_dispatch__", "Instruction::LoadConst#1"]),
                CallPath::from_segments(["__opcode_dispatch__", "Instruction::BuildList#30"]),
                CallPath::from_segments(["__opcode_dispatch__", "Instruction::PopTop#14"]),
            ] {
                let jitcode = reg
                    .by_path
                    .get(&path)
                    .unwrap_or_else(|| panic!("missing JitCode for path {:?}", path.segments));
                assert_eq!(
                    jitcode.fnaddr,
                    symbolic_fnaddr_for_path(&path),
                    "generated::with_all_jitcodes unexpectedly stopped using the symbolic fnaddr \
                     fallback for {:?}; if this became a real fnaddr, update the contract \
                     comment above with the chosen monomorphization strategy",
                    path.segments
                );
            }
        });
    }

    /// Phase D acceptance: `eval_loop_jit` is the portal (upstream
    /// `warmspot.py::portal_runner` analogue, seeded into
    /// `find_all_graphs(portal, policy)` at `call.py:57`). When the
    /// portal runner lives in `PYRE_JIT_GRAPH_SOURCES`
    /// (`pyre-jit/src/eval.rs`), the production pipeline must register a
    /// JitCode for it *and* keep the match-arm callee
    /// `execute_opcode_step` reachable as a BFS-discovered inlinee (not
    /// as a separate entry point). A missing `eval_loop_jit` JitCode
    /// here would indicate that `setup_jitdriver` silently fell back to
    /// `execute_opcode_step`, reintroducing the "handler-as-portal"
    /// shape Phase D eliminates.
    #[test]
    fn eval_loop_jit_portal_produces_jitcode_with_handler_closure() {
        with_all_jitcodes(|reg| {
            // Module-qualified paths — see comment on
            // `generic_handler_graphs_keep_symbolic_fnaddr_surface`.
            // `pyre-jit/src/eval.rs` is declared under `"eval"` in
            // `PYRE_JIT_GRAPH_SOURCES`; `pyre-interpreter/src/pyopcode.rs`
            // under `"pyopcode"`.
            let portal = CallPath::from_segments(["eval", "eval_loop_jit"]);
            assert!(
                reg.by_path.contains_key(&portal),
                "Phase D parity: eval_loop_jit must have a JitCode after \
                 analyze_multiple_pipeline runs (production seeds portal via \
                 setup_jitdriver, find_all_graphs BFSes from it). Missing the \
                 entry means the portal fallback unexpectedly chose \
                 execute_opcode_step."
            );
            // `execute_opcode_step` is decomposed by
            // `build_canonical_opcode_dispatch` (`lib.rs:965-1027`) into
            // synthetic per-arm graphs keyed under `__opcode_dispatch__`. The
            // wrapper itself is not registered; the dispatch arms are what
            // make the BFS surface from `eval_loop_jit` reach individual
            // opcode handlers. A missing arm here means
            // `extract_opcode_dispatch_arms` no longer sees the match body, or
            // the synthetic registration step in
            // `build_canonical_opcode_dispatch` regressed.
            let has_arm = reg.by_path.keys().any(|k| {
                k.segments
                    .first()
                    .map(|s| s == "__opcode_dispatch__")
                    .unwrap_or(false)
            });
            assert!(
                has_arm,
                "Phase D parity: per-opcode dispatch arms must be registered \
                 under [\"__opcode_dispatch__\", ...]; absence means the \
                 eval_loop_jit → execute_opcode_step match arms never landed."
            );
        });
    }
}
