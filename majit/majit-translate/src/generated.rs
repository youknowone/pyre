//! Phase F: static registry of `Arc<JitCode>`s produced from
//! pyre-interpreter handler graphs.
//!
//! ## Positioning (PRE-EXISTING-ADAPTATION)
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
//! `all_jitcodes()` → `&'static AllJitCodes`, the process-wide registry
//! keyed by `CallPath`. First call performs the full pipeline via
//! `analyze_multiple_pipeline`. Subsequent calls are O(1) reads of a
//! `OnceLock`.
//!
//! `AllJitCodes` itself lives on the parity layer at
//! `crate::codewriter::AllJitCodes` and is re-exported here as a
//! convenience for downstream consumers that already import
//! `crate::generated`.
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

use std::sync::OnceLock;

pub use crate::codewriter::AllJitCodes;

/// PRE-EXISTING-ADAPTATION: Rust source → FunctionGraph bridge. RPython's
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
/// - `pyopcode.rs` — freestanding `opcode_*` handlers.
/// - `eval.rs` — `PyFrame` trait impls (LocalOpcodeHandler /
///   SharedOpcodeHandler / ControlOpcodeHandler / …).
/// - `pyframe.rs` — inherent `impl PyFrame` helpers (push / pop /
///   peek / check_exc_match).
/// - `shared_opcode.rs` — freestanding `opcode_make_function`,
///   `opcode_call`, `opcode_build_{list,tuple,map}`,
///   `opcode_store_subscr`, `opcode_list_append`,
///   `opcode_unpack_sequence`, `opcode_load_attr`, `opcode_store_attr`.
///   These are imported at `pyopcode.rs:6` and called directly from
///   default trait methods (pyopcode.rs:821). Before their inclusion,
///   `analyze_multiple_pipeline` would report them as unresolved
///   `direct_call` targets.
///
/// `build.rs` carries a parallel `cargo:rerun-if-changed=...` entry for
/// every string in this manifest; keep the two lists in lock-step.
const PYRE_JIT_GRAPH_SOURCES: &[&str] = &[
    include_str!("../../../pyre/pyre-interpreter/src/pyopcode.rs"),
    include_str!("../../../pyre/pyre-interpreter/src/eval.rs"),
    include_str!("../../../pyre/pyre-interpreter/src/pyframe.rs"),
    include_str!("../../../pyre/pyre-interpreter/src/shared_opcode.rs"),
];

static ALL_JITCODES: OnceLock<AllJitCodes> = OnceLock::new();

/// Access the process-wide pyre-interpreter JitCode registry.
///
/// First call performs the full pipeline (see [`build`]). Subsequent calls
/// are O(1). A panic inside `build` poisons the `OnceLock` and every
/// subsequent caller will panic too — by design, since a malformed
/// handler graph is a hard parity violation that should surface loudly.
pub fn all_jitcodes() -> &'static AllJitCodes {
    ALL_JITCODES.get_or_init(build)
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
    // KNOWN DEVIATION (tracked by Task #100): this path uses the
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
    //   caller. `generated::all_jitcodes()` is different: it caches one
    //   process-wide, monomorphization-neutral registry in a `OnceLock`,
    //   so it cannot pick one concrete instantiation without changing the
    //   meaning of the graph set it exposes.
    //
    // In short, this registry is parity-accurate for graph discovery and
    // JitCode bodies, but intentionally not for `fnaddr`. The symbolic
    // fallback is therefore part of the current API contract and is locked
    // down by the unit tests below.
    let result = crate::analyze_multiple_pipeline(PYRE_JIT_GRAPH_SOURCES);
    AllJitCodes {
        by_path: result.jitcodes_by_path,
        in_order: result.jitcodes,
    }
}

#[cfg(test)]
mod tests {
    use super::all_jitcodes;
    use crate::call::symbolic_fnaddr_for_path;
    use crate::parse::CallPath;

    #[test]
    fn generic_handler_graphs_keep_symbolic_fnaddr_surface() {
        let reg = all_jitcodes();

        for path in [
            CallPath::from_segments(["execute_opcode_step"]),
            CallPath::from_segments(["opcode_load_const"]),
            CallPath::from_segments(["opcode_build_list"]),
        ] {
            let jitcode = reg
                .by_path
                .get(&path)
                .unwrap_or_else(|| panic!("missing JitCode for path {:?}", path.segments));
            assert_eq!(
                jitcode.fnaddr,
                symbolic_fnaddr_for_path(&path),
                "generated::all_jitcodes() unexpectedly stopped using the symbolic fnaddr \
                 fallback for {:?}; if this became a real fnaddr, update the contract \
                 comment above with the chosen monomorphization strategy",
                path.segments
            );
        }
    }
}
