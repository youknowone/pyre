//! Front-end scaffolding for semantic graph construction.
//!
//! ## Positioning
//!
//! This module bridges **Rust source (`syn::ItemFn`)** to the **`FunctionGraph`** type that the rest of the codewriter pipeline (`jtransform`, `flatten`, `regalloc`, `liveness`, `assembler`) consumes.
//!
//! RPython has no direct counterpart. In upstream, `rpython/jit/codewriter/codewriter.py:74 CodeWriter.make_jitcodes()` is handed `translator.graphs` — graphs already produced by `rpython/rtyper/` from RPython source. The codewriter never sees interpreter source files.
//!
//! pyre cannot inherit that assumption. Rust functions must become `FunctionGraph`s somewhere, and this module is where. Every file under `front/` is Rust-specific lowering that has no RPython structural match.
//!
//! ## Why this is the right layer
//!
//! - The boundary coincides with an upstream boundary: `FunctionGraph` is the line-by-line analogue of RPython `FlowGraph` / rtyper's post-translation graph form. Everything downstream (jit_codewriter) consumes the same shape RPython consumes.
//! - Keeping the adapter inside `front/` means no new opnames, no new `OpKind`, no new jitcode keys leak into the parity layer.
//! - Upstream conventions (`SpaceOperation`, `Block.inputargs`, `Terminator`) are re-used without modification.
//!
//! ## Out of scope
//!
//! - New IR opkinds (`OpKind::TryOp` and similar) are forbidden here. `?` / `PyResult` must be lowered to exceptional successor edges of the existing `Terminator`, matching `rpython/translator/exceptiontransform.py` + `rpython/jit/codewriter/jtransform.py:rewrite_op_direct_call`.
//! - New JitCode key schemas (variant-keyed maps, opcode-to-fragment lookups) are forbidden. The canonical output is `{graph: JitCode}` per `rpython/jit/codewriter/call.py:155 enum_pending_graphs` and `rpython/jit/codewriter/codewriter.py:33 transform_func_to_jitcode`.
//!
//! ## Maintenance rule
//!
//! Every non-trivial addition to this module must include a comment citing the RPython file:line it replaces or bridges. If no such line exists, the addition is further pyre-specific deviation and must be justified explicitly in the commit message.
//!
//! ## Step 1 — dyn-Trait / impl-Trait classification (closed)
//!
//! Issue #97 framed a source refactor — chiefly enum-ifying the 8-impl
//! `dyn DictStrategy` hierarchy — as a prerequisite for Charon.  The
//! Step 0/1 extraction audit refuted that premise: Charon extracts every
//! `dyn Trait` / `impl Trait` site in the JIT-consumed crates
//! (`pyre-object`, `pyre-interpreter`, `pyre-module`) cleanly, and
//! **zero** of the 36 `Dynamic` fat-pointer calls in
//! `pyre-interpreter.ullbc` are `DictStrategy::*` — dict dispatch is
//! statically-resolved `Trait`-kind, not virtual.  No RPITIT / GAT /
//! trait alias appears in scope; the only extraction gap is std
//! `thread_local!` accessor stubs, treated as opaque ops.  (The
//! call-classification counts that back this live in
//! `majit-charon-reader`'s README.)
//!
//! Decision: **no source refactor is required to unblock Charon.**  This
//! driver targets the as-extracted ULLBC and treats `Dynamic` calls as
//! first-class opaque indirect calls (type-flow devirtualization only if
//! a hot path later demands it).  The `dyn Trait` refactors the issue
//! proposed are optional post-cutover polish, not gating work:
//! `dyn DictStrategy` (8 impls) deferred; `dyn ActionFlagOps` (1 impl) a
//! mechanical retire; `dyn AsyncActionOps` (pluggable registry) and
//! `dyn FnMut` (GC visitors) stay `dyn`; `impl Trait` iterator returns
//! extract fine.
//!
//! ## Step 6 cutover status (2026-05-25)
//!
//! - **Step 6.A** — `majit/charon-corpus` retained as the corpus
//!   + reader integration in
//!   `majit-charon-reader::tests::corpus` and
//!   `majit-translate::tests::test_mir_frontend` are now the
//!   authoritative regression oracle.
//! - **Step 6.B** — `auto_discover_workspace_llbc_paths` in
//!   `lib.rs` resolves `<workspace>/build/llbc/{pyre-object,
//!   pyre-interpreter}.ullbc` when `PYRE_MIR_FRONTEND_LLBC` is
//!   unset, so production builds engage the MIR cutover without
//!   per-invocation env-var setup.  Test fixtures stay on AST via
//!   the `module_path`-empty + `parsed_files.len() < 50`
//!   fingerprint.
//! - **Step 6.C** — `default = ["mir-frontend"]` on
//!   `majit-translate` (commit `cd90129089`).  `cargo build
//!   --workspace`, `cargo test -p majit-translate --lib` (2786
//!   passed), and `pyre/check.py --backend {dynasm,cranelift}`
//!   (39/39 each) all run through the MIR cutover.
//! - **Step 6.D** — AST is fallback-only.
//!   `build_semantic_program_via_active_frontend` (lib.rs:137)
//!   routes through `front::mir::build_semantic_program_from_llbcs`
//!   whenever an LLBC source resolves; the AST builder runs only
//!   for contributors on stable Rust without Charon installed.
//!   Production binaries built through `pyre-jit-trace/build.rs`
//!   always engage MIR because `auto_discover_workspace_llbc_paths`
//!   fingerprints production input.
//!
//! ### Correctness gates that previously held the cutover back
//!
//! - `function_hints` from `#[majit_macros::elidable*]` /
//!   `#[oopspec]` — closed via Step 4.5.b hybrid pass.
//! - `immutable_fields` from `#[majit_macros::immutable]` — closed
//!   via Step 4.5.c hybrid pass.
//! - `fn_return_types` — retired entirely (Slice ③, Task #61).  The
//!   `SemanticProgram` field and its `parse::extract_*` threading
//!   were removed once it was confirmed that no production consumer
//!   read the map; the per-callsite return type is taken from the
//!   MIR-built `SemanticFunction` signature directly.
//! - `classdef` hybrid pass — closed via Step 4.5.e (Task #31,
//!   commit `7242bde6a7`).  Root cause was BFS path resolution,
//!   not classdef binding: MIR routed `CallKind::Trait` to a
//!   synthetic path matching no registered graph; fix routes to
//!   `[trait_leaf, method_leaf]` matching `extract_trait_impls`'s
//!   direct-path key.
//! - `merge_ast_only_functions` (Step 4.5.d hybrid backfill) —
//!   retired 2026-05-25 (commit `8a85391cdb`).  Dead in production
//!   (LLBC covers all functions) and unreached by tests (no test
//!   sets `PYRE_MIR_FRONTEND_LLBC`).
//! - `tyref_to_value_type` literal-type schema — widened to
//!   `Literal::{Int,UInt}` / atom `"Bool"` / atom `"Char"` (Step
//!   6.C.1, commit `cd90129089`).  Earlier the helper only
//!   matched the pre-split `Literal::Integer` shape, so every
//!   `usize` / `isize` argument fell back to `Ref` and tripped
//!   `flatten.rs:1155 "switch exitswitch must be int"` for graphs
//!   that switched on integer-typed arguments.
//!
//! ### Step 6.E — AST graph builder retirement (in progress)
//!
//! AST's `build_function_graph_*` family is still reachable through
//! `parse::extract_trait_impls` / `extract_inherent_impl_methods`
//! which build per-method graphs parallel to the MIR-built
//! `program.functions`.  Replacement is multi-slice:
//!
//! - **Slice 1** (`382b21e0ec`) — carve `SemanticProgram`,
//!   `SemanticFunction`, `StructFieldRegistry`, `AstGraphOptions`,
//!   `FlowingError`, `ProgramMetadata` and the
//!   `qualify_type_name_with_imports` helper out of `front::ast`
//!   into `front::semantic`.  `front::ast` retains `pub use`
//!   re-exports.
//! - **Slice 2 (in progress, multiple commits)** — incremental
//!   extractions out of `front::ast` into `front::syn_metadata`,
//!   shrinking the AST surface so eventual deletion is mechanical.
//!   Leaf string-utility helpers `transparent_result_ok_type` /
//!   `nolength_from_array_type_id` landed first (`7a538de2a9`);
//!   subsequent commits moved metadata collectors, syn-walk
//!   operator + path helpers, match-arm classifiers, type-string
//!   rendering helpers, `LoopBodyLocals` + visitors, pure
//!   type-string parsers (with `first_top_level_generic_arg`
//!   dedup), pure path/intrinsic classifiers + `cast_builtin_name`,
//!   `UnaryNotOperandKind` + const/op value-type helpers, and
//!   `classify_fn_arg_ty` + `extract_dyn_trait_root` +
//!   `canonical_pat_name`.  The remaining `ast.rs` core is the
//!   `GraphBuildContext`-bound `lower_*` walker which is structurally
//!   tied to the opcode-arm AST extractor at
//!   `parse.rs:extract_match_arms`.
//! - **Slice 3 (in progress)** — populate
//!   `SemanticFunction.self_ty_root` (`66423f0fc4`) and
//!   `SemanticFunction.trait_root` (`36e3a6520b`) on MIR-built
//!   impl methods so the canonical registration loop in
//!   `lib.rs:1010-1190` can substitute MIR graphs for AST graphs.
//!   - **3.A** (`6fc8da1327`) — trait-default body detection in MIR
//!     via `trait_default_owner_for_fundecl(fd, known_trait_names)`.
//!     Audit on test_phase_f LLBC fixture: trait covered 26→150 / 275
//!     (9% → 55%) after MIR populates `trait_root` for default
//!     bodies; inherent unchanged at 102/125 (82%).
//!   - **3.B** (`fce8698259`) — registration loop in
//!     `analyze_pipeline_from_parsed` (`lib.rs:1010+`) indexes
//!     `program.functions` and substitutes MIR graphs for the
//!     AST graphs `extract_trait_impls`/`extract_inherent_impl_methods`
//!     carried.  AST graph remains the fallback when MIR has no entry.
//!   - **3.C** (`24ecf9e865`) — new `front::semantic::MirGraphLookup`
//!     threaded into `extract_*`; on hit the collector returns the
//!     MIR graph and computes hints from syn attrs directly, skipping
//!     `build_function_graph_with_self_ty_pub`.  The AST graph build
//!     now runs only for entries MIR did not lower (the residual
//!     coverage gap below).
//!   - **3.D** (`102883ec7b`) — `build_semantic_program_from_llbcs`
//!     was deduping merged entries by the bare `SemanticFunction.name`
//!     leaf, so `pyre-object::FloatArray::push` seeded the dedup map
//!     with `"push"` and every later same-named pyre-interpreter impl
//!     method (`PyFrame::{push,__init__,new,ncells,…}`, etc.) was
//!     silently dropped.  Switched dedup key to the full qualified
//!     path (`{module_path}::{name}`).  Audit on the test_phase_f
//!     LLBC fixture:
//!       - trait covered 150 → 242 / 275 (55% → 88%)
//!       - inherent covered 102 → 114 / 125 (82% → 91%)
//!   - **3.E (in progress)** — close the residual MIR-uncovered
//!     entries.  Latest audit (29 trait_missing / 0 inherent_missing
//!     after extract_inherent_impl_methods reached full coverage) on
//!     the `test_make_jitcodes_produces_graph_keyed_output` fixture:
//!       - **26 entries**: `PyreBlackholeAllocator::bh_*` /
//!         `os_str_concat` / `os_str_slice` / `os_uni_concat` /
//!         `os_uni_slice` / `allocate_*` / `box_int` / `box_float`
//!         (the `impl BlackholeAllocator for PyreBlackholeAllocator`
//!         block in `pyre/pyre-jit/src/eval.rs:6537`).
//!       - **3 entries**: `JitSuppressionGuard::drop`,
//!         `GuardCompilingScope::drop`,
//!         `eval::tests::TestJitParamsGuard::drop` (Drop impls in
//!         `pyre/pyre-jit/src/eval.rs`).
//!     All 29 live in the `pyre-jit` crate, which the auto-discover
//!     LLBC set did not include.  Adding `pyre-jit` to
//!     `scripts/extract-llbc.sh` and treating `pyre-jit.ullbc` as an
//!     OPTIONAL artefact in `auto_discover_workspace_llbc_paths`
//!     closes the gap once contributors re-run extraction; the
//!     canonical pair (`pyre-object.ullbc`, `pyre-interpreter.ullbc`)
//!     remains REQUIRED.  AST fallback covers the entries today via
//!     the `extract_*` `graph: None` placeholder path, so production
//!     stays green between extraction and consumption.
//!   - **3.F** — `extract_trait_impls` /
//!     `extract_inherent_impl_methods` no longer call
//!     `build_function_graph_with_self_ty_pub` when the caller
//!     passes a `MirGraphLookup` and the lookup misses.  The
//!     trait-impl branch emits `graph: None` on miss; the inherent
//!     branch skips the method entirely (its `graph` field is
//!     non-Option).  `analyze_pipeline_from_parsed` only constructs
//!     `MirGraphLookup` when the active frontend is genuinely MIR
//!     (`build_semantic_program_via_active_frontend` now returns
//!     `(SemanticProgram, ActiveFrontend)`).  In AST-only mode
//!     (test fixtures with no LLBC) the extractors still run the
//!     legacy AST graph build for every method.  This shrinks the
//!     AST graph builder reach in production to the residual ≈44
//!     entries from 3.E.
//! - **Slice 4** (LANDED, `bde9afa8dd`) — env-gated panic in the
//!   AST-fallback arm of `build_semantic_program_via_active_frontend`.
//!   Setting `PYRE_REQUIRE_MIR_FRONTEND=1` makes the function panic
//!   when no LLBC source resolves.  `pyre/check.py::build_backend`
//!   sets the flag on the cargo build subprocess so production
//!   cutover asserts MIR-only.  Tests that exercise the AST builder
//!   call `front::build_semantic_program_from_parsed_files` directly
//!   and bypass this dispatch — they remain unaffected by the flag.
//! - **Slice 5 (mostly landed)** — AST graph builder unreachable
//!   and entry points deleted.  Net ~9k LOC removed across:
//!     - 12 AST-builder-dependent integration tests
//!       (~2869 LOC).
//!     - `front/ast.rs::mod tests` block (~4396 LOC).
//!     - In-tree `mod tests` blocks in `lib.rs`,
//!       `translator/rtyper/{legacy_pipeline,cutover}.rs`,
//!       `jit_codewriter/call.rs`, `generated.rs`,
//!       `parse.rs` (~2035 LOC + ~80 LOC).
//!     - `parse::collect_function_graphs`,
//!       `tests/test_extractor_flowing_error_propagation.rs`.
//!     - `build_semantic_program*` /
//!       `build_function_graph_*` / `build_graphs_from_items` /
//!       `collect_types_from_items` / `collect_generic_trait_roots`
//!       / `LoweringFnNameGuard` (~620 LOC in `front/ast.rs`).
//!   The `build_semantic_program_via_active_frontend` fallback is
//!   an unconditional panic; `MirGraphLookup` construction in
//!   `analyze_pipeline_from_parsed` is unconditional; `extract_*`
//!   no longer falls back to AST graph builds; production 39/39
//!   dynasm + 39/39 cranelift.
//!
//!   front/ast.rs is now DELETED.  The per-arm body-graph walker
//!   (`lower_expr_into_graph_with_signature`) was retired once every
//!   `execute_opcode_step` arm routed through a seam synthesizer
//!   (tail-call / const-return / raise-stub / wildcard tail-call in
//!   `front::opcode_wrapper`), so no arm needs syn-AST
//!   lowering.  The metadata/type helpers it re-exported live in
//!   `front/semantic` and `front/syn_metadata`; callers reference those
//!   modules directly.  The three AST-side CFG-reconstruction shims
//!   (`lazy_install_local_at_current_block_var`, `lower_if_expr`'s
//!   fallback branch, `GraphBuildContext` per-scope binding tracking)
//!   were retired with the walker (Step 6.E complete).
//!

pub mod llbc_hints;
pub mod mir;
pub mod mir_dispatch;
pub mod opcode_wrapper;
pub mod raise;
pub mod semantic;
pub mod syn_metadata;

pub use semantic::{AstGraphOptions, SemanticFunction, SemanticProgram, StructFieldRegistry};
