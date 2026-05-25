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
//! ## Step 6 cutover status (2026-05-25)
//!
//! - **Step 6.A** — `majit/charon-spike/prototype` marked as deletion
//!   candidate; the corpus + reader integration in
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
//! - `fn_return_types` MIR-native primitive resolution — closed via
//!   Step 4.3.c.ext dedup-body widening (Task #30, commit
//!   `654a65ba80`).  Hybrid AST pass still fills `Result<T, E>` /
//!   `Option<T>` / non-primitive shapes Charon cannot reconstruct.
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
//! ### Remaining AST-side CFG shims (Step 6.E)
//!
//! Four AST-side CFG-reconstruction shims remain reachable through
//! the AST front-end:
//!
//! - `front::ast::lazy_install_local_at_current_block_var`
//! - `front::ast::can_thread_variable_to_block`
//! - `front::ast::lower_if_expr`'s fallback branch
//! - `front::ast::GraphBuildContext` per-scope binding tracking
//!
//! These are AST builder internals compensating for the recursive
//! walker's CFG-blindness.  MIR has CFG explicit and never needs
//! them.  Removing them requires either:
//!
//! - Integrating Charon into the cargo build (separate
//!   infrastructure project), or
//! - Shipping pre-extracted `.ullbc` artefacts in the source tree
//!   (CI-scoped), or
//! - Rewriting the AST builder to not need CFG-blindness
//!   compensation (broad refactor of the recursive walker).
//!
//! Each path is its own multi-session epic outside the Charon
//! mission's stable-Rust acceptance criterion.  The mission ships
//! with the AST fallback intact — issue #97 acceptance criterion
//! "removed only if production lowering no longer depends on it"
//! is honoured because the AST builder is reachable only when no
//! LLBC source resolves.
//!

pub mod ast;
pub mod mir;
pub mod raise;
pub mod semantic;

pub use ast::{
    build_semantic_program, build_semantic_program_from_parsed_files,
    build_semantic_program_from_parsed_files_with_statics,
};
pub use semantic::{AstGraphOptions, SemanticFunction, SemanticProgram, StructFieldRegistry};
