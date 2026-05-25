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
//! ## Step 5 retirement status (2026-05-25)
//!
//! All correctness gates closed:
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
//! - Production validation: `check.py --backend dynasm` 39/39 +
//!   `--backend cranelift` 39/39 under both default and
//!   `--features mir-frontend
//!   PYRE_MIR_FRONTEND_LLBC=pyre-object.ullbc:pyre-interpreter.ullbc`.
//! - `merge_ast_only_functions` (Step 4.5.d hybrid backfill) —
//!   retired 2026-05-25 (commit `8a85391cdb`).  Dead in production
//!   (LLBC covers all functions) and unreached by tests (no test
//!   sets `PYRE_MIR_FRONTEND_LLBC`).
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
//! them.  AST stays the cargo-default front-end because Charon
//! extraction is a separate out-of-band build step
//! (`scripts/extract-llbc.sh` requires the pinned Charon nightly).
//! Removing the shims requires either:
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
//! with the cutover feature-gated and shim-removal deferred — issue
//! #97 acceptance criterion "removed only if production lowering no
//! longer depends on it" is honoured by leaving the shims in place.
//!

pub mod ast;
pub mod mir;
pub mod raise;

pub use ast::{
    AstGraphOptions, SemanticFunction, SemanticProgram, StructFieldRegistry,
    build_semantic_program, build_semantic_program_from_parsed_files,
    build_semantic_program_from_parsed_files_with_statics,
};
