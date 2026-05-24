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
//! ## Step 5 retirement plan (gated on Step 4.5 production cutover)
//!
//! Once the MIR-driven driver (`front::mir`) reaches production
//! coverage parity, the four AST-side CFG-reconstruction workarounds
//! become reachable from the AST path only and can be lifted out:
//!
//! - `front::ast::lazy_install_local_at_current_block_var`
//! - `front::ast::can_thread_variable_to_block`
//! - `front::ast::lower_if_expr`'s fallback branch
//! - `front::ast::GraphBuildContext` per-scope binding tracking
//!
//! Each compensates for the recursive walk's inability to see the CFG
//! ahead of time.  MIR has the CFG explicit, so these have no analog
//! in `front::mir`.  Retirement waits on:
//!
//! 1. Step 4.5 downstream-consumer widenings: `portal_targets`
//!    (needs Charon to surface `#[majit_macros::portal]`),
//!    `function_hints` (needs `elidable*` / `oopspec` attribute
//!    surfacing — closed via Step 4.5.b hybrid pass),
//!    `immutable_fields` (`#[majit_macros::immutable]` — closed via
//!    Step 4.5.c hybrid pass).
//! 2. The Charon dedup-table widening (Step 4.3.c.ext) so
//!    `fn_return_types` carries resolved ADT paths rather than
//!    `ty#N` labels — closed via Step 4.5.c hybrid pass (AST
//!    populates `fn_return_types` from syn-source type strings;
//!    same surface treatment as struct_fields).
//! 3. Production validation under `--features mir-frontend
//!    PYRE_MIR_FRONTEND_LLBC=...` passing `check.py` — LANDED
//!    2026-05-25 (`check.py` 39/39 dynasm + 39/39 cranelift under
//!    `PYRE_MIR_FRONTEND_LLBC=pyre-object.ullbc:pyre-interpreter.ullbc`).
//! 4. Step 4.5.e classdef hybrid pass — pending.  MIR-derived
//!    `Variable.annotation` carries `SomeInstance{ classdef: None }`;
//!    downstream annotator passes that resolve `.field` access via
//!    `SomeInstance.getattr` panic (one remaining lib test:
//!    `generated::tests::generic_handler_graphs_keep_symbolic_fnaddr_surface`).
//!    Closing this needs a pass that walks parsed_files and pre-
//!    populates the annotator's classdef bookkeeper so
//!    `init_someinstance_overrides` can resolve field reads on
//!    MIR-derived `SomeInstance`.
//!
//! Until (4) lands, the AST front-end remains the source of the
//! four shims above and Step 5 deletion stays gated.  The hybrid
//! cutover (Step 4.6, `8e3c3731b1`) reduced MIR-only-mode lib test
//! failures from 9 to 1 while keeping production validation green.
//!

pub mod ast;
pub mod mir;
pub mod raise;

pub use ast::{
    AstGraphOptions, SemanticFunction, SemanticProgram, StructFieldRegistry,
    build_semantic_program, build_semantic_program_from_parsed_files,
    build_semantic_program_from_parsed_files_with_statics,
};
