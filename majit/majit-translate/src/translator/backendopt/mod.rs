//! `translator/backendopt/` — mirror of upstream
//! `rpython/translator/backendopt/`. Houses the post-annotator graph
//! optimisations (SSA conversion, escape analysis, etc.).
//!
//! Each file name matches upstream (e.g. `ssa.py` → `ssa.rs`).
//!
//! # None of these passes runs in the JIT codegen path
//!
//! Every pass here consumes [`crate::flowspace::model::FunctionGraph`]
//! (`GraphRef` — the upstream `Rc<RefCell<Block>>` / `Link` / `GraphFunc`
//! shape).  The codewriter consumes a *different* type,
//! [`crate::model::FunctionGraph`] — the slot-indexed `Vec<Block>` /
//! `BlockId` IR — and that is the graph a `JitCode` is built from
//! (`codewriter::CodeWriter::transform_graph_to_jitcode`).  The only
//! bridge between them is one-directional:
//! `rtyper::flowspace_adapter::function_graph_to_flowspace` lifts legacy
//! → flowspace, and nothing converts back.  The flowspace arm exists
//! solely as the dual-gate *type resolver*: it hands back a
//! `LegacyToTyped` variable map whose `Variable.concretetype` cells the
//! codewriter copies onto the legacy graph, then discards the flowspace
//! graph.  So a pass that rewrites graph structure here rewrites graphs
//! that are thrown away.
//!
//! Reachability compounds this.  `all::backend_optimizations` is called
//! only from `driver::TranslationDriver::task_backendopt_lltype`, and a
//! `TranslationDriver` is constructed only by `translator::interactive`,
//! `driver::from_targetspec`, and tests — the translate-to-C port.  The
//! production path (`pyre-jit-trace/build.rs` →
//! `analyze_multiple_pipeline_with_modules` →
//! `CodeWriter::drain_pending_graphs`) never builds one.  Concretely:
//! `inline::always_inline` reads `GraphFunc._always_inline_`, which
//! `rtyper::cutover::lift_callee_to_pygraph` is the sole writer of, so
//! `#[majit_macros::always_inline]` has no effect on shipped jitcodes —
//! hinted functions still emit their own standalone jitcode.  Before
//! reaching for any pass in this directory, check which
//! `FunctionGraph` your consumer holds.  The legacy-IR graph rewriter
//! is [`crate::inline`] — also unwired, and it does not handle a
//! raising callee (it has no `exceptblock` arm at all), so tests
//! passing there is not evidence it is safe to wire up.

pub mod all;
pub mod canraise;
pub mod collectanalyze;
pub mod constfold;
pub mod finalizer;
pub mod gilanalysis;
pub mod graphanalyze;
pub mod inline;
pub mod innerloop;
pub mod malloc;
pub mod merge_if_blocks;
pub mod removeassert;
pub mod removenoops;
pub mod ssa;
pub mod stat;
pub mod storesink;
pub mod support;
pub mod writeanalyze;
