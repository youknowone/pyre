//! `flowspace/rust_source/` — Rust AST → `flowspace::FunctionGraph`
//! adapter.
//!
//! pyre-interpreter's source is Rust, not Python, so upstream
//! `rpython/flowspace/objspace.py:38-53 build_flow(func)` cannot consume
//! it directly. This module is the Position-2 adaptation (see
//! `.claude/plans/annotator-monomorphization-tier1-abstract-lake.md` —
//! "Position 2 — Rust AST adapter into unchanged flowspace"): walk a
//! `syn::ItemFn` and emit the same `flowspace::FunctionGraph` shape
//! `build_flow(GraphFunc)` would have emitted for the equivalent
//! Python bytecode.
//!
//! The downstream consumers (`annotator/*`, `classdesc`, rtyper) run
//! unchanged — the adapter is the only place the "input side is Rust"
//! divergence lives.
//!
//! ## Scope of the M2.5a skeleton
//!
//! This is the initial `build_flow_from_rust` entry point per the
//! plan's M2.5a bullet. It covers the "no control flow" core:
//!
//! - Function signature: non-generic, non-async, non-unsafe,
//!   identifier-only parameters, no `self` receiver.
//! - Body: sequence of `let <ident> = <expr>;` statements followed by
//!   an expression-tail `return`.
//! - Expressions: integer / bool literals, identifier paths (local
//!   reference), 16 `BinOp`s matching upstream `operation.py:475
//!   add_operator` entries.
//!
//! Anything outside that scope (control flow, method calls, struct
//! literals, closures, …) rejects via [`AdapterError::Unsupported`].
//! Control flow lands in M2.5b; method calls + trait dispatch in
//! M2.5c; struct/enum/tuple literals in M2.5d; full
//! `execute_opcode_step` in M2.5e.
//!
//! ## Output shape
//!
//! A `FunctionGraph` with exactly one non-terminal block:
//!
//! ```text
//!   startblock([p_0, p_1, …, p_n])
//!     op_0
//!     op_1
//!     …
//!     op_m
//!     → returnblock(tail_value)
//! ```
//!
//! Matching upstream `test_model.py:13-43` + `test_ssa.py:55-88`
//! construction idioms for straight-line functions.

pub mod build_flow;
pub mod register;

pub use build_flow::{AdapterError, build_flow_from_rust};
pub use register::build_host_function_from_rust;
