//! majit — Meta-tracing JIT compiler framework.
//!
//! This is the facade crate that re-exports all majit sub-crates under
//! a single dependency. Instead of depending on `majit-ir`, `majit-opt`,
//! `majit-meta`, etc. individually, users can depend on `majit` alone:
//!
//! ```toml
//! [dependencies]
//! majit = { path = "path/to/majit/majit" }
//! ```
//!
//! Then use the sub-crates as modules:
//!
//! ```rust,ignore
//! use majit::ir::{OpCode, OpRef, Type, Value};
//! use majit::meta::MetaInterp;
//! use majit::opt::optimizer::Optimizer;
//! use majit::codegen::Backend;
//! use majit::cranelift::CraneliftBackend;
//! use majit::trace::recorder::Trace;
//! use majit::gc;
//! use majit::runtime;
//! ```

/// IR definitions: opcodes, types, values, descriptors.
pub use majit_ir as ir;

/// Optimization pipeline: 8-pass default + auxiliary passes.
pub use majit_opt as opt;

/// Trace recording: hot counting, recorder, warm state.
pub use majit_trace as trace;

/// Backend abstraction: Backend trait, JitCellToken, DeadFrame.
pub use majit_codegen as codegen;

/// Cranelift backend: native code generation.
pub use majit_codegen_cranelift as cranelift;

/// Meta-interpreter: JitDriver, MetaInterp, JitState, resume, blackhole.
pub use majit_metainterp as metainterp;

/// GC support: nursery, oldgen, write barriers, card marking.
pub use majit_gc as gc;

/// Runtime helpers: jit_merge_point, can_enter_jit.
pub use majit_runtime as runtime;

/// Proc macros: #[jit_driver], #[jit_interp], #[jit_inline], #[jit_module].
pub use majit_macros as macros;
