//! pyre-jit-trace: Trace-time JIT for pyre.
//!
//! This crate contains MIFrame (the meta-interpreter frame) and all
//! trace-time logic. It is compiled as a separate compilation unit
//! from pyre-jit's eval_loop_jit to prevent MIFrame's monomorphization
//! of `execute_opcode_step<E>` from bloating the eval loop's codegen.

pub mod callbacks;
pub mod descr;
pub mod dispatch_manifest;
pub mod driver;
pub mod frame_layout;
pub mod helpers;
pub mod liveness;
pub mod metainterp;
pub mod state;
mod trace_opcode;
pub use state::set_majit_jitcode;
pub mod trace;
pub mod virtualizable_gen;
pub mod virtualizable_spec;

/// Auto-generated trace functions from majit-codewriter.
#[allow(dead_code, unused_imports)]
pub mod generated {
    use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}

// Re-export top-level auto-generated functions for crate-level access
use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));

// Auto-generated `OpcodeHandler` trait impls. Lives in a separate file
// because jit_trace_gen.rs is `include!`d twice (once inside `pub mod
// generated`, once at crate root) and trait impls cannot be duplicated.
include!(concat!(env!("OUT_DIR"), "/jit_trace_trait_impls.rs"));
