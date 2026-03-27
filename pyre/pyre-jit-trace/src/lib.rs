//! pyre-jit-trace: Trace-time JIT for pyre.
//!
//! This crate contains MIFrame (the meta-interpreter frame) and all
//! trace-time logic. It is compiled as a separate compilation unit
//! from pyre-jit's eval_loop_jit to prevent MIFrame's monomorphization
//! of `execute_opcode_step<E>` from bloating the eval loop's codegen.

pub mod callbacks;
pub mod descr;
pub mod driver;
pub mod frame_layout;
pub mod helpers;
pub mod metainterp;
pub mod state;
pub mod trace;
pub mod virtualizable_spec;

/// Auto-generated trace functions from majit-analyze.
#[allow(dead_code, unused_imports)]
pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}

// Re-export top-level auto-generated functions for crate-level access
include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
