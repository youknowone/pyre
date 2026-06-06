//! pyre-jit-trace: Trace-time JIT for pyre.
//!
//! This crate contains MIFrame (the meta-interpreter frame) and all
//! trace-time logic. It is compiled as a separate compilation unit
//! from pyre-jit's eval_loop_jit to prevent MIFrame's monomorphization
//! of `execute_opcode_step<E>` from bloating the eval loop's codegen.

pub mod assembler;
pub mod callbacks;
pub mod canonical_bridge;
pub mod descr;
pub mod driver;
pub mod frame_layout;
pub mod helpers;
pub mod jitcode_dispatch;
pub mod jitcode_runtime;
pub mod liveness;
pub mod metainterp;
pub mod pyjitcode;
pub mod pyre_cpu;
pub mod runtime_fnaddr_patch;
pub mod shadow_walker;
pub mod state;
pub mod super_inst_expand;
mod trace_opcode;
pub use pyjitcode::{PyJitCode, PyJitCodeMetadata};
pub use trace_opcode::{production_blackhole_handles, production_walker_handles};
pub mod trace;
pub mod virtualizable_gen;
pub mod virtualizable_spec;

// pyre-jit-trace local invariant: PyFrame's `_virtualizable_` declares
// exactly one extra red (ec, see `virtualizable_gen.rs:29-31` and
// `pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`).
// `majit-macros::virtualizable!` itself is generic over `extra_reds.len()`
// (mod.rs:515), so this assertion is *pyre-local*. Tracing-time helpers
// that seed/push the ec slot rely on this invariant; bumping it requires
// re-auditing every ec wiring callsite — see the v3 plan at
// `~/.claude/plans/ec-wiring-gentle-wave.md`.
const _: () = assert!(
    virtualizable_gen::NUM_EXTRA_REDS == 1,
    "pyre's PyFrame virtualizable layout requires exactly one extra red (ec)",
);

/// Auto-generated trace functions from majit-translate.
#[allow(dead_code, unsafe_op_in_unsafe_fn, unused_imports, unused_variables)]
pub mod generated {
    use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}

// Re-export top-level auto-generated functions for crate-level access.
// Keep generated-code lint allowances scoped to this include wrapper.
#[allow(dead_code, unsafe_op_in_unsafe_fn, unused_variables)]
mod generated_root {
    use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}
pub use generated_root::*;

// Auto-generated `OpcodeHandler` trait impls. Lives in a separate file
// because jit_trace_gen.rs is `include!`d twice (once inside `pub mod
// generated`, once at crate root) and trait impls cannot be duplicated.
use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
include!(concat!(env!("OUT_DIR"), "/jit_trace_trait_impls.rs"));
