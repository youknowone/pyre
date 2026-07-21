#![allow(
    dead_code,
    non_snake_case,
    private_interfaces,
    unsafe_op_in_unsafe_fn,
    unused_doc_comments,
    unused_imports,
    unused_unsafe,
    unused_variables
)]

//! pyre-jit-trace: Trace-time JIT for pyre.
//!
//! This crate contains MIFrame (the meta-interpreter frame) and all
//! trace-time logic. It is compiled as a separate compilation unit
//! from pyre-jit's eval_loop_jit to prevent MIFrame's monomorphization
//! of `execute_opcode_step<E>` from bloating the eval loop's codegen.

// Self-alias so include!()'d codegen written for `majit-translate`'s
// crate name keeps compiling when its source is also `include!`d into
// this crate's `generated*` modules (jit_trace_gen.rs).  Allows generic
// bounds like `F: pyre_jit_trace::walker_frame_ops::WalkerFrameOps` to
// resolve from both sides.
extern crate self as pyre_jit_trace;

pub mod assembler;
pub mod callbacks;
pub mod descr;
pub mod driver;
pub mod frame_layout;
pub mod helpers;
pub mod jitcode_dispatch;
pub mod jitcode_runtime;
pub mod liveness;
pub mod pyjitcode;
pub mod pyjitpl;
pub mod pyre_cpu;
pub mod runtime_fnaddr_patch;
pub mod state;
pub mod super_inst_expand;
mod trace_opcode;
pub mod unpack_state;
pub use pyjitcode::{PyJitCode, PyJitCodeMetadata};
pub mod trace;
pub mod virtualizable_gen;
pub mod virtualizable_spec;
pub mod walker_frame_ops;

/// Build a standalone trace context on a libtest thread after installing the
/// interpreter's test-only dict hash hooks for that thread.
#[cfg(test)]
pub(crate) fn trace_ctx_for_test(num_inputs: usize) -> majit_metainterp::TraceCtx {
    pyre_interpreter::test_hooks::install_hash_hook();
    majit_metainterp::TraceCtx::for_test(num_inputs)
}

// pyre-jit-trace local invariant: PyFrame's `_virtualizable_` declares
// exactly one extra red (ec, see `virtualizable_gen.rs:29-31` and
// `pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`).
// `majit-macros::virtualizable!` itself is generic over `extra_reds.len()`
// (mod.rs), so this assertion is *pyre-local*. Tracing-time helpers
// that seed/push the ec slot rely on this invariant; bumping it requires
// re-auditing every ec wiring callsite.
const _: () = assert!(
    virtualizable_gen::NUM_EXTRA_REDS == 1,
    "pyre's PyFrame virtualizable layout requires exactly one extra red (ec)",
);

/// `PYRE_PROBE_SUBSCR` env-var gate cached once on first read. The
/// state.rs/jitcode_dispatch.rs probe sites are on hot paths; sampling
/// `std::env::var_os` on every cache hit would dominate the cost when
/// the probe is off.
pub(crate) fn probe_subscr_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_PROBE_SUBSCR").is_some())
}

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

// Fixed trace helpers (operator dispatch tables, concrete computation,
// unbox/box/binop trace primitives, typed `generated_*` operations). These
// are hand-maintained Rust — the `pyjitpl.py`/`executor.py` analogs — not
// translator output, so they live in a real module. Re-exported at crate
// root because call sites reference them as `crate::<name>`.
#[allow(dead_code, unsafe_op_in_unsafe_fn, unused_imports, unused_variables)]
mod trace_helpers;
pub use trace_helpers::*;
