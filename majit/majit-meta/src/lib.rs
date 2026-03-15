//! `majit-meta`: Meta-tracing automation layer for the majit JIT framework.
//!
//! Provides [`MetaInterp`] — a high-level JIT engine that handles the full
//! lifecycle: warm counting → tracing → optimization → compilation → execution.
//!
//! Interpreter authors only need to:
//! 1. Call [`MetaInterp::on_back_edge`] at backward jumps
//! 2. Record IR ops via [`TraceCtx`] during tracing
//! 3. Provide state extraction/restoration logic
//!
//! Everything else (constant management, FailDescr/CallDescr creation,
//! optimizer invocation, backend compilation, I/O buffering) is automated.

extern crate self as majit_meta;

use majit_ir::{OpRef, Type};

pub mod blackhole;
mod call_descr;
mod constant_pool;
mod driver;
mod fail_descr;
pub mod io_buffer;
mod jit_state;
mod jitcode;
mod meta_interp;
pub mod parity;
pub mod quasi_immut;
pub mod resume;
mod symbolic_stack;
mod trace_ctx;
pub mod virtual_ref;
pub mod virtualizable;

pub use call_descr::{make_call_assembler_descr, make_call_descr};
pub use constant_pool::ConstantPool;
pub use driver::JitDriver;
pub use fail_descr::{make_fail_descr, make_fail_descr_typed};
pub use io_buffer::{
    emit_commit_io, io_buffer_commit, io_buffer_discard, io_buffer_write, io_buffer_write_fmt,
};
pub use jit_state::{JitState, PendingFieldWriteLayout};
pub use jitcode::{
    trace_jitcode, ClosureRuntime, JitArgKind, JitCallArg, JitCode, JitCodeBuilder, JitCodeMachine,
    JitCodeRuntime, JitCodeSym, LivenessInfo, MIFrame, MIFrameStack,
};
pub use majit_codegen::CompiledTraceInfo;
pub use meta_interp::{
    BackEdgeAction, BlackholeRunResult, CompiledExitLayout, CompiledTerminalExitLayout,
    CompiledTraceLayout, DeadFrameArtifacts, DetailedDriverRunOutcome, DriverRunOutcome,
    GuardRecoveryAction, InlineDecision, JitHooks, MetaInterp, RawCompileResult,
};
pub use parity::{assert_trace_parity, normalize_ops, normalize_trace, TraceParityCase};
pub use quasi_immut::QuasiImmut;
pub use symbolic_stack::SymbolicStack;
pub use trace_ctx::{DeclarativeJitDriver, JitDriverDescriptor, TraceCtx, VableSyncField};

/// Whether `MAJIT_LOG` is set, cached at first access.
pub fn majit_log_enabled() -> bool {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("MAJIT_LOG").is_ok());
    *ENABLED
}

/// Result of tracing a single instruction.
///
/// Returned by the interpreter's `trace_instruction()` function
/// to indicate what the framework should do next.
#[derive(Debug)]
pub enum TraceAction {
    /// Continue tracing the next instruction.
    Continue,
    /// Close the loop (back-edge to header detected).
    CloseLoop,
    /// Close the loop with explicit jump arguments supplied by the tracer.
    CloseLoopWithArgs { jump_args: Vec<OpRef> },
    /// Finish the trace with terminal output values.
    Finish {
        finish_args: Vec<OpRef>,
        finish_arg_types: Vec<Type>,
    },
    /// Abort the current trace (recoverable — may retry later).
    Abort,
    /// Abort the current trace permanently (never trace this location again).
    AbortPermanent,
}

/// Marker macro for the tracing merge point.
///
/// When used with `#[jit_interp]`, this is replaced with `driver.merge_point(...)`.
/// When used standalone, this is a no-op (interpreter runs without tracing).
#[macro_export]
macro_rules! jit_merge_point {
    () => {};
    ($($tt:tt)*) => {};
}

/// Marker macro for the back-edge entry point.
///
/// When used with `#[jit_interp]`, this is replaced with `driver.back_edge(...)`.
/// When used standalone, this is a no-op.
#[macro_export]
macro_rules! can_enter_jit {
    ($($tt:tt)*) => {};
}

// ── we_are_jitted / JIT mode flag ──
// Re-exported from majit-codegen so both meta and backend can access it.
pub use majit_codegen::{set_jitted, we_are_jitted, JittedGuard};
