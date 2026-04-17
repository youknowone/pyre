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

extern crate self as majit_metainterp;

use majit_ir::{OpRef, Type};

pub mod blackhole;
pub mod call_descr;
pub(crate) mod compile;
mod constant_pool;
pub(crate) mod executor;
mod fail_descr;
pub mod greenfield;
pub mod io_buffer;
pub mod jit;
mod jit_state;
pub mod jitcode;
mod jitdriver;
pub mod jitexc;
pub mod jitframe;
pub mod optimize;
pub mod optimizeopt;
pub mod parity;
mod pyjitpl;
pub mod quasiimmut;
pub mod resume;
pub mod rvmprof;
mod trace_ctx;
pub mod virtualizable;
pub mod virtualref;
pub mod walkvirtual;

pub use call_descr::{
    make_call_assembler_descr, make_call_assembler_descr_with_vable, make_call_descr,
};
pub use constant_pool::ConstantPool;
pub use fail_descr::{make_fail_descr, make_fail_descr_typed};
pub use io_buffer::{
    emit_commit_io, encode_decimal_i64, io_buffer_commit, io_buffer_discard, io_buffer_write,
    io_buffer_write_fmt, jit_write_number_i64, jit_write_utf8_codepoint,
};
pub use jit_state::{
    DeoptMaterializationCache, JitState, PendingFieldWriteLayout, ResidualVirtualizableSync,
    ResumeDataResult,
};
pub use jitcode::{JitArgKind, JitCallArg, JitCode, JitCodeBuilder, LivenessInfo};
pub use jitdriver::{DeclarativeJitDriver, JitDriver, JitDriverStaticData};
pub use majit_backend::CompiledTraceInfo;
pub use parity::{TraceParityCase, assert_trace_parity, normalize_ops, normalize_trace};
pub use pyjitpl::{
    BackEdgeAction, BlackholeRunResult, BridgeRetraceResult, ClosureRuntime, CompileOutcome,
    CompiledExitLayout, CompiledTerminalExitLayout, CompiledTraceLayout, DeadFrameArtifacts,
    DetailedDriverRunOutcome, DriverRunOutcome, InlineDecision, JitCodeMachine, JitCodeRuntime,
    JitCodeSym, JitHooks, JitStats, MIFrame, MIFrameStack, MetaInterp, MetaInterpGlobalData,
    MetaInterpStaticData, RawCompileResult, StandaloneFrameStack, trace_jitcode,
};
pub use quasiimmut::QuasiImmut;
pub use trace_ctx::{MergePoint, TraceCtx};

/// Whether `MAJIT_LOG` is set, cached at first access.
/// Compute green key from code pointer and PC.
/// Must use the same hash as the front-end's make_green_key.
pub fn green_key_from_code_ptr(code_ptr: usize, pc: usize) -> u64 {
    (code_ptr as u64).wrapping_mul(1000003) ^ (pc as u64)
}

pub fn majit_log_enabled() -> bool {
    std::env::var_os("MAJIT_LOG").is_some()
}

/// Result of tracing a single instruction.
///
/// Returned by the interpreter's `trace_instruction()` function
/// to indicate what the framework should do next.
#[derive(Debug)]
pub enum TraceAction {
    /// Continue tracing the next instruction.
    Continue,
    /// reached_loop_header() compiled the current trace into an existing
    /// target and tracing must stop immediately.
    ///
    /// RPython parity: pyjitpl.py says compile_trace() "raises in case it
    /// works". pyre surfaces that control-flow edge explicitly.
    CompileTrace,
    /// Close the loop (back-edge to header detected).
    CloseLoop,
    /// Close the loop with explicit jump arguments supplied by the tracer.
    ///
    /// RPython parity: the tracer can also pass the explicit loop-header PC
    /// (the backward-jump target / reached loop header).  This lets the
    /// tracing context retarget its green key from the true merge point,
    /// instead of trying to recover it later from virtualizable state.
    CloseLoopWithArgs {
        jump_args: Vec<OpRef>,
        loop_header_pc: Option<usize>,
    },
    /// Finish the trace with terminal output values.
    Finish {
        finish_args: Vec<OpRef>,
        finish_arg_types: Vec<Type>,
    },
    /// Close and compile a segmented loop (force_finish_trace).
    /// pyjitpl.py:1622 _create_segmented_trace_and_blackhole parity.
    /// The trace has GUARD_ALWAYS_FAILS + unreachable FINISH appended.
    /// compile_simple_loop inserts a LABEL at entry for bridge attachment.
    SegmentedLoop,
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

/// Assure the JIT that `func(args...)` will produce `result`.
/// `func` must be an elidable function.
///
/// rlib/jit.py:1224 — `record_known_result(result, func, *args)`
///
/// At runtime (non-JIT), verifies `func(args) == result` (debug builds).
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `record_known_result_{i|r}` opcode with func and args visible
/// as separate operands — matching RPython's rtyper decomposition.
///
/// Usage: `record_known_result!(result, my_elidable_fn, arg1, arg2)`
#[macro_export]
macro_rules! record_known_result {
    ($result:expr, $func:path $(, $arg:expr)*) => {
        // rlib/jit.py:1229-1232 — untranslated consistency check
        debug_assert_eq!(
            $func($($arg),*), $result,
            "record_known_result: func(...) != result"
        );
    };
}

/// rlib/jit.py:1301 — `conditional_call(condition, function, *args)`
///
/// At runtime: `if condition { function(args...) }`.
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `conditional_call_ir_v` opcode with func and args as
/// separate operands — matching RPython's ConditionalCallEntry decomposition.
///
/// Usage: `conditional_call!(cond, my_func, arg1, arg2)`
#[macro_export]
macro_rules! conditional_call {
    ($condition:expr, $func:path $(, $arg:expr)*) => {
        if $condition {
            $func($($arg),*);
        }
    };
}

/// rlib/jit.py:1322 — `conditional_call_elidable(value, function, *args)`
///
/// At runtime: `if value is falsy { value = function(args...) }; return value`.
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `conditional_call_value_ir_{i|r}` opcode with func and args as
/// separate operands.
///
/// Usage: `let v = conditional_call_elidable!(cached, compute_fn, arg1, arg2)`
#[macro_export]
macro_rules! conditional_call_elidable {
    ($value:expr, $func:path $(, $arg:expr)*) => {{
        let __val = $value;
        if __val == 0 {
            $func($($arg),*)
        } else {
            __val
        }
    }};
}

/// Hash a green key from i64 slice values.
///
/// Uses the same algorithm as [`GreenKey::hash_u64`](majit_ir::GreenKey::hash_u64),
/// so callers can compute a key hash without constructing a full `GreenKey`.
#[inline]
pub fn green_key_hash(values: &[i64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    values.hash(&mut hasher);
    hasher.finish()
}

// ── we_are_jitted / JIT mode flag ──
// Re-exported from majit-codegen so both meta and backend can access it.
pub use majit_backend::{JittedGuard, set_jitted, we_are_jitted};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn green_key_hash_deterministic() {
        let a = green_key_hash(&[10, 20]);
        let b = green_key_hash(&[10, 20]);
        assert_eq!(a, b);
    }

    #[test]
    fn green_key_hash_different_values() {
        let a = green_key_hash(&[10, 20]);
        let b = green_key_hash(&[10, 21]);
        assert_ne!(a, b);
    }

    #[test]
    fn green_key_hash_matches_green_key() {
        let hash = green_key_hash(&[42, 7]);
        let gk = majit_ir::GreenKey::new(vec![42, 7]);
        assert_eq!(hash, gk.hash_u64());
    }
}
pub mod resumecode;
