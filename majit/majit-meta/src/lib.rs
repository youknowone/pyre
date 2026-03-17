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
    emit_commit_io, encode_decimal_i64, io_buffer_commit, io_buffer_discard, io_buffer_write,
    io_buffer_write_fmt, jit_write_number_i64, jit_write_utf8_codepoint,
};
pub use jit_state::{DeoptMaterializationCache, JitState, PendingFieldWriteLayout};
pub use jitcode::{
    trace_jitcode, trace_jitcode_with_data_ptr, ClosureRuntime, JitArgKind, JitCallArg, JitCode, JitCodeBuilder, JitCodeMachine,
    JitCodeRuntime, JitCodeSym, LivenessInfo, MIFrame, MIFrameStack,
};
pub use majit_codegen::CompiledTraceInfo;
pub use meta_interp::{
    BackEdgeAction, BlackholeRunResult, CompiledExitLayout, CompiledTerminalExitLayout,
    CompiledTraceLayout, DeadFrameArtifacts, DetailedDriverRunOutcome, DriverRunOutcome,
    GuardRecoveryAction, InlineDecision, JitHooks, JitStats, MetaInterp, RawCompileResult,
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

/// Hash a green key from i64 slice values.
///
/// Uses the same algorithm as [`GreenKey::hash_u64`](majit_ir::GreenKey::hash_u64),
/// so callers can compute a key hash without constructing a full `GreenKey`.
pub fn green_key_hash(values: &[i64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    values.hash(&mut hasher);
    hasher.finish()
}

// ── we_are_jitted / JIT mode flag ──
// Re-exported from majit-codegen so both meta and backend can access it.
pub use majit_codegen::{set_jitted, we_are_jitted, JittedGuard};

/// Generic guard state restore for storage-pool interpreters.
///
/// Decodes a flat `values` array produced by a guard failure back into
/// per-storage arrays, plus a selected-storage index and resume PC.
///
/// Values layout: `[storage_0_vals..., storage_1_vals..., ...,
///                  len_0, len_1, ..., selected, resume_pc]`
///
/// `storage_layout` lists `(storage_idx, num_traced)` in the same order
/// that the tracer recorded them.
///
/// Returns `(resume_pc, selected_storage)` on success, or `None` if the
/// values are malformed.
pub fn restore_storage_pool_guard_state(
    values: &[i64],
    storage_layout: &[(usize, usize)],
    max_selected: usize,
    max_pc: usize,
    mut push_fn: impl FnMut(usize, &[i64]),
) -> Option<(usize, usize)> {
    let storage_count = storage_layout.len();
    let lengths_start = values.len().checked_sub(storage_count + 2)?;
    let lengths = &values[lengths_start..values.len() - 2];
    let selected = usize::try_from(values[values.len() - 2]).ok()?;
    let resume_pc = usize::try_from(values[values.len() - 1]).ok()?;
    if selected >= max_selected || resume_pc > max_pc {
        return None;
    }

    let flat_values = &values[..lengths_start];
    let mut offset = 0;
    let mut total = 0usize;
    let mut decoded_lengths = Vec::with_capacity(lengths.len());
    for &len_raw in lengths {
        let len = usize::try_from(len_raw).ok()?;
        total += len;
        decoded_lengths.push(len);
    }
    if total != flat_values.len() {
        return None;
    }

    for (i, &len) in decoded_lengths.iter().enumerate() {
        let end = offset + len;
        let (storage_idx, _) = storage_layout[i];
        push_fn(storage_idx, &flat_values[offset..end]);
        offset = end;
    }
    Some((resume_pc, selected))
}

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

    #[test]
    fn test_restore_storage_pool_basic() {
        // 2 storages: storage 0 has [10, 20], storage 1 has [30]
        // layout: [(0, 2), (1, 1)]
        // values: [10, 20, 30, 2, 1, 3, 42]
        //          ^flat         ^lens ^sel ^pc
        let values = [10, 20, 30, 2, 1, 3, 42];
        let layout = [(0, 2), (1, 1)];
        let mut result: Vec<(usize, Vec<i64>)> = Vec::new();

        let outcome = restore_storage_pool_guard_state(&values, &layout, 10, 100, |idx, vals| {
            result.push((idx, vals.to_vec()))
        });

        assert_eq!(outcome, Some((42, 3)));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0, vec![10, 20]));
        assert_eq!(result[1], (1, vec![30]));
    }

    #[test]
    fn test_restore_storage_pool_invalid_selected() {
        let values = [10, 20, 2, 5, 42];
        let layout = [(0, 2)];

        let outcome = restore_storage_pool_guard_state(
            &values,
            &layout,
            5, // max_selected = 5, selected = 5 => out of range
            100,
            |_, _| {},
        );

        assert_eq!(outcome, None);
    }

    #[test]
    fn test_restore_storage_pool_length_mismatch() {
        // lengths say 3 total but flat area only has 2 values
        let values = [10, 20, 3, 5, 42];
        let layout = [(0, 2)];

        let outcome = restore_storage_pool_guard_state(&values, &layout, 10, 100, |_, _| {
            panic!("should not be called")
        });

        assert_eq!(outcome, None);
    }
}
