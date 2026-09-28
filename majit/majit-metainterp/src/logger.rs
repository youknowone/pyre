//! RPython parity module for `rpython/jit/metainterp/logger.py`.
//!
//! The implementation lives in `majit_trace` with the trace recorder, but the
//! upstream import path is `metainterp.logger`.

// This module is an import-path parity surface; callers may use the upstream
// path even when this crate itself does not.
#[allow(unused_imports)]
pub use majit_trace::logger::{
    JitTimer, LogOperations, Logger, TraceRecord, int_could_be_an_address, stats_enabled,
};

/// logger.py `Logger.log_loop_from_trace`: the `jit-log-noopt` section, headed
/// by the traced op count.
pub fn log_loop_from_trace<
    T: AsRef<majit_ir::Op>,
    V: std::fmt::Debug,
    C: majit_ir::resoperation::ConstLookup<V>,
>(
    ops: &[T],
    constants: &C,
) {
    let _s = crate::debug::scope("jit-log-noopt");
    if !crate::debug::have_debug_prints() {
        return;
    }
    crate::debug::debug_print(&format!("# Traced loop or bridge with {} ops", ops.len()));
    for line in majit_ir::format_trace(ops, constants).lines() {
        crate::debug::debug_print(line);
    }
}
