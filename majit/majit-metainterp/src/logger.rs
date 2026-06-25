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
