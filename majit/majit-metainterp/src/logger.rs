//! RPython parity module for `rpython/jit/metainterp/logger.py`.
//!
//! The implementation lives in `majit_trace` with the trace recorder, but the
//! upstream import path is `metainterp.logger`.

pub use majit_trace::logger::*;
