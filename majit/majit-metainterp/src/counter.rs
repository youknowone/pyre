//! RPython parity module for `rpython/jit/metainterp/counter.py`.
//!
//! The implementation lives in `majit_trace` with the trace recorder, but the
//! upstream import path is `metainterp.counter`.

pub use majit_trace::counter::*;
