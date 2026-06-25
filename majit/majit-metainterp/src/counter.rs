//! RPython parity module for `rpython/jit/metainterp/counter.py`.
//!
//! The implementation lives in `majit_trace` with the trace recorder, but the
//! upstream import path is `metainterp.counter`. Keep the public re-export
//! explicit so parity tooling can see the upstream class names.

pub use majit_trace::counter::{DEFAULT_SIZE, DeterministicJitCounter, JitCounter};
