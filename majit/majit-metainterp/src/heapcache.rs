//! RPython parity module for `rpython/jit/metainterp/heapcache.py`.
//!
//! The implementation lives in `majit_trace` with the trace recorder, but the
//! upstream import path is `metainterp.heapcache`.

pub use majit_trace::heapcache::*;
