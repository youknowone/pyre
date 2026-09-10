//! RPython parity module for `rpython/jit/metainterp/heapcache.py`.
//!
//! The implementation lives in `majit_trace` with the trace recorder, but the
//! upstream import path is `metainterp.heapcache`.

// This module is an import-path parity surface; callers may use the upstream
// path even when this crate itself does not.
#[allow(unused_imports)]
pub use majit_trace::heapcache::{
    CacheEntry, FieldUpdater, HF_VERSION_MAX, HeapCache, HeapFlags, SameConstantOracle,
};
