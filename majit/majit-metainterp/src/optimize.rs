//! Optimization exceptions — re-exported from `majit_ir::optimize`.
//!
//! Mirrors RPython's `optimize.py`: `InvalidLoop`, `SpeculativeError`.
//! The definitions live in `majit-ir` so the bound / info types that
//! reference them can be hosted alongside without a circular dep.

pub use majit_ir::optimize::{InvalidLoop, SpeculativeError};
