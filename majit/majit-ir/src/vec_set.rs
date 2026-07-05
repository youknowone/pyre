//! Insertion-order membership set used by majit.
//!
//! Previously backed by `vecmap_rs::VecSet` (linear-scan Vec). Replaced by
//! `indexmap::IndexSet` for O(1) membership checks while preserving insertion
//! order. The `VecSet` name is kept as a type alias to minimise churn.

pub use indexmap::IndexSet as VecSet;
