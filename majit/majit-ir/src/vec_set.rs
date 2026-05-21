//! Insertion-order, Vec-backed membership set used to replace small
//! `HashSet`s per the house no-HashMap rule.
//!
//! Re-export `vecmap-rs`'s `VecSet` instead of carrying a local copy.  This
//! crate's set has the semantics we need here: insertion order is preserved
//! and membership operations require only `Eq`, not `Ord`.

pub use vecmap_rs::VecSet;
