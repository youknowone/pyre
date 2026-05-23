//! Optimizer exceptions and nullness constants.
//!
//! Mirrors RPython's `optimize.py` (exceptions) and `info.py:13-15`
//! (INFO_* nullness constants). They live in `majit-ir` so the bound /
//! info types that reference them can be hosted here without a
//! circular dep back into `majit-metainterp`.

/// Raised when an intersection or constraint leads to an empty set,
/// meaning the current trace is impossible and should be abandoned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidLoop(pub &'static str);

impl std::fmt::Display for InvalidLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InvalidLoop: {}", self.0)
    }
}

impl std::error::Error for InvalidLoop {}

/// Raised when a speculative optimization turned out to be wrong.
///
/// The trace must be recompiled without the speculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeError(pub &'static str);

impl std::fmt::Display for SpeculativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpeculativeError: {}", self.0)
    }
}

impl std::error::Error for SpeculativeError {}

/// info.py:13-15 INFO_NULL / INFO_NONNULL / INFO_UNKNOWN constants.
///
/// Used by `PtrInfo::getnullness` and `IntBound::getnullness` to
/// report whether a slot is known null, known non-null, or unknown.
/// Matches the upstream integer enum values exactly so majit code
/// can be ported line-by-line from `optimizer.py:127` / `rewrite.py:496-503`
/// `_optimize_nullness` switches.
pub const INFO_NULL: i8 = 0;
pub const INFO_NONNULL: i8 = 1;
pub const INFO_UNKNOWN: i8 = 2;
