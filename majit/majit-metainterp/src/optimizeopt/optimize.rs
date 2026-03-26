//! Optimization exceptions.
//!
//! Mirrors RPython's `optimize.py`: InvalidLoop, SpeculativeError.

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
