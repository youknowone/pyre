//! Re-export of `rpython/rtyper/exceptiondata.py`.
//!
//! The concrete port lives in `rtyper.rs` because `RPythonTyper` owns the
//! initialization and finish lifecycle.  Keep this module so imports match
//! PyPy's source layout.

pub use super::rtyper::ExceptionData;

/// RPython `class UnknownException(Exception)` (`exceptiondata.py:7-8`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnknownException(pub String);

impl std::fmt::Display for UnknownException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UnknownException({})", self.0)
    }
}

impl std::error::Error for UnknownException {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceptiondata_type_is_exposed_at_parity_path() {
        let name = std::any::type_name::<ExceptionData>();
        assert!(name.ends_with("rtyper::rtyper::ExceptionData"));
    }

    #[test]
    fn unknown_exception_type_is_exposed_at_parity_path() {
        let err = UnknownException("ValueError".to_string());
        assert_eq!(err.to_string(), "UnknownException(ValueError)");
    }
}
