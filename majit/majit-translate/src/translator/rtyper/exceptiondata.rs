//! Re-export of `rpython/rtyper/exceptiondata.py`.
//!
//! The concrete port lives in `rtyper.rs` because `RPythonTyper` owns the
//! initialization and finish lifecycle.  Keep this module so imports match
//! PyPy's source layout.

pub use super::rtyper::ExceptionData;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceptiondata_type_is_exposed_at_parity_path() {
        let name = std::any::type_name::<ExceptionData>();
        assert!(name.ends_with("rtyper::rtyper::ExceptionData"));
    }
}
