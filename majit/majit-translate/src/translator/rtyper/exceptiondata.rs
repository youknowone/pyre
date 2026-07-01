//! RPython `rpython/rtyper/exceptiondata.py`.

use std::cell::RefCell;
use std::sync::Arc;

use super::lltypesystem::lltype::LowLevelType;
use super::rmodel::Repr;
use super::rtyper::LowLevelFunction;

/// RPython `class ExceptionData` (`exceptiondata.py:11-26`).
///
/// Upstream's `__init__` obtains `r_type = rtyper.rootclass_repr` and
/// `r_instance = getinstancerepr(rtyper, None)`, then freezes their
/// lltypes into the last two fields. `RPythonTyper` owns construction
/// because it has to seed `rootclass_repr` first, but the concrete type
/// lives in this module to match PyPy's source layout.
#[derive(Clone, Debug)]
pub struct ExceptionData {
    /// RPython `self.r_exception_type = rtyper.rootclass_repr` — the
    /// class repr used for every exception vtable pointer.
    pub(crate) r_exception_type: Arc<dyn Repr>,
    /// RPython `self.r_exception_value = getinstancerepr(rtyper, None)`
    /// — the instance repr shared by every exception value.
    pub(crate) r_exception_value: Arc<dyn Repr>,
    /// RPython `self.lltype_of_exception_type = r_type.lowleveltype`.
    pub(crate) lltype_of_exception_type: LowLevelType,
    /// RPython `self.lltype_of_exception_value = r_instance.lowleveltype`.
    pub(crate) lltype_of_exception_value: LowLevelType,
    /// RPython `self.fn_exception_match` assigned by
    /// `ExceptionData.make_helpers()`.
    pub(crate) fn_exception_match: RefCell<Option<LowLevelFunction>>,
    /// RPython `self.fn_type_of_exc_inst` assigned by
    /// `ExceptionData.make_helpers()`.
    pub(crate) fn_type_of_exc_inst: RefCell<Option<LowLevelFunction>>,
}

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
        assert!(name.ends_with("rtyper::exceptiondata::ExceptionData"));
    }

    #[test]
    fn unknown_exception_type_is_exposed_at_parity_path() {
        let err = UnknownException("ValueError".to_string());
        assert_eq!(err.to_string(), "UnknownException(ValueError)");
    }
}
