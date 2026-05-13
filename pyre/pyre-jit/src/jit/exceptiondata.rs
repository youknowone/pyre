//! `rpython/rtyper/exceptiondata.py` `class ExceptionData(object)` +
//! `rpython/rtyper/rtyper.py:71 self.exceptiondata = ExceptionData(self)`.
//!
//! Pyre-side shim for the only attribute chain `flatten_graph` reads
//! from `cpu`: `cpu.rtyper.exceptiondata.get_standard_ll_exc_instance_by_class(...)`
//! at `rpython/jit/codewriter/flatten.py:166-170`.  Other RPython
//! `ExceptionData` methods (`make_helpers`, `make_exception_matcher`,
//! `make_type_of_exc_inst`, …) are intentionally absent; they get added
//! one method at a time when a future port reads them.

use super::flatten::Kind;
use super::flow::Constant;

/// `rpython/rtyper/exceptiondata.py:7 class UnknownException(Exception)`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UnknownException(pub String);

/// `rpython/annotator/exception.py:standardexceptions` — names of the
/// exceptions the rtyper preallocates an instance for.  Pyre only needs
/// `OverflowError` today (`flatten.py:167`); other names are added when
/// a future port consumes them.
const STANDARD_EXCEPTIONS: &[&str] = &["OverflowError"];

/// `rpython/rtyper/exceptiondata.py:11 class ExceptionData(object)`.
#[derive(Debug)]
pub struct ExceptionData {
    /// `exceptiondata.py:14 standardexceptions = standardexceptions`.
    pub standardexceptions: &'static [&'static str],
}

impl Default for ExceptionData {
    fn default() -> Self {
        Self::new()
    }
}

impl ExceptionData {
    /// `exceptiondata.py:16 def __init__(self, rtyper)`.  Pyre's rtyper
    /// shim carries no state, so `rtyper` is implicit here; the fields
    /// upstream populates (`r_exception_type`, `r_exception_value`,
    /// `lltype_of_exception_type`, `lltype_of_exception_value`) all
    /// belong to the LL type system that pyre-jit does not model.
    pub fn new() -> Self {
        Self {
            standardexceptions: STANDARD_EXCEPTIONS,
        }
    }

    /// `exceptiondata.py:40-45 def get_standard_ll_exc_instance_by_class(self, exceptionclass)`.
    ///
    /// Upstream walks the bookkeeper to obtain `clsdef` then calls
    /// `get_standard_ll_exc_instance(rtyper, clsdef)` which returns the
    /// reusable prebuilt LL instance pointer wrapped at the caller in
    /// `Constant(ll_ovf, concretetype=lltype.typeOf(ll_ovf))`
    /// (`flatten.py:168-169`).  Pyre has no LL type system; the
    /// production `lower_constant` closure threaded through
    /// `GraphFlattener` resolves the opaque `Constant` to the runtime
    /// PyObject pointer for the exception class.
    pub fn get_standard_ll_exc_instance_by_class(
        &self,
        exceptionclass: &str,
    ) -> Result<Constant, UnknownException> {
        if !self.standardexceptions.contains(&exceptionclass) {
            return Err(UnknownException(exceptionclass.to_owned()));
        }
        Ok(Constant::opaque(exceptionclass, Some(Kind::Ref)))
    }
}

/// `rpython/rtyper/rtyper.py:33 class RPythonTyper(object)`.
///
/// Pyre-jit operates on the flowspace graph directly without a typed
/// low-level rewrite; the rtyper shim exists only to satisfy the
/// `cpu.rtyper.exceptiondata` attribute chain that `flatten_graph` reads
/// from at `flatten.py:166`.  Future attributes are added one at a time
/// when a flatten / codewriter consumer materializes them.
#[derive(Debug, Default)]
pub struct Rtyper {
    /// `rtyper.py:71 self.exceptiondata = ExceptionData(self)`.
    pub exceptiondata: ExceptionData,
}

impl Rtyper {
    pub fn new() -> Self {
        Self {
            exceptiondata: ExceptionData::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_standard_ll_exc_instance_by_class_returns_overflow_error() {
        let data = ExceptionData::new();
        let constant = data
            .get_standard_ll_exc_instance_by_class("OverflowError")
            .expect("OverflowError must be a standard exception");
        match &constant.value {
            crate::jit::flow::ConstantValue::Opaque(opaque) => {
                assert_eq!(opaque.repr(), "OverflowError");
            }
            other => panic!("expected Opaque, got {other:?}"),
        }
        assert_eq!(constant.kind, Some(Kind::Ref));
    }

    #[test]
    fn get_standard_ll_exc_instance_by_class_rejects_unknown_class() {
        let data = ExceptionData::new();
        let err = data
            .get_standard_ll_exc_instance_by_class("NotAStandardException")
            .expect_err("non-standard class must error");
        assert_eq!(err, UnknownException("NotAStandardException".to_owned()));
    }
}
