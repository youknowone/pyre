//! `rpython/rtyper/exceptiondata.py` `class ExceptionData(object)` +
//! `rpython/rtyper/rtyper.py:71 self.exceptiondata = ExceptionData(self)`.
//!
//! Pyre-side shim for the only attribute chain `flatten_graph` reads
//! from `cpu`: `cpu.rtyper.exceptiondata.get_standard_ll_exc_instance_by_class(...)`
//! at `rpython/jit/codewriter/flatten.py:166-170`.  Other RPython
//! `ExceptionData` methods (`make_helpers`, `make_exception_matcher`,
//! `make_type_of_exc_inst`, …) are intentionally absent; they get added
//! one method at a time when a future port reads them.

use std::collections::HashMap;

use super::flatten::Kind;
use super::flow::{Constant, ConstantValue};

/// `rpython/rtyper/exceptiondata.py:7 class UnknownException(Exception)`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UnknownException(pub String);

/// `rpython/annotator/exception.py:standardexceptions` — names of the
/// exceptions the rtyper preallocates an instance for.  Mirrors the
/// upstream set verbatim: every exception that can be implicitly raised
/// by some flow-space operation, in the same order upstream lists them.
/// Today only `OverflowError` has a caller via the `_ovf` rewrite at
/// `flatten.py:167`; the rest are present so future ports that read
/// `standardexceptions` find a parity-complete table.
const STANDARD_EXCEPTIONS: &[&str] = &[
    "TypeError",
    "OverflowError",
    "ValueError",
    "ZeroDivisionError",
    "MemoryError",
    "IOError",
    "OSError",
    "StopIteration",
    "KeyError",
    "IndexError",
    "AssertionError",
    "RuntimeError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "NotImplementedError",
    "_StackOverflow",
];

/// `rpython/rtyper/exceptiondata.py:11 class ExceptionData(object)`.
#[derive(Debug)]
pub struct ExceptionData {
    /// `exceptiondata.py:14 standardexceptions = standardexceptions`.
    pub standardexceptions: &'static [&'static str],
    /// Pre-resolved class pointer per standard exception name.  When
    /// populated, `get_standard_ll_exc_instance_by_class` returns the
    /// rtyped form `Constant(Signed(pointer), Some(Ref))` matching
    /// upstream's `Constant(ll_ovf, concretetype=lltype.typeOf(ll_ovf))`
    /// at `flatten.py:168-169`.  When absent, the same call returns the
    /// pre-rtype opaque shape, deferring resolution to the production
    /// `lower_constant` closure threaded through `GraphFlattener`.
    instance_pointers: HashMap<&'static str, i64>,
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
            instance_pointers: HashMap::new(),
        }
    }

    /// Pre-resolve every standard exception class to its runtime
    /// PyObject pointer via `resolve`, matching upstream's
    /// `get_standard_ll_exc_instance(rtyper, clsdef)` which materialises
    /// the LL instance pointer at rtyper construction time
    /// (`exceptiondata.py:34-42`).  Production callers invoke this
    /// after the runtime exception classes are loaded so the canonical
    /// `flatten_graph` driver's `handling_ovf=True` arm reaches
    /// `get_standard_ll_exc_instance_by_class("OverflowError")` and
    /// receives the rtyped shape directly.
    pub fn resolve_standard_exception_pointers<F>(&mut self, mut resolve: F)
    where
        F: FnMut(&str) -> i64,
    {
        for &name in self.standardexceptions {
            let pointer = resolve(name);
            self.instance_pointers.insert(name, pointer);
        }
    }

    /// `exceptiondata.py:40-45 def get_standard_ll_exc_instance_by_class(self, exceptionclass)`.
    ///
    /// Upstream walks the bookkeeper to obtain `clsdef` then calls
    /// `get_standard_ll_exc_instance(rtyper, clsdef)` which returns the
    /// LL instance pointer wrapped at the caller in `Constant(ll_ovf,
    /// concretetype=lltype.typeOf(ll_ovf))` (`flatten.py:168-169`).
    ///
    /// When `resolve_standard_exception_pointers` has been called, this
    /// returns the rtyped shape `Constant(Signed(pointer), Some(Ref))`
    /// directly.  Otherwise it returns the pre-rtype opaque shape and
    /// expects the caller's `lower_constant` closure to resolve.
    pub fn get_standard_ll_exc_instance_by_class(
        &self,
        exceptionclass: &str,
    ) -> Result<Constant, UnknownException> {
        if !self.standardexceptions.contains(&exceptionclass) {
            return Err(UnknownException(exceptionclass.to_owned()));
        }
        if let Some(&pointer) = self.instance_pointers.get(exceptionclass) {
            return Ok(Constant::new(
                ConstantValue::Signed(pointer),
                Some(Kind::Ref),
            ));
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

    #[test]
    fn resolve_standard_exception_pointers_returns_rtyped_signed_ref() {
        let mut data = ExceptionData::new();
        data.resolve_standard_exception_pointers(|name| match name {
            "OverflowError" => 0xc0de,
            _ => 0,
        });
        let constant = data
            .get_standard_ll_exc_instance_by_class("OverflowError")
            .expect("OverflowError must be a standard exception");
        match (&constant.value, constant.kind) {
            (ConstantValue::Signed(p), Some(Kind::Ref)) => assert_eq!(*p, 0xc0de),
            other => panic!("expected Signed(0xc0de)/Ref after resolve, got {other:?}"),
        }
    }
}
