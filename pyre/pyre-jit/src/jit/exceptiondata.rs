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
///
/// Mirrors upstream's attribute-based layout: each pre-allocated
/// standard exception instance is held in a *named* field.  Today only
/// `OverflowError` has a `_ovf`-rewrite caller (`flatten.py:167`); other
/// `standardexceptions` entries lack a dedicated field until a future
/// caller materialises.  This explicit-field shape (rather than a
/// `HashMap<&'static str, i64>`) matches RPython's `exceptiondata.py`
/// where each pre-allocated instance is stored as a class attribute
/// (`self.ll_TypeError_inst`, `self.ll_OverflowError_inst`, ...).
#[derive(Debug, Default)]
pub struct ExceptionData {
    /// `exceptiondata.py:14 standardexceptions = standardexceptions`.
    pub standardexceptions: &'static [&'static str],
    /// Resolved runtime pointer for the OverflowError class instance —
    /// the `ll_ovf` value at `flatten.py:167-169`.  `None` means
    /// `resolve_standard_exception_pointers` has not been called yet;
    /// calling `get_standard_ll_exc_instance_by_class("OverflowError")`
    /// in that state panics (production invariant: every
    /// `flatten_graph` reachable on the `_ovf` path must see a resolved
    /// pointer).
    overflow_error_instance: Option<i64>,
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
            overflow_error_instance: None,
        }
    }

    /// Pre-resolve every standard exception class with a caller-side
    /// pointer reachable through `resolve`, matching upstream's
    /// `get_standard_ll_exc_instance(rtyper, clsdef)` which materialises
    /// the LL instance pointer at rtyper construction time
    /// (`exceptiondata.py:34-42`).  Production callers invoke this
    /// after the runtime exception classes are loaded (e.g. from
    /// `Cpu::new` via `pyre_interpreter::lookup_exc_class`) so the
    /// `handling_ovf=True` arm of `flatten_graph` reaches
    /// `get_standard_ll_exc_instance_by_class("OverflowError")` and
    /// receives the rtyped shape directly.
    ///
    /// Currently only the `OverflowError` slot is consumed —
    /// `resolve(name)` may return `None` for the other entries until a
    /// future caller wires a dedicated field for them; the resolver's
    /// `Some` value for `"OverflowError"` is the only one currently
    /// recorded.
    pub fn resolve_standard_exception_pointers<F>(&mut self, mut resolve: F)
    where
        F: FnMut(&str) -> Option<i64>,
    {
        if let Some(pointer) = resolve("OverflowError") {
            self.overflow_error_instance = Some(pointer);
        }
    }

    /// `exceptiondata.py:40-45 def get_standard_ll_exc_instance_by_class(self, exceptionclass)`.
    ///
    /// Upstream walks the bookkeeper to obtain `clsdef` then calls
    /// `get_standard_ll_exc_instance(rtyper, clsdef)` which returns the
    /// LL instance pointer wrapped at the caller in `Constant(ll_ovf,
    /// concretetype=lltype.typeOf(ll_ovf))` (`flatten.py:168-169`).
    ///
    /// Pyre's contract: the standard pointer MUST be pre-resolved via
    /// `resolve_standard_exception_pointers` before `flatten_graph`
    /// reaches the `_ovf` rewrite (currently the only caller).  An
    /// unresolved pointer panics — production must never return the
    /// pre-rtype opaque shape because the canonical `lower_constant`
    /// closure cannot recover a runtime pointer from an opaque token
    /// the rtyper would have resolved upstream.
    pub fn get_standard_ll_exc_instance_by_class(
        &self,
        exceptionclass: &str,
    ) -> Result<Constant, UnknownException> {
        if !self.standardexceptions.contains(&exceptionclass) {
            return Err(UnknownException(exceptionclass.to_owned()));
        }
        match exceptionclass {
            "OverflowError" => {
                let pointer = self.overflow_error_instance.unwrap_or_else(|| {
                    panic!(
                        "ExceptionData::get_standard_ll_exc_instance_by_class\
                         (\"OverflowError\") called before \
                         resolve_standard_exception_pointers — production \
                         pipelines must wire Cpu::new -> \
                         pyre_interpreter::lookup_exc_class(\"OverflowError\") \
                         so flatten_graph never reaches an opaque shape per \
                         rpython/rtyper/rtyper.py:specialize"
                    )
                });
                Ok(Constant::new(
                    ConstantValue::Signed(pointer),
                    Some(Kind::Ref),
                ))
            }
            other => panic!(
                "ExceptionData: standard exception {other:?} has no resolved \
                 instance field yet — add a dedicated field + \
                 resolve_standard_exception_pointers branch when a caller \
                 materialises (rpython/rtyper/exceptiondata.py stores each \
                 standard instance as a named class attribute, not a generic \
                 map)"
            ),
        }
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
    fn get_standard_ll_exc_instance_by_class_returns_resolved_overflow_pointer() {
        let mut data = ExceptionData::new();
        data.resolve_standard_exception_pointers(|name| match name {
            "OverflowError" => Some(0xc0de),
            _ => None,
        });
        let constant = data
            .get_standard_ll_exc_instance_by_class("OverflowError")
            .expect("OverflowError must be a standard exception");
        match (&constant.value, constant.kind) {
            (ConstantValue::Signed(p), Some(Kind::Ref)) => assert_eq!(*p, 0xc0de),
            other => panic!("expected Signed(0xc0de)/Ref after resolve, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "called before resolve_standard_exception_pointers")]
    fn get_standard_ll_exc_instance_by_class_panics_when_unresolved() {
        let data = ExceptionData::new();
        let _ = data.get_standard_ll_exc_instance_by_class("OverflowError");
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
