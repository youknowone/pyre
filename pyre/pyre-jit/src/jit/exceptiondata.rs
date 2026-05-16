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
/// Storage layout mirrors RPython's "pre-allocated standard exception
/// instances on the rtyper" — each entry in `STANDARD_EXCEPTIONS` has
/// a parallel slot in `instance_pointers`.  Resolution lookup walks
/// `STANDARD_EXCEPTIONS` linearly (15 entries) and reads the matching
/// slot, matching upstream's `get_standard_ll_exc_instance(clsdef)`
/// shape where the clsdef key uniquely identifies the LL instance.
/// Position-indexed parallel `Vec` matches RPython's `_bigints` /
/// `_floats` / `_refs` idiom for rtyper-resolved values, and satisfies
/// the pyre/majit no-HashMap invariant (memory `feedback-no-hashmap-
/// ever`).  Both `OverflowError` (the `_ovf` rewrite caller today) and
/// any other future caller for a standard exception receive the same
/// resolved-pointer shape.
#[derive(Debug, Default)]
pub struct ExceptionData {
    /// `exceptiondata.py:14 standardexceptions = standardexceptions`.
    pub standardexceptions: &'static [&'static str],
    /// Resolved runtime pointer per standard exception, indexed
    /// parallel to `standardexceptions`.  `None` means the resolver
    /// has not populated this slot yet; calling
    /// `get_standard_ll_exc_instance_by_class` for that name panics
    /// (production invariant: every `flatten_graph` reachable on a
    /// standard-exception rewrite path must see a resolved pointer).
    instance_pointers: Vec<Option<i64>>,
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
            instance_pointers: vec![None; STANDARD_EXCEPTIONS.len()],
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
    /// `resolve(name)` may return `None` for a `standardexceptions`
    /// entry whose runtime registration isn't installed yet on this
    /// thread (test paths that never reach the corresponding rewrite);
    /// the slot stays `None` and `get_standard_ll_exc_instance_by_class`
    /// panics if it's later queried for that name.
    pub fn resolve_standard_exception_pointers<F>(&mut self, mut resolve: F)
    where
        F: FnMut(&str) -> Option<i64>,
    {
        for (idx, &name) in self.standardexceptions.iter().enumerate() {
            if let Some(pointer) = resolve(name) {
                self.instance_pointers[idx] = Some(pointer);
            }
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
    /// reaches the rewrite for `exceptionclass`.  An unresolved
    /// pointer panics — production must never return the pre-rtype
    /// opaque shape because the canonical `lower_constant` closure
    /// cannot recover a runtime pointer from an opaque token the
    /// rtyper would have resolved upstream.
    pub fn get_standard_ll_exc_instance_by_class(
        &self,
        exceptionclass: &str,
    ) -> Result<Constant, UnknownException> {
        let Some(idx) = self
            .standardexceptions
            .iter()
            .position(|&name| name == exceptionclass)
        else {
            return Err(UnknownException(exceptionclass.to_owned()));
        };
        let pointer = self.instance_pointers[idx].unwrap_or_else(|| {
            panic!(
                "ExceptionData::get_standard_ll_exc_instance_by_class\
                 ({exceptionclass:?}) called before \
                 resolve_standard_exception_pointers populated this \
                 slot — production pipelines must wire Cpu::new -> \
                 pyre_interpreter::lookup_exc_class so flatten_graph \
                 never reaches an opaque shape per \
                 rpython/rtyper/rtyper.py:specialize"
            )
        });
        Ok(Constant::new(
            ConstantValue::Signed(pointer),
            Some(Kind::Ref),
        ))
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
    fn get_standard_ll_exc_instance_by_class_returns_resolved_other_standard_pointer() {
        // R3 parity: RPython's get_standard_ll_exc_instance_by_class
        // accepts any class in standardexceptions, not just
        // OverflowError. Verify the generic lookup works for a
        // second entry (the resolver decides which slots are
        // populated).
        let mut data = ExceptionData::new();
        data.resolve_standard_exception_pointers(|name| match name {
            "ZeroDivisionError" => Some(0xbeef),
            _ => None,
        });
        let constant = data
            .get_standard_ll_exc_instance_by_class("ZeroDivisionError")
            .expect("ZeroDivisionError is a standard exception");
        match (&constant.value, constant.kind) {
            (ConstantValue::Signed(p), Some(Kind::Ref)) => assert_eq!(*p, 0xbeef),
            other => panic!("expected Signed(0xbeef)/Ref, got {other:?}"),
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
