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
/// `STANDARD_EXCEPTIONS` linearly (16 entries) and reads the matching
/// slot, matching upstream's `get_standard_ll_exc_instance(clsdef)`
/// shape where the clsdef key uniquely identifies the LL instance.
/// Position-indexed parallel `Vec` matches RPython's `_bigints` /
/// `_floats` / `_refs` idiom for rtyper-resolved values, and satisfies
/// the pyre/majit no-HashMap invariant (memory `feedback-no-hashmap-
/// ever`).  Both `OverflowError` (the `_ovf` rewrite caller today) and
/// any other future caller for a standard exception receive the same
/// resolved-pointer shape.
///
/// Resolution is **lazy**: `instance_pointers` is populated on the
/// first `get_standard_ll_exc_instance_by_class` lookup per slot, by
/// calling the resolver thunk installed via `set_lazy_resolver`.
/// Production wires `Cpu::new` to install a resolver that hands out
/// per-`ExcKind` singleton W_ExceptionObject instance pointers
/// (`pyre_interpreter::lookup_exc_instance`); the singletons are
/// allocated only when the resolver is actually invoked, which under
/// the current walker-driven pipeline is never (the
/// `handling_ovf=true` arm of `flatten.rs::make_exception_link` lives
/// behind the canonical `flatten_graph` entry that production does
/// not yet flip to).  Tests pre-populate by calling
/// `resolve_standard_exception_pointers` directly — that path takes
/// precedence over the lazy resolver because it writes the slot
/// outright.
pub struct ExceptionData {
    /// `exceptiondata.py:14 standardexceptions = standardexceptions`.
    pub standardexceptions: &'static [&'static str],
    /// Resolved runtime pointer per standard exception, indexed
    /// parallel to `standardexceptions`.  `None` means the resolver
    /// has not been called for this slot yet (and no eager
    /// `resolve_standard_exception_pointers` populated it).  The
    /// interior mutability lets `get_standard_ll_exc_instance_by_class`
    /// take `&self` while caching the lazy-resolver's first answer.
    instance_pointers: std::cell::RefCell<Vec<Option<i64>>>,
    /// Optional lazy resolver consulted by
    /// `get_standard_ll_exc_instance_by_class` when a slot is unresolved.
    /// `None` (default) means "fail loud on unresolved lookup"; production
    /// callers install a resolver in `Cpu::new`.  Wrapped in `Box<dyn Fn>`
    /// so the closure can capture per-`Cpu` state without leaking the
    /// concrete callback type through the public API.
    #[allow(clippy::type_complexity)]
    lazy_resolver: Option<Box<dyn Fn(&str) -> Option<i64> + Send + Sync>>,
}

impl std::fmt::Debug for ExceptionData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExceptionData")
            .field("standardexceptions", &self.standardexceptions)
            .field("instance_pointers", &self.instance_pointers)
            .field(
                "lazy_resolver",
                &self.lazy_resolver.as_ref().map(|_| "<closure>"),
            )
            .finish()
    }
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
            instance_pointers: std::cell::RefCell::new(vec![None; STANDARD_EXCEPTIONS.len()]),
            lazy_resolver: None,
        }
    }

    /// Install a lazy resolver that
    /// `get_standard_ll_exc_instance_by_class` consults on the first
    /// unresolved lookup per slot.  Used by `Cpu::new` to wire
    /// `pyre_interpreter::lookup_exc_instance` without paying the
    /// per-`ExcKind` `W_ExceptionObject` singleton allocation cost up
    /// front (production walker doesn't reach the `_ovf` direct-raise
    /// rewrite, so the singletons stay unallocated under today's
    /// pipeline; deferring keeps the heap layout stable, which the
    /// cranelift backend's trace compilation is sensitive to).
    pub fn set_lazy_resolver<F>(&mut self, resolver: F)
    where
        F: Fn(&str) -> Option<i64> + Send + Sync + 'static,
    {
        self.lazy_resolver = Some(Box::new(resolver));
    }

    /// Pre-resolve every standard exception class with a caller-side
    /// pointer reachable through `resolve`, matching upstream's
    /// `get_standard_ll_exc_instance(rtyper, clsdef)` which materialises
    /// the LL instance pointer at rtyper construction time
    /// (`exceptiondata.py:34-42`).  Pyre's lazy-resolver path
    /// (`set_lazy_resolver`) is the production analog; this eager
    /// variant exists for tests that want to control the pointer
    /// values directly without going through the live interpreter.
    ///
    /// `resolve(name)` may return `None` for a `standardexceptions`
    /// entry the caller doesn't want to populate; the slot stays
    /// `None` and a subsequent `get_standard_ll_exc_instance_by_class`
    /// falls back to the lazy resolver (or panics if none is installed).
    pub fn resolve_standard_exception_pointers<F>(&mut self, mut resolve: F)
    where
        F: FnMut(&str) -> Option<i64>,
    {
        let mut slots = self.instance_pointers.borrow_mut();
        for (idx, &name) in self.standardexceptions.iter().enumerate() {
            if let Some(pointer) = resolve(name) {
                slots[idx] = Some(pointer);
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
    /// Pyre lazily populates the slot from the resolver installed via
    /// `set_lazy_resolver` on first lookup; once cached, subsequent
    /// lookups reuse the cached pointer.  An unresolved slot with no
    /// resolver panics — production must wire a resolver via
    /// `set_lazy_resolver` before `flatten_graph` reaches the rewrite
    /// for `exceptionclass`.
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
        let mut slots = self.instance_pointers.borrow_mut();
        let pointer = match slots[idx] {
            Some(p) => p,
            None => {
                let resolved = self
                    .lazy_resolver
                    .as_ref()
                    .and_then(|resolver| resolver(exceptionclass));
                match resolved {
                    Some(p) => {
                        slots[idx] = Some(p);
                        p
                    }
                    None => panic!(
                        "ExceptionData::get_standard_ll_exc_instance_by_class\
                         ({exceptionclass:?}) called before the slot was \
                         populated and no lazy resolver yielded a pointer — \
                         production pipelines must wire Cpu::new -> \
                         set_lazy_resolver(lookup_exc_instance) so \
                         flatten_graph never reaches an opaque shape per \
                         rpython/rtyper/rtyper.py:specialize"
                    ),
                }
            }
        };
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
    #[should_panic(expected = "called before the slot was populated")]
    fn get_standard_ll_exc_instance_by_class_panics_when_unresolved() {
        let data = ExceptionData::new();
        let _ = data.get_standard_ll_exc_instance_by_class("OverflowError");
    }

    #[test]
    fn get_standard_ll_exc_instance_by_class_lazy_resolver_fills_slot_on_demand() {
        // Production wiring: a resolver installed via
        // `set_lazy_resolver` is consulted on the first lookup per
        // slot.  Subsequent lookups read the cached pointer without
        // re-invoking the resolver, so a counting resolver here pins
        // both the lazy fill (first call returns the resolver's
        // pointer) and the cache (second call doesn't call the
        // resolver a second time).
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let calls_in_closure = calls.clone();
        let mut data = ExceptionData::new();
        data.set_lazy_resolver(move |name| {
            calls_in_closure.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            match name {
                "OverflowError" => Some(0xdeadbeef),
                _ => None,
            }
        });
        let first = data
            .get_standard_ll_exc_instance_by_class("OverflowError")
            .expect("OverflowError must resolve via lazy resolver");
        let second = data
            .get_standard_ll_exc_instance_by_class("OverflowError")
            .expect("second lookup must hit the cache");
        match (&first.value, &second.value) {
            (ConstantValue::Signed(a), ConstantValue::Signed(b)) => {
                assert_eq!(*a, 0xdeadbeef);
                assert_eq!(*a, *b);
            }
            other => panic!("expected Signed pointer pair, got {other:?}"),
        }
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "lazy resolver must be called exactly once per slot"
        );
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
