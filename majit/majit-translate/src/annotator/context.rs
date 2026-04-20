//! RPython `RPythonAnnotator` callback surface consumed by the
//! operation dispatchers.
//!
//! Upstream `rpython/flowspace/operation.py` calls
//! `annotator.annotation(arg)` and `annotator.bookkeeper.…` from inside
//! `HLOperation.consider` / `get_specialization` / `transform`.
//! Porting `consider` etc. to Rust requires these callbacks to exist
//! before `RPythonAnnotator` itself lands (Commit 7). This trait is
//! the minimal hook surface so the Rust `annotator::operations`
//! helpers can be wired incrementally.
//!
//! Parity deviation: upstream has a single concrete class
//! `RPythonAnnotator`. Rust uses a trait object because the operation
//! dispatchers need access to the annotator before it exists as a
//! struct. Final layout: `impl AnnotatorContext for RPythonAnnotator`
//! lands with Commit 7; the trait surface does not grow beyond what
//! upstream invokes on `self`.

use super::super::flowspace::model::Hlvalue;
use super::model::SomeValue;

/// RPython `class RPythonAnnotator` surface consumed by dispatchers.
///
/// Each method mirrors the read side of the annotator used by
/// `operation.py`. The write side (`setbinding`, `addpendingblock`)
/// lands with the driver itself.
pub trait AnnotatorContext {
    /// RPython `annotator.annotation(arg)` (annrpython.py:273-280).
    ///
    /// Returns the `SomeValue` bound to a variable, or
    /// `bookkeeper.immutablevalue(c.value)` for a constant. Absence
    /// (unbound variable) is `None` — callers can distinguish from
    /// `Some(Impossible)` which is a real lattice value.
    fn annotation(&self, arg: &Hlvalue) -> Option<SomeValue>;
}
