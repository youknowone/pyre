//! RPython `HLOperation.{consider, transform, get_can_only_throw}`
//! ported as free functions over the operation + annotator pair.
//!
//! Upstream these live on `rpython/flowspace/operation.py:66-301` as
//! methods on `HLOperation` / `SingleDispatchMixin` /
//! `DoubleDispatchMixin`. The Rust port relocates them to the
//! annotator crate because `get_specialization` walks the `SomeValue`
//! hierarchy — a piece of annotator state that `flowspace` cannot see
//! (parity rule #1: minimum crate-boundary deviation).
//!
//! Commit split:
//!
//! * **Commit 2a (this file, initial version)** — trait hook
//!   ([`super::context::AnnotatorContext`]) + entry points
//!   ([`consider_hlop`], [`transform_hlop`], [`get_can_only_throw`])
//!   + [`read_can_only_throw`] helper.  Bodies are stubbed until the
//!   dispatch registry (Commit 2b) lands.
//! * **Commit 2b** — `Registry` / `TransformRegistry` + [`get_specialization`] /
//!   `get_transformer` — fills the stubbed bodies.

use super::super::flowspace::operation::{BuiltinException, HLOperation};
use super::context::AnnotatorContext;
use super::model::SomeValue;

/// RPython `SingleDispatchMixin.get_specialization` /
/// `DoubleDispatchMixin.get_specialization` return value
/// (operation.py:226-239 / 268-281).
///
/// Upstream specializations are Python closures carrying a
/// `can_only_throw` attribute (list or callable). The Rust port lands
/// with Commit 2b; this alias reserves the name.
pub type SpecializationFn = dyn Fn(&dyn AnnotatorContext, &HLOperation) -> SomeValue;

/// RPython `read_can_only_throw(opimpl, *args)` (model.py:837-841).
///
/// An `opimpl` (a specialization closure produced by `get_specialization`)
/// carries an optional `can_only_throw` attribute. It is either a plain
/// list of exception classes or a callable that, given the argument
/// annotations, returns such a list.
///
/// The Rust port represents this as an enum over the two branches so
/// callers do not need to reach into Python attribute space. Commit 2b
/// populates this when it builds the specialization registry.
pub enum CanOnlyThrow {
    /// `can_only_throw` missing or `None` upstream.
    Absent,
    /// `can_only_throw` is a fixed list of exception classes.
    List(Vec<BuiltinException>),
    /// `can_only_throw` is a callable computed per-call site.
    /// Boxed closure; evaluated lazily via [`read_can_only_throw`].
    Callable(Box<dyn Fn(&[SomeValue]) -> Vec<BuiltinException>>),
}

/// RPython `read_can_only_throw(opimpl, *args)` — resolve the
/// specialization's exception set at call time.
pub fn read_can_only_throw(
    can_only_throw: &CanOnlyThrow,
    args_s: &[SomeValue],
) -> Option<Vec<BuiltinException>> {
    match can_only_throw {
        CanOnlyThrow::Absent => None,
        CanOnlyThrow::List(xs) => Some(xs.clone()),
        CanOnlyThrow::Callable(f) => Some(f(args_s)),
    }
}

/// RPython `HLOperation.consider(self, annotator)` (operation.py:101-104).
///
/// ```python
/// def consider(self, annotator):
///     args_s = [annotator.annotation(arg) for arg in self.args]
///     spec = type(self).get_specialization(*args_s)
///     return spec(annotator, *self.args)
/// ```
///
/// Body lands with Commit 2b.
pub fn consider_hlop(_ann: &dyn AnnotatorContext, _hlop: &HLOperation) -> SomeValue {
    todo!("consider_hlop: registry lookup lands with Commit 2b")
}

/// RPython `HLOperation.transform(self, annotator)` (operation.py:112-115).
///
/// Body lands with Commit 2b.
pub fn transform_hlop(_ann: &dyn AnnotatorContext, _hlop: &HLOperation) {
    todo!("transform_hlop: registry lookup lands with Commit 2b")
}

/// RPython `HLOperation.get_can_only_throw(self, annotator)`
/// (operation.py:106-107, SingleDispatchMixin:221-224,
/// DoubleDispatchMixin:283-286).
///
/// The default `HLOperation.get_can_only_throw` returns `None` — the
/// annotator falls back on `op.canraise`. Single/Double dispatch
/// subclasses route through `read_can_only_throw(spec, *args_s)`.
/// Body lands with Commit 2b.
pub fn get_can_only_throw(
    _ann: &dyn AnnotatorContext,
    _hlop: &HLOperation,
) -> Option<Vec<BuiltinException>> {
    // Default HLOperation override returns None (operation.py:106-107).
    None
}

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::operation::BuiltinException;
    use super::*;

    #[test]
    fn read_can_only_throw_absent_is_none() {
        assert!(read_can_only_throw(&CanOnlyThrow::Absent, &[]).is_none());
    }

    #[test]
    fn read_can_only_throw_list_is_clone() {
        let xs = vec![BuiltinException::IndexError, BuiltinException::KeyError];
        let got = read_can_only_throw(&CanOnlyThrow::List(xs.clone()), &[]).unwrap();
        assert_eq!(got, xs);
    }

    #[test]
    fn read_can_only_throw_callable_is_called() {
        let cot = CanOnlyThrow::Callable(Box::new(|_args_s| vec![BuiltinException::OverflowError]));
        let got = read_can_only_throw(&cot, &[]).unwrap();
        assert_eq!(got, vec![BuiltinException::OverflowError]);
    }
}
