//! RPython `HLOperation.{consider, transform, get_can_only_throw}`
//! ported as free functions over the operation + annotator pair, plus
//! the dispatch registry that backs `get_specialization`.
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
//! * **Commit 2a** — trait hook ([`super::context::AnnotatorContext`])
//!   + entry-point signatures + [`read_can_only_throw`] helper.
//! * **Commit 2b (this update)** — [`UnaryRegistry`] / [`BinaryRegistry`]
//!   + `register_*` / `get_specialization_*` + [`consider_hlop`] body.
//!   Registries are populated lazily on first access; binaryop /
//!   unaryop modules feed them via [`BinaryRegistry::populate_defaults`]
//!   and [`UnaryRegistry::populate_defaults`].

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use super::super::flowspace::operation::{BuiltinException, Dispatch, HLOperation, OpKind};
use super::context::AnnotatorContext;
use super::model::{SomeValue, SomeValueTag};

/// RPython `SingleDispatchMixin.get_specialization` /
/// `DoubleDispatchMixin.get_specialization` return value
/// (operation.py:226-239 / 268-281).
///
/// Upstream specializations are Python closures carrying a
/// `can_only_throw` attribute (list or callable). The Rust port
/// represents the closure body with `Fn` and packages the
/// `can_only_throw` side-band next to it.
pub struct Specialization {
    /// Actual annotation handler. `(ann, hlop)` → `SomeValue`.
    pub apply: Box<dyn Fn(&dyn AnnotatorContext, &HLOperation) -> SomeValue + Send + Sync>,
    /// RPython `specialized.can_only_throw` side-band (operation.py:234).
    pub can_only_throw: CanOnlyThrow,
}

/// RPython `read_can_only_throw(opimpl, *args)` (model.py:837-841).
///
/// An `opimpl` (a specialization closure produced by `get_specialization`)
/// carries an optional `can_only_throw` attribute. It is either a plain
/// list of exception classes or a callable that, given the argument
/// annotations, returns such a list.
pub enum CanOnlyThrow {
    /// `can_only_throw` missing or `None` upstream.
    Absent,
    /// `can_only_throw` is a fixed list of exception classes.
    List(Vec<BuiltinException>),
    /// `can_only_throw` is a callable computed per-call site.
    Callable(Box<dyn Fn(&[SomeValue]) -> Vec<BuiltinException> + Send + Sync>),
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

/// Registry backing [`OpKind`] whose upstream `dispatch = 1`
/// (`SingleDispatchMixin`, operation.py:202-255).
///
/// Key: `(OpKind, SomeValueTag)` — the op plus the tag of the first
/// argument's annotation. Lookup emulates upstream's MRO walk by
/// iterating [`SomeValueTag::mro`].
pub struct UnaryRegistry {
    entries: HashMap<(OpKind, SomeValueTag), Specialization>,
}

impl UnaryRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// RPython `@op.<name>.register(Some_cls)` (operation.py:205-210).
    pub fn register(&mut self, op: OpKind, tag: SomeValueTag, spec: Specialization) {
        self.entries.insert((op, tag), spec);
    }

    /// RPython `SingleDispatchMixin._dispatch(type(s_arg))` MRO walk
    /// (operation.py:212-219). Returns the registered specialization
    /// for the most specific tag in `SomeValueTag::mro`.
    pub fn get(&self, op: OpKind, arg_tag: SomeValueTag) -> Option<&Specialization> {
        for tag in arg_tag.mro() {
            if let Some(spec) = self.entries.get(&(op, *tag)) {
                return Some(spec);
            }
        }
        None
    }
}

/// Registry backing [`OpKind`] whose upstream `dispatch = 2`
/// (`DoubleDispatchMixin`, operation.py:258-300).
pub struct BinaryRegistry {
    entries: HashMap<(OpKind, SomeValueTag, SomeValueTag), Specialization>,
}

impl BinaryRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// RPython `@op.<name>.register(Some1, Some2)` (operation.py:261-266).
    pub fn register(
        &mut self,
        op: OpKind,
        lhs: SomeValueTag,
        rhs: SomeValueTag,
        spec: Specialization,
    ) {
        self.entries.insert((op, lhs, rhs), spec);
    }

    /// RPython `DoubleDispatchMixin` registry lookup
    /// (operation.py:268-281). Walks the MRO cross-product so a
    /// `(SomeBool, SomeInt)` call falls back through
    /// `(SomeInteger, SomeInteger)` to `(SomeObject, SomeObject)`.
    pub fn get(
        &self,
        op: OpKind,
        lhs_tag: SomeValueTag,
        rhs_tag: SomeValueTag,
    ) -> Option<&Specialization> {
        for l in lhs_tag.mro() {
            for r in rhs_tag.mro() {
                if let Some(spec) = self.entries.get(&(op, *l, *r)) {
                    return Some(spec);
                }
            }
        }
        None
    }
}

/// Global single-dispatch registry.
pub static UNARY_REGISTRY: LazyLock<RwLock<UnaryRegistry>> =
    LazyLock::new(|| RwLock::new(UnaryRegistry::new()));

/// Global double-dispatch registry.
pub static BINARY_REGISTRY: LazyLock<RwLock<BinaryRegistry>> = LazyLock::new(|| {
    let mut reg = BinaryRegistry::new();
    super::binaryop::populate_defaults(&mut reg);
    RwLock::new(reg)
});

/// RPython `HLOperation.consider(self, annotator)` (operation.py:101-104,
/// with the SingleDispatch / DoubleDispatch overrides at 226-239 /
/// 268-281).
///
/// ```python
/// def consider(self, annotator):
///     args_s = [annotator.annotation(arg) for arg in self.args]
///     spec = type(self).get_specialization(*args_s)
///     return spec(annotator, *self.args)
/// ```
pub fn consider_hlop(ann: &dyn AnnotatorContext, hlop: &HLOperation) -> SomeValue {
    let args_s: Vec<SomeValue> = hlop
        .args
        .iter()
        .map(|a| {
            ann.annotation(a)
                .unwrap_or_else(|| panic!("consider: unbound arg in {:?}", hlop.kind))
        })
        .collect();
    match hlop.kind.dispatch() {
        Dispatch::Single => {
            let tag = args_s.first().expect("dispatch=1 op with 0 args").tag();
            let reg = UNARY_REGISTRY.read().expect("UNARY_REGISTRY poisoned");
            let spec = reg
                .get(hlop.kind, tag)
                .unwrap_or_else(|| panic!("no unary spec for {:?}({:?})", hlop.kind, tag));
            (spec.apply)(ann, hlop)
        }
        Dispatch::Double => {
            let tag_l = args_s.first().expect("dispatch=2 op with 0 args").tag();
            let tag_r = args_s.get(1).expect("dispatch=2 op with 1 arg").tag();
            let reg = BINARY_REGISTRY.read().expect("BINARY_REGISTRY poisoned");
            let spec = reg.get(hlop.kind, tag_l, tag_r).unwrap_or_else(|| {
                panic!(
                    "no binary spec for {:?}({:?}, {:?})",
                    hlop.kind, tag_l, tag_r
                )
            });
            (spec.apply)(ann, hlop)
        }
        Dispatch::None => {
            // Special-cased ops (NewDict/NewTuple/NewList/NewSlice/Pow/
            // SimpleCall/CallArgs/Contains/Trunc/Format/Get/Set/Delete/
            // UserDel/Buffer/Yield/Hint). Upstream per-class consider().
            todo!("consider_hlop: Dispatch::None special cases land with their own commits")
        }
    }
}

/// RPython `HLOperation.transform(self, annotator)` (operation.py:112-115).
pub fn transform_hlop(_ann: &dyn AnnotatorContext, _hlop: &HLOperation) {
    // `cls._transform` registries are populated lazily by
    // `register_unary_transform` / `register_binary_transform` — empty
    // registry means the upstream `lambda *args: None` default applies.
    // A follow-up commit wires the transform tables alongside the
    // unary/binary operation bodies.
}

/// RPython `HLOperation.get_can_only_throw(self, annotator)`
/// (operation.py:106-107, SingleDispatchMixin:221-224,
/// DoubleDispatchMixin:283-286).
pub fn get_can_only_throw(
    ann: &dyn AnnotatorContext,
    hlop: &HLOperation,
) -> Option<Vec<BuiltinException>> {
    let args_s: Vec<SomeValue> = hlop.args.iter().filter_map(|a| ann.annotation(a)).collect();
    match hlop.kind.dispatch() {
        Dispatch::Single => {
            if args_s.is_empty() {
                return None;
            }
            let reg = UNARY_REGISTRY.read().expect("UNARY_REGISTRY poisoned");
            reg.get(hlop.kind, args_s[0].tag())
                .and_then(|spec| read_can_only_throw(&spec.can_only_throw, &args_s))
        }
        Dispatch::Double => {
            if args_s.len() < 2 {
                return None;
            }
            let reg = BINARY_REGISTRY.read().expect("BINARY_REGISTRY poisoned");
            reg.get(hlop.kind, args_s[0].tag(), args_s[1].tag())
                .and_then(|spec| read_can_only_throw(&spec.can_only_throw, &args_s))
        }
        // Default HLOperation override returns None (operation.py:106-107).
        Dispatch::None => None,
    }
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

    #[test]
    fn binary_registry_mro_fallback_resolves_object_default() {
        use super::super::model::{SomeInteger, SomeValueTag};
        let mut reg = BinaryRegistry::new();
        reg.register(
            OpKind::Is,
            SomeValueTag::Object,
            SomeValueTag::Object,
            Specialization {
                apply: Box::new(|_ann, _hl| SomeValue::object()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
        // SomeBool args hit the Object default via the MRO walk.
        assert!(
            reg.get(OpKind::Is, SomeValueTag::Bool, SomeValueTag::Bool)
                .is_some()
        );
        // Dispatch on a specific integer also falls back.
        let _ = SomeInteger::default();
        assert!(
            reg.get(OpKind::Is, SomeValueTag::Integer, SomeValueTag::Object)
                .is_some()
        );
    }

    #[test]
    fn unary_registry_mro_fallback_resolves_object_default() {
        let mut reg = UnaryRegistry::new();
        reg.register(
            OpKind::Neg,
            SomeValueTag::Object,
            Specialization {
                apply: Box::new(|_ann, _hl| SomeValue::object()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
        assert!(reg.get(OpKind::Neg, SomeValueTag::Integer).is_some());
    }
}
