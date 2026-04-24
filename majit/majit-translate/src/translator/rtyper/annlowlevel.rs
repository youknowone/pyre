//! RPython `rpython/rtyper/annlowlevel.py` — low-level helper annotation
//! + mix-level helper injection bridge.
//!
//! Upstream docstring:
//!
//! > The code needed to flow and annotate low-level helpers -- the
//! > ll_*() functions
//!
//! This module is the structural bridge between
//! [`crate::annotator::annrpython::RPythonAnnotator`] and
//! [`crate::translator::rtyper::rtyper::RPythonTyper`] for any code
//! that needs to inject a freshly annotated + rtyped helper graph into
//! an in-flight translation:
//!
//! * `ExceptionData.make_helpers` (exceptiondata.py:47-62) →
//!   `annmixlevel.getgraph(ll_exception_match, …)` etc.
//! * `normalizecalls.create_instantiate_functions` (normalizecalls.py:
//!   :~300-370) — generates `my_instantiate_graph` on every `ClassDef`
//!   with an `__init__`.
//! * `RPythonTyper.getannmixlevel` (rtyper.py:179-181) and the
//!   `self.annmixlevel.finish()` fixpoint step inside
//!   `specialize_more_blocks` (rtyper.py:238-241).
//! * `rpython/translator/backendopt/all.py::backend_optimizations`
//!   `secondary=True` path (via `backend_optimize`).
//!
//! ## Status of this port
//!
//! **Skeleton landed; bodies deferred.** Every public function and
//! method below surfaces as `TyperError::message("annlowlevel.py:N
//! — <symbol> port pending")` so downstream consumers can call into
//! the module in upstream shape without silently taking an adapted
//! code path. Filling the bodies is the orthodox R4 work — the
//! skeleton exists so each method can be swapped in against its
//! upstream counterpart without call-site churn.
//!
//! The items intentionally **not** included in the first skeleton
//! (scheduled as their own follow-ups):
//!
//! * `PseudoHighLevelCallable` + `PseudoHighLevelCallableEntry`
//!   (annlowlevel.py:286-320) — depends on
//!   [`crate::translator::rtyper::extregistry::ExtRegistryEntry`] and
//!   `hop.genop('direct_call', …)` wiring.
//! * `llhelper` / `llhelper_args` + `LLHelperEntry` (annlowlevel.py:
//!   :325-376) — depends on `r.get_unique_llfn()` on
//!   `FunctionsPBCRepr` (unported).
//! * `hlstr` / `llstr` / `hlunicode` / `llunicode` and their entries
//!   (annlowlevel.py:380-449) — depend on
//!   `rpython/rtyper/lltypesystem/rstr.py::STR` / `UNICODE` (unported).
//! * `cast_object_to_ptr`, `cast_instance_to_base_ptr`,
//!   `cast_instance_to_gcref`, `cast_nongc_instance_to_base_ptr`,
//!   `cast_nongc_instance_to_adr`, `cast_base_ptr_to_instance`,
//!   `cast_gcref_to_instance`, `cast_adr_to_nongc_instance`, and
//!   their entries (annlowlevel.py:453-568) — depend on
//!   `lltype.cast_pointer` / `cast_opaque_ptr` ops at the rtyper
//!   opcode dispatch level.
//! * `placeholder_sigarg` / `typemeth_placeholder_sigarg` /
//!   `ADTInterface` (annlowlevel.py:573-640) — depend on
//!   `rpython/annotator/signature.py::Sig` wiring and the `adtmeths`
//!   attribute of low-level types.
//! * `cachedtype` metaclass (annlowlevel.py:644-668) — metaclass
//!   shape has no direct Rust counterpart; callers reach for
//!   `once_cell::sync::Lazy` / `std::sync::LazyLock` + a manual cache
//!   map instead.
//!
//! ## Dependencies already in place
//!
//! * [`crate::annotator::annrpython::RPythonAnnotator::annotate_helper`]
//!   (annrpython.py:99-110) — primitive that `getgraph` wraps.
//! * [`crate::annotator::annrpython::RPythonAnnotator::complete_helpers`]
//!   (annrpython.py:112-120) — called by `finish_annotate`.
//! * [`crate::annotator::annrpython::RPythonAnnotator::using_policy`]
//!   (annrpython.py:122-128) — RAII policy swap guard used by both
//!   `getgraph` and `finish_annotate`.
//! * [`crate::translator::rtyper::rtyper::RPythonTyper::getrepr`]
//!   (rtyper.py:143-165) — consumed by `getdelayedrepr`.
//! * [`crate::translator::rtyper::rtyper::RPythonTyper::getcallable`]
//!   (rtyper.py:569-581) — consumed by `finish_rtype` to resolve
//!   `delayedfuncs`.
//! * [`crate::translator::rtyper::rtyper::RPythonTyper::call_all_setups`]
//!   (rtyper.py:543) — called by `finish_rtype` between delayed
//!   resolution passes.
//! * [`crate::translator::rtyper::rtyper::RPythonTyper::specialize_more_blocks`]
//!   (rtyper.py:198-241) — called by `finish_rtype` to flow the
//!   newly annotated helpers.
//! * [`crate::translator::rtyper::normalizecalls::perform_normalizations`]
//!   (normalizecalls.py:1-20) — called by `finish_rtype`.
//! * [`crate::translator::rtyper::rmodel::Repr::set_setup_delayed`] +
//!   `set_setup_maybe_delayed` / `is_setup_delayed` (rmodel.py:79-93).
//!
//! Filling the method bodies from these primitives is strictly a
//! line-by-line transliteration of upstream.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::description::{FunctionDesc, GraphCacheKey};
use crate::annotator::model::{SomeObjectTrait, SomeValue, not_const};
use crate::annotator::policy::{
    AnnotatorPolicy, PolicyError, PolicyHandle, PolicyOps, Specializer,
    parse_specializer_directive, specializer_from_normalized,
};
use crate::flowspace::model::{BlockKey, ConstValue, Constant, GraphKey, GraphRef, HostObject};
use crate::flowspace::pygraph::PyGraph;

use super::error::TyperError;
use super::llannotation::{annotation_to_lltype, lltype_to_annotation};
use super::lltypesystem::lltype::{
    self, _ptr, DelayedPointer, ForwardReference, FuncType, LowLevelType, Ptr,
};
use super::rmodel::Repr;
use super::rtyper::RPythonTyper;

/// RPython `rpython.tool.sourcetools.valid_identifier` — fold any
/// non-alphanumeric character to `_`, prepend `_` if the first byte
/// would otherwise be a digit, truncate to 120 characters. Shared
/// between [`self`] and [`super::normalizecalls`] (both upstream
/// modules import from `rpython.tool.sourcetools`).
pub(super) fn valid_identifier(stuff: impl std::fmt::Display) -> String {
    let mut stuff: String = stuff
        .to_string()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    if stuff.is_empty() || stuff.as_bytes()[0].is_ascii_digit() {
        stuff.insert(0, '_');
    }
    stuff.truncate(120);
    stuff
}

// ---------------------------------------------------------------------
// annlowlevel.py:20-41 — KeyComp
// ---------------------------------------------------------------------

/// RPython `class KeyComp(object)` (annlowlevel.py:20-41).
///
/// Wrapper used by `LowLevelAnnotatorPolicy.lowlevelspecialize` to
/// build cache keys out of PBC constants and low-level types. The
/// upstream class overrides `__eq__` / `__hash__` / `__str__` so that
/// cache-key comparisons are value-based rather than identity-based
/// even when the wrapped value is a mutable Python object.
///
/// **Port pending.** The concrete `value` payload is one of:
///
/// * a [`LowLevelType`] (stringified via `_short_name` + `'LlT'`),
/// * a Python object with `__name__` or `compact_repr()` + `'Const'`,
/// * or a general `repr(val) + 'Const'`.
///
/// Representing that union as a Rust enum with the same `Eq + Hash`
/// semantics is straightforward once the repr-based path is ported —
/// until then, constructors return this stub.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyComp {
    /// Original upstream attribute `self.val`. Mirrored here verbatim;
    /// kept `pub` so test helpers can observe the seed value.
    pub val: KeyCompValue,
}

/// Internal tag for the wrapped value. Keeps the [`KeyComp`] API
/// uniform while the various branches are ported one at a time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyCompValue {
    /// `lltype.LowLevelType` branch (annlowlevel.py:31-32).
    Lltype(LowLevelType),
    /// Generic constant branch (annlowlevel.py:33-39).
    Const(ConstValue),
}

impl KeyComp {
    /// Construct a `KeyComp` from a wrapped value.
    pub fn new(val: KeyCompValue) -> Self {
        KeyComp { val }
    }
}

/// Upstream `KeyComp.__str__` / `__repr__` (annlowlevel.py:29-41):
/// `LowLevelType` → `'<short_name>LlT'`; general consts append `'Const'`.
impl std::fmt::Display for KeyComp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.val {
            KeyCompValue::Lltype(t) => write!(f, "{}LlT", t.short_name()),
            KeyCompValue::Const(ConstValue::HostObject(h)) => {
                let name = h.qualname().rsplit('.').next().unwrap_or(h.qualname());
                write!(f, "{}Const", name)
            }
            KeyCompValue::Const(value) => write!(f, "{}Const", value),
        }
    }
}

// ---------------------------------------------------------------------
// annlowlevel.py:43-90 — LowLevelAnnotatorPolicy
// ---------------------------------------------------------------------

/// RPython `class LowLevelAnnotatorPolicy(AnnotatorPolicy)`
/// (annlowlevel.py:43-90).
///
/// Specialization policy that keys the cached graph by the lltype of
/// each positional argument, with constant-PBC arguments promoted to
/// a [`KeyComp`]-wrapped identity key.
///
/// Upstream is a subclass of `AnnotatorPolicy`; the Rust port composes
/// rather than inherits — `base` holds the default policy and
/// overrides are implemented as direct methods on this struct.
#[derive(Clone, Debug)]
pub struct LowLevelAnnotatorPolicy {
    /// RPython `self.rtyper = rtyper` (annlowlevel.py:44-45). Weak
    /// because the rtyper owns the strong reference and this policy
    /// lives inside `MixLevelHelperAnnotator` whose lifetime is
    /// ≤ rtyper's.
    pub rtyper: Weak<RPythonTyper>,
    /// Base-policy fallback for non-`ll_`-prefixed callees in the
    /// `MixLevelAnnotatorPolicy` subclass override. Stored on the
    /// struct rather than derived per-call because `AnnotatorPolicy`
    /// is trivially cloneable.
    pub base: AnnotatorPolicy,
}

impl LowLevelAnnotatorPolicy {
    /// RPython `LowLevelAnnotatorPolicy.__init__(self, rtyper=None)`
    /// (annlowlevel.py:44-45).
    pub fn new(rtyper: Option<&Rc<RPythonTyper>>) -> Self {
        LowLevelAnnotatorPolicy {
            rtyper: rtyper.map(Rc::downgrade).unwrap_or_default(),
            base: AnnotatorPolicy::new(),
        }
    }

    /// RPython `LowLevelAnnotatorPolicy.lowlevelspecialize(funcdesc,
    /// args_s, key_for_args)` (annlowlevel.py:47-76).
    ///
    /// ```python
    /// @staticmethod
    /// def lowlevelspecialize(funcdesc, args_s, key_for_args):
    ///     args_s, key1, builder = flatten_star_args(funcdesc, args_s)
    ///     key = []
    ///     new_args_s = []
    ///     for i, s_obj in enumerate(args_s):
    ///         if i in key_for_args:
    ///             key.append(key_for_args[i])
    ///             new_args_s.append(s_obj)
    ///         elif isinstance(s_obj, annmodel.SomePBC):
    ///             assert s_obj.is_constant(), "ambiguous low-level helper specialization"
    ///             key.append(KeyComp(s_obj.const))
    ///             new_args_s.append(s_obj)
    ///         elif isinstance(s_obj, annmodel.SomeNone):
    ///             key.append(KeyComp(None))
    ///             new_args_s.append(s_obj)
    ///         else:
    ///             new_args_s.append(annmodel.not_const(s_obj))
    ///             try:
    ///                 key.append(annotation_to_lltype(s_obj))
    ///             except ValueError:
    ///                 key.append(s_obj.__class__)
    ///     key = (tuple(key),)
    ///     if key1 is not None:
    ///         key += (key1,)
    ///     flowgraph = funcdesc.cachedgraph(key, builder=builder)
    ///     args_s[:] = new_args_s
    ///     return flowgraph
    /// ```
    ///
    /// Port pending. Depends on
    /// [`crate::annotator::specialize::flatten_star_args`],
    /// [`crate::translator::rtyper::llannotation::annotation_to_lltype`],
    /// and `FunctionDesc::cachedgraph` — all already ported.
    pub fn lowlevelspecialize(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &mut Vec<SomeValue>,
        key_for_args: &HashMap<usize, KeyComp>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let (args_s_flat, key1, builder) = funcdesc
            .flatten_star_args(args_s)
            .map_err(|e| TyperError::message(e.to_string()))?;
        let mut key = Vec::new();
        let mut new_args_s = Vec::new();
        for (i, s_obj) in args_s_flat.iter().enumerate() {
            if let Some(keycomp) = key_for_args.get(&i) {
                key.push(GraphCacheKey::KeyComp(keycomp.clone()));
                new_args_s.push(s_obj.clone());
            } else if let SomeValue::PBC(_pbc) = s_obj {
                if !s_obj.is_constant() {
                    return Err(TyperError::message(
                        "ambiguous low-level helper specialization",
                    ));
                }
                let const_value = s_obj
                    .const_()
                    .cloned()
                    .expect("constant SomePBC must carry const()");
                key.push(GraphCacheKey::KeyComp(KeyComp::new(KeyCompValue::Const(
                    const_value,
                ))));
                new_args_s.push(s_obj.clone());
            } else if matches!(s_obj, SomeValue::None_(_)) {
                key.push(GraphCacheKey::KeyComp(KeyComp::new(KeyCompValue::Const(
                    ConstValue::None,
                ))));
                new_args_s.push(s_obj.clone());
            } else {
                new_args_s.push(not_const(s_obj));
                match annotation_to_lltype(s_obj, None) {
                    Ok(lltype) => key.push(GraphCacheKey::LowLevelType(lltype)),
                    Err(_) => key.push(GraphCacheKey::SomeValueTag(s_obj.tag())),
                }
            }
        }
        let mut key = vec![GraphCacheKey::Tuple(key)];
        if !matches!(key1, GraphCacheKey::None) {
            key.push(key1);
        }
        let flowgraph = funcdesc
            .cachedgraph(GraphCacheKey::Tuple(key), None, builder)
            .map_err(|e| TyperError::message(e.to_string()))?;
        *args_s = new_args_s;
        Ok(flowgraph)
    }

    /// RPython `LowLevelAnnotatorPolicy.default_specialize(funcdesc,
    /// args_s)` (annlowlevel.py:78-80). Thin alias:
    ///
    /// ```python
    /// return LowLevelAnnotatorPolicy.lowlevelspecialize(funcdesc, args_s, {})
    /// ```
    pub fn default_specialize(
        &self,
        funcdesc: &crate::annotator::description::FunctionDesc,
        args_s: &mut Vec<SomeValue>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        self.lowlevelspecialize(funcdesc, args_s, &std::collections::HashMap::new())
    }

    /// RPython `specialize__ll = default_specialize` (annlowlevel.py:82).
    /// Kept as a distinct method so the
    /// [`crate::annotator::policy::Specializer::Ll`] dispatch can
    /// call it by its upstream attribute name.
    pub fn specialize_ll(
        &self,
        funcdesc: &crate::annotator::description::FunctionDesc,
        args_s: &mut Vec<SomeValue>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        self.default_specialize(funcdesc, args_s)
    }

    /// RPython `LowLevelAnnotatorPolicy.specialize__ll_and_arg(funcdesc,
    /// args_s, *argindices)` (annlowlevel.py:84-90).
    ///
    /// ```python
    /// @staticmethod
    /// def specialize__ll_and_arg(funcdesc, args_s, *argindices):
    ///     keys = {}
    ///     for i in argindices:
    ///         keys[i] = args_s[i].const
    ///     return LowLevelAnnotatorPolicy.lowlevelspecialize(
    ///         funcdesc, args_s, keys)
    /// ```
    ///
    /// Port pending.
    pub fn specialize_ll_and_arg(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &mut Vec<SomeValue>,
        argindices: &[usize],
    ) -> Result<Rc<PyGraph>, TyperError> {
        let mut keys = HashMap::new();
        for &i in argindices {
            let const_value = args_s
                .get(i)
                .and_then(SomeValue::const_)
                .cloned()
                .ok_or_else(|| {
                    TyperError::message(format!(
                        "specialize__ll_and_arg: argument {i} must be constant"
                    ))
                })?;
            keys.insert(i, KeyComp::new(KeyCompValue::Const(const_value)));
        }
        self.lowlevelspecialize(funcdesc, args_s, &keys)
    }
}

impl PolicyOps for LowLevelAnnotatorPolicy {
    fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError> {
        let Some(directive) = directive else {
            return Ok(Specializer::LowLevelDefault);
        };
        let (normalized, parms) = parse_specializer_directive(directive)?;
        match normalized.as_str() {
            "default_specialize" => Ok(Specializer::LowLevelDefault),
            _ => specializer_from_normalized(&normalized, parms),
        }
    }

    fn no_more_blocks_to_annotate(&self, ann: &RPythonAnnotator) {
        self.base.no_more_blocks_to_annotate(ann);
    }
}

impl From<LowLevelAnnotatorPolicy> for PolicyHandle {
    fn from(value: LowLevelAnnotatorPolicy) -> Self {
        PolicyHandle::new(value)
    }
}

// ---------------------------------------------------------------------
// annlowlevel.py:92-95 — annotate_lowlevel_helper
// ---------------------------------------------------------------------

/// RPython `annotate_lowlevel_helper(annotator, ll_function, args_s,
/// policy=None)` (annlowlevel.py:92-95).
///
/// ```python
/// def annotate_lowlevel_helper(annotator, ll_function, args_s, policy=None):
///     if policy is None:
///         policy = LowLevelAnnotatorPolicy()
///     return annotator.annotate_helper(ll_function, args_s, policy)
/// ```
///
/// Port pending. The Rust side already has
/// [`RPythonAnnotator::annotate_helper`], so the body reduces to:
/// construct a default `LowLevelAnnotatorPolicy` when `policy` is
/// `None`, then delegate. The one obstacle is that
/// `annotate_helper` currently consumes `AnnotatorPolicy` by value
/// and the low-level variant would need to pass through the ll-policy
/// methods — follow-up work threads the policy as a trait object.
pub fn annotate_lowlevel_helper(
    annotator: &Rc<RPythonAnnotator>,
    ll_function: &HostObject,
    args_s: Vec<SomeValue>,
    policy: Option<LowLevelAnnotatorPolicy>,
) -> Result<Rc<PyGraph>, TyperError> {
    let policy = policy.unwrap_or_else(|| LowLevelAnnotatorPolicy::new(None));
    annotator
        .annotate_helper(ll_function, args_s, Some(PolicyHandle::from(policy)))
        .map_err(|e| TyperError::message(e.to_string()))
}

// ---------------------------------------------------------------------
// annlowlevel.py:100-123 — MixLevelAnnotatorPolicy
// ---------------------------------------------------------------------

/// RPython `class MixLevelAnnotatorPolicy(LowLevelAnnotatorPolicy)`
/// (annlowlevel.py:100-123).
///
/// Specialization policy used by [`MixLevelHelperAnnotator`]. Extends
/// `LowLevelAnnotatorPolicy` so that:
///
/// * Callees whose name starts with `ll_` or `_ll_` go through
///   `LowLevelAnnotatorPolicy.default_specialize` (i.e. lltype-keyed
///   cache).
/// * Every other callee falls back to plain
///   `AnnotatorPolicy.default_specialize`.
///
/// Upstream `__init__` takes an `annhelper` (the
/// `MixLevelHelperAnnotator` instance) and copies its
/// `annhelper.rtyper` onto `self.rtyper`. The Rust port wraps a
/// `LowLevelAnnotatorPolicy` directly because the only rtyper
/// reference the policy ever reads is `self.rtyper` — inheriting
/// from `LowLevelAnnotatorPolicy` via composition captures that
/// shape without the `annhelper` → `rtyper` indirection.
#[derive(Clone, Debug)]
pub struct MixLevelAnnotatorPolicy {
    /// Composed base — `LowLevelAnnotatorPolicy.rtyper` is the only
    /// field upstream inherits via `super().__init__`.
    pub ll: LowLevelAnnotatorPolicy,
}

impl MixLevelAnnotatorPolicy {
    /// RPython `MixLevelAnnotatorPolicy.__init__(self, annhelper)`
    /// (annlowlevel.py:102-103).
    pub fn new(annhelper: &MixLevelHelperAnnotator) -> Self {
        MixLevelAnnotatorPolicy {
            ll: LowLevelAnnotatorPolicy {
                rtyper: annhelper.rtyper.clone(),
                base: AnnotatorPolicy::new(),
            },
        }
    }

    /// RPython `MixLevelAnnotatorPolicy.default_specialize(self,
    /// funcdesc, args_s)` (annlowlevel.py:105-111).
    ///
    /// ```python
    /// def default_specialize(self, funcdesc, args_s):
    ///     name = funcdesc.name
    ///     if name.startswith('ll_') or name.startswith('_ll_'):
    ///         return super(MixLevelAnnotatorPolicy, self) \
    ///             .default_specialize(funcdesc, args_s)
    ///     else:
    ///         return AnnotatorPolicy.default_specialize(funcdesc, args_s)
    /// ```
    ///
    /// Port pending.
    pub fn default_specialize(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &mut Vec<SomeValue>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let name = &funcdesc.name;
        if name.starts_with("ll_") || name.starts_with("_ll_") {
            self.ll.default_specialize(funcdesc, args_s)
        } else {
            funcdesc
                .default_specialize(args_s)
                .map_err(|e| TyperError::message(e.to_string()))
        }
    }

    /// RPython `MixLevelAnnotatorPolicy.specialize__arglltype(self,
    /// funcdesc, args_s, i)` (annlowlevel.py:113-116).
    ///
    /// ```python
    /// def specialize__arglltype(self, funcdesc, args_s, i):
    ///     key = self.rtyper.getrepr(args_s[i]).lowleveltype
    ///     alt_name = funcdesc.name + "__for_%sLlT" % key._short_name()
    ///     return funcdesc.cachedgraph(key, alt_name=valid_identifier(alt_name))
    /// ```
    ///
    /// Port pending. Consumers already reach
    /// [`RPythonTyper::getrepr`] via
    /// [`LowLevelAnnotatorPolicy::rtyper`].
    pub fn specialize_arglltype(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &[SomeValue],
        i: usize,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let rtyper = self
            .ll
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelAnnotatorPolicy: rtyper dropped"))?;
        let key = rtyper.getrepr(&args_s[i])?.lowleveltype().clone();
        let alt_name = valid_identifier(format!("{}__for_{}LlT", funcdesc.name, key.short_name()));
        funcdesc
            .cachedgraph(GraphCacheKey::LowLevelType(key), Some(&alt_name), None)
            .map_err(|e| TyperError::message(e.to_string()))
    }

    /// RPython `MixLevelAnnotatorPolicy.specialize__genconst(self,
    /// funcdesc, args_s, i)` (annlowlevel.py:118-123).
    ///
    /// Upstream comment: `# XXX this is specific to the JIT`. Coerces
    /// `args_s[i]` to a `lltype_to_annotation(TYPE)` and keys on the
    /// resulting lltype.
    ///
    /// Port pending.
    pub fn specialize_genconst(
        &self,
        funcdesc: &FunctionDesc,
        args_s: &mut Vec<SomeValue>,
        i: usize,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let typ = annotation_to_lltype(&args_s[i], Some("genconst"))
            .map_err(|e| TyperError::message(e.to_string()))?;
        args_s[i] = lltype_to_annotation(typ.clone());
        let alt_name = valid_identifier(format!("{}__{}", funcdesc.name, typ.short_name()));
        funcdesc
            .cachedgraph(GraphCacheKey::LowLevelType(typ), Some(&alt_name), None)
            .map_err(|e| TyperError::message(e.to_string()))
    }
}

impl PolicyOps for MixLevelAnnotatorPolicy {
    fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError> {
        let Some(directive) = directive else {
            return Ok(Specializer::MixLevelDefault);
        };
        let (normalized, parms) = parse_specializer_directive(directive)?;
        match normalized.as_str() {
            "default_specialize" => Ok(Specializer::MixLevelDefault),
            "specialize__arglltype" => Ok(Specializer::ArgLltype { parms }),
            "specialize__genconst" => Ok(Specializer::GenConst { parms }),
            _ => specializer_from_normalized(&normalized, parms),
        }
    }

    fn no_more_blocks_to_annotate(&self, ann: &RPythonAnnotator) {
        self.ll.base.no_more_blocks_to_annotate(ann);
    }
}

impl From<MixLevelAnnotatorPolicy> for PolicyHandle {
    fn from(value: MixLevelAnnotatorPolicy) -> Self {
        PolicyHandle::new(value)
    }
}

// ---------------------------------------------------------------------
// annlowlevel.py:126-284 — MixLevelHelperAnnotator
// ---------------------------------------------------------------------

/// One entry of [`MixLevelHelperAnnotator::pending`] — matches the
/// upstream 4-tuple `(ll_function, graph, args_s, s_result)`
/// (annlowlevel.py:131).
#[derive(Debug, Clone)]
pub struct PendingHelper {
    pub ll_function: HostObject,
    pub graph: Rc<PyGraph>,
    pub args_s: Vec<SomeValue>,
    pub s_result: SomeValue,
}

/// One entry of [`MixLevelHelperAnnotator::delayedconsts`] — matches
/// upstream `(delayedptr, repr, obj)` (annlowlevel.py:212).
#[derive(Debug, Clone)]
pub struct DelayedConst {
    pub delayedptr: _ptr,
    pub repr: Arc<dyn Repr>,
    /// Upstream `obj` is a Python value of arbitrary type that the
    /// repr's `convert_const` later lowers.
    pub obj: ConstValue,
}

/// One entry of [`MixLevelHelperAnnotator::delayedfuncs`] — matches
/// upstream `(delayedptr, graph)` (annlowlevel.py:176).
#[derive(Debug, Clone)]
pub struct DelayedFunc {
    pub delayedptr: _ptr,
    pub graph: Rc<PyGraph>,
}

/// RPython `class MixLevelHelperAnnotator(object)` (annlowlevel.py:
/// :126-284).
///
/// Stateful coordinator that buffers helper-graph registrations,
/// annotates them in one pass, and then rtypes them in a second
/// pass. Callers interleave `getgraph` / `delayedfunction` /
/// `constfunc` / `getdelayedrepr` / `delayedconst` calls with
/// regular rtyper work, then invoke [`MixLevelHelperAnnotator::finish`]
/// once when the surrounding phase is done to flush everything.
///
/// Upstream field mapping:
///
/// | Python | Rust |
/// |---|---|
/// | `self.rtyper` | `rtyper: Weak<RPythonTyper>` |
/// | `self.policy` | `policy: MixLevelAnnotatorPolicy` |
/// | `self.pending: list` | `pending: RefCell<Vec<PendingHelper>>` |
/// | `self.delayedreprs: set` | `delayedreprs: RefCell<Vec<Arc<dyn Repr>>>` (pointer-id dedup) |
/// | `self.delayedconsts: list` | `delayedconsts: RefCell<Vec<DelayedConst>>` |
/// | `self.delayedfuncs: list` | `delayedfuncs: RefCell<Vec<DelayedFunc>>` |
/// | `self.newgraphs: set` | `newgraphs: RefCell<Vec<GraphRef>>` (pointer-id dedup) |
///
/// The upstream `set` containers become `Vec` because `dyn Repr` /
/// `GraphRef` have no trivially-hashable identity; Rust dedup mirrors
/// the existing `RPythonTyper::seen_reprs_must_call_setup` pattern.
pub struct MixLevelHelperAnnotator {
    /// RPython `self.rtyper = rtyper` (annlowlevel.py:129).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.policy = MixLevelAnnotatorPolicy(self)`
    /// (annlowlevel.py:130). Wrapped in `RefCell` so `finish_annotate`
    /// can borrow-swap it into `annotator.using_policy` without an
    /// owned clone per call.
    pub policy: RefCell<MixLevelAnnotatorPolicy>,
    /// RPython `self.pending = []` (annlowlevel.py:131). Queue of
    /// helper graphs awaiting `finish_annotate`.
    pub pending: RefCell<Vec<PendingHelper>>,
    /// RPython `self.delayedreprs = set()` (annlowlevel.py:132).
    /// Pointer-identity dedup via the caller; entries are
    /// `Arc<dyn Repr>` whose `set_setup_delayed(false)` is flipped by
    /// `finish_rtype`.
    pub delayedreprs: RefCell<Vec<Arc<dyn Repr>>>,
    /// RPython `self.delayedconsts = []` (annlowlevel.py:133).
    pub delayedconsts: RefCell<Vec<DelayedConst>>,
    /// RPython `self.delayedfuncs = []` (annlowlevel.py:134).
    pub delayedfuncs: RefCell<Vec<DelayedFunc>>,
    /// RPython `self.newgraphs = set()` (annlowlevel.py:135).
    /// Upstream populates this from the `original_graph_count`
    /// checkpoint inside `finish_annotate` / `finish_rtype`.
    pub newgraphs: RefCell<Vec<GraphRef>>,
}

impl MixLevelHelperAnnotator {
    /// RPython `MixLevelHelperAnnotator.__init__(self, rtyper)`
    /// (annlowlevel.py:128-135).
    pub fn new(rtyper: &Rc<RPythonTyper>) -> Self {
        // Build the policy against an incomplete self — upstream
        // passes `self` to `MixLevelAnnotatorPolicy(self)` at the same
        // construction site. The policy only reads
        // `annhelper.rtyper`, so fabricating it from a temporary
        // ref-only struct here mirrors the Python semantics without a
        // partially-initialised moved value.
        let ll = LowLevelAnnotatorPolicy::new(Some(rtyper));
        let policy = MixLevelAnnotatorPolicy { ll };
        MixLevelHelperAnnotator {
            rtyper: Rc::downgrade(rtyper),
            policy: RefCell::new(policy),
            pending: RefCell::new(Vec::new()),
            delayedreprs: RefCell::new(Vec::new()),
            delayedconsts: RefCell::new(Vec::new()),
            delayedfuncs: RefCell::new(Vec::new()),
            newgraphs: RefCell::new(Vec::new()),
        }
    }

    fn add_newgraph(&self, graph: &GraphRef) {
        let mut newgraphs = self.newgraphs.borrow_mut();
        if !newgraphs.iter().any(|existing| Rc::ptr_eq(existing, graph)) {
            newgraphs.push(graph.clone());
        }
    }

    /// RPython `MixLevelHelperAnnotator.getgraph(self, ll_function,
    /// args_s, s_result)` (annlowlevel.py:137-149).
    ///
    /// ```python
    /// def getgraph(self, ll_function, args_s, s_result):
    ///     ann = self.rtyper.annotator
    ///     with ann.using_policy(self.policy):
    ///         graph, args_s = ann.get_call_parameters(ll_function, args_s)
    ///     for v_arg, s_arg in zip(graph.getargs(), args_s):
    ///         ann.setbinding(v_arg, s_arg)
    ///     ann.setbinding(graph.getreturnvar(), s_result)
    ///     self.pending.append((ll_function, graph, args_s, s_result))
    ///     return graph
    /// ```
    ///
    /// Port pending.
    pub fn getgraph(
        &self,
        ll_function: &HostObject,
        args_s: Vec<SomeValue>,
        s_result: SomeValue,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let ann = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
        let policy = self.policy.borrow().clone();
        let (graph, args_s) = {
            let _guard = ann.using_policy(policy);
            ann.get_call_parameters(ll_function, args_s)
                .map_err(|e| TyperError::message(e.to_string()))?
        };
        for (v_arg, s_arg) in graph
            .graph
            .borrow()
            .getargs()
            .into_iter()
            .zip(args_s.iter())
        {
            if let crate::flowspace::model::Hlvalue::Variable(mut v_arg) = v_arg {
                ann.setbinding(&mut v_arg, s_arg.clone());
            }
        }
        if let crate::flowspace::model::Hlvalue::Variable(mut retvar) =
            graph.graph.borrow().getreturnvar()
        {
            ann.setbinding(&mut retvar, s_result.clone());
        }
        self.pending.borrow_mut().push(PendingHelper {
            ll_function: ll_function.clone(),
            graph: graph.clone(),
            args_s,
            s_result,
        });
        Ok(graph)
    }

    /// RPython `MixLevelHelperAnnotator.delayedfunction(self,
    /// ll_function, args_s, s_result, needtype=False)`
    /// (annlowlevel.py:151-162).
    ///
    /// Delegates to [`MixLevelHelperAnnotator::getgraph`] then
    /// [`MixLevelHelperAnnotator::graph2delayed`]; when `needtype` is
    /// true the returned delayed pointer is typed eagerly by
    /// consulting [`MixLevelHelperAnnotator::getdelayedrepr`] for
    /// each arg.
    ///
    /// Port pending.
    pub fn delayedfunction(
        &self,
        ll_function: &HostObject,
        args_s: Vec<SomeValue>,
        s_result: SomeValue,
        needtype: bool,
    ) -> Result<_ptr, TyperError> {
        let graph = self.getgraph(ll_function, args_s.clone(), s_result.clone())?;
        let functype = if needtype {
            let args = args_s
                .iter()
                .map(|s_arg| {
                    self.getdelayedrepr(s_arg, false)
                        .map(|r| r.lowleveltype().clone())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let result = self
                .getdelayedrepr(&s_result, false)?
                .lowleveltype()
                .clone();
            Some(LowLevelType::Func(Box::new(FuncType { args, result })))
        } else {
            None
        };
        self.graph2delayed(&graph, functype)
    }

    /// RPython `MixLevelHelperAnnotator.constfunc(self, ll_function,
    /// args_s, s_result)` (annlowlevel.py:164-166).
    ///
    /// ```python
    /// def constfunc(self, ll_function, args_s, s_result):
    ///     p = self.delayedfunction(ll_function, args_s, s_result)
    ///     return Constant(p, lltype.typeOf(p))
    /// ```
    ///
    /// Port pending. Returns a typed `Constant`-wrapping lltype
    /// pointer; the real return type once ported is most likely
    /// `(Hlvalue::Constant, LowLevelType)` so both the Python
    /// `Constant(p, typeOf(p))` halves survive the Rust lowering.
    pub fn constfunc(
        &self,
        ll_function: &HostObject,
        args_s: Vec<SomeValue>,
        s_result: SomeValue,
    ) -> Result<(_ptr, LowLevelType), TyperError> {
        let p = self.delayedfunction(ll_function, args_s, s_result, false)?;
        Ok((p.clone(), LowLevelType::Ptr(Box::new(lltype::typeOf(&p)))))
    }

    /// RPython `MixLevelHelperAnnotator.graph2delayed(self, graph,
    /// FUNCTYPE=None)` (annlowlevel.py:168-177).
    ///
    /// ```python
    /// def graph2delayed(self, graph, FUNCTYPE=None):
    ///     if FUNCTYPE is None:
    ///         FUNCTYPE = lltype.ForwardReference()
    ///     name = "delayed!%s" % (graph.name,)
    ///     delayedptr = lltype._ptr(lltype.Ptr(FUNCTYPE), name, solid=True)
    ///     self.delayedfuncs.append((delayedptr, graph))
    ///     return delayedptr
    /// ```
    ///
    /// Port pending.
    pub fn graph2delayed(
        &self,
        graph: &Rc<PyGraph>,
        functype: Option<LowLevelType>,
    ) -> Result<_ptr, TyperError> {
        let functype = functype
            .unwrap_or_else(|| LowLevelType::ForwardReference(Box::new(ForwardReference::new())));
        let _name = format!("delayed!{}", graph.graph.borrow().name);
        let delayedptr = _ptr::new_with_solid(
            Ptr::from_container_type(functype).map_err(TyperError::message)?,
            Err(DelayedPointer),
            true,
        );
        self.delayedfuncs.borrow_mut().push(DelayedFunc {
            delayedptr: delayedptr.clone(),
            graph: graph.clone(),
        });
        Ok(delayedptr)
    }

    /// RPython `MixLevelHelperAnnotator.graph2const(self, graph)`
    /// (annlowlevel.py:179-181).
    ///
    /// ```python
    /// def graph2const(self, graph):
    ///     p = self.graph2delayed(graph)
    ///     return Constant(p, lltype.typeOf(p))
    /// ```
    ///
    /// Port pending.
    pub fn graph2const(&self, graph: &Rc<PyGraph>) -> Result<(_ptr, LowLevelType), TyperError> {
        let p = self.graph2delayed(graph, None)?;
        Ok((p.clone(), LowLevelType::Ptr(Box::new(lltype::typeOf(&p)))))
    }

    /// RPython `MixLevelHelperAnnotator.getdelayedrepr(self, s_value,
    /// check_never_seen=True)` (annlowlevel.py:183-195).
    ///
    /// ```python
    /// def getdelayedrepr(self, s_value, check_never_seen=True):
    ///     r = self.rtyper.getrepr(s_value)
    ///     if check_never_seen:
    ///         r.set_setup_delayed(True)
    ///         delayed = True
    ///     else:
    ///         delayed = r.set_setup_maybe_delayed()
    ///     if delayed:
    ///         self.delayedreprs.add(r)
    ///     return r
    /// ```
    ///
    /// Port pending.
    pub fn getdelayedrepr(
        &self,
        s_value: &SomeValue,
        check_never_seen: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let r = rtyper.getrepr(s_value)?;
        let delayed = if check_never_seen {
            r.set_setup_delayed(true);
            true
        } else {
            r.set_setup_maybe_delayed()
        };
        if delayed {
            let mut delayedreprs = self.delayedreprs.borrow_mut();
            if !delayedreprs
                .iter()
                .any(|existing| Arc::ptr_eq(existing, &r))
            {
                delayedreprs.push(r.clone());
            }
        }
        Ok(r)
    }

    /// RPython `MixLevelHelperAnnotator.s_r_instanceof(self, cls,
    /// can_be_None=True, check_never_seen=True)` (annlowlevel.py:
    /// :197-202).
    ///
    /// ```python
    /// def s_r_instanceof(self, cls, can_be_None=True, check_never_seen=True):
    ///     classdesc = self.rtyper.annotator.bookkeeper.getdesc(cls)
    ///     classdef = classdesc.getuniqueclassdef()
    ///     s_instance = annmodel.SomeInstance(classdef, can_be_None)
    ///     r_instance = self.getdelayedrepr(s_instance, check_never_seen)
    ///     return s_instance, r_instance
    /// ```
    ///
    /// Port pending.
    pub fn s_r_instanceof(
        &self,
        cls: &HostObject,
        can_be_none: bool,
        check_never_seen: bool,
    ) -> Result<(SomeValue, Arc<dyn Repr>), TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let ann = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
        let classdesc = ann
            .bookkeeper
            .getdesc(cls)
            .map_err(|e| TyperError::message(e.to_string()))?
            .as_class()
            .ok_or_else(|| {
                TyperError::message(format!(
                    "MixLevelHelperAnnotator.s_r_instanceof: {:?} is not a class descriptor",
                    cls.qualname()
                ))
            })?;
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(&classdesc)
            .map_err(|e| TyperError::message(e.to_string()))?;
        let s_instance = SomeValue::Instance(crate::annotator::model::SomeInstance::new(
            Some(classdef),
            can_be_none,
            Default::default(),
        ));
        let r_instance = self.getdelayedrepr(&s_instance, check_never_seen)?;
        Ok((s_instance, r_instance))
    }

    /// RPython `MixLevelHelperAnnotator.delayedconst(self, repr, obj)`
    /// (annlowlevel.py:204-215).
    ///
    /// ```python
    /// def delayedconst(self, repr, obj):
    ///     if repr.is_setup_delayed():
    ///         bk = self.rtyper.annotator.bookkeeper
    ///         bk.immutablevalue(obj)
    ///         delayedptr = lltype._ptr(repr.lowleveltype, "delayed!")
    ///         self.delayedconsts.append((delayedptr, repr, obj))
    ///         return delayedptr
    ///     else:
    ///         return repr.convert_const(obj)
    /// ```
    ///
    /// Port pending.
    pub fn delayedconst(
        &self,
        repr: &Arc<dyn Repr>,
        obj: &ConstValue,
    ) -> Result<Constant, TyperError> {
        if repr.is_setup_delayed() {
            let rtyper = self
                .rtyper
                .upgrade()
                .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
            let ann = rtyper
                .annotator
                .upgrade()
                .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
            ann.bookkeeper
                .immutablevalue(obj)
                .map_err(|e| TyperError::message(e.to_string()))?;
            let LowLevelType::Ptr(ptr_type) = repr.lowleveltype() else {
                return Err(TyperError::message(format!(
                    "MixLevelHelperAnnotator.delayedconst: expected Ptr lowleveltype, got {:?}",
                    repr.lowleveltype()
                )));
            };
            let delayedptr = _ptr::new((**ptr_type).clone(), Err(DelayedPointer));
            self.delayedconsts.borrow_mut().push(DelayedConst {
                delayedptr: delayedptr.clone(),
                repr: repr.clone(),
                obj: obj.clone(),
            });
            Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(delayedptr)),
                repr.lowleveltype().clone(),
            ))
        } else {
            repr.convert_const(obj)
        }
    }

    /// RPython `MixLevelHelperAnnotator.finish(self)`
    /// (annlowlevel.py:217-219).
    ///
    /// ```python
    /// def finish(self):
    ///     self.finish_annotate()
    ///     self.finish_rtype()
    /// ```
    pub fn finish(&self) -> Result<(), TyperError> {
        self.finish_annotate()?;
        self.finish_rtype()?;
        Ok(())
    }

    /// RPython `MixLevelHelperAnnotator.finish_annotate(self)`
    /// (annlowlevel.py:221-248).
    ///
    /// Port pending. Upstream body:
    ///
    /// ```python
    /// def finish_annotate(self):
    ///     rtyper = self.rtyper
    ///     ann = rtyper.annotator
    ///     bk = ann.bookkeeper
    ///     translator = ann.translator
    ///     original_graph_count = len(translator.graphs)
    ///     with ann.using_policy(self.policy):
    ///         for ll_function, graph, args_s, s_result in self.pending:
    ///             ann.annotated[graph.returnblock] = graph
    ///             s_function = bk.immutablevalue(ll_function)
    ///             bk.emulate_pbc_call(graph, s_function, args_s)
    ///             self.newgraphs.add(graph)
    ///         ann.complete_helpers()
    ///     for ll_function, graph, args_s, s_result in self.pending:
    ///         s_real_result = ann.binding(graph.getreturnvar())
    ///         if s_real_result != s_result:
    ///             raise Exception("wrong annotation for the result of %r:\n"
    ///                             "originally specified: %r\n"
    ///                             " found by annotating: %r" %
    ///                             (graph, s_result, s_real_result))
    ///     del self.pending[:]
    ///     for graph in translator.graphs[original_graph_count:]:
    ///         self.newgraphs.add(graph)
    /// ```
    pub fn finish_annotate(&self) -> Result<(), TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let ann = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
        let bk = ann.bookkeeper.clone();
        let translator = ann.translator.clone();
        let original_graph_count = translator.graphs.borrow().len();
        let pending = self.pending.borrow().clone();
        {
            let _guard = ann.using_policy(self.policy.borrow().clone());
            for PendingHelper {
                ll_function,
                graph,
                args_s,
                s_result: _,
            } in &pending
            {
                let returnblock = graph.graph.borrow().returnblock.clone();
                let returnblock_key = BlockKey::of(&returnblock);
                ann.annotated
                    .borrow_mut()
                    .insert(returnblock_key.clone(), Some(graph.graph.clone()));
                ann.all_blocks
                    .borrow_mut()
                    .insert(returnblock_key, returnblock);
                let s_function = bk
                    .immutablevalue(&ConstValue::HostObject(ll_function.clone()))
                    .map_err(|e| TyperError::message(e.to_string()))?;
                bk.emulate_pbc_call(
                    crate::annotator::bookkeeper::EmulatedPbcCallKey::Graph(GraphKey::of(
                        &graph.graph,
                    )),
                    &s_function,
                    args_s,
                    &[],
                    None,
                )
                .map_err(|e| TyperError::message(e.to_string()))?;
                self.add_newgraph(&graph.graph);
            }
            ann.complete_helpers()
                .map_err(|e| TyperError::message(e.to_string()))?;
        }
        for PendingHelper {
            ll_function: _,
            graph,
            args_s: _,
            s_result,
        } in &pending
        {
            let s_real_result = ann.binding(&graph.graph.borrow().getreturnvar());
            if &s_real_result != s_result {
                return Err(TyperError::message(format!(
                    "wrong annotation for the result of {:?}:\noriginally specified: {:?}\n found by annotating: {:?}",
                    graph.graph.borrow().name,
                    s_result,
                    s_real_result,
                )));
            }
        }
        self.pending.borrow_mut().clear();
        let fresh_graphs: Vec<GraphRef> = translator
            .graphs
            .borrow()
            .iter()
            .skip(original_graph_count)
            .cloned()
            .collect();
        for graph in fresh_graphs {
            self.add_newgraph(&graph);
        }
        Ok(())
    }

    /// RPython `MixLevelHelperAnnotator.finish_rtype(self)`
    /// (annlowlevel.py:250-275).
    ///
    /// Port pending. Upstream body:
    ///
    /// ```python
    /// def finish_rtype(self):
    ///     rtyper = self.rtyper
    ///     translator = rtyper.annotator.translator
    ///     original_graph_count = len(translator.graphs)
    ///     perform_normalizations(rtyper.annotator)
    ///     for r in self.delayedreprs:
    ///         r.set_setup_delayed(False)
    ///     rtyper.call_all_setups()
    ///     for p, repr, obj in self.delayedconsts:
    ///         p._become(repr.convert_const(obj))
    ///     rtyper.call_all_setups()
    ///     for p, graph in self.delayedfuncs:
    ///         self.newgraphs.add(graph)
    ///         real_p = rtyper.getcallable(graph)
    ///         REAL = lltype.typeOf(real_p).TO
    ///         FUNCTYPE = lltype.typeOf(p).TO
    ///         if isinstance(FUNCTYPE, lltype.ForwardReference):
    ///             FUNCTYPE.become(REAL)
    ///         assert FUNCTYPE == REAL
    ///         p._become(real_p)
    ///     rtyper.specialize_more_blocks()
    ///     self.delayedreprs.clear()
    ///     del self.delayedconsts[:]
    ///     del self.delayedfuncs[:]
    ///     for graph in translator.graphs[original_graph_count:]:
    ///         self.newgraphs.add(graph)
    /// ```
    pub fn finish_rtype(&self) -> Result<(), TyperError> {
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: rtyper dropped"))?;
        let ann = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MixLevelHelperAnnotator: annotator dropped"))?;
        let translator = ann.translator.clone();
        let original_graph_count = translator.graphs.borrow().len();
        super::normalizecalls::perform_normalizations(ann.as_ref())
            .map_err(|e| TyperError::message(e.to_string()))?;
        let delayedreprs = self.delayedreprs.borrow().clone();
        for r in &delayedreprs {
            r.set_setup_delayed(false);
        }
        rtyper.call_all_setups()?;
        let delayedconsts = self.delayedconsts.borrow().clone();
        for delayed in &delayedconsts {
            let real = delayed.repr.convert_const(&delayed.obj)?;
            let ConstValue::LLPtr(real_p) = &real.value else {
                return Err(TyperError::message(format!(
                    "MixLevelHelperAnnotator.finish_rtype: delayed const for {} did not lower to LLPtr",
                    delayed.repr.repr_string()
                )));
            };
            delayed.delayedptr._become(real_p);
        }
        rtyper.call_all_setups()?;
        let delayedfuncs = self.delayedfuncs.borrow().clone();
        for delayed in &delayedfuncs {
            self.add_newgraph(&delayed.graph.graph);
            let real_p = rtyper.getcallable(&delayed.graph)?;
            let real = lltype::typeOf(&real_p).TO.clone();
            let functype = lltype::typeOf(&delayed.delayedptr).TO.clone();
            if let lltype::PtrTarget::ForwardReference(forward_ref) = &functype {
                forward_ref
                    .r#become(real.clone().into())
                    .map_err(TyperError::message)?;
            }
            assert_eq!(functype, real);
            delayed.delayedptr._become(&real_p);
        }
        rtyper.specialize_more_blocks()?;
        self.delayedreprs.borrow_mut().clear();
        self.delayedconsts.borrow_mut().clear();
        self.delayedfuncs.borrow_mut().clear();
        let fresh_graphs: Vec<GraphRef> = translator
            .graphs
            .borrow()
            .iter()
            .skip(original_graph_count)
            .cloned()
            .collect();
        for graph in fresh_graphs {
            self.add_newgraph(&graph);
        }
        Ok(())
    }

    /// RPython `MixLevelHelperAnnotator.backend_optimize(self,
    /// **flags)` (annlowlevel.py:277-284).
    ///
    /// ```python
    /// def backend_optimize(self, **flags):
    ///     from rpython.translator.backendopt.all import backend_optimizations
    ///     translator = self.rtyper.annotator.translator
    ///     newgraphs = list(self.newgraphs)
    ///     backend_optimizations(translator, newgraphs, secondary=True,
    ///                           inline_graph_from_anywhere=True, **flags)
    ///     self.newgraphs.clear()
    /// ```
    ///
    /// Port blocker: the body forwards into
    /// `rpython/translator/backendopt/all.py:backend_optimizations`,
    /// which itself chains the whole backend-opt pipeline (inlining,
    /// mallocs-removal, constant folding, gc transform, stack check
    /// insertion, …). None of that subsystem has been ported yet, and
    /// the existing pyre graph pipeline (`translator/` passes like
    /// `simplify`, `transform`, `unsimplify`) only covers a subset.
    /// Until the backendopt.all port lands — which is a
    /// cross-subsystem effort on its own — this surfaces as a typed
    /// error so call sites (e.g. `rtyper_finish_helpers`) fail loudly
    /// instead of silently running un-optimised helper graphs.
    pub fn backend_optimize(&self) -> Result<(), TyperError> {
        Err(TyperError::message(
            "annlowlevel.py:277 MixLevelHelperAnnotator.backend_optimize port pending \
             (blocked on rpython/translator/backendopt/all.py port)",
        ))
    }
}

// ---------------------------------------------------------------------
// Unit tests — lock in that every stub surface reports its upstream
// line number so follow-up body fills can trip on the exact error
// string.
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::SomeInteger;
    use crate::flowspace::model::{ConstValue, Constant, GraphFunc};
    use rustpython_compiler::{Mode, compile as rp_compile};
    use rustpython_compiler_core::bytecode::ConstantData;

    fn make_rtyper() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let annotator = RPythonAnnotator::new(None, None, None, false);
        let rtyper = annotator.translator.buildrtyper();
        rtyper.initialize_exceptiondata().unwrap();
        (annotator, rtyper)
    }

    fn compiled_host_function(src: &str) -> HostObject {
        let code = rp_compile(src, Mode::Exec, "<test>".into(), Default::default())
            .expect("compile should succeed");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body should be a code constant");
        let func = GraphFunc::from_host_code(
            crate::flowspace::bytecode::HostCode::from_code(inner),
            Constant::new(ConstValue::Dict(Default::default())),
            vec![],
        );
        HostObject::new_user_function(func)
    }

    #[test]
    fn mix_level_helper_annotator_pending_queues_start_empty() {
        let (_annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        assert!(h.pending.borrow().is_empty());
        assert!(h.delayedreprs.borrow().is_empty());
        assert!(h.delayedconsts.borrow().is_empty());
        assert!(h.delayedfuncs.borrow().is_empty());
        assert!(h.newgraphs.borrow().is_empty());
    }

    #[test]
    fn mix_level_helper_annotator_finish_annotate_marks_returnblock_and_clears_pending() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = h
            .getgraph(
                &ll_function,
                vec![SomeValue::Integer(SomeInteger::default())],
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect("graph");

        h.finish_annotate().expect("finish_annotate");
        assert!(h.pending.borrow().is_empty());
        let returnblock_key = BlockKey::of(&graph.graph.borrow().returnblock);
        let annotated = annotator.annotated.borrow();
        let Some(Some(done_graph)) = annotated.get(&returnblock_key) else {
            panic!("return block should be marked annotated");
        };
        assert!(std::rc::Rc::ptr_eq(done_graph, &graph.graph));
        assert!(
            h.newgraphs
                .borrow()
                .iter()
                .any(|newgraph| std::rc::Rc::ptr_eq(newgraph, &graph.graph))
        );
    }

    #[test]
    fn mix_level_helper_annotator_finish_rtype_resolves_delayed_function_and_clears_queues() {
        let (_annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = h
            .getgraph(
                &ll_function,
                vec![SomeValue::Integer(SomeInteger::default())],
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect("graph");
        let delayed = h.graph2delayed(&graph, None).expect("delayed pointer");

        h.finish_annotate().expect("finish_annotate");
        h.finish_rtype().expect("finish_rtype");

        assert!(h.delayedreprs.borrow().is_empty());
        assert!(h.delayedconsts.borrow().is_empty());
        assert!(h.delayedfuncs.borrow().is_empty());
        let lltype::PtrTarget::ForwardReference(forward_ref) = &lltype::typeOf(&delayed).TO else {
            panic!("expected delayed function type to start as ForwardReference");
        };
        assert!(forward_ref.resolved().is_some());
        let lltype::_ptr_obj::Func(funcobj) = delayed._obj().expect("resolved delayed ptr") else {
            panic!("resolved delayed ptr should expose a func object");
        };
        assert_eq!(
            funcobj.graph,
            Some(crate::flowspace::model::GraphKey::of(&graph.graph).as_usize())
        );
    }

    #[test]
    fn mix_level_helper_annotator_getgraph_populates_pending_queue() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = h
            .getgraph(
                &ll_function,
                vec![SomeValue::Integer(SomeInteger::default())],
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect("getgraph should return a helper graph");
        assert!(graph.graph.borrow().name.starts_with("ll_identity"));
        assert_eq!(h.pending.borrow().len(), 1);
        let returnvar = graph.graph.borrow().getreturnvar();
        assert!(matches!(
            annotator.annotation(&returnvar),
            Some(SomeValue::Integer(_))
        ));
    }

    #[test]
    fn mix_level_helper_annotator_graph2delayed_enqueues_delayed_func() {
        let (_annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = h
            .getgraph(
                &ll_function,
                vec![SomeValue::Integer(SomeInteger::default())],
                SomeValue::Integer(SomeInteger::default()),
            )
            .expect("graph");
        let delayed = h.graph2delayed(&graph, None).expect("delayed ptr");
        assert_eq!(h.delayedfuncs.borrow().len(), 1);
        assert!(matches!(
            lltype::typeOf(&delayed).TO,
            lltype::PtrTarget::ForwardReference(_)
        ));
    }

    #[test]
    fn mix_level_helper_annotator_getdelayedrepr_marks_repr_delayed_and_dedups() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let cls = HostObject::new_class("pkg.Box", vec![]);
        let classdef = annotator.bookkeeper.getuniqueclassdef(&cls).unwrap();
        let s_value = SomeValue::Instance(crate::annotator::model::SomeInstance::new(
            Some(classdef),
            false,
            Default::default(),
        ));

        let r1 = h
            .getdelayedrepr(&s_value, true)
            .expect("first delayed repr");
        assert!(r1.is_setup_delayed());
        assert_eq!(h.delayedreprs.borrow().len(), 1);

        let r2 = h
            .getdelayedrepr(&s_value, false)
            .expect("second delayed repr");
        assert!(std::sync::Arc::ptr_eq(&r1, &r2));
        assert_eq!(h.delayedreprs.borrow().len(), 1);
    }

    #[test]
    fn mix_level_helper_annotator_s_r_instanceof_builds_someinstance_and_repr() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let cls = HostObject::new_class("pkg.Target", vec![]);
        let expected = annotator.bookkeeper.getuniqueclassdef(&cls).unwrap();

        let (s_instance, r_instance) = h
            .s_r_instanceof(&cls, true, true)
            .expect("s_r_instanceof should succeed");
        let SomeValue::Instance(s_instance) = s_instance else {
            panic!("expected SomeInstance");
        };
        assert!(s_instance.can_be_none);
        let actual = s_instance.classdef.expect("classdef");
        assert!(std::rc::Rc::ptr_eq(&actual, &expected));
        assert!(r_instance.is_setup_delayed());
        assert_eq!(h.delayedreprs.borrow().len(), 1);
    }

    #[test]
    fn mix_level_helper_annotator_delayedconst_enqueues_pointer_for_delayed_repr() {
        let (annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let cls = HostObject::new_class("pkg.Delayed", vec![]);
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let classdef = annotator.bookkeeper.getuniqueclassdef(&cls).unwrap();
        let s_value = SomeValue::Instance(crate::annotator::model::SomeInstance::new(
            Some(classdef),
            false,
            Default::default(),
        ));
        let repr = h.getdelayedrepr(&s_value, true).expect("delayed repr");

        let constant = h
            .delayedconst(&repr, &ConstValue::HostObject(inst.clone()))
            .expect("delayed const");
        assert_eq!(h.delayedconsts.borrow().len(), 1);
        let delayedconsts = h.delayedconsts.borrow();
        assert!(matches!(
            &delayedconsts[0].obj,
            ConstValue::HostObject(host) if host == &inst
        ));
        assert_eq!(constant.concretetype, Some(repr.lowleveltype().clone()));
        let ConstValue::LLPtr(ptr) = &constant.value else {
            panic!("expected LLPtr constant");
        };
        assert_eq!(ptr._TYPE, delayedconsts[0].delayedptr._TYPE);
    }

    #[test]
    fn mix_level_helper_annotator_delayedconst_converts_immediately_for_ready_repr() {
        let (_annotator, rtyper) = make_rtyper();
        let h = MixLevelHelperAnnotator::new(&rtyper);
        let s_value = SomeValue::Integer(SomeInteger::default());
        let repr = rtyper.getrepr(&s_value).expect("int repr");
        let expected = repr
            .convert_const(&ConstValue::Int(7))
            .expect("expected const");

        let actual = h
            .delayedconst(&repr, &ConstValue::Int(7))
            .expect("immediate const");
        assert_eq!(actual, expected);
        assert!(h.delayedconsts.borrow().is_empty());
    }

    #[test]
    fn annotate_lowlevel_helper_free_function_builds_helper_graph() {
        let (annotator, _rtyper) = make_rtyper();
        let ll_function = compiled_host_function("def ll_identity(x):\n    return x\n");
        let graph = annotate_lowlevel_helper(
            &annotator,
            &ll_function,
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
        )
        .expect("annotate_lowlevel_helper should annotate the helper");
        assert!(graph.graph.borrow().name.starts_with("ll_identity"));
        let returnvar = graph.graph.borrow().getreturnvar();
        assert!(matches!(
            annotator.annotation(&returnvar),
            Some(SomeValue::Integer(_))
        ));
        assert!(annotator.added_blocks.borrow().is_none());
    }
}
