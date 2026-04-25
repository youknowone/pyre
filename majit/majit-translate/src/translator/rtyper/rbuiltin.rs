//! RPython `rpython/rtyper/rbuiltin.py` — reprs for `SomeBuiltin`
//! / `SomeBuiltinMethod` annotations.
//!
//! Upstream dispatches every `SomeBuiltin(const)` to
//! [`BuiltinFunctionRepr`] (rbuiltin.py:23-33) and every
//! `SomeBuiltinMethod` to `BuiltinMethodRepr` (rbuiltin.py:35-50).
//! Both repr types carry no runtime value (the function / bound
//! receiver is known statically) — the calling hop dispatches into a
//! module-level `BUILTIN_TYPER` registry.
//!
//! ## Status of this port
//!
//! Only the narrow surface needed to lift `SomeBuiltin.rtyper_makerepr`
//! out of the `MissingRTypeOperation` fallback in
//! [`crate::translator::rtyper::rmodel::rtyper_makerepr`] is ported
//! today:
//!
//! * [`BuiltinFunctionRepr`] — stateless Void-typed repr that carries
//!   the builtin identifier (pyre: [`HostObject`]).
//! * [`somebuiltin_rtyper_makerepr`] — upstream `SomeBuiltin.
//!   rtyper_makerepr(self, rtyper)` dispatcher, factored into a free
//!   function so [`rmodel`] can route into it without an import cycle.
//!
//! ## Deferred
//!
//! * Concrete `rtype_method_<name>` overrides on each `Repr` subclass
//!   (e.g. `ListRepr.rtype_method_append`, `StringRepr.rtype_method_join`)
//!   — route through `Repr::rtype_method` once the `r<type>.py` ports
//!   land.
//! * `extregistry.specialize_call` (rbuiltin.py:78-81) — the
//!   [`crate::translator::rtyper::extregistry::ExtRegistryEntry`] type
//!   has `is_registered` / `lookup` wired for the `_ptr` entry only
//!   (via `_ptrEntry` at lltype.py:1513-1518, which does not override
//!   `specialize_call`, matching upstream `AttributeError` when this
//!   path is hit). The dispatch to `entry.specialize_call` for entry
//!   types whose upstream subclasses do override it (e.g. `llhelper`,
//!   `llmemory.*` family, `objectmodel.*` hints) is not ported yet;
//!   those entry types must first be added as new `ExtRegistryEntry`
//!   enum variants before `findbltintyper` can route them.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

use crate::annotator::model::{SomeBuiltin, SomeBuiltinMethod, SomeValue};
use crate::flowspace::model::{ConstValue, Constant, HOST_ENV, HostObject};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::extregistry;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{HighLevelOp, RPythonTyper};

/// RPython `BUILTIN_TYPER = {}` (rbuiltin.py:14).
///
/// Module-level registry mapping builtin callables to their
/// `rtype_builtin_*` specializer. Upstream populates this dict via
/// the `@typer_for(func)` decorator at module import time
/// (rbuiltin.py:16-20); the Rust port populates it via
/// [`typer_for`] calls from module initializers (to be wired when
/// the first concrete `@typer_for` port lands).
///
/// Keyed by [`HostObject`] because upstream uses the Python builtin
/// function object itself as the dict key (Python dict uses `id()`
/// for unhashable callables — Rust mirrors via
/// [`Arc::ptr_eq`]-based identity on [`HostObject`]).
static BUILTIN_TYPER: OnceLock<Mutex<HashMap<HostObject, BuiltinTyperFn>>> = OnceLock::new();

fn builtin_typer_map() -> &'static Mutex<HashMap<HostObject, BuiltinTyperFn>> {
    BUILTIN_TYPER.get_or_init(|| {
        let mut map = HashMap::new();
        install_default_typers(&mut map);
        Mutex::new(map)
    })
}

/// Register the `@typer_for(...)` entries from upstream rbuiltin.py
/// that the Rust port has ported so far. Called once via the
/// [`BUILTIN_TYPER`] `OnceLock` initializer.
///
/// Each `(builtin_name, typer_fn)` pair mirrors one
/// `@typer_for(<python builtin>)` decorator in upstream. The Python
/// builtin is resolved via [`HOST_ENV::lookup_builtin`]; missing
/// entries are silently skipped so bootstrap stays robust when the
/// host environment is partially populated.
///
/// Upstream rbuiltin.py continues with many more `@typer_for` entries
/// past `list` (the last one registered here). The outstanding
/// backlog — all needing additional [`HOST_ENV`] plumbing (module-
/// qualified names + per-callable `HostObject`s) plus per-typer body
/// ports:
///
///   * rbuiltin.py:221-231 — `rarithmetic.intmask` /
///     `rarithmetic.longlongmask`
///   * rbuiltin.py:234-255 — `min` / `max` (need `ll_min` / `ll_max`
///     helper-graph registration via `gendirectcall`)
///   * rbuiltin.py:258-261 — `reversed`
///   * rbuiltin.py:264-305 — `object.__init__` /
///     `EnvironmentError.__init__` / `WindowsError.__init__`
///   * rbuiltin.py:307-340 — `objectmodel.hlinvoke`
///   * rbuiltin.py:342-344 — `range` / `xrange` / `enumerate`
///     (delegates to `rrange.py`)
///   * rbuiltin.py:349-460 — `lltype.malloc` / `free` / `typeOf` /
///     `nullptr` / `cast_*` family
///   * rbuiltin.py:462-600 — `llmemory.*` + `objectmodel.instantiate`
///   * rbuiltin.py:700-782 — weakref family
///
/// Each batch is a natural stand-alone commit once its dependent
/// infra (HOST_ENV entry, helper-graph registration hook, or inputarg
/// coercion primitives) lands.
fn install_default_typers(map: &mut HashMap<HostObject, BuiltinTyperFn>) {
    let entries: &[(&str, BuiltinTyperFn)] = &[
        // rbuiltin.py:172-176
        ("bool", rtype_builtin_bool),
        // rbuiltin.py:178-184
        ("int", rtype_builtin_int),
        // rbuiltin.py:186-189
        ("float", rtype_builtin_float),
        // rbuiltin.py:191-194
        ("chr", rtype_builtin_chr),
        // rbuiltin.py:196-199
        ("unichr", rtype_builtin_unichr),
        // rbuiltin.py:201-203
        ("unicode", rtype_builtin_unicode),
        // rbuiltin.py:205-207
        ("bytearray", rtype_builtin_bytearray),
        // rbuiltin.py:209-211
        ("list", rtype_builtin_list),
    ];
    for (name, typer) in entries {
        if let Some(host) = HOST_ENV.lookup_builtin(name) {
            map.insert(host, *typer);
        }
    }
}

/// Signature of every `rtype_builtin_*` dispatcher in the
/// `BUILTIN_TYPER` registry.
///
/// Upstream typers have the signature `def rtype_builtin_xxx(hop,
/// **kwds_i)` (rbuiltin.py:173-onwards). The Rust port surfaces
/// `kwds_i` as an explicit `&HashMap<String, usize>` that maps
/// upstream `'i_<name>'` keys to their index in `hop.args_v`. Simple
/// calls pass an empty map; keyword-aware typers (e.g.
/// `rtype_malloc`) read specific keys.
pub type BuiltinTyperFn = fn(&HighLevelOp, &HashMap<String, usize>) -> RTypeResult;

/// RPython `typer_for(func)` decorator (rbuiltin.py:16-20).
///
/// ```python
/// def typer_for(func):
///     def wrapped(rtyper_func):
///         BUILTIN_TYPER[func] = rtyper_func
///         return rtyper_func
///     return wrapped
/// ```
///
/// The Rust port folds the two-stage decorator into a single
/// registration call — downstream modules invoke
/// `typer_for(host_obj, rtype_builtin_fn)` at startup.
pub fn typer_for(func: HostObject, rtyper_func: BuiltinTyperFn) {
    builtin_typer_map()
        .lock()
        .expect("BUILTIN_TYPER poisoned")
        .insert(func, rtyper_func);
}

/// Non-upstream helper: read-only registry lookup, used by
/// [`BuiltinFunctionRepr::findbltintyper`]. Kept separate so tests
/// can probe the registry without re-entering `findbltintyper`'s
/// fallback branches.
fn lookup_typer(func: &HostObject) -> Option<BuiltinTyperFn> {
    builtin_typer_map()
        .lock()
        .expect("BUILTIN_TYPER poisoned")
        .get(func)
        .copied()
}

/// RPython `class BuiltinFunctionRepr(Repr)` (rbuiltin.py:67-110).
///
/// Void-typed repr for a statically known Python builtin callable.
/// `lowleveltype = Void` because the builtin identifier is resolved
/// at typing-time via the `BUILTIN_TYPER` registry (not ported yet).
#[derive(Debug)]
pub struct BuiltinFunctionRepr {
    /// RPython `self.builtinfunc = builtinfunc` (rbuiltin.py:70-71).
    /// Carried as the pyre [`HostObject`] wrapping the Python builtin;
    /// future `findbltintyper` ports will read this to pick the
    /// right `rtype_builtin_*` dispatcher.
    pub builtinfunc: HostObject,
    state: ReprState,
    lltype: LowLevelType,
}

impl BuiltinFunctionRepr {
    /// RPython `BuiltinFunctionRepr.__init__(self, builtinfunc)`
    /// (rbuiltin.py:70-71).
    pub fn new(builtinfunc: HostObject) -> Self {
        BuiltinFunctionRepr {
            builtinfunc,
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        }
    }

    /// RPython `BuiltinFunctionRepr.findbltintyper(self, rtyper)`
    /// (rbuiltin.py:73-83).
    ///
    /// ```python
    /// def findbltintyper(self, rtyper):
    ///     "Find the function to use to specialize calls to this built-in func."
    ///     try:
    ///         return BUILTIN_TYPER[self.builtinfunc]
    ///     except (KeyError, TypeError):
    ///         pass
    ///     if extregistry.is_registered(self.builtinfunc):
    ///         entry = extregistry.lookup(self.builtinfunc)
    ///         return entry.specialize_call
    ///     raise TyperError("don't know about built-in function %r" % (
    ///         self.builtinfunc,))
    /// ```
    ///
    /// Upstream accepts `rtyper` as a parameter but never reads it —
    /// the Rust port drops the arg. The `entry.specialize_call`
    /// attribute lookup is delegated to
    /// [`ExtRegistryEntry::specialize_call`], which mirrors upstream's
    /// per-subclass override pattern: subclasses that define
    /// `specialize_call` add an arm returning `Ok(typer_fn)`; subclasses
    /// that do not (e.g. `_ptrEntry` in `lltype.py:1513-1518`, the only
    /// registered variant today) yield the upstream `AttributeError`
    /// surface as a `TyperError`. `ExtRegistryEntry` (extregistry.py:33-72)
    /// defines no base `specialize_call`, so this is a per-arm decision.
    pub fn findbltintyper(&self) -> Result<BuiltinTyperFn, TyperError> {
        if let Some(f) = lookup_typer(&self.builtinfunc) {
            return Ok(f);
        }
        let as_const = ConstValue::HostObject(self.builtinfunc.clone());
        if extregistry::is_registered(&as_const) {
            let entry = extregistry::lookup(&as_const)
                .expect("extregistry::is_registered returned true but lookup returned None");
            return entry.specialize_call();
        }
        Err(TyperError::message(format!(
            "don't know about built-in function {:?}",
            self.builtinfunc
        )))
    }

    /// RPython `BuiltinFunctionRepr._call(self, hop2, **kwds_i)`
    /// (rbuiltin.py:85-92).
    ///
    /// ```python
    /// def _call(self, hop2, **kwds_i):
    ///     bltintyper = self.findbltintyper(hop2.rtyper)
    ///     hop2.llops._called_exception_is_here_or_cannot_occur = False
    ///     v_result = bltintyper(hop2, **kwds_i)
    ///     if not hop2.llops._called_exception_is_here_or_cannot_occur:
    ///         raise TyperError("missing hop.exception_cannot_occur() or "
    ///                          "hop.exception_is_here() in %s" % bltintyper)
    ///     return v_result
    /// ```
    ///
    fn _call(&self, hop2: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
        let bltintyper = self.findbltintyper()?;
        hop2.llops
            .borrow_mut()
            ._called_exception_is_here_or_cannot_occur = false;
        let v_result = bltintyper(hop2, kwds_i)?;
        let checked = hop2
            .llops
            .borrow()
            ._called_exception_is_here_or_cannot_occur;
        if !checked {
            return Err(TyperError::message(
                "missing hop.exception_cannot_occur() or hop.exception_is_here() in builtin typer",
            ));
        }
        Ok(v_result)
    }
}

impl Repr for BuiltinFunctionRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "BuiltinFunctionRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::BuiltinFunctionRepr
    }

    /// RPython `BuiltinFunctionRepr.rtype_simple_call(self, hop)`
    /// (rbuiltin.py:94-97).
    ///
    /// ```python
    /// def rtype_simple_call(self, hop):
    ///     hop2 = hop.copy()
    ///     hop2.r_s_popfirstarg()
    ///     return self._call(hop2)
    /// ```
    fn rtype_simple_call(&self, hop: &HighLevelOp) -> RTypeResult {
        let hop2 = hop.copy();
        hop2.r_s_popfirstarg();
        self._call(&hop2, &HashMap::new())
    }

    /// RPython `BuiltinFunctionRepr.rtype_call_args(self, hop)`
    /// (rbuiltin.py:99-110).
    ///
    /// ```python
    /// def rtype_call_args(self, hop):
    ///     # calling a built-in function with keyword arguments:
    ///     # mostly for rpython.objectmodel.hint()
    ///     hop, kwds_i = call_args_expand(hop)
    ///
    ///     hop2 = hop.copy()
    ///     hop2.r_s_popfirstarg()
    ///     hop2.r_s_popfirstarg()
    ///     return self._call(hop2, **kwds_i)
    /// ```
    fn rtype_call_args(&self, hop: &HighLevelOp) -> RTypeResult {
        let (hop, kwds_i) = call_args_expand(hop)?;
        let hop2 = hop.copy();
        hop2.r_s_popfirstarg();
        hop2.r_s_popfirstarg();
        self._call(&hop2, &kwds_i)
    }
}

/// RPython `call_args_expand(hop)` (rbuiltin.py:52-64).
///
/// ```python
/// def call_args_expand(hop):
///     hop = hop.copy()
///     from rpython.annotator.argument import ArgumentsForTranslation
///     arguments = ArgumentsForTranslation.fromshape(
///             hop.args_s[1].const, # shape
///             range(hop.nb_args-2))
///     assert arguments.w_stararg is None
///     keywords = arguments.keywords
///     # prefix keyword arguments with 'i_'
///     kwds_i = {}
///     for key in keywords:
///         kwds_i['i_' + key] = keywords[key]
///     return hop, kwds_i
/// ```
///
/// The Rust port short-circuits the `ArgumentsForTranslation.fromshape(
/// shape, range(N))` trick — upstream abuses `data_w=range(N)` to let
/// `fromshape` place integer indices into `keywords`. We compute the
/// same `i_<name> → index` map directly from the decoded
/// [`crate::flowspace::argument::CallShape`].
pub fn call_args_expand(
    hop: &HighLevelOp,
) -> Result<(HighLevelOp, HashMap<String, usize>), TyperError> {
    let hop = hop.copy();
    let shape_const = {
        let args_s = hop.args_s.borrow();
        if args_s.len() < 2 {
            return Err(TyperError::message(
                "call_args_expand: hop.args_s has fewer than 2 entries",
            ));
        }
        let Some(cv) = args_s[1].const_() else {
            return Err(TyperError::message(
                "call_args_expand: hop.args_s[1] is not a constant",
            ));
        };
        cv.clone()
    };
    let Some(shape) = crate::annotator::unaryop::decode_call_shape(&shape_const) else {
        return Err(TyperError::message(format!(
            "call_args_expand: hop.args_s[1].const does not decode as a CallShape: {shape_const:?}"
        )));
    };
    if shape.shape_star {
        return Err(TyperError::message(
            "call_args_expand: arguments.w_stararg is None assertion failed",
        ));
    }
    let mut kwds_i = HashMap::new();
    for (offset, key) in shape.shape_keys.iter().enumerate() {
        let idx = shape.shape_cnt + offset;
        kwds_i.insert(format!("i_{key}"), idx);
    }
    Ok((hop, kwds_i))
}

/// RPython `parse_kwds(hop, *argspec_i_r)` (rbuiltin.py:153-168).
///
/// ```python
/// def parse_kwds(hop, *argspec_i_r):
///     lst = [i for (i, r) in argspec_i_r if i is not None]
///     lst.sort()
///     if lst != range(hop.nb_args - len(lst), hop.nb_args):
///         raise TyperError("keyword args are expected to be at the end of "
///                          "the 'hop' arg list")
///     result = []
///     for i, r in argspec_i_r:
///         if i is not None:
///             if r is None:
///                 r = hop.args_r[i]
///             result.append(hop.inputarg(r, arg=i))
///         else:
///             result.append(None)
///     del hop.args_v[hop.nb_args - len(lst):]
///     return result
/// ```
///
/// Each `argspec_i_r` entry is `(Some(i), Some(r))` to extract arg `i`
/// converted into `r`, `(Some(i), None)` to use `hop.args_r[i]`, or
/// `(None, None)` for a placeholder (keyword argument not supplied).
///
/// The trailing `del hop.args_v[...]` truncates only `args_v` upstream
/// (asymmetry is upstream-faithful: `args_r` / `args_s` keep their
/// original lengths after keyword consumption).
pub fn parse_kwds(
    hop: &HighLevelOp,
    argspec_i_r: &[(Option<usize>, Option<Arc<dyn Repr>>)],
) -> Result<Vec<Option<crate::flowspace::model::Hlvalue>>, TyperError> {
    let mut lst: Vec<usize> = argspec_i_r.iter().filter_map(|(i, _)| *i).collect();
    lst.sort();
    let nb_args = hop.nb_args();
    let tail_start = nb_args - lst.len();
    let expected: Vec<usize> = (tail_start..nb_args).collect();
    if lst != expected {
        return Err(TyperError::message(
            "keyword args are expected to be at the end of the 'hop' arg list",
        ));
    }
    let mut result: Vec<Option<crate::flowspace::model::Hlvalue>> =
        Vec::with_capacity(argspec_i_r.len());
    for (i_opt, r_opt) in argspec_i_r.iter() {
        if let Some(i) = *i_opt {
            let r_effective: Arc<dyn Repr> = match r_opt {
                Some(r) => r.clone(),
                None => hop.args_r.borrow()[i].clone().ok_or_else(|| {
                    TyperError::message("parse_kwds: hop.args_r[i] is None and no override given")
                })?,
            };
            let v = hop.inputarg(&r_effective, i)?;
            result.push(Some(v));
        } else {
            result.push(None);
        }
    }
    hop.args_v.borrow_mut().truncate(tail_start);
    Ok(result)
}

/// RPython `SomeBuiltin.rtyper_makerepr(self, rtyper)` (rbuiltin.py:
/// :23-27).
///
/// ```python
/// def rtyper_makerepr(self, rtyper):
///     if not self.is_constant():
///         raise TyperError("non-constant built-in function!")
///     return BuiltinFunctionRepr(self.const)
/// ```
///
/// Pyre's port delegates to [`BuiltinFunctionRepr::new`] after
/// asserting `is_constant()`. The [`HostObject`] carrier pulled off
/// `SomeBuiltin.base.const_box` is the pyre equivalent of the Python
/// builtin function object `self.const`.
pub fn somebuiltin_rtyper_makerepr(s_builtin: &SomeBuiltin) -> Result<Arc<dyn Repr>, TyperError> {
    let Some(const_box) = &s_builtin.base.const_box else {
        return Err(TyperError::message("non-constant built-in function!"));
    };
    let ConstValue::HostObject(host) = &const_box.value else {
        return Err(TyperError::message(format!(
            "SomeBuiltin.rtyper_makerepr: expected HostObject const, got {:?}",
            const_box.value
        )));
    };
    Ok(Arc::new(BuiltinFunctionRepr::new(host.clone())) as Arc<dyn Repr>)
}

/// RPython `class BuiltinMethodRepr(Repr)` (rbuiltin.py:113-142).
///
/// Bound builtin method repr — stores the receiver annotation and its
/// concrete repr; dispatches `rtype_simple_call` via the `self_repr`'s
/// `rtype_method_<methodname>` lookup.
#[derive(Debug)]
pub struct BuiltinMethodRepr {
    /// RPython `self.s_self = s_self` (rbuiltin.py:116). Shared
    /// [`Rc`] mirroring upstream's identity-preserved receiver
    /// annotation.
    pub s_self: Rc<SomeValue>,
    /// RPython `self.self_repr = rtyper.getrepr(s_self)` (rbuiltin.py:117).
    pub self_repr: Arc<dyn Repr>,
    /// RPython `self.methodname = methodname` (rbuiltin.py:118).
    pub methodname: String,
    state: ReprState,
    /// RPython `self.lowleveltype = self.self_repr.lowleveltype`
    /// (rbuiltin.py:120) — bound methods have no runtime identity
    /// separate from their receiver, so the lowleveltype mirrors the
    /// receiver's directly.
    lltype: LowLevelType,
}

impl BuiltinMethodRepr {
    /// RPython `BuiltinMethodRepr.__init__(self, rtyper, s_self,
    /// methodname)` (rbuiltin.py:115-120).
    pub fn new(
        rtyper: &RPythonTyper,
        s_self: Rc<SomeValue>,
        methodname: String,
    ) -> Result<Self, TyperError> {
        let self_repr = rtyper.getrepr(&s_self)?;
        let lltype = self_repr.lowleveltype().clone();
        Ok(BuiltinMethodRepr {
            s_self,
            self_repr,
            methodname,
            state: ReprState::new(),
            lltype,
        })
    }
}

impl Repr for BuiltinMethodRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "BuiltinMethodRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::BuiltinMethodRepr
    }

    /// RPython `BuiltinMethodRepr.convert_const(self, obj)`
    /// (rbuiltin.py:122-123).
    ///
    /// ```python
    /// def convert_const(self, obj):
    ///     return self.self_repr.convert_const(obj.__self__)
    /// ```
    ///
    /// `obj` is a bound-method constant; `obj.__self__` is the
    /// receiver. The Rust port unwraps
    /// [`HostObject::bound_method_self`] and delegates to
    /// `self.self_repr.convert_const` on the receiver wrapped as
    /// [`ConstValue::HostObject`].
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "BuiltinMethodRepr.convert_const: expected HostObject bound method, got {value:?}"
            )));
        };
        let receiver = host.bound_method_self().ok_or_else(|| {
            TyperError::message(
                "BuiltinMethodRepr.convert_const: HostObject is not a bound method (no __self__)",
            )
        })?;
        self.self_repr
            .convert_const(&ConstValue::HostObject(receiver.clone()))
    }

    /// RPython `BuiltinMethodRepr.rtype_simple_call(self, hop)`
    /// (rbuiltin.py:125-142).
    ///
    /// ```python
    /// def rtype_simple_call(self, hop):
    ///     # methods: look up the rtype_method_xxx()
    ///     name = 'rtype_method_' + self.methodname
    ///     try:
    ///         bltintyper = getattr(self.self_repr, name)
    ///     except AttributeError:
    ///         raise TyperError("missing %s.%s" % (
    ///             self.self_repr.__class__.__name__, name))
    ///     # hack based on the fact that 'lowleveltype == self_repr.lowleveltype'
    ///     hop2 = hop.copy()
    ///     assert hop2.args_r[0] is self
    ///     if isinstance(hop2.args_v[0], Constant):
    ///         c = hop2.args_v[0].value    # get object from bound method
    ///         c = c.__self__
    ///         hop2.args_v[0] = Constant(c)
    ///     hop2.args_s[0] = self.s_self
    ///     hop2.args_r[0] = self.self_repr
    ///     return bltintyper(hop2)
    /// ```
    ///
    /// The `getattr(self.self_repr, 'rtype_method_' + methodname)`
    /// upstream lookup maps to the [`Repr::rtype_method`] trait method
    /// — each concrete `Repr` overrides `rtype_method` to route by
    /// `method_name`. The `assert hop2.args_r[0] is self` upstream
    /// identity check is dropped in Rust; arg0 rewriting is the
    /// observable effect.
    fn rtype_simple_call(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::flowspace::model::Hlvalue;

        let hop2 = hop.copy();
        // `hack based on the fact that lowleveltype == self_repr.lowleveltype`:
        // if args_v[0] is a Constant bound method, pull __self__ and
        // rebind.
        {
            let mut args_v = hop2.args_v.borrow_mut();
            if let Hlvalue::Constant(c) = &args_v[0] {
                if let ConstValue::HostObject(host) = &c.value {
                    if let Some(receiver) = host.bound_method_self() {
                        let new_const = Constant::new(ConstValue::HostObject(receiver.clone()));
                        args_v[0] = Hlvalue::Constant(new_const);
                    }
                }
            }
        }
        *hop2.args_s.borrow_mut().get_mut(0).ok_or_else(|| {
            TyperError::message("BuiltinMethodRepr.rtype_simple_call: hop.args_s is empty")
        })? = (*self.s_self).clone();
        *hop2.args_r.borrow_mut().get_mut(0).ok_or_else(|| {
            TyperError::message("BuiltinMethodRepr.rtype_simple_call: hop.args_r is empty")
        })? = Some(self.self_repr.clone());
        self.self_repr.rtype_method(&self.methodname, &hop2)
    }
}

/// RPython `SomeBuiltinMethod.rtyper_makerepr(self, rtyper)`
/// (rbuiltin.py:36-39).
///
/// ```python
/// def rtyper_makerepr(self, rtyper):
///     assert self.methodname is not None
///     result = BuiltinMethodRepr(rtyper, self.s_self, self.methodname)
///     return result
/// ```
///
/// `methodname` is a non-optional `String` in the Rust port, so the
/// `assert self.methodname is not None` is structurally enforced.
pub fn somebuiltinmethod_rtyper_makerepr(
    s_method: &SomeBuiltinMethod,
    rtyper: &RPythonTyper,
) -> Result<Arc<dyn Repr>, TyperError> {
    let repr =
        BuiltinMethodRepr::new(rtyper, s_method.s_self.clone(), s_method.methodname.clone())?;
    Ok(Arc::new(repr) as Arc<dyn Repr>)
}

/// RPython `pairtype(BuiltinMethodRepr, BuiltinMethodRepr).convert_from_to`
/// (rbuiltin.py:144-151).
///
/// ```python
/// class __extend__(pairtype(BuiltinMethodRepr, BuiltinMethodRepr)):
///     def convert_from_to((r_from, r_to), v, llops):
///         # convert between two MethodReprs only if they are about the
///         # same methodname.
///         if r_from.methodname != r_to.methodname:
///             return NotImplemented
///         return llops.convertvar(v, r_from.self_repr, r_to.self_repr)
/// ```
///
/// Upstream's `NotImplemented` maps to `Ok(None)` — the pairtype
/// dispatcher ([`crate::translator::rtyper::pairtype::pair_convert_from_to`])
/// treats this as "keep walking `pair_mro`".
///
/// Downcasts `&dyn Repr` → `&dyn Any` → `&BuiltinMethodRepr` via the
/// `Any` supertrait on [`Repr`]. Returns `Ok(None)` if either side
/// isn't actually a [`BuiltinMethodRepr`] (defensive — callers guard
/// with `ReprClassId` but the cast keeps this robust to misuse).
pub fn pair_builtin_method_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &crate::flowspace::model::Hlvalue,
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    let any_from: &dyn std::any::Any = r_from;
    let any_to: &dyn std::any::Any = r_to;
    let Some(from) = any_from.downcast_ref::<BuiltinMethodRepr>() else {
        return Ok(None);
    };
    let Some(to) = any_to.downcast_ref::<BuiltinMethodRepr>() else {
        return Ok(None);
    };
    if from.methodname != to.methodname {
        return Ok(None);
    }
    llops
        .convertvar(v.clone(), from.self_repr.as_ref(), to.self_repr.as_ref())
        .map(Some)
}

/// Top-level dispatcher used by [`crate::translator::rtyper::rmodel::rtyper_makerepr`]
/// — routes `SomeBuiltin` and `SomeBuiltinMethod` to their respective
/// ports. Keeps the dispatch surface inside this module so the
/// rmodel-side arm is a one-liner.
pub fn dispatch_rtyper_makerepr(
    s: &SomeValue,
    rtyper: &RPythonTyper,
) -> Result<Arc<dyn Repr>, TyperError> {
    match s {
        SomeValue::Builtin(b) => somebuiltin_rtyper_makerepr(b),
        SomeValue::BuiltinMethod(m) => somebuiltinmethod_rtyper_makerepr(m, rtyper),
        other => Err(TyperError::message(format!(
            "rbuiltin::dispatch_rtyper_makerepr: unexpected SomeValue variant {other:?}"
        ))),
    }
}

// =====================================================================
// @typer_for(...) registrations — rbuiltin.py:172- onwards.
// Each function mirrors one upstream `@typer_for(<python builtin>)`
// decorated `rtype_builtin_*` body. Registration is driven by
// [`install_default_typers`] via the [`BUILTIN_TYPER`] OnceLock init.
// =====================================================================

fn arg_repr(hop: &HighLevelOp, index: usize) -> Result<Arc<dyn Repr>, TyperError> {
    hop.args_r
        .borrow()
        .get(index)
        .cloned()
        .flatten()
        .ok_or_else(|| {
            TyperError::message(format!(
                "builtin typer: hop.args_r[{index}] is None or out of range"
            ))
        })
}

/// RPython `@typer_for(bool) def rtype_builtin_bool(hop)`
/// (rbuiltin.py:172-176).
///
/// ```python
/// @typer_for(bool)
/// def rtype_builtin_bool(hop):
///     # not called any more?
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_bool(hop)
/// ```
fn rtype_builtin_bool(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_bool: expected nb_args == 1, got {}",
            hop.nb_args()
        )));
    }
    arg_repr(hop, 0)?.rtype_bool(hop)
}

/// RPython `@typer_for(int) def rtype_builtin_int(hop)`
/// (rbuiltin.py:178-184).
///
/// ```python
/// @typer_for(int)
/// def rtype_builtin_int(hop):
///     if isinstance(hop.args_s[0], annmodel.SomeString):
///         assert 1 <= hop.nb_args <= 2
///         return hop.args_r[0].rtype_int(hop)
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_int(hop)
/// ```
///
/// The two branches call `rtype_int` identically; only the
/// `nb_args` assertion range differs. `SomeString` is matched on the
/// enum tag to match upstream's `isinstance` check.
fn rtype_builtin_int(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    let is_string = matches!(hop.args_s.borrow().first(), Some(SomeValue::String(_)));
    let nb = hop.nb_args();
    if is_string {
        if !(1..=2).contains(&nb) {
            return Err(TyperError::message(format!(
                "rtype_builtin_int: SomeString branch expects 1 <= nb_args <= 2, got {nb}"
            )));
        }
    } else if nb != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_int: expected nb_args == 1, got {nb}"
        )));
    }
    arg_repr(hop, 0)?.rtype_int(hop)
}

/// RPython `@typer_for(float) def rtype_builtin_float(hop)`
/// (rbuiltin.py:186-189).
///
/// ```python
/// @typer_for(float)
/// def rtype_builtin_float(hop):
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_float(hop)
/// ```
fn rtype_builtin_float(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_float: expected nb_args == 1, got {}",
            hop.nb_args()
        )));
    }
    arg_repr(hop, 0)?.rtype_float(hop)
}

/// RPython `@typer_for(chr) def rtype_builtin_chr(hop)`
/// (rbuiltin.py:191-194).
///
/// ```python
/// @typer_for(chr)
/// def rtype_builtin_chr(hop):
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_chr(hop)
/// ```
fn rtype_builtin_chr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_chr: expected nb_args == 1, got {}",
            hop.nb_args()
        )));
    }
    arg_repr(hop, 0)?.rtype_chr(hop)
}

/// RPython `@typer_for(unichr) def rtype_builtin_unichr(hop)`
/// (rbuiltin.py:196-199).
///
/// ```python
/// @typer_for(unichr)
/// def rtype_builtin_unichr(hop):
///     assert hop.nb_args == 1
///     return hop.args_r[0].rtype_unichr(hop)
/// ```
fn rtype_builtin_unichr(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_builtin_unichr: expected nb_args == 1, got {}",
            hop.nb_args()
        )));
    }
    arg_repr(hop, 0)?.rtype_unichr(hop)
}

/// RPython `@typer_for(unicode) def rtype_builtin_unicode(hop)`
/// (rbuiltin.py:201-203).
///
/// ```python
/// @typer_for(unicode)
/// def rtype_builtin_unicode(hop):
///     return hop.args_r[0].rtype_unicode(hop)
/// ```
///
/// Upstream does not assert `nb_args`; both `unicode(x)` and
/// `unicode(x, encoding)` flow through the same dispatch.
fn rtype_builtin_unicode(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    arg_repr(hop, 0)?.rtype_unicode(hop)
}

/// RPython `@typer_for(bytearray) def rtype_builtin_bytearray(hop)`
/// (rbuiltin.py:205-207).
///
/// ```python
/// @typer_for(bytearray)
/// def rtype_builtin_bytearray(hop):
///     return hop.args_r[0].rtype_bytearray(hop)
/// ```
fn rtype_builtin_bytearray(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    arg_repr(hop, 0)?.rtype_bytearray(hop)
}

/// RPython `@typer_for(list) def rtype_builtin_list(hop)`
/// (rbuiltin.py:209-211).
///
/// ```python
/// @typer_for(list)
/// def rtype_builtin_list(hop):
///     return hop.args_r[0].rtype_bltn_list(hop)
/// ```
fn rtype_builtin_list(hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
    arg_repr(hop, 0)?.rtype_bltn_list(hop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeBuiltin, SomeBuiltinMethod};
    use crate::flowspace::model::HostObject;

    fn host_bltin(name: &str) -> HostObject {
        HostObject::new_builtin_callable(name)
    }

    #[test]
    fn builtin_function_repr_has_void_lowleveltype() {
        let r = BuiltinFunctionRepr::new(host_bltin("len"));
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        assert_eq!(r.class_name(), "BuiltinFunctionRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::BuiltinFunctionRepr);
    }

    #[test]
    fn builtin_function_repr_convert_const_default_preserves_value_on_void() {
        // Upstream `Repr.convert_const` default (`rmodel.py:124-130`) keeps
        // the Python value; `Void._contains_value` (lltype.py:194-197)
        // accepts anything, so Void-typed reprs pass through the value
        // unchanged. No `BuiltinFunctionRepr.convert_const` override
        // exists upstream (rbuiltin.py:67-110).
        let r = BuiltinFunctionRepr::new(host_bltin("len"));
        let c = r.convert_const(&ConstValue::Int(42)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::Int(42)));
    }

    #[test]
    fn somebuiltin_rtyper_makerepr_constant_branch_returns_builtin_function_repr() {
        let mut s = SomeBuiltin::new("len", None, None);
        // Populate `const_box` so `is_constant()` succeeds.
        s.base.const_box = Some(Constant::new(ConstValue::HostObject(host_bltin("len"))));
        let r = somebuiltin_rtyper_makerepr(&s).unwrap();
        assert_eq!(r.class_name(), "BuiltinFunctionRepr");
    }

    #[test]
    fn somebuiltin_rtyper_makerepr_non_constant_branch_errors() {
        // No const_box set → upstream raises `TyperError("non-constant built-in function!")`.
        let s = SomeBuiltin::new("len", None, None);
        let err = somebuiltin_rtyper_makerepr(&s).unwrap_err();
        assert!(err.to_string().contains("non-constant built-in function"));
    }

    fn dummy_rtyper() -> RPythonTyper {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        RPythonTyper::new(&ann)
    }

    #[test]
    fn somebuiltinmethod_rtyper_makerepr_builds_builtin_method_repr_with_self_repr() {
        use crate::annotator::model::SomeInteger;
        let rtyper = dummy_rtyper();
        // s_self = SomeInteger → self_repr = IntegerRepr(Signed).
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let s_method = SomeBuiltinMethod::new("foo", s_self, "foo");
        let r = somebuiltinmethod_rtyper_makerepr(&s_method, &rtyper).unwrap();
        assert_eq!(r.class_name(), "BuiltinMethodRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::BuiltinMethodRepr);
        // lowleveltype mirrors the receiver's repr (IntegerRepr → Signed).
        assert_eq!(r.lowleveltype(), &LowLevelType::Signed);
    }

    #[test]
    fn builtin_method_repr_convert_const_rejects_non_hostobject() {
        use crate::annotator::model::SomeInteger;
        let rtyper = dummy_rtyper();
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let repr = BuiltinMethodRepr::new(&rtyper, Rc::new(s_self), "foo".into()).unwrap();
        let err = repr.convert_const(&ConstValue::Int(42)).unwrap_err();
        assert!(err.to_string().contains("expected HostObject bound method"));
    }

    #[test]
    fn builtin_method_repr_convert_const_rejects_hostobject_not_bound_method() {
        use crate::annotator::model::SomeInteger;
        let rtyper = dummy_rtyper();
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let repr = BuiltinMethodRepr::new(&rtyper, Rc::new(s_self), "foo".into()).unwrap();
        // A plain builtin HostObject (not a bound method) → error.
        let err = repr
            .convert_const(&ConstValue::HostObject(host_bltin("len")))
            .unwrap_err();
        assert!(err.to_string().contains("HostObject is not a bound method"));
    }

    fn noop_typer(_hop: &HighLevelOp, _kwds_i: &HashMap<String, usize>) -> RTypeResult {
        Ok(None)
    }

    #[test]
    fn typer_for_registers_and_lookup_typer_reads_back() {
        let key = host_bltin("rbuiltin_test_registers_and_reads_back");
        assert!(lookup_typer(&key).is_none());
        typer_for(key.clone(), noop_typer);
        assert!(lookup_typer(&key).is_some());
    }

    #[test]
    fn findbltintyper_returns_registered_typer() {
        let key = host_bltin("rbuiltin_test_findbltintyper_returns_registered");
        typer_for(key.clone(), noop_typer);
        let repr = BuiltinFunctionRepr::new(key);
        let f = repr.findbltintyper().expect("typer found");
        // Function pointers compare by address — registered fn must
        // match `noop_typer`.
        assert!(std::ptr::eq(f as *const (), noop_typer as *const ()));
    }

    #[test]
    fn findbltintyper_unknown_host_raises_typererror() {
        // HostObject with no registry entry and not ConstValue::LLPtr
        // (so extregistry::is_registered returns false) → upstream
        // `TyperError("don't know about built-in function %r" % ...)`.
        let key = host_bltin("rbuiltin_test_findbltintyper_unknown_host");
        let repr = BuiltinFunctionRepr::new(key);
        let err = repr.findbltintyper().unwrap_err();
        assert!(
            err.to_string()
                .contains("don't know about built-in function")
        );
    }

    fn dummy_hop() -> HighLevelOp {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let spaceop = SpaceOperation::new(
            OpKind::SimpleCall.opname(),
            Vec::new(),
            crate::flowspace::model::Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops)
    }

    fn typer_calls_exception_cannot_occur(
        hop: &HighLevelOp,
        _kwds_i: &HashMap<String, usize>,
    ) -> RTypeResult {
        hop.exception_cannot_occur().unwrap();
        Ok(None)
    }

    #[test]
    fn call_returns_v_result_when_typer_calls_exception_cannot_occur() {
        let key = host_bltin("rbuiltin_test_call_returns_v_result");
        typer_for(key.clone(), typer_calls_exception_cannot_occur);
        let repr = BuiltinFunctionRepr::new(key);
        let hop = dummy_hop();
        let result = repr._call(&hop, &HashMap::new()).unwrap();
        assert!(result.is_none());
        // Flag was flipped to true by exception_cannot_occur (set back
        // to true after the initial `= false` reset in `_call`).
        assert!(hop.llops.borrow()._called_exception_is_here_or_cannot_occur);
    }

    #[test]
    fn call_errors_when_typer_skips_exception_guard() {
        let key = host_bltin("rbuiltin_test_call_errors_when_skipped");
        // `noop_typer` never calls exception_is_here/exception_cannot_occur.
        typer_for(key.clone(), noop_typer);
        let repr = BuiltinFunctionRepr::new(key);
        let hop = dummy_hop();
        let err = repr._call(&hop, &HashMap::new()).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing hop.exception_cannot_occur()")
        );
    }

    #[test]
    fn rtype_simple_call_pops_first_arg_before_dispatching_typer() {
        use crate::flowspace::model::{Hlvalue, Variable};
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NB_ARGS_AT_CALL: AtomicUsize = AtomicUsize::new(usize::MAX);
        fn record_nb_args_typer(
            hop: &HighLevelOp,
            _kwds_i: &HashMap<String, usize>,
        ) -> RTypeResult {
            NB_ARGS_AT_CALL.store(hop.nb_args(), Ordering::SeqCst);
            hop.exception_cannot_occur().unwrap();
            Ok(None)
        }

        let key = host_bltin("rbuiltin_test_rtype_simple_call_pops");
        typer_for(key.clone(), record_nb_args_typer);
        let repr = BuiltinFunctionRepr::new(key);
        let hop = dummy_hop();
        // Seed hop with a single arg (the builtin callable itself) so
        // rtype_simple_call -> r_s_popfirstarg leaves 0 args for the typer.
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Builtin(SomeBuiltin::new("unused", None, None)));
        hop.args_r.borrow_mut().push(None);
        NB_ARGS_AT_CALL.store(usize::MAX, Ordering::SeqCst);
        repr.rtype_simple_call(&hop).unwrap();
        assert_eq!(NB_ARGS_AT_CALL.load(Ordering::SeqCst), 0);
        // Original hop is unaffected (upstream `hop.copy()` semantics).
        assert_eq!(hop.nb_args(), 1);
    }

    fn seed_hop_with_shape(
        hop: &HighLevelOp,
        shape_cnt: usize,
        shape_keys: &[&str],
        shape_star: bool,
        extra_args: usize,
    ) {
        use crate::annotator::model::{SomeInteger, SomeTuple};
        use crate::flowspace::model::{Constant, Hlvalue, Variable};

        // Position 0: the builtin callable itself (SomeValue::Builtin,
        // value unused here — just needs to exist for r_s_popfirstarg).
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Builtin(SomeBuiltin::new("unused", None, None)));
        hop.args_r.borrow_mut().push(None);

        // Position 1: the shape constant — SomeTuple of
        // (Int(shape_cnt), Tuple(Str(...)), Bool(shape_star)).
        let shape_const = ConstValue::Tuple(vec![
            ConstValue::Int(shape_cnt as i64),
            ConstValue::Tuple(
                shape_keys
                    .iter()
                    .map(|k| ConstValue::Str((*k).to_string()))
                    .collect(),
            ),
            ConstValue::Bool(shape_star),
        ]);
        let mut s_shape = SomeTuple::new(Vec::new());
        s_shape.base.const_box = Some(Constant::new(shape_const));
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s.borrow_mut().push(SomeValue::Tuple(s_shape));
        hop.args_r.borrow_mut().push(None);

        // Extra positions: placeholders, so hop.nb_args() reads as
        // expected by typers.
        for _ in 0..extra_args {
            hop.args_v
                .borrow_mut()
                .push(Hlvalue::Variable(Variable::new()));
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::new(false, false)));
            hop.args_r.borrow_mut().push(None);
        }
    }

    #[test]
    fn call_args_expand_builds_kwds_i_from_shape_keys() {
        let hop = dummy_hop();
        // hint(x, category='foo') → shape_cnt=1, shape_keys=['category'].
        seed_hop_with_shape(&hop, 1, &["category"], false, 2);
        let (out_hop, kwds_i) = call_args_expand(&hop).unwrap();
        assert_eq!(kwds_i.len(), 1);
        // shape_cnt=1 → first keyword occupies index 1 (post-shape-header,
        // relative to the shape's own flatten).
        assert_eq!(kwds_i.get("i_category"), Some(&1));
        // hop is copied, not mutated.
        assert_eq!(out_hop.nb_args(), hop.nb_args());
    }

    #[test]
    fn call_args_expand_empty_keywords_yields_empty_kwds_i() {
        let hop = dummy_hop();
        seed_hop_with_shape(&hop, 2, &[], false, 2);
        let (_out_hop, kwds_i) = call_args_expand(&hop).unwrap();
        assert!(kwds_i.is_empty());
    }

    #[test]
    fn call_args_expand_rejects_stararg_shape() {
        let hop = dummy_hop();
        seed_hop_with_shape(&hop, 1, &[], true, 2);
        // HighLevelOp is not Debug, so `.unwrap_err()` won't compile —
        // pattern-match the Result manually.
        match call_args_expand(&hop) {
            Err(err) => assert!(err.to_string().contains("w_stararg is None")),
            Ok(_) => panic!("expected TyperError for shape_star=true"),
        }
    }

    #[test]
    fn rtype_call_args_forwards_kwds_i_and_pops_two_leading_args() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static LAST_CATEGORY_IDX: AtomicUsize = AtomicUsize::new(usize::MAX);
        static LAST_NB_ARGS: AtomicUsize = AtomicUsize::new(usize::MAX);

        fn kwds_observing_typer(hop: &HighLevelOp, kwds_i: &HashMap<String, usize>) -> RTypeResult {
            LAST_NB_ARGS.store(hop.nb_args(), Ordering::SeqCst);
            LAST_CATEGORY_IDX.store(
                kwds_i.get("i_category").copied().unwrap_or(usize::MAX),
                Ordering::SeqCst,
            );
            hop.exception_cannot_occur().unwrap();
            Ok(None)
        }

        let key = host_bltin("rbuiltin_test_rtype_call_args_forwards");
        typer_for(key.clone(), kwds_observing_typer);
        let repr = BuiltinFunctionRepr::new(key);
        let hop = dummy_hop();
        // hint(x, category='foo') → shape_cnt=1, shape_keys=['category'],
        // extra_args=2 so hop.nb_args()=4 → hop2.nb_args()=2 after the
        // two r_s_popfirstarg calls.
        seed_hop_with_shape(&hop, 1, &["category"], false, 2);
        LAST_CATEGORY_IDX.store(usize::MAX, Ordering::SeqCst);
        LAST_NB_ARGS.store(usize::MAX, Ordering::SeqCst);
        repr.rtype_call_args(&hop).unwrap();
        assert_eq!(LAST_NB_ARGS.load(Ordering::SeqCst), 2);
        assert_eq!(LAST_CATEGORY_IDX.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parse_kwds_all_none_specs_yield_none_results_and_preserve_args_v() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(None);

        let specs: Vec<(Option<usize>, Option<Arc<dyn Repr>>)> = vec![(None, None), (None, None)];
        let result = parse_kwds(&hop, &specs).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].is_none());
        assert!(result[1].is_none());
        // lst is empty → tail_start == nb_args, truncate is a no-op.
        assert_eq!(hop.nb_args(), 1);
    }

    #[test]
    fn parse_kwds_rejects_misordered_keyword_indices() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        let hop = dummy_hop();
        for _ in 0..3 {
            hop.args_v
                .borrow_mut()
                .push(Hlvalue::Variable(Variable::new()));
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::new(false, false)));
            hop.args_r.borrow_mut().push(None);
        }
        // nb_args=3, specify i=0 (should be 2) — lst=[0] must equal [2].
        let specs: Vec<(Option<usize>, Option<Arc<dyn Repr>>)> = vec![(Some(0), None)];
        match parse_kwds(&hop, &specs) {
            Err(err) => assert!(
                err.to_string()
                    .contains("keyword args are expected to be at the end")
            ),
            Ok(_) => panic!("expected parse_kwds to reject misordered keyword index"),
        }
    }

    #[test]
    fn parse_kwds_consumes_tail_args_and_truncates_args_v() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Constant, Hlvalue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        // nb_args = 2: [positional, keyword]. Seed the keyword as a
        // Hlvalue::Constant so inputarg takes the early-return path
        // (avoids convertvar wiring).
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(None);

        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(42),
                LowLevelType::Signed,
            )));
        let mut s_const = SomeInteger::new(false, false);
        s_const.base.const_box = Some(Constant::new(ConstValue::Int(42)));
        hop.args_s.borrow_mut().push(SomeValue::Integer(s_const));
        hop.args_r.borrow_mut().push(None);

        let r: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        let specs: Vec<(Option<usize>, Option<Arc<dyn Repr>>)> = vec![(Some(1), Some(r.clone()))];
        let result = parse_kwds(&hop, &specs).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_some());
        // args_v truncated to 1 (tail_start = nb_args(2) - len(lst)(1)).
        assert_eq!(hop.args_v.borrow().len(), 1);
    }

    #[test]
    fn builtin_method_repr_rtype_simple_call_surfaces_missing_rtype_method() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        let rtyper = dummy_rtyper();
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let repr = BuiltinMethodRepr::new(&rtyper, Rc::new(s_self), "foo".into()).unwrap();

        let hop = dummy_hop();
        // Seed args_v[0] with a Variable (non-constant) — no bound-method
        // unwrap path.
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(None);

        let err = repr.rtype_simple_call(&hop).unwrap_err();
        // IntegerRepr has no rtype_method_foo override → default trait
        // method raises the "missing <class>.rtype_method_<name>" error.
        let msg = err.to_string();
        assert!(msg.contains("rtype_method_foo"));
    }

    #[test]
    fn builtin_method_repr_rtype_simple_call_rewrites_constant_bound_method_arg0() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Constant, Hlvalue, Variable};
        use crate::translator::rtyper::rmodel::Repr;

        // Observer repr to inspect args_v[0] after rtype_simple_call's
        // rewrite. Captures the constant via interior mutability.
        #[derive(Debug)]
        struct ObserverRepr {
            state: ReprState,
            lltype: LowLevelType,
            captured: std::sync::Mutex<Option<ConstValue>>,
        }
        impl ObserverRepr {
            fn new() -> Arc<Self> {
                Arc::new(ObserverRepr {
                    state: ReprState::new(),
                    lltype: LowLevelType::Signed,
                    captured: std::sync::Mutex::new(None),
                })
            }
        }
        impl Repr for ObserverRepr {
            fn lowleveltype(&self) -> &LowLevelType {
                &self.lltype
            }
            fn state(&self) -> &ReprState {
                &self.state
            }
            fn class_name(&self) -> &'static str {
                "ObserverRepr"
            }
            fn repr_class_id(&self) -> ReprClassId {
                ReprClassId::Repr
            }
            fn convert_const(&self, _v: &ConstValue) -> Result<Constant, TyperError> {
                Err(TyperError::message("ObserverRepr.convert_const unused"))
            }
            fn rtype_method(&self, _name: &str, hop: &HighLevelOp) -> RTypeResult {
                if let Hlvalue::Constant(c) = &hop.args_v.borrow()[0] {
                    *self.captured.lock().unwrap() = Some(c.value.clone());
                }
                hop.exception_cannot_occur().unwrap();
                Ok(None)
            }
        }

        let observer: Arc<ObserverRepr> = ObserverRepr::new();
        let self_repr: Arc<dyn Repr> = observer.clone();
        let s_self = Rc::new(SomeValue::Integer(SomeInteger::new(false, false)));
        let repr = BuiltinMethodRepr {
            s_self: s_self.clone(),
            self_repr,
            methodname: "foo".into(),
            state: ReprState::new(),
            lltype: LowLevelType::Signed,
        };

        // Build a bound-method HostObject whose __self__ is an int-like
        // builtin-callable (unused — only identity matters for the
        // capture).
        let receiver = host_bltin("receiver_obj");
        let func = host_bltin("receiver_method");
        let origin_class = host_bltin("origin_class");
        let bound = HostObject::new_bound_method(
            "m.receiver_method",
            receiver.clone(),
            func,
            "foo",
            origin_class,
        );

        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                bound,
            ))));
        hop.args_s.borrow_mut().push((*s_self).clone());
        hop.args_r.borrow_mut().push(None);

        repr.rtype_simple_call(&hop).unwrap();

        // ObserverRepr captured the rewritten arg0: bound method's
        // HostObject was replaced by its __self__.
        let captured = observer.captured.lock().unwrap().clone().unwrap();
        match captured {
            ConstValue::HostObject(h) => assert_eq!(h, receiver),
            other => panic!("expected rewritten HostObject, got {other:?}"),
        }
    }

    #[test]
    fn pair_builtin_method_convert_from_to_returns_none_when_methodnames_differ() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let rtyper = Rc::new(dummy_rtyper());
        let s_self = Rc::new(SomeValue::Integer(SomeInteger::new(false, false)));
        let from = BuiltinMethodRepr::new(&rtyper, s_self.clone(), "foo".into()).unwrap();
        let to = BuiltinMethodRepr::new(&rtyper, s_self, "bar".into()).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let v = Hlvalue::Variable(Variable::new());
        let result = pair_builtin_method_convert_from_to(&from, &to, &v, &mut llops).unwrap();
        // NotImplemented → Ok(None), pair_mro walker falls through.
        assert!(result.is_none());
    }

    #[test]
    fn pair_builtin_method_convert_from_to_same_methodname_delegates_to_self_repr_convertvar() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let rtyper = Rc::new(dummy_rtyper());
        let s_self = Rc::new(SomeValue::Integer(SomeInteger::new(false, false)));
        let from = BuiltinMethodRepr::new(&rtyper, s_self.clone(), "same".into()).unwrap();
        let to = BuiltinMethodRepr::new(&rtyper, s_self, "same".into()).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let v = Hlvalue::Variable(Variable::new());
        // Both self_reprs are IntegerRepr(Signed) → convertvar's
        // same-repr short-circuit returns orig_v unchanged.
        let result = pair_builtin_method_convert_from_to(&from, &to, &v, &mut llops)
            .unwrap()
            .expect("same-methodname path returns Some");
        match (&v, &result) {
            (Hlvalue::Variable(a), Hlvalue::Variable(b)) => assert_eq!(a, b),
            _ => panic!("expected Variable round-trip"),
        }
    }

    #[test]
    fn install_default_typers_registers_bool_int_float_chr_from_host_env() {
        // HOST_ENV.lookup_builtin populates `bool`/`int`/`float`/`chr`
        // at bootstrap — the BUILTIN_TYPER map must carry a typer for
        // each after the OnceLock init fires.
        for name in ["bool", "int", "float", "chr"] {
            let host = HOST_ENV
                .lookup_builtin(name)
                .unwrap_or_else(|| panic!("HOST_ENV missing builtin {name}"));
            assert!(
                lookup_typer(&host).is_some(),
                "BUILTIN_TYPER missing entry for `{name}`"
            );
        }
    }

    #[test]
    fn rtype_builtin_bool_rejects_nb_args_mismatch() {
        let hop = dummy_hop();
        // nb_args = 0 → should err.
        let err = rtype_builtin_bool(&hop, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("rtype_builtin_bool"));
    }

    #[test]
    fn rtype_builtin_int_non_string_branch_delegates_to_args_r_rtype_int() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        // Seed the variable with a concretetype so hop.genop's
        // assertion doesn't panic inside IntegerRepr::rtype_int.
        let var = Variable::new();
        var.set_concretetype(Some(LowLevelType::Signed));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(var));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let int_repr: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(int_repr));

        // IntegerRepr::rtype_int is implemented — non-string branch
        // delegates and returns a variable.
        let result = rtype_builtin_int(&hop, &HashMap::new()).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn rtype_builtin_float_rejects_nb_args_2() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        let hop = dummy_hop();
        for _ in 0..2 {
            hop.args_v
                .borrow_mut()
                .push(Hlvalue::Variable(Variable::new()));
            hop.args_s
                .borrow_mut()
                .push(SomeValue::Integer(SomeInteger::new(false, false)));
            hop.args_r.borrow_mut().push(None);
        }
        let err = rtype_builtin_float(&hop, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("expected nb_args == 1"));
    }

    #[test]
    fn rtype_builtin_unichr_rejects_nb_args_mismatch() {
        let hop = dummy_hop();
        let err = rtype_builtin_unichr(&hop, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("rtype_builtin_unichr"));
    }

    #[test]
    fn rtype_builtin_unicode_delegates_to_default_rtype_unicode_stub() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let int_repr: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(int_repr));

        let err = rtype_builtin_unicode(&hop, &HashMap::new()).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("unicode"));
    }

    #[test]
    fn rtype_builtin_bytearray_delegates_to_default_rtype_bytearray_stub() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let int_repr: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(int_repr));

        let err = rtype_builtin_bytearray(&hop, &HashMap::new()).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("bytearray"));
    }

    #[test]
    fn rtype_builtin_list_delegates_to_default_rtype_bltn_list_stub() {
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;

        let hop = dummy_hop();
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        let int_repr: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int")));
        hop.args_r.borrow_mut().push(Some(int_repr));

        let err = rtype_builtin_list(&hop, &HashMap::new()).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("bltn_list"));
    }

    #[test]
    fn dispatch_routes_somebuiltin_to_function_repr_and_somebuiltinmethod_to_method_repr() {
        use crate::annotator::model::SomeInteger;
        let rtyper = dummy_rtyper();

        // SomeBuiltin — constant branch.
        let mut sb = SomeBuiltin::new("abs", None, None);
        sb.base.const_box = Some(Constant::new(ConstValue::HostObject(host_bltin("abs"))));
        let r = dispatch_rtyper_makerepr(&SomeValue::Builtin(sb), &rtyper).unwrap();
        assert_eq!(r.class_name(), "BuiltinFunctionRepr");

        // SomeBuiltinMethod — now ported.
        let s_self = SomeValue::Integer(SomeInteger::new(false, false));
        let sbm = SomeBuiltinMethod::new("foo", s_self, "foo");
        let r = dispatch_rtyper_makerepr(&SomeValue::BuiltinMethod(sbm), &rtyper).unwrap();
        assert_eq!(r.class_name(), "BuiltinMethodRepr");
    }
}
