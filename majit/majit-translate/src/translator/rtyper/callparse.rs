//! Port of `rpython/rtyper/callparse.py`.
//!
//! Structure mirrors upstream line-by-line:
//!
//! * `getrinputs` / `getrresult` — callparse.py:16-25.
//! * `ArgumentsForRtype` — callparse.py:7-14. Holder-list analogue of
//!   [`crate::annotator::argument::ArgumentsForTranslation`]; inherits
//!   `match_signature` semantics with upstream `_match_signature`
//!   order (argument.py:43-120): positional → vararg → keyword
//!   conflict detection → fill with kwds-or-defaults.
//! * `Holder` + 4 variants — callparse.py:73-164. The `Holder` struct
//!   wraps an [`Rc<HolderInner>`] so `clone()` shares both
//!   upstream-Python identity and the `_cache` slot. `VarHolder` /
//!   `ConstHolder` emit directly; `NewTupleHolder.__new__` ports the
//!   stararg-passthrough fast-path (callparse.py:127-137).
//! * `callparse` — callparse.py:34-70. Dispatches simple_call /
//!   call_args, builds `defs_h` from `graph.defaults`, calls
//!   `match_signature` + `Holder.emit`.
//!
//! ## Deferred — blocked on `rtuple::TupleRepr`
//!
//! `NewTupleHolder._emit` (callparse.py:145-152) and
//! `ItemHolder._emit` (callparse.py:160-164) require
//! `rtuple::TupleRepr` to materialise tuple components. Both arms
//! surface `MissingRTypeOperation` until that port lands. The
//! `Holder` enum carries the variants so the rest of callparse stays
//! upstream-shaped (NewTupleHolder.__new__ identity comparison +
//! ItemHolder construction in `Holder::items`); only the
//! `Hlvalue`-emitting half waits.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::annotator::argument::ArgErr;
use crate::annotator::model::SomeValue;
use crate::flowspace::argument::{CallShape, Signature};
use crate::flowspace::model::{ConstValue, Constant, Hlvalue};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::Repr;
use crate::translator::rtyper::rtyper::{HighLevelOp, RPythonTyper};

/// RPython `getrinputs(rtyper, graph)` (callparse.py:16-18):
///
/// ```python
/// def getrinputs(rtyper, graph):
///     return [rtyper.bindingrepr(v) for v in graph.getargs()]
/// ```
pub fn getrinputs(
    rtyper: &RPythonTyper,
    graph: &PyGraph,
) -> Result<Vec<Arc<dyn Repr>>, TyperError> {
    graph
        .graph
        .borrow()
        .getargs()
        .iter()
        .map(|arg| rtyper.bindingrepr(arg))
        .collect()
}

/// RPython `getrresult(rtyper, graph)` (callparse.py:20-25):
///
/// ```python
/// def getrresult(rtyper, graph):
///     if graph.getreturnvar().annotation is not None:
///         return rtyper.bindingrepr(graph.getreturnvar())
///     else:
///         return lltype.Void
/// ```
pub enum GetrresultKind {
    /// Annotated return — carries a concrete repr.
    Repr(Arc<dyn Repr>),
    /// Unannotated return — `lltype.Void` sentinel.
    Void,
}

pub fn getrresult(rtyper: &RPythonTyper, graph: &PyGraph) -> Result<GetrresultKind, TyperError> {
    let ret_var = graph.graph.borrow().getreturnvar();
    let annotated = match &ret_var {
        Hlvalue::Variable(v) => v.annotation.borrow().is_some(),
        Hlvalue::Constant(_) => true,
    };
    if annotated {
        Ok(GetrresultKind::Repr(rtyper.bindingrepr(&ret_var)?))
    } else {
        Ok(GetrresultKind::Void)
    }
}

impl GetrresultKind {
    pub fn lowleveltype(&self) -> LowLevelType {
        match self {
            GetrresultKind::Repr(r) => r.lowleveltype().clone(),
            GetrresultKind::Void => LowLevelType::Void,
        }
    }

    pub fn as_repr(&self) -> Option<&dyn Repr> {
        match self {
            GetrresultKind::Repr(r) => Some(r.as_ref()),
            GetrresultKind::Void => None,
        }
    }
}

/// RPython `class Holder(object)` (callparse.py:73-88).
///
/// `Holder` is a thin Rc-backed wrapper so two `Holder` values created
/// by `clone()` share the same upstream Python object identity — i.e.
/// they share their `_cache` and compare equal under
/// [`Holder::ptr_eq`]. Identity sharing matches upstream's class-
/// instance semantics, where `clone` doesn't exist and every
/// `holder1 == holder2` comparison is `id(holder1) == id(holder2)`.
#[derive(Debug)]
pub struct Holder {
    inner: Rc<HolderInner>,
}

#[derive(Debug)]
struct HolderInner {
    kind: HolderKind,
    /// Upstream `Holder._cache` (callparse.py:80-82). Keyed by the
    /// target repr's `Arc::as_ptr` identity because `&dyn Repr` isn't
    /// directly hashable.
    _cache: RefCell<HashMap<usize, Hlvalue>>,
}

impl Clone for Holder {
    fn clone(&self) -> Self {
        // Share identity + cache with the source — upstream's Python
        // semantics. NewTupleHolder.__new__ depends on this for its
        // ItemHolder-passthrough optimisation.
        Holder {
            inner: Rc::clone(&self.inner),
        }
    }
}

/// RPython `VarHolder` / `ConstHolder` / `NewTupleHolder` / `ItemHolder`
/// (callparse.py:91-164). The four upstream subclasses collapse into
/// this enum because dispatch goes through `Holder` methods.
#[derive(Clone, Debug)]
pub enum HolderKind {
    /// RPython `class VarHolder(Holder)` (callparse.py:91-110). Holds
    /// the hop-arg index + the source SomeValue binding (used only by
    /// `is_tuple()` to branch tuple-unpack logic).
    Var { num: usize, s_obj: SomeValue },
    /// RPython `class ConstHolder(Holder)` (callparse.py:112-124).
    Const { value: Constant },
    /// RPython `class NewTupleHolder(Holder)` (callparse.py:127-152).
    /// Carries the tuple-item holders that `ArgumentsForRtype.newtuple`
    /// collected from a stararg.
    NewTuple { holders: Vec<Holder> },
    /// RPython `class ItemHolder(Holder)` (callparse.py:155-164).
    Item { parent: Holder, index: usize },
}

impl Holder {
    pub fn with_kind(kind: HolderKind) -> Self {
        Holder {
            inner: Rc::new(HolderInner {
                kind,
                _cache: RefCell::new(HashMap::new()),
            }),
        }
    }

    /// Read-only view of the variant payload. Used by the
    /// `NewTupleHolder.__new__` passthrough check and external
    /// inspectors; the kind is immutable after construction.
    pub fn kind(&self) -> &HolderKind {
        &self.inner.kind
    }

    /// Identity comparison — matches upstream `id(a) == id(b)`. Two
    /// Holder values produced by `clone()` share inner state so this
    /// returns `true`; two independently constructed Holders are
    /// distinct even if their kinds compare structurally equal.
    pub fn ptr_eq(a: &Holder, b: &Holder) -> bool {
        Rc::ptr_eq(&a.inner, &b.inner)
    }

    pub fn var(num: usize, s_obj: SomeValue) -> Self {
        Self::with_kind(HolderKind::Var { num, s_obj })
    }

    pub fn const_(value: Constant) -> Self {
        Self::with_kind(HolderKind::Const { value })
    }

    /// RPython `NewTupleHolder.__new__(cls, holders)` (callparse.py:127-137):
    ///
    /// ```python
    /// def __new__(cls, holders):
    ///     for h in holders:
    ///         if not isinstance(h, ItemHolder) or not h.holder == holders[0].holder:
    ///             break
    ///     else:
    ///         if 0 < len(holders) == len(holders[0].holder.items()):
    ///             return holders[0].holder
    ///     inst = Holder.__new__(cls)
    ///     inst.holders = tuple(holders)
    ///     return inst
    /// ```
    ///
    /// Passthrough fast-path: if every holder is an `ItemHolder`
    /// pointing at the same parent and the count matches the parent's
    /// item count, return the parent holder unchanged. Mirrors
    /// upstream's `f(*x)` optimisation where unpacking-then-repacking
    /// `x` collapses to passing `x` itself. Identity comparison is
    /// [`Holder::ptr_eq`] (Rc-backed) so this only fires when the
    /// caller's items() came from the *same* Holder instance.
    pub fn new_tuple(holders: Vec<Holder>) -> Self {
        if !holders.is_empty()
            && let HolderKind::Item {
                parent: first_parent,
                ..
            } = holders[0].kind()
        {
            let first_parent = first_parent.clone();
            let all_share_parent = holders.iter().all(|h| match h.kind() {
                HolderKind::Item { parent, .. } => Holder::ptr_eq(parent, &first_parent),
                _ => false,
            });
            if all_share_parent {
                // upstream callparse.py:133 — `holders[0].holder.items()`.
                // ItemHolder is only ever produced by `Holder::items()`,
                // which itself only succeeds for tuple-shaped parents
                // (VarHolder/SomeTuple, ConstHolder/Tuple, NewTuple).
                // So this `items()` is an invariant call: if a caller
                // hand-built an ItemHolder pointing at a non-tuple
                // parent, that's a programmer error — propagate via
                // panic, matching upstream's `assert isinstance(s_obj,
                // SomeTuple)` in `VarHolder.items()` (callparse.py:103).
                let parent_items = first_parent
                    .items()
                    .expect("ItemHolder parent must be tuple-shaped (callparse.py:133)");
                if !parent_items.is_empty() && holders.len() == parent_items.len() {
                    return first_parent;
                }
            }
        }
        Self::with_kind(HolderKind::NewTuple { holders })
    }

    pub fn item(parent: Holder, index: usize) -> Self {
        Self::with_kind(HolderKind::Item { parent, index })
    }

    /// RPython `Holder.is_tuple(self)` (callparse.py:75-76) + override
    /// on `VarHolder` / `ConstHolder` / `NewTupleHolder`
    /// (callparse.py:97-98, 116-117, 139-141).
    pub fn is_tuple(&self) -> bool {
        match &self.inner.kind {
            HolderKind::Var { s_obj, .. } => matches!(s_obj, SomeValue::Tuple(_)),
            HolderKind::Const { value } => matches!(&value.value, ConstValue::Tuple(_)),
            HolderKind::NewTuple { .. } => true,
            HolderKind::Item { .. } => false,
        }
    }

    /// RPython `VarHolder.items(self)` (callparse.py:100-103) +
    /// `ConstHolder.items(self)` (callparse.py:119-121) +
    /// `NewTupleHolder.items(self)` (callparse.py:142-143).
    /// Materialise the tuple components.
    ///
    /// `VarHolder` materialises `ItemHolder(self, i)` per slot —
    /// `ItemHolder.parent` shares this Holder's identity (via
    /// `Rc::clone`), which is what
    /// `NewTupleHolder.__new__`'s ItemHolder-passthrough check
    /// depends on (callparse.py:130-134).
    pub fn items(&self) -> Result<Vec<Holder>, TyperError> {
        match &self.inner.kind {
            HolderKind::Var { s_obj, .. } => {
                let SomeValue::Tuple(t) = s_obj else {
                    return Err(TyperError::message("Holder::items: not a tuple holder"));
                };
                let len = t.items.len();
                Ok((0..len).map(|i| Holder::item(self.clone(), i)).collect())
            }
            HolderKind::Const { value } => {
                let ConstValue::Tuple(items) = &value.value else {
                    return Err(TyperError::message("Holder::items: not a tuple holder"));
                };
                Ok(items
                    .iter()
                    .map(|v| Holder::const_(Constant::new(v.clone())))
                    .collect())
            }
            HolderKind::NewTuple { holders } => Ok(holders.clone()),
            HolderKind::Item { .. } => Err(TyperError::message(
                "Holder::items: ItemHolder does not expose items()",
            )),
        }
    }

    /// RPython `Holder.emit(self, repr, hop)` (callparse.py:78-88).
    /// Per-instance cache keyed by the target repr's pointer identity
    /// so re-emitting the same holder against the same repr returns
    /// the identical `Hlvalue`. Cache lives in `inner._cache` and is
    /// shared across `clone()`s — matches upstream's
    /// `self._cache[repr]` semantics where `self` is the Python
    /// object identity.
    pub fn emit(&self, repr: &dyn Repr, hop: &HighLevelOp) -> Result<Hlvalue, TyperError> {
        let key = repr as *const dyn Repr as *const () as usize;
        if let Some(cached) = self.inner._cache.borrow().get(&key).cloned() {
            return Ok(cached);
        }
        let v = self._emit(repr, hop)?;
        self.inner._cache.borrow_mut().insert(key, v.clone());
        Ok(v)
    }

    /// RPython `VarHolder._emit` (callparse.py:105-106) /
    /// `ConstHolder._emit` (callparse.py:123-124) /
    /// `NewTupleHolder._emit` (callparse.py:145-152) /
    /// `ItemHolder._emit` (callparse.py:160-164).
    fn _emit(&self, repr: &dyn Repr, hop: &HighLevelOp) -> Result<Hlvalue, TyperError> {
        match &self.inner.kind {
            HolderKind::Var { num, .. } => hop.inputarg(repr, *num),
            HolderKind::Const { value } => {
                let c = HighLevelOp::inputconst(repr, &value.value)?;
                Ok(Hlvalue::Constant(c))
            }
            HolderKind::NewTuple { .. } => {
                // upstream callparse.py:145-152 — requires
                // `rtuple::TupleRepr.newtuple(llops, repr, items_v)`.
                // TupleRepr is not yet ported; this arm stays
                // unreachable in practice because callsites only
                // build NewTupleHolder via stararg, and the
                // simple_call / call_args golden paths don't trigger
                // it under the simple-positional + matching-arg-count
                // signatures the present consumers use.
                Err(TyperError::missing_rtype_operation(
                    "NewTupleHolder._emit (callparse.py:145-152) port pending — \
                     blocked on rtuple::TupleRepr",
                ))
            }
            HolderKind::Item { .. } => {
                // upstream callparse.py:160-164 — needs
                // `parent.access(hop)` + `TupleRepr.getitem_internal`
                // + `llops.convertvar`. Same TupleRepr blocker as
                // NewTupleHolder._emit above.
                Err(TyperError::missing_rtype_operation(
                    "ItemHolder._emit (callparse.py:160-164) port pending — \
                     blocked on rtuple::TupleRepr",
                ))
            }
        }
    }
}

/// RPython `class ArgumentsForRtype(ArgumentsForTranslation)`
/// (callparse.py:7-14). Holder-list analogue of the SomeValue-based
/// [`crate::annotator::argument::ArgumentsForTranslation`].
#[derive(Clone, Debug)]
pub struct ArgumentsForRtype {
    pub arguments: Vec<Holder>,
    pub keywords: HashMap<String, Holder>,
    pub stararg: Option<Holder>,
}

impl ArgumentsForRtype {
    /// `ArgumentsForRtype(args_h)` — the simple_call call path.
    pub fn simple(args: Vec<Holder>) -> Self {
        ArgumentsForRtype {
            arguments: args,
            keywords: HashMap::new(),
            stararg: None,
        }
    }

    /// RPython `ArgumentsForTranslation.fromshape(shape, data_w)`
    /// (flowspace/argument.py). Holder-typed mirror.
    pub fn fromshape(shape: &CallShape, data_w: Vec<Holder>) -> Self {
        let shape_cnt = shape.shape_cnt;
        let end_keys = shape_cnt + shape.shape_keys.len();
        let args_w = data_w[..shape_cnt].to_vec();
        let keyword_slice = &data_w[shape_cnt..end_keys];
        let mut keywords: HashMap<String, Holder> = HashMap::new();
        for (name, value) in shape.shape_keys.iter().zip(keyword_slice.iter()) {
            keywords.insert(name.clone(), value.clone());
        }
        let stararg = if shape.shape_star {
            Some(data_w[end_keys].clone())
        } else {
            None
        };
        ArgumentsForRtype {
            arguments: args_w,
            keywords,
            stararg,
        }
    }

    /// RPython `ArgumentsForRtype.newtuple(self, items)` (callparse.py:8-9):
    /// `return NewTupleHolder(items)`.
    pub fn newtuple(items: Vec<Holder>) -> Holder {
        Holder::new_tuple(items)
    }

    /// RPython `ArgumentsForRtype.unpackiterable(self, it)` (callparse.py:11-14):
    ///
    /// ```python
    /// def unpackiterable(self, it):
    ///     assert it.is_tuple()
    ///     items = it.items()
    ///     return list(items)
    /// ```
    pub fn unpackiterable(it: &Holder) -> Result<Vec<Holder>, TyperError> {
        if !it.is_tuple() {
            return Err(TyperError::message(
                "ArgumentsForRtype.unpackiterable: argument is not a tuple holder",
            ));
        }
        it.items()
    }

    /// RPython `positional_args` property (argument.py:8-14). Combines
    /// `self.arguments` with any unpacked `*args` holder.
    fn positional_args(&self) -> Result<Vec<Holder>, TyperError> {
        if let Some(stararg) = &self.stararg {
            let mut out = self.arguments.clone();
            out.extend(Self::unpackiterable(stararg)?);
            Ok(out)
        } else {
            Ok(self.arguments.clone())
        }
    }

    /// RPython `match_signature(signature, defaults_w)`
    /// (argument.py:43-120) — `_match_signature` order:
    ///
    /// 1. `scope_w[:take] = args_w[:take]`, `input_argcount = take`.
    /// 2. If `has_vararg`: `scope_w[co_argcount] = newtuple(extras)`.
    /// 3. Keyword loop — conflict detect, build `kwds_mapping`.
    /// 4. Missing-fill loop `for i in range(input_argcount, co_argcount)`:
    ///    keyword first, then default, else `missing += 1`.
    pub fn match_signature(
        &self,
        signature: &Signature,
        defaults_w: &[Holder],
    ) -> Result<Vec<Holder>, MatchSignatureError> {
        let co_argcount = signature.num_argnames();
        let scopelen = signature.scope_length();
        let mut scope_w: Vec<Option<Holder>> = vec![None; scopelen];

        // upstream argument.py:52-55 — positional + keyword counts.
        let args_w = self.positional_args().map_err(MatchSignatureError::Typer)?;
        let num_args = args_w.len();
        let num_kwds = self.keywords.len();

        // upstream argument.py:57-60 — scope_w[:take] = args_w[:take].
        let take = num_args.min(co_argcount);
        for (slot, value) in scope_w.iter_mut().zip(args_w.iter()).take(take) {
            *slot = Some(value.clone());
        }
        let input_argcount = take;

        // upstream argument.py:62-70 — stararg collection.
        if signature.has_vararg() {
            let stararg_items: Vec<Holder> = if num_args > co_argcount {
                args_w[co_argcount..].to_vec()
            } else {
                Vec::new()
            };
            scope_w[co_argcount] = Some(Self::newtuple(stararg_items));
        } else if num_args > co_argcount {
            return Err(MatchSignatureError::Arg(ArgErr::Count {
                signature: signature.clone(),
                num_defaults: defaults_w.len(),
                missing_args: 0,
                num_args,
                num_kwds,
            }));
        }

        // upstream argument.py:72 — assert not signature.has_kwarg().
        if signature.has_kwarg() {
            return Err(MatchSignatureError::Typer(TyperError::message(
                "callparse: **kwargs signatures not supported (upstream asserts this)",
            )));
        }

        // upstream argument.py:74-101 — keyword conflict detection.
        let mut remaining_kwds: Vec<String> = Vec::new();
        if num_kwds > 0 {
            for name in self.keywords.keys() {
                let j = signature.argnames.iter().position(|n| n == name);
                match j {
                    Some(idx) if idx < input_argcount => {
                        return Err(MatchSignatureError::Arg(ArgErr::MultipleValues {
                            argname: name.clone(),
                        }));
                    }
                    Some(_) => {
                        // keyword matches a slot >= input_argcount;
                        // handled by the fill loop below.
                    }
                    None => {
                        remaining_kwds.push(name.clone());
                    }
                }
            }
            if !remaining_kwds.is_empty() {
                // upstream argument.py:97-101 — unknown kwargs (no
                // **kwargs sink). The `co_argcount == 0` special case
                // (argument.py:98-99) raises `ArgErrCount` instead, so
                // that "no positional slots, but caller passed kwargs"
                // surfaces as a count mismatch rather than UnknownKwds.
                if co_argcount == 0 {
                    return Err(MatchSignatureError::Arg(ArgErr::Count {
                        signature: signature.clone(),
                        num_defaults: defaults_w.len(),
                        missing_args: 0,
                        num_args,
                        num_kwds,
                    }));
                }
                return Err(MatchSignatureError::Arg(ArgErr::UnknownKwds {
                    num_kwds: remaining_kwds.len(),
                    kwd_name: remaining_kwds.into_iter().next().unwrap(),
                }));
            }
        }

        // upstream argument.py:103-120 — missing-fill loop.
        let mut missing = 0usize;
        if input_argcount < co_argcount {
            let def_first = co_argcount - defaults_w.len();
            for i in input_argcount..co_argcount {
                let name = &signature.argnames[i];
                if let Some(kwd) = self.keywords.get(name) {
                    scope_w[i] = Some(kwd.clone());
                    continue;
                }
                if i >= def_first {
                    let defnum = i - def_first;
                    scope_w[i] = Some(defaults_w[defnum].clone());
                } else {
                    missing += 1;
                }
            }
            if missing > 0 {
                return Err(MatchSignatureError::Arg(ArgErr::Count {
                    signature: signature.clone(),
                    num_defaults: defaults_w.len(),
                    missing_args: missing,
                    num_args,
                    num_kwds,
                }));
            }
        }

        scope_w
            .into_iter()
            .enumerate()
            .map(|(i, slot)| {
                slot.ok_or_else(|| {
                    MatchSignatureError::Typer(TyperError::message(format!(
                        "match_signature: scope_w[{i}] is unfilled (unreachable)",
                    )))
                })
            })
            .collect()
    }
}

/// Wrapper around `ArgErr` that also carries `TyperError` escape paths
/// (unpackiterable failures, panic-style asserts). Upstream raises
/// `ArgErr` and `TyperError` through separate code paths; Rust
/// merges them here so `match_signature` stays a single-return fn.
#[derive(Debug)]
pub enum MatchSignatureError {
    Arg(ArgErr),
    Typer(TyperError),
}

impl std::fmt::Display for MatchSignatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchSignatureError::Arg(e) => f.write_str(&e.getmsg()),
            MatchSignatureError::Typer(e) => write!(f, "{e}"),
        }
    }
}

/// RPython `callparse(rtyper, graph, hop, r_self=None)` (callparse.py:34-70).
pub fn callparse(
    rtyper: &Rc<RPythonTyper>,
    graph: &Rc<PyGraph>,
    hop: &HighLevelOp,
    r_self: Option<Arc<dyn Repr>>,
) -> Result<Vec<Hlvalue>, TyperError> {
    let mut rinputs = getrinputs(rtyper, graph)?;

    // upstream callparse.py:41-45 — `rinputs[0] = r_self` then start = 0.
    let start = match r_self {
        Some(rs) => {
            if rinputs.is_empty() {
                return Err(TyperError::message(
                    "callparse: r_self is set but graph has no formal arg slot",
                ));
            }
            rinputs[0] = rs;
            0usize
        }
        None => 1usize,
    };

    // upstream callparse.py:38-40 — `args_h(start) = [VarHolder(i,
    // hop.args_s[i]) for i in range(start, hop.nb_args)]`.
    let args_h_from = |base_start: usize| -> Vec<Holder> {
        (base_start..hop.nb_args())
            .map(|i| Holder::var(i, hop.args_s.borrow()[i].clone()))
            .collect()
    };

    // upstream callparse.py:46-52 — opname dispatch.
    let arguments = match hop.spaceop.opname.as_str() {
        "simple_call" => ArgumentsForRtype::simple(args_h_from(start)),
        "call_args" => {
            // `hop.args_s[start].const` carries the CallShape tuple
            // encoded by flowcontext::build_call_shape_constant.
            let shape_const = hop
                .args_s
                .borrow()
                .get(start)
                .and_then(|sv| sv.const_().cloned())
                .ok_or_else(|| {
                    TyperError::message("callparse: call_args shape slot is not a Constant")
                })?;
            let shape = call_shape_from_const(&shape_const).map_err(TyperError::message)?;
            // `args_h(start+1)` — the shape slot lives at args[start].
            let data_h = args_h_from(start + 1);
            ArgumentsForRtype::fromshape(&shape, data_h)
        }
        other => {
            return Err(TyperError::message(format!(
                "callparse: unsupported call opname {other:?}"
            )));
        }
    };

    // upstream callparse.py:54-58 — gather defaults as ConstHolders.
    let defs_h: Vec<Holder> = match graph.defaults.borrow().as_ref() {
        None => Vec::new(),
        Some(defs) => defs.iter().map(|c| Holder::const_(c.clone())).collect(),
    };

    // upstream callparse.py:59-63 — match_signature; TyperError wrap on ArgErr.
    let holders = arguments
        .match_signature(&graph.signature.borrow(), &defs_h)
        .map_err(|e| {
            TyperError::message(format!(
                "callparse: signature mismatch for {}: {}",
                graph.graph.borrow().name,
                e
            ))
        })?;

    // upstream callparse.py:65 — `assert len(holders) == len(rinputs)`.
    if holders.len() != rinputs.len() {
        return Err(TyperError::message(format!(
            "callparse: holder count ({}) != rinputs count ({})",
            holders.len(),
            rinputs.len()
        )));
    }

    // upstream callparse.py:66-70 — `vlist = [h.emit(r, hop) for ...]`.
    let mut vlist = Vec::with_capacity(holders.len());
    for (h, r) in holders.iter().zip(rinputs.iter()) {
        vlist.push(h.emit(r.as_ref(), hop)?);
    }
    Ok(vlist)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn const_holder(value: i64) -> Holder {
        Holder::const_(Constant::new(ConstValue::Int(value)))
    }

    fn var_holder(num: usize) -> Holder {
        Holder::var(num, SomeValue::Impossible)
    }

    #[test]
    fn match_signature_simple_positional_maps_each_arg_to_signature_slot() {
        let sig = Signature::new(vec!["a".into(), "b".into()], None, None);
        let args = ArgumentsForRtype::simple(vec![var_holder(1), var_holder(2)]);
        let holders = args.match_signature(&sig, &[]).unwrap();
        assert_eq!(holders.len(), 2);
        assert!(matches!(holders[0].kind(), HolderKind::Var { num: 1, .. }));
        assert!(matches!(holders[1].kind(), HolderKind::Var { num: 2, .. }));
    }

    #[test]
    fn match_signature_fills_trailing_defaults_when_caller_omits_tail() {
        // upstream argument.py:106-120 — def_first = co_argcount -
        // len(defaults); trailing slots pick from defaults_w[i - def_first].
        let sig = Signature::new(vec!["a".into(), "b".into(), "c".into()], None, None);
        let args = ArgumentsForRtype::simple(vec![var_holder(1)]);
        let defs = vec![const_holder(10), const_holder(20)];
        let holders = args.match_signature(&sig, &defs).unwrap();
        assert_eq!(holders.len(), 3);
        assert!(matches!(holders[0].kind(), HolderKind::Var { num: 1, .. }));
        match &holders[1].kind() {
            HolderKind::Const { value } => assert_eq!(value.value, ConstValue::Int(10)),
            other => panic!("expected Const, got {other:?}"),
        }
        match &holders[2].kind() {
            HolderKind::Const { value } => assert_eq!(value.value, ConstValue::Int(20)),
            other => panic!("expected Const, got {other:?}"),
        }
    }

    #[test]
    fn match_signature_errors_on_too_many_positional_args_without_vararg() {
        let sig = Signature::new(vec!["a".into()], None, None);
        let args = ArgumentsForRtype::simple(vec![var_holder(1), var_holder(2)]);
        let err = args.match_signature(&sig, &[]).unwrap_err();
        assert!(matches!(
            err,
            MatchSignatureError::Arg(ArgErr::Count { .. })
        ));
    }

    #[test]
    fn match_signature_errors_on_missing_positional_without_default() {
        let sig = Signature::new(vec!["a".into(), "b".into()], None, None);
        let args = ArgumentsForRtype::simple(vec![var_holder(1)]);
        let err = args.match_signature(&sig, &[]).unwrap_err();
        assert!(matches!(
            err,
            MatchSignatureError::Arg(ArgErr::Count { .. })
        ));
    }

    #[test]
    fn match_signature_routes_keyword_to_named_slot() {
        let sig = Signature::new(vec!["a".into(), "b".into(), "c".into()], None, None);
        let mut args = ArgumentsForRtype::simple(vec![var_holder(1), var_holder(2)]);
        args.keywords.insert("c".into(), const_holder(42));
        let holders = args.match_signature(&sig, &[]).unwrap();
        match &holders[2].kind() {
            HolderKind::Const { value } => assert_eq!(value.value, ConstValue::Int(42)),
            other => panic!("expected Const, got {other:?}"),
        }
    }

    #[test]
    fn match_signature_errors_on_keyword_collision_with_positional() {
        let sig = Signature::new(vec!["a".into()], None, None);
        let mut args = ArgumentsForRtype::simple(vec![var_holder(1)]);
        args.keywords.insert("a".into(), const_holder(2));
        let err = args.match_signature(&sig, &[]).unwrap_err();
        assert!(matches!(
            err,
            MatchSignatureError::Arg(ArgErr::MultipleValues { .. })
        ));
    }

    #[test]
    fn match_signature_vararg_without_extras_binds_empty_tuple_holder() {
        // upstream argument.py:62-68 — has_vararg + num_args <= co_argcount
        // ⇒ starargs_w = [] ⇒ scope_w[co_argcount] = newtuple([]).
        let sig = Signature::new(vec!["a".into()], Some("args".into()), None);
        let args = ArgumentsForRtype::simple(vec![var_holder(1)]);
        let holders = args.match_signature(&sig, &[]).unwrap();
        // Two slots: argname 'a', then the vararg sink.
        assert_eq!(holders.len(), 2);
        assert!(matches!(holders[0].kind(), HolderKind::Var { num: 1, .. }));
        let HolderKind::NewTuple { holders: inner } = &holders[1].kind() else {
            panic!("expected NewTupleHolder, got {:?}", holders[1].kind());
        };
        assert!(inner.is_empty(), "empty vararg tuple holder");
    }

    #[test]
    fn match_signature_keyword_default_interleave_prefers_keyword_over_default() {
        // keyword 'b' provided, default only for 'b' exists. Fill loop
        // must pick the keyword, not the default.
        let sig = Signature::new(vec!["a".into(), "b".into()], None, None);
        let mut args = ArgumentsForRtype::simple(vec![var_holder(1)]);
        args.keywords.insert("b".into(), const_holder(77));
        let defs = vec![const_holder(999)]; // would fill 'b' if keyword wasn't there
        let holders = args.match_signature(&sig, &defs).unwrap();
        match &holders[1].kind() {
            HolderKind::Const { value } => assert_eq!(value.value, ConstValue::Int(77)),
            other => panic!("expected keyword const 77, got {other:?}"),
        }
    }

    #[test]
    fn match_signature_zero_co_argcount_with_unknown_kwd_raises_count_not_unknownkwds() {
        // upstream argument.py:97-101 — when the signature accepts no
        // positional slots (`co_argcount == 0`) but the caller passes a
        // keyword that has no matching slot, raise ArgErrCount instead
        // of ArgErrUnknownKwds. Bookkeeper-side ArgumentsForTranslation
        // already matches this; the new callparse.rs Holder path must
        // too.
        let sig = Signature::new(vec![], None, None);
        let mut args = ArgumentsForRtype::simple(vec![]);
        args.keywords.insert("anything".into(), const_holder(1));
        let err = args.match_signature(&sig, &[]).unwrap_err();
        match err {
            MatchSignatureError::Arg(ArgErr::Count { .. }) => {}
            other => panic!("expected ArgErr::Count, got {other:?}"),
        }
    }

    #[test]
    fn match_signature_nonzero_co_argcount_with_unknown_kwd_still_raises_unknownkwds() {
        // Sanity check the inverse: when co_argcount > 0, an unknown
        // keyword must surface as ArgErr::UnknownKwds (upstream
        // argument.py:100-101).
        let sig = Signature::new(vec!["a".into()], None, None);
        let mut args = ArgumentsForRtype::simple(vec![var_holder(1)]);
        args.keywords.insert("zzz".into(), const_holder(2));
        let err = args.match_signature(&sig, &[]).unwrap_err();
        assert!(matches!(
            err,
            MatchSignatureError::Arg(ArgErr::UnknownKwds { .. })
        ));
    }

    fn tuple_var_holder() -> Holder {
        use crate::annotator::model::SomeTuple;
        // Build a SomeValue::Tuple of length 2 so `items()` materialises
        // two ItemHolders pointing at the same parent.
        let s_obj = SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Impossible,
            SomeValue::Impossible,
        ]));
        Holder::var(0, s_obj)
    }

    #[test]
    fn new_tuple_passthrough_returns_parent_when_holders_unpack_full_tuple() {
        // upstream callparse.py:127-137 — NewTupleHolder.__new__
        // returns the parent holder when every input is an ItemHolder
        // pointing at the same parent AND the count covers the
        // parent's full item list.
        let parent = tuple_var_holder();
        let items = parent.items().expect("VarHolder of SomeTuple yields items");
        assert_eq!(items.len(), 2);

        let nt = Holder::new_tuple(items);
        // The passthrough must return parent identity (Rc::ptr_eq).
        assert!(
            Holder::ptr_eq(&nt, &parent),
            "passthrough should return the parent holder identity"
        );
    }

    #[test]
    fn new_tuple_no_passthrough_when_count_mismatch() {
        // Partial unpack — only one item out of two — must fall
        // through to the NewTupleHolder construction (no passthrough).
        let parent = tuple_var_holder();
        let mut items = parent.items().unwrap();
        items.pop(); // drop second item, length now 1
        let nt = Holder::new_tuple(items);
        assert!(matches!(nt.kind(), HolderKind::NewTuple { holders } if holders.len() == 1));
        assert!(!Holder::ptr_eq(&nt, &parent));
    }

    #[test]
    fn new_tuple_no_passthrough_when_parent_identities_differ() {
        // Two ItemHolders pointing at distinct parents must not
        // collapse — upstream's `h.holder == holders[0].holder`
        // check is Python-identity, mirrored here via Holder::ptr_eq.
        let parent_a = tuple_var_holder();
        let parent_b = tuple_var_holder();
        assert!(!Holder::ptr_eq(&parent_a, &parent_b));
        let items_a = parent_a.items().unwrap();
        let items_b = parent_b.items().unwrap();
        // Mix one item from A with one from B.
        let mixed = vec![items_a[0].clone(), items_b[0].clone()];
        let nt = Holder::new_tuple(mixed);
        assert!(matches!(nt.kind(), HolderKind::NewTuple { .. }));
    }

    #[test]
    fn fromshape_decodes_call_args_triple_into_arguments_keywords_stararg() {
        let shape = CallShape {
            shape_cnt: 1,
            shape_keys: vec!["name".into()],
            shape_star: false,
        };
        let args = ArgumentsForRtype::fromshape(&shape, vec![var_holder(1), const_holder(7)]);
        assert_eq!(args.arguments.len(), 1);
        assert!(args.keywords.contains_key("name"));
        assert!(args.stararg.is_none());
    }

    #[test]
    fn callparse_r_self_override_replaces_rinputs_slot_zero() {
        // upstream callparse.py:41-45 — `r_self is not None` path:
        // `start = 0`, `rinputs[0] = r_self`. The caller passes the
        // method receiver's repr directly; our port must NOT fall
        // back to `hop.args_r[0]`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::description::{DescEntry, FunctionDesc, GraphCacheKey};
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable as FlowVariable};
        use crate::translator::rtyper::rmodel::ReprState;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::Cell;
        use std::cell::RefCell as StdRef;
        use std::rc::Rc;

        #[derive(Debug)]
        struct TaggedRepr {
            tag: &'static str,
            lltype: LowLevelType,
            state: ReprState,
        }
        impl Repr for TaggedRepr {
            fn lowleveltype(&self) -> &LowLevelType {
                &self.lltype
            }
            fn state(&self) -> &ReprState {
                &self.state
            }
            fn class_name(&self) -> &'static str {
                self.tag
            }
            fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
                crate::translator::rtyper::pairtype::ReprClassId::Repr
            }
            fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
                Ok(Constant::with_concretetype(
                    value.clone(),
                    self.lltype.clone(),
                ))
            }
        }
        fn tagged(tag: &'static str, lltype: LowLevelType) -> Arc<dyn Repr> {
            Arc::new(TaggedRepr {
                tag,
                lltype,
                state: ReprState::new(),
            })
        }

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));

        // Build a two-arg graph whose first arg is int-annotated.
        let sig = Signature::new(vec!["self".into(), "x".into()], None, None);
        let fd = Rc::new(StdRef::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "method",
            sig.clone(),
            None,
            None,
        )));
        let mut arg_self = FlowVariable::named("self");
        arg_self
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::default()))));
        let mut arg_x = FlowVariable::named("x");
        arg_x
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::default()))));
        let startblock = crate::flowspace::model::Block::shared(vec![
            Hlvalue::Variable(arg_self),
            Hlvalue::Variable(arg_x),
        ]);
        let mut ret_var = FlowVariable::new();
        ret_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::default()))));
        let graph = crate::flowspace::model::FunctionGraph::with_return_var(
            "method",
            startblock.clone(),
            Hlvalue::Variable(ret_var),
        );
        let pygraph = Rc::new(PyGraph {
            graph: Rc::new(StdRef::new(graph)),
            func: crate::flowspace::model::GraphFunc::new(
                "method",
                Constant::new(ConstValue::Dict(Default::default())),
            ),
            signature: StdRef::new(sig),
            defaults: StdRef::new(None),
            access_directly: Cell::new(false),
        });
        fd.borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, pygraph.clone());

        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            Vec::new(),
            Hlvalue::Variable(FlowVariable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Make arg[0] a Constant so inputarg routes through
        // inputconst(converted_to_repr, value) — that path carries
        // the converted_to repr's lowleveltype into the emitted
        // Constant, so we can assert downstream which repr was used.
        let hop_arg0 = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(42),
            LowLevelType::Signed,
        ));
        let hop_arg1 = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(99),
            LowLevelType::Signed,
        ));
        hop.args_v
            .borrow_mut()
            .extend([hop_arg0.clone(), hop_arg1.clone()]);
        let mut args_s0 = SomeInteger::default();
        args_s0.base.const_box = Some(Constant::new(ConstValue::Int(42)));
        let mut args_s1 = SomeInteger::default();
        args_s1.base.const_box = Some(Constant::new(ConstValue::Int(99)));
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::Integer(args_s0), SomeValue::Integer(args_s1)]);
        // hop.args_r[0] is a wrong-lowleveltype repr (Bool). If the
        // port falls back to hop.args_r[0], it would materialise a
        // Bool-typed constant — distinct from the Signed-typed
        // constant r_self demands.
        let hop_r0_wrong = tagged("HopArg0Repr_Bool", LowLevelType::Bool);
        let hop_r1 = tagged("HopArg1Repr", LowLevelType::Signed);
        hop.args_r
            .borrow_mut()
            .extend([Some(hop_r0_wrong.clone()), Some(hop_r1.clone())]);
        *hop.r_result.borrow_mut() = Some(hop_r1.clone());

        // r_self path: pass an explicit Signed-typed receiver. The
        // emitted vlist[0] must carry Signed — proving callparse
        // wrote r_self into rinputs[0] rather than reading
        // hop.args_r[0].
        let r_self = tagged("SelfRepr_Signed", LowLevelType::Signed);
        let vlist = super::callparse(&rtyper, &pygraph, &hop, Some(r_self.clone())).unwrap();
        assert_eq!(vlist.len(), 2);
        match &vlist[0] {
            Hlvalue::Constant(c) => {
                assert_eq!(
                    c.concretetype,
                    Some(LowLevelType::Signed),
                    "r_self path must route through the Signed-typed receiver repr, \
                     not hop.args_r[0] (Bool)"
                );
            }
            other => panic!("expected Constant via inputconst(r_self), got {other:?}"),
        }
    }

    #[test]
    fn holder_emit_caches_result_per_target_repr() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                Vec::new(),
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops,
        );

        let r = rtyper.getprimitiverepr(&LowLevelType::Signed).unwrap();
        let h = Holder::const_(Constant::new(ConstValue::Int(5)));
        let first = h.emit(r.as_ref(), &hop).unwrap();
        let second = h.emit(r.as_ref(), &hop).unwrap();
        assert_eq!(first, second);
    }
}

/// Mirror of [`crate::annotator::bookkeeper::call_shape_from_const`],
/// duplicated here because that helper is private to bookkeeper.
fn call_shape_from_const(cv: &ConstValue) -> Result<CallShape, String> {
    let items = match cv {
        ConstValue::Tuple(items) => items,
        _ => {
            return Err(format!(
                "callparse: CallShape constant must be Tuple, got {cv:?}"
            ));
        }
    };
    if items.len() != 3 {
        return Err(format!(
            "callparse: CallShape tuple must have 3 slots, got {}",
            items.len()
        ));
    }
    let shape_cnt = match &items[0] {
        ConstValue::Int(n) => *n as usize,
        other => {
            return Err(format!(
                "callparse: CallShape[0] must be Int, got {other:?}"
            ));
        }
    };
    let shape_keys = match &items[1] {
        ConstValue::Tuple(keys) => keys
            .iter()
            .map(|k| match k {
                ConstValue::Str(s) => Ok(s.clone()),
                other => Err(format!(
                    "callparse: CallShape keyword slot must be Str, got {other:?}"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?,
        other => {
            return Err(format!(
                "callparse: CallShape[1] must be Tuple, got {other:?}"
            ));
        }
    };
    let shape_star = match &items[2] {
        ConstValue::Bool(b) => *b,
        other => {
            return Err(format!(
                "callparse: CallShape[2] must be Bool, got {other:?}"
            ));
        }
    };
    Ok(CallShape {
        shape_cnt,
        shape_keys,
        shape_star,
    })
}
