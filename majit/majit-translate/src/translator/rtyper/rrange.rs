//! RPython `rpython/rtyper/rrange.py` + `lltypesystem/rrange.py` —
//! minimal `AbstractRangeRepr` slice covering the `len(range(...))` lowering for
//! the step-1 case (`range(n)` / `range(a, b)`), which is the form
//! `builtin_range` mints for the overwhelmingly common call shapes
//! (`annotator/builtin.rs:523-528`).
//!
//! A `range()` result that is never mutated annotates as a `SomeList`
//! carrying a non-`None` `range_step` (`annotator/listdef.rs:177`); its
//! repr is NOT array-backed (`FixedSizeListRepr`) but an immutable
//! `GcStruct("range", start, stop)` (`lltypesystem/rrange.py:51-57`).
//!
//! Landed: `rtype_len` (`step == 1`), `rtype_getitem` (constant-step,
//! nonneg, `dum_nocheck`), and constant-step `RangeIteratorRepr`
//! (`make_iterator_repr` / `newiter` / `rtype_next` via the `ll_rangeiter`
//! + `ll_rangenext_up` / `_down` helper graphs).
//!
//! Deferred to follow-on slices (matching how `FixedSizeListRepr` landed
//! `rtype_len` first), all blocked on `_ll_rangelen`'s `int_floordiv`
//! (not yet a recognised low-level op) or the variable-step `RANGEST`
//! path: the general-step `ll_rangelen` length, the `rtype_getitem`
//! `checkidx` (implicit-IndexError) and negative-index (`ll_rangeitem`)
//! branches, the `RANGEST` variable-step `_getstep` getitem, and the
//! variable-step `RANGESTITER` iterator (`ll_rangenext_updown`).

#![allow(non_camel_case_types)]

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr, PtrTarget, Struct};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, HighLevelOp, constant_with_lltype, exception_args, helper_pygraph_from_graph,
    variable_with_lltype,
};
use std::sync::Arc;

fn rrange_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!("rrange.{name} helper surface deferred"))
}

/// RPython `_ll_rangelen(start, stop, step)`.
fn _ll_rangelen(start: i64, stop: i64, step: i64) -> i64 {
    let mut result = if step > 0 {
        (stop - start + (step - 1)) / step
    } else {
        (start - stop - (step + 1)) / (-step)
    };
    if result < 0 {
        result = 0;
    }
    result
}

/// RPython `ll_rangelen(l, step)` for any carrier exposing start/stop fields.
pub fn ll_rangelen(start: i64, stop: i64, step: i64) -> i64 {
    _ll_rangelen(start, stop, step)
}

/// RPython `ll_rangelen1(l)` for any carrier exposing start/stop fields.
pub fn ll_rangelen1(start: i64, stop: i64) -> i64 {
    (stop - start).max(0)
}

/// RPython `ll_rangeitem_nonneg(func, l, index, step)` in the `dum_nocheck`
/// case; the checkidx branch is represented by `checked=true`.
pub fn ll_rangeitem_nonneg(
    start: i64,
    stop: i64,
    index: i64,
    step: i64,
    checked: bool,
) -> Result<i64, TyperError> {
    if checked && index >= _ll_rangelen(start, stop, step) {
        return Err(TyperError::message("IndexError"));
    }
    Ok(start + index * step)
}

/// RPython `ll_rangeitem(func, l, index, step)`.
pub fn ll_rangeitem(
    start: i64,
    stop: i64,
    mut index: i64,
    step: i64,
    checked: bool,
) -> Result<i64, TyperError> {
    if checked {
        let length = _ll_rangelen(start, stop, step);
        if index < 0 {
            index += length;
        }
        if index < 0 || index >= length {
            return Err(TyperError::message("IndexError"));
        }
    } else if index < 0 {
        index += _ll_rangelen(start, stop, step);
    }
    Ok(start + index * step)
}

/// RPython `rtype_builtin_range(hop)`.
pub fn rtype_builtin_range(_hop: &HighLevelOp) -> RTypeResult {
    Err(rrange_deferred("rtype_builtin_range"))
}

/// RPython `rtype_builtin_xrange = rtype_builtin_range`.
pub use rtype_builtin_range as rtype_builtin_xrange;

/// RPython `ll_range2list(LIST, start, stop, step)`; the Rust carrier is the
/// immutable integer payload the helper would write through `ll_setitem_fast`.
pub fn ll_range2list(start: i64, stop: i64, step: i64) -> Result<Vec<i64>, TyperError> {
    if step == 0 {
        return Err(TyperError::message("ValueError"));
    }
    let length = _ll_rangelen(start, stop, step);
    let mut out = Vec::with_capacity(length as usize);
    let mut value = start;
    for _ in 0..length {
        out.push(value);
        value += step;
    }
    Ok(out)
}

/// Lightweight carrier for `ll_rangenext_*` tests and deferred iterator reprs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeIter {
    pub next: i64,
    pub stop: i64,
    pub step: i64,
}

/// RPython `class AbstractRangeIteratorRepr(IteratorRepr)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbstractRangeIteratorRepr {
    pub step: i64,
    pub lowleveltype: LowLevelType,
}

/// RPython `ll_rangenext_up(iter, step)`.
pub fn ll_rangenext_up(iter: &mut RangeIter, step: i64) -> Result<i64, TyperError> {
    let next = iter.next;
    if next >= iter.stop {
        return Err(TyperError::message("StopIteration"));
    }
    iter.next = next + step;
    Ok(next)
}

/// RPython `ll_rangenext_down(iter, step)`.
pub fn ll_rangenext_down(iter: &mut RangeIter, step: i64) -> Result<i64, TyperError> {
    let next = iter.next;
    if next <= iter.stop {
        return Err(TyperError::message("StopIteration"));
    }
    iter.next = next + step;
    Ok(next)
}

/// RPython `ll_rangenext_updown(iter)`.
pub fn ll_rangenext_updown(iter: &mut RangeIter) -> Result<i64, TyperError> {
    let step = iter.step;
    if step > 0 {
        ll_rangenext_up(iter, step)
    } else {
        ll_rangenext_down(iter, step)
    }
}

/// RPython `class EnumerateIteratorRepr(IteratorRepr)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumerateIteratorRepr {
    pub const_startindex: Option<i64>,
}

/// RPython `rtype_builtin_enumerate(hop)`.
pub fn rtype_builtin_enumerate(_hop: &HighLevelOp) -> RTypeResult {
    Err(rrange_deferred("rtype_builtin_enumerate"))
}

/// RPython `class AbstractRangeRepr(Repr)` (`rrange.py:10-30`), with the
/// lltypesystem concrete fields stored here until the concrete module grows
/// its own wrapper type:
///
/// ```python
/// class AbstractRangeRepr(Repr):
///     def __init__(self, step):
///         self.step = step
///         if step != 0:
///             self.lowleveltype = self.RANGE
///         else:
///             self.lowleveltype = self.RANGEST
/// ```
///
/// where (`lltypesystem/rrange.py:51-57`):
///
/// ```python
/// self.RANGE = Ptr(GcStruct("range", ("start", Signed), ("stop", Signed),
///                           ..., hints = {'immutable': True}))
/// ```
///
/// The `RANGEST` (variable-step `("start", "stop", "step")`) shape and
/// the `adtmeths` / iterator surface are deferred.
#[derive(Debug)]
pub struct AbstractRangeRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// `self.step` (`rrange.py:12`) — the constant range step. `0`
    /// signals upstream's "variable step" (`RANGEST`).
    step: i64,
}

impl AbstractRangeRepr {
    /// `AbstractRangeRepr.__init__(self, step)` — picks `RANGE`
    /// (constant step) or `RANGEST` (variable step) as the low-level
    /// type. Both are immutable `GcStruct("range", ...)`.
    pub fn new(step: i64) -> Result<Self, TyperError> {
        let signed = LowLevelType::Signed;
        let fields = if step != 0 {
            vec![
                ("start".to_string(), signed.clone()),
                ("stop".to_string(), signed),
            ]
        } else {
            vec![
                ("start".to_string(), signed.clone()),
                ("stop".to_string(), signed.clone()),
                ("step".to_string(), signed),
            ]
        };
        let st = Struct::gc_with_hints(
            "range",
            fields,
            vec![("immutable".to_string(), ConstValue::Bool(true))],
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(st),
        }));
        Ok(AbstractRangeRepr {
            state: ReprState::new(),
            lltype,
            step,
        })
    }
}

impl Repr for AbstractRangeRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "RangeRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::RangeRepr
    }

    /// RPython `AbstractRangeRepr.rtype_len(self, hop)`
    /// (`rrange.py:22-30`):
    ///
    /// ```python
    /// def rtype_len(self, hop):
    ///     v_rng, = hop.inputargs(self)
    ///     if self.step == 1:
    ///         return hop.gendirectcall(ll_rangelen1, v_rng)
    ///     elif self.step != 0:
    ///         v_step = hop.inputconst(Signed, self.step)
    ///     else:
    ///         v_step = self._getstep(v_rng, hop)
    ///     return hop.gendirectcall(ll_rangelen, v_rng, v_step)
    /// ```
    ///
    /// Only the `step == 1` (`ll_rangelen1`) path lands today; the
    /// general `ll_rangelen` (`_ll_rangelen`'s floor-division) is
    /// deferred until `int_floordiv` is a recognised low-level op.
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        if self.step != 1 {
            return Err(TyperError::missing_rtype_operation(format!(
                "AbstractRangeRepr.rtype_len(step={}) — general ll_rangelen path \
                 deferred (needs int_floordiv lowering)",
                self.step
            )));
        }
        let v_rng = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_rangelen1".to_string(),
            vec![ptr_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_rangelen1_helper_graph("ll_rangelen1", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, v_rng)
    }

    /// RPython `pair(AbstractRangeRepr, IntegerRepr).rtype_getitem`
    /// (`rrange.py:34-50`):
    ///
    /// ```python
    /// def rtype_getitem((r_rng, r_int), hop):
    ///     if hop.has_implicit_exception(IndexError):
    ///         spec = dum_checkidx
    ///     else:
    ///         spec = dum_nocheck
    ///     v_func = hop.inputconst(Void, spec)
    ///     v_lst, v_index = hop.inputargs(r_rng, Signed)
    ///     if r_rng.step != 0:
    ///         cstep = hop.inputconst(Signed, r_rng.step)
    ///     else:
    ///         cstep = r_rng._getstep(v_lst, hop)
    ///     if hop.args_s[1].nonneg:
    ///         llfn = ll_rangeitem_nonneg
    ///     else:
    ///         llfn = ll_rangeitem
    ///     hop.exception_is_here()
    ///     return hop.gendirectcall(llfn, v_func, v_lst, v_index, cstep)
    /// ```
    ///
    /// The constant-step (`step != 0`) + nonneg + `dum_nocheck` fast path:
    /// `ll_rangeitem_nonneg(dum_nocheck, l, index, step)` collapses (no
    /// IndexError branch) to `l.start + index * step` (`rrange.py:74-77`).
    /// Unlike `rtype_len`, the formula is `int_mul` + `int_add` with no
    /// floor-division, so any constant step lowers here. `step` is baked
    /// as the `inputconst(Signed, self.step)` runtime arg. The `checkidx`
    /// (implicit-IndexError, needs `_ll_rangelen` floor-division), the
    /// negative-index (`ll_rangeitem`), and the variable-step `RANGEST`
    /// (`step == 0`, needs `_getstep`) branches surface a `TyperError`
    /// until those land.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::SomeValue;
        if hop.has_implicit_exception("IndexError") {
            return Err(TyperError::message(
                "AbstractRangeRepr.rtype_getitem: checkidx IndexError branch not yet ported",
            ));
        }
        if self.step == 0 {
            return Err(TyperError::message(
                "AbstractRangeRepr.rtype_getitem: variable-step RANGEST (_getstep) not yet ported",
            ));
        }
        let s1 = hop.args_s.borrow().get(1).cloned().ok_or_else(|| {
            TyperError::message("AbstractRangeRepr.rtype_getitem: args_s[1] missing")
        })?;
        let nonneg = match &s1 {
            SomeValue::Integer(i) => i.nonneg,
            other => {
                return Err(TyperError::message(format!(
                    "AbstractRangeRepr.rtype_getitem: args_s[1] must be SomeInteger, got {other:?}"
                )));
            }
        };
        if !nonneg {
            return Err(TyperError::message(
                "AbstractRangeRepr.rtype_getitem: negative-index ll_rangeitem branch not yet ported",
            ));
        }
        let mut args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ])?;
        args.push(constant_with_lltype(
            ConstValue::Int(self.step),
            LowLevelType::Signed,
        ));
        hop.exception_is_here()?;
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_rangeitem_nonneg".to_string(),
            vec![ptr_lltype, LowLevelType::Signed, LowLevelType::Signed],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_rangeitem_nonneg_helper_graph(
                    "ll_rangeitem_nonneg",
                    ptr_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, args)
    }

    /// RPython `RangeRepr.make_iterator_repr(self, variant=None)`
    /// (`lltypesystem/rrange.py:63-67`):
    ///
    /// ```python
    /// def make_iterator_repr(self, variant=None):
    ///     if variant is not None:
    ///         raise TyperError("unsupported %r iterator over a range list" %
    ///                          (variant,))
    ///     return RangeIteratorRepr(self)
    /// ```
    ///
    /// The constant-step (`step != 0`) iterator lands; the variable-step
    /// `RANGESTITER` iterator (`step == 0`) is deferred inside
    /// [`RangeIteratorRepr::new`].
    fn make_iterator_repr(&self, variant: &[String]) -> Result<Arc<dyn Repr>, TyperError> {
        if !variant.is_empty() {
            return Err(TyperError::message(format!(
                "unsupported {variant:?} iterator over a range list"
            )));
        }
        Ok(Arc::new(RangeIteratorRepr::new(self)?))
    }
}

/// Synthesise the `ll_rangelen1` helper graph (`rrange.py:68-72`):
///
/// ```python
/// def ll_rangelen1(l):
///     result = l.stop - l.start
///     if result < 0:
///         result = 0
///     return result
/// ```
///
/// 2-block CFG:
/// - **start**: `start = getfield(l, 'start'); stop = getfield(l, 'stop');
///   result = int_sub(stop, start); neg = int_lt(result, 0)`. Switch on
///   `neg`: True → returnblock with const `0`; False → returnblock with
///   `result`.
pub(crate) fn build_ll_rangelen1_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("l", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);

    // start block: result = l.stop - l.start; neg = result < 0.
    let v_start = variable_with_lltype("start", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg.clone()), field_const("start")],
        Hlvalue::Variable(v_start.clone()),
    ));
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg), field_const("stop")],
        Hlvalue::Variable(v_stop.clone()),
    ));
    let v_result = variable_with_lltype("result", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(v_stop), Hlvalue::Variable(v_start)],
        Hlvalue::Variable(v_result.clone()),
    ));
    let v_neg = variable_with_lltype("neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(v_result.clone()), signed_const(0)],
        Hlvalue::Variable(v_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_neg));

    // True (result < 0): clamp to 0.
    let true_link = Link::new(
        vec![signed_const(0)],
        Some(graph.returnblock.clone()),
        Some(bool_const(true)),
    )
    .into_ref();
    // False: return result unchanged.
    let false_link = Link::new(
        vec![Hlvalue::Variable(v_result)],
        Some(graph.returnblock.clone()),
        Some(bool_const(false)),
    )
    .into_ref();
    startblock.closeblock(vec![true_link, false_link]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

/// Synthesise the `ll_rangeitem_nonneg` helper graph (`rrange.py:74-77`)
/// for the `dum_nocheck` case (the `dum_checkidx` IndexError branch is
/// folded out):
///
/// ```python
/// def ll_rangeitem_nonneg(func, l, index, step):
///     ...
///     return l.start + index * step
/// ```
///
/// Single block: `start = getfield(l, 'start'); prod = int_mul(index,
/// step); result = int_add(start, prod)`. `step` is a runtime arg (the
/// `inputconst(Signed, self.step)` the caller appends).
pub(crate) fn build_ll_rangeitem_nonneg_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let step_arg = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
        Hlvalue::Variable(step_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);

    let v_start = variable_with_lltype("start", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg), field_const("start")],
        Hlvalue::Variable(v_start.clone()),
    ));
    let v_prod = variable_with_lltype("prod", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_mul",
        vec![Hlvalue::Variable(index_arg), Hlvalue::Variable(step_arg)],
        Hlvalue::Variable(v_prod.clone()),
    ));
    let v_result = variable_with_lltype("result", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(v_start), Hlvalue::Variable(v_prod)],
        Hlvalue::Variable(v_result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string(), "step".to_string()],
        func,
    ))
}

/// RPython `class RangeIteratorRepr(AbstractRangeIteratorRepr)`
/// (`rrange.py:145-156` + `lltypesystem/rrange.py:85-97`), constant-step
/// (`step != 0`, `RANGEITER`) slice.
///
/// ```python
/// class AbstractRangeIteratorRepr(IteratorRepr):
///     def __init__(self, r_rng):
///         self.r_rng = r_rng
///         if r_rng.step != 0:
///             self.lowleveltype = r_rng.RANGEITER
///         else:
///             self.lowleveltype = r_rng.RANGESTITER
/// ```
///
/// where (`lltypesystem/rrange.py:58`) `RANGEITER = Ptr(GcStruct("range",
/// ("next", Signed), ("stop", Signed)))`. Like
/// [`super::rtuple::Length1TupleIteratorRepr`], pyre collapses the
/// abstract/concrete split into one concrete repr. The variable-step
/// `RANGESTITER` (`("next", "stop", "step")`) shape and its
/// `ll_rangenext_updown` advance are deferred.
#[derive(Debug)]
pub struct RangeIteratorRepr {
    /// `self.r_rng.step` (`rrange.py:147`) — the constant range step,
    /// nonzero for this repr. Selects `ll_rangenext_up` / `_down` and is
    /// baked as the advance `step` runtime arg.
    step: i64,
    /// `self.lowleveltype = r_rng.RANGEITER` (`lltypesystem/rrange.py:58`)
    /// — `Ptr(GcStruct("range", ("next", Signed), ("stop", Signed)))`.
    lowleveltype: LowLevelType,
    state: ReprState,
}

impl RangeIteratorRepr {
    /// RPython `AbstractRangeIteratorRepr.__init__(self, r_rng)`
    /// (`rrange.py:146-151`) for the `step != 0` `RANGEITER` shape.
    pub fn new(r_rng: &AbstractRangeRepr) -> Result<Self, TyperError> {
        if r_rng.step == 0 {
            return Err(rrange_deferred(
                "RangeIteratorRepr (variable-step RANGESTITER / ll_rangenext_updown)",
            ));
        }
        // RANGEITER = GcStruct("range", ("next", Signed), ("stop", Signed)).
        // Mutable (no immutable hint): `ll_rangenext_*` writes `iter.next`.
        let signed = LowLevelType::Signed;
        let st = Struct::gc(
            "range",
            vec![
                ("next".to_string(), signed.clone()),
                ("stop".to_string(), signed),
            ],
        );
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(st),
        }));
        Ok(RangeIteratorRepr {
            step: r_rng.step,
            lowleveltype,
            state: ReprState::new(),
        })
    }
}

impl Repr for RangeIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "RangeIteratorRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::RangeIteratorRepr
    }

    /// RPython `IteratorRepr.rtype_iter(self, hop)` (rmodel.py:266-268) —
    /// `iter(iter(x)) <==> iter(x)`: an iterator is its own iterator, so
    /// the op is the identity on the receiver (mirroring
    /// [`super::rlist::ListIteratorRepr::rtype_iter`]).
    fn rtype_iter(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(Some(vlist[0].clone()))
    }

    /// RPython `IteratorRepr.rtype_method_next(self, hop)`
    /// (rmodel.py:270-271) — `iter.next()` delegates to `rtype_next`.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            "next" => self.rtype_next(hop),
            other => Err(TyperError::message(format!(
                "missing RangeIteratorRepr.rtype_method_{other}"
            ))),
        }
    }

    /// RPython `AbstractRangeIteratorRepr.newiter(self, hop)`
    /// (`rrange.py:153-156`):
    ///
    /// ```python
    /// def newiter(self, hop):
    ///     v_rng, = hop.inputargs(self.r_rng)
    ///     citerptr = hop.inputconst(Void, self.lowleveltype)
    ///     return hop.gendirectcall(self.ll_rangeiter, citerptr, v_rng)
    /// ```
    ///
    /// As with [`super::rtuple::Length1TupleIteratorRepr::newiter`], the
    /// iterator low-level type is baked into the helper builder rather
    /// than threaded as a `citerptr` Void const; the source range repr
    /// is read from `args_r[0]`.
    fn newiter(&self, hop: &HighLevelOp) -> RTypeResult {
        let r_rng = {
            let args_r = hop.args_r.borrow();
            args_r.first().and_then(|o| o.clone()).ok_or_else(|| {
                TyperError::message("RangeIteratorRepr.newiter: arg0 repr missing")
            })?
        };
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_rng.as_ref())])?;
        let range_lltype = r_rng.lowleveltype().clone();
        let iter_lltype = self.lowleveltype.clone();
        let range_for_builder = range_lltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_rangeiter".to_string(),
            vec![range_lltype],
            iter_lltype,
            move |_rtyper, _args, _result| {
                build_ll_rangeiter_helper_graph(
                    "ll_rangeiter",
                    range_for_builder.clone(),
                    iter_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `AbstractRangeIteratorRepr.rtype_next(self, hop)`
    /// (`rrange.py:158-170`):
    ///
    /// ```python
    /// def rtype_next(self, hop):
    ///     v_iter, = hop.inputargs(self)
    ///     args = hop.inputconst(Signed, self.r_rng.step),
    ///     if self.r_rng.step > 0:
    ///         llfn = ll_rangenext_up
    ///     elif self.r_rng.step < 0:
    ///         llfn = ll_rangenext_down
    ///     else:
    ///         llfn = ll_rangenext_updown
    ///         args = ()
    ///     hop.has_implicit_exception(StopIteration)
    ///     hop.exception_is_here()
    ///     return hop.gendirectcall(llfn, v_iter, *args)
    /// ```
    ///
    /// `step != 0` here (the repr rejects variable step), so the
    /// `ll_rangenext_updown` arm is unreachable; `up` (`step > 0`) and
    /// `down` (`step < 0`) bake the constant step as the advance arg.
    fn rtype_next(&self, hop: &HighLevelOp) -> RTypeResult {
        let mut args = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        args.push(constant_with_lltype(
            ConstValue::Int(self.step),
            LowLevelType::Signed,
        ));
        hop.has_implicit_exception("StopIteration");
        hop.exception_is_here()?;
        let name = if self.step > 0 {
            "ll_rangenext_up"
        } else {
            "ll_rangenext_down"
        };
        let up = self.step > 0;
        let iter_lltype = self.lowleveltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            name.to_string(),
            vec![iter_lltype, LowLevelType::Signed],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_rangenext_helper_graph(name, iter_for_builder.clone(), up)
            },
        )?;
        hop.gendirectcall(&helper, args)
    }
}

/// Synthesise the `ll_rangeiter` helper graph
/// (`lltypesystem/rrange.py:91-97`) for the constant-step `RANGEITER`
/// shape:
///
/// ```python
/// def ll_rangeiter(ITERPTR, rng):
///     iter = malloc(ITERPTR.TO)
///     iter.next = rng.start
///     iter.stop = rng.stop
///     return iter
/// ```
///
/// Single block: `start = getfield(rng, 'start'); stop = getfield(rng,
/// 'stop'); iter = malloc(RANGEITER, flavor=gc); setfield(iter, 'next',
/// start); setfield(iter, 'stop', stop)`. `ITERPTR` is baked as
/// `iter_lltype`; the `RANGESTITER` `iter.step = rng.step` write is
/// elided (constant-step only).
pub(crate) fn build_ll_rangeiter_helper_graph(
    name: &str,
    range_lltype: LowLevelType,
    iter_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let rng_arg = variable_with_lltype("rng", range_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(rng_arg.clone())]);
    let return_var = variable_with_lltype("result", iter_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let LowLevelType::Ptr(ptr) = &iter_lltype else {
        return Err(TyperError::message(
            "build_ll_rangeiter_helper_graph: iter lltype is not Ptr",
        ));
    };
    let inner_struct = match &ptr.TO {
        PtrTarget::Struct(body) => body.clone(),
        other => {
            return Err(TyperError::message(format!(
                "build_ll_rangeiter_helper_graph: Ptr target must be Struct, got {other:?}"
            )));
        }
    };

    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);

    // start = rng.start; stop = rng.stop.
    let v_start = variable_with_lltype("start", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(rng_arg.clone()), field_const("start")],
        Hlvalue::Variable(v_start.clone()),
    ));
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(rng_arg), field_const("stop")],
        Hlvalue::Variable(v_stop.clone()),
    ));

    // iter = malloc(RANGEITER, flavor=gc).
    let c_type = Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
        LowLevelType::Void,
    );
    let c_flags =
        Constant::with_concretetype(ConstValue::byte_str("flavor=gc"), LowLevelType::Void);
    let v_iter = variable_with_lltype("iter", iter_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![Hlvalue::Constant(c_type), Hlvalue::Constant(c_flags)],
        Hlvalue::Variable(v_iter.clone()),
    ));

    // iter.next = start; iter.stop = stop.
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            field_const("next"),
            Hlvalue::Variable(v_start),
        ],
        Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            field_const("stop"),
            Hlvalue::Variable(v_stop),
        ],
        Hlvalue::Variable(variable_with_lltype("v1", LowLevelType::Void)),
    ));

    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_iter)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["rng".to_string()],
        func,
    ))
}

/// Synthesise the `ll_rangenext_up` / `ll_rangenext_down` helper graph
/// (`rrange.py:172-184`):
///
/// ```python
/// def ll_rangenext_up(iter, step):    # `_down` flips `>=` to `<=`
///     next = iter.next
///     if next >= iter.stop:
///         raise StopIteration
///     iter.next = next + step
///     return next
/// ```
///
/// 2-block CFG. `up` selects `int_ge` (`next >= stop`) vs `int_le`
/// (`next <= stop`) for the StopIteration guard:
/// - **start**: `next = getfield(iter, 'next'); stop = getfield(iter,
///   'stop'); atend = int_ge/int_le(next, stop)`. Switch on `atend`:
///   True → exceptblock (StopIteration); False → cont, carrying `iter,
///   next, step`.
/// - **cont**: `newnext = int_add(next, step); setfield(iter, 'next',
///   newnext)`; return `next`.
pub(crate) fn build_ll_rangenext_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
    up: bool,
) -> Result<PyGraph, TyperError> {
    let iter_arg = variable_with_lltype("iter", iter_lltype.clone());
    let step_arg = variable_with_lltype("step", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(iter_arg.clone()),
        Hlvalue::Variable(step_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let field_const = |f: &str| constant_with_lltype(ConstValue::byte_str(f), LowLevelType::Void);
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);

    // next = iter.next; stop = iter.stop; atend = next >=/<= stop.
    let v_next = variable_with_lltype("next", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(iter_arg.clone()), field_const("next")],
        Hlvalue::Variable(v_next.clone()),
    ));
    let v_stop = variable_with_lltype("stop", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(iter_arg.clone()), field_const("stop")],
        Hlvalue::Variable(v_stop.clone()),
    ));
    let v_atend = variable_with_lltype("atend", LowLevelType::Bool);
    let cmp_op = if up { "int_ge" } else { "int_le" };
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        cmp_op,
        vec![Hlvalue::Variable(v_next.clone()), Hlvalue::Variable(v_stop)],
        Hlvalue::Variable(v_atend.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_atend));

    // cont block args: (iter, next, step).
    let c_iter = variable_with_lltype("iter", iter_lltype);
    let c_next = variable_with_lltype("next", LowLevelType::Signed);
    let c_step = variable_with_lltype("step", LowLevelType::Signed);
    let cont = Block::shared(vec![
        Hlvalue::Variable(c_iter.clone()),
        Hlvalue::Variable(c_next.clone()),
        Hlvalue::Variable(c_step.clone()),
    ]);

    let exc_args = exception_args("StopIteration")?;
    startblock.closeblock(vec![
        // atend == True: raise StopIteration.
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        // atend == False: advance and return.
        Link::new(
            vec![
                Hlvalue::Variable(iter_arg),
                Hlvalue::Variable(v_next),
                Hlvalue::Variable(step_arg),
            ],
            Some(cont.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // newnext = next + step; iter.next = newnext.
    let v_newnext = variable_with_lltype("newnext", LowLevelType::Signed);
    {
        let mut b = cont.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(c_next.clone()), Hlvalue::Variable(c_step)],
            Hlvalue::Variable(v_newnext.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(c_iter),
                field_const("next"),
                Hlvalue::Variable(v_newnext),
            ],
            Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
        ));
    }
    cont.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c_next)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["iter".to_string(), "step".to_string()],
        func,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    #[test]
    fn rangerepr_step1_lowleveltype_is_immutable_range_gcstruct() {
        let rr = AbstractRangeRepr::new(1).unwrap();
        assert_eq!(rr.repr_class_id(), ReprClassId::RangeRepr);
        match rr.lowleveltype() {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(st) => {
                    // RANGE = GcStruct("range", start, stop) — two Signed fields.
                    assert_eq!(st._names_without_voids(), vec!["start", "stop"]);
                }
                other => panic!("AbstractRangeRepr lltype TO not Struct: {other:?}"),
            },
            other => panic!("AbstractRangeRepr lltype not Ptr: {other:?}"),
        }
    }

    #[test]
    fn rangerepr_variable_step_lowleveltype_is_rangest_gcstruct() {
        // step == 0 → RANGEST with the extra `step` field.
        let rr = AbstractRangeRepr::new(0).unwrap();
        match rr.lowleveltype() {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(st) => {
                    assert_eq!(st._names_without_voids(), vec!["start", "stop", "step"]);
                }
                other => panic!("not Struct: {other:?}"),
            },
            other => panic!("not Ptr: {other:?}"),
        }
    }

    #[test]
    fn build_ll_rangelen1_helper_graph_synthesizes_2_block_cfg() {
        let ptr = AbstractRangeRepr::new(1).unwrap().lowleveltype().clone();
        let g = build_ll_rangelen1_helper_graph("ll_rangelen1", ptr).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "getfield", "int_sub", "int_lt"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);
    }

    #[test]
    fn build_ll_rangeitem_nonneg_helper_graph_synthesizes_getfield_mul_add() {
        let ptr = AbstractRangeRepr::new(1).unwrap().lowleveltype().clone();
        let g = build_ll_rangeitem_nonneg_helper_graph("ll_rangeitem_nonneg", ptr).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        // start = l.start; prod = index * step; result = start + prod.
        assert_eq!(ops, vec!["getfield", "int_mul", "int_add"]);
        assert!(startblock.exitswitch.is_none());
        assert_eq!(startblock.exits.len(), 1);
        // (l, index, step) inputargs.
        assert_eq!(startblock.inputargs.len(), 3);
    }

    /// rrange.py:34-50 constant-step + nonneg branch — `getitem` on a
    /// `AbstractRangeRepr` lowers to a `direct_call` of `ll_rangeitem_nonneg`
    /// (`start + index*step`), preceded by `hop.exception_is_here()`, with
    /// the constant step appended as the trailing arg.
    #[test]
    fn rangerepr_getitem_nonneg_const_step_emits_direct_call_to_ll_rangeitem_nonneg() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        // step == 2 (constant step != 1) — the formula handles any step.
        let range_repr: Arc<AbstractRangeRepr> =
            Arc::new(AbstractRangeRepr::new(2).expect("AbstractRangeRepr::new(2)"));
        let range_lltype = range_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_rng = Variable::new();
        v_rng.set_concretetype(Some(range_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_rng), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        // args_s[0] (the range sequence) is not inspected by rtype_getitem
        // (only args_s[1].nonneg); a generic SomeList stands in.
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(range_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = range_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("range getitem nonneg: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(ops._called_exception_is_here_or_cannot_occur);
        // direct_call args: funcptr, v_rng, v_index, cstep (the baked step).
        assert_eq!(ops.ops[0].args.len(), 4);
        let Hlvalue::Constant(cstep) = &ops.ops[0].args[3] else {
            panic!("expected Constant step as last direct_call arg");
        };
        assert!(matches!(cstep.value, ConstValue::Int(2)));
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_rangeitem_nonneg"),
            "expected 'll_rangeitem_nonneg' in {dbg}"
        );
    }

    /// Negative-index (`args_s[1].nonneg == false`) needs the `ll_rangeitem`
    /// length normalisation (`_ll_rangelen` floor-division), deferred; like
    /// the checkidx and variable-step branches it surfaces a `TyperError`.
    #[test]
    fn rangerepr_getitem_negative_index_is_deferred() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let range_repr: Arc<AbstractRangeRepr> =
            Arc::new(AbstractRangeRepr::new(1).expect("AbstractRangeRepr::new(1)"));
        let range_lltype = range_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_rng = Variable::new();
        v_rng.set_concretetype(Some(range_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_rng), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(range_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);
        assert!(range_repr.rtype_getitem(&hop).is_err());
    }

    /// `make_iterator_repr` mints a `RangeIteratorRepr` whose lowleveltype
    /// is the `RANGEITER` struct (`next`, `stop`). A non-None variant is
    /// rejected (`rrange.py:64-66`); the variable-step (`step == 0`)
    /// iterator is deferred.
    #[test]
    fn make_iterator_repr_constant_step_yields_rangeiter_with_next_stop_struct() {
        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let it = r_rng.make_iterator_repr(&[]).unwrap();
        assert_eq!(it.class_name(), "RangeIteratorRepr");
        assert_eq!(it.repr_class_id(), ReprClassId::RangeIteratorRepr);
        match it.lowleveltype() {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(st) => {
                    assert_eq!(st._names_without_voids(), vec!["next", "stop"]);
                }
                other => panic!("RANGEITER TO not Struct: {other:?}"),
            },
            other => panic!("RANGEITER not Ptr: {other:?}"),
        }
        // unsupported variant → TyperError.
        assert!(r_rng.make_iterator_repr(&["reversed".to_string()]).is_err());
        // variable-step RANGESTITER iterator is deferred.
        let rst = AbstractRangeRepr::new(0).unwrap();
        assert!(rst.make_iterator_repr(&[]).is_err());
    }

    /// `ll_rangeiter` single block: `getfield start; getfield stop;
    /// malloc RANGEITER; setfield next; setfield stop`.
    #[test]
    fn build_ll_rangeiter_helper_graph_synthesizes_getfields_malloc_setfields() {
        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let iter_lltype = RangeIteratorRepr::new(&r_rng)
            .unwrap()
            .lowleveltype()
            .clone();
        let range_lltype = r_rng.lowleveltype().clone();
        let g = build_ll_rangeiter_helper_graph("ll_rangeiter", range_lltype, iter_lltype).unwrap();
        let inner = g.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec!["getfield", "getfield", "malloc", "setfield", "setfield"]
        );
        assert!(startblock.exitswitch.is_none());
        assert_eq!(startblock.exits.len(), 1);
        assert_eq!(startblock.inputargs.len(), 1);
    }

    /// `ll_rangenext_up` start block guards with `int_ge` and branches to
    /// the StopIteration (exceptblock) and advance (cont) exits;
    /// `ll_rangenext_down` flips the guard to `int_le`.
    #[test]
    fn build_ll_rangenext_helper_graph_guards_with_int_ge_up_int_le_down() {
        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let iter_lltype = RangeIteratorRepr::new(&r_rng)
            .unwrap()
            .lowleveltype()
            .clone();

        for (up, cmp) in [(true, "int_ge"), (false, "int_le")] {
            let name = if up {
                "ll_rangenext_up"
            } else {
                "ll_rangenext_down"
            };
            let g = build_ll_rangenext_helper_graph(name, iter_lltype.clone(), up).unwrap();
            let inner = g.graph.borrow();
            let startblock = inner.startblock.borrow();
            let ops: Vec<&str> = startblock
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(ops, vec!["getfield", "getfield", cmp]);
            assert!(startblock.exitswitch.is_some());
            assert_eq!(startblock.exits.len(), 2);
            // (iter, step) inputargs.
            assert_eq!(startblock.inputargs.len(), 2);
        }
    }

    /// rrange.py:158-170 constant-step `rtype_next` lowers to a
    /// `direct_call` of `ll_rangenext_up` (`step > 0`) with the constant
    /// step appended, preceded by `hop.exception_is_here()`.
    #[test]
    fn rangeiter_rtype_next_emits_direct_call_to_ll_rangenext_up() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let iter_repr: Arc<RangeIteratorRepr> = Arc::new(RangeIteratorRepr::new(&r_rng).unwrap());
        let iter_lltype = iter_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_iter = Variable::new();
        v_iter.set_concretetype(Some(iter_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "next".to_string(),
                vec![Hlvalue::Variable(v_iter)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        // Non-constant binding so inputarg reaches convertvar (identity, no-op).
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r
            .borrow_mut()
            .push(Some(iter_repr.clone() as Arc<dyn Repr>));

        let result = iter_repr
            .rtype_next(&hop)
            .unwrap_or_else(|err| panic!("range rtype_next: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(ops._called_exception_is_here_or_cannot_occur);
        // direct_call args: funcptr, v_iter, cstep (the baked step).
        assert_eq!(ops.ops[0].args.len(), 3);
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_rangenext_up"),
            "expected 'll_rangenext_up' in {dbg}"
        );
        let Hlvalue::Constant(cstep) = &ops.ops[0].args[2] else {
            panic!("expected Constant step as last direct_call arg");
        };
        assert!(matches!(cstep.value, ConstValue::Int(1)));
    }

    /// rmodel.py:266-268 `IteratorRepr.rtype_iter` — `iter()` on a range
    /// iterator is the identity (returns the iterator unchanged, emits no
    /// op), and the iterator carries its own `RangeIteratorRepr` class id.
    #[test]
    fn rangeiter_rtype_iter_returns_the_iterator_itself() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        let r_rng = AbstractRangeRepr::new(1).unwrap();
        let iter_repr: Arc<RangeIteratorRepr> = Arc::new(RangeIteratorRepr::new(&r_rng).unwrap());
        assert_eq!(iter_repr.repr_class_id(), ReprClassId::RangeIteratorRepr);
        let iter_lltype = iter_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_iter = Variable::new();
        v_iter.set_concretetype(Some(iter_lltype.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(iter_lltype));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "iter".to_string(),
                vec![Hlvalue::Variable(v_iter)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r
            .borrow_mut()
            .push(Some(iter_repr.clone() as Arc<dyn Repr>));

        let result = iter_repr
            .rtype_iter(&hop)
            .unwrap_or_else(|err| panic!("range rtype_iter: {err:?}"));
        // identity: the iterator is returned unchanged, no op emitted.
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_eq!(llops.borrow().ops.len(), 0);

        // `it.next()` method call routes to rtype_next.
        let method_err = iter_repr.rtype_method("unknown_method", &hop);
        assert!(method_err.is_err());
    }
}
