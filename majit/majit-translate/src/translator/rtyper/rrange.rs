//! RPython `rpython/rtyper/rrange.py` + `lltypesystem/rrange.py` —
//! minimal `RangeRepr` slice covering the `len(range(...))` lowering for
//! the step-1 case (`range(n)` / `range(a, b)`), which is the form
//! `builtin_range` mints for the overwhelmingly common call shapes
//! (`annotator/builtin.rs:523-528`).
//!
//! A `range()` result that is never mutated annotates as a `SomeList`
//! carrying a non-`None` `range_step` (`annotator/listdef.rs:177`); its
//! repr is NOT array-backed (`FixedSizeListRepr`) but an immutable
//! `GcStruct("range", start, stop)` (`lltypesystem/rrange.py:51-57`).
//!
//! Deferred to follow-on slices (matching how `FixedSizeListRepr` landed
//! `rtype_len` first): the general-step `ll_rangelen` length (needs the
//! `int_floordiv` lowering, not yet a recognised low-level op), the
//! `RANGEST` variable-step path, `pairtype(RangeRepr, IntegerRepr)`
//! `rtype_getitem`, and `RangeIteratorRepr`.

use std::rc::Rc;
use std::sync::Arc;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr, PtrTarget, StructType};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, HighLevelOp, RPythonTyper, constant_with_lltype, helper_pygraph_from_graph,
    variable_with_lltype,
};

/// RPython `class RangeRepr(AbstractRangeRepr)` (`rrange.py:43-67` +
/// `rrange.py:10-16`):
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
pub struct RangeRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// `self.step` (`rrange.py:12`) — the constant range step. `0`
    /// signals upstream's "variable step" (`RANGEST`).
    step: i64,
}

impl RangeRepr {
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
        let st = StructType::gc_with_hints(
            "range",
            fields,
            vec![("immutable".to_string(), ConstValue::Bool(true))],
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(st),
        }));
        Ok(RangeRepr {
            state: ReprState::new(),
            lltype,
            step,
        })
    }
}

impl Repr for RangeRepr {
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
                "RangeRepr.rtype_len(step={}) — general ll_rangelen path \
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
                "RangeRepr.rtype_getitem: checkidx IndexError branch not yet ported",
            ));
        }
        if self.step == 0 {
            return Err(TyperError::message(
                "RangeRepr.rtype_getitem: variable-step RANGEST (_getstep) not yet ported",
            ));
        }
        let s1 = hop
            .args_s
            .borrow()
            .get(1)
            .cloned()
            .ok_or_else(|| TyperError::message("RangeRepr.rtype_getitem: args_s[1] missing"))?;
        let nonneg = match &s1 {
            SomeValue::Integer(i) => i.nonneg,
            other => {
                return Err(TyperError::message(format!(
                    "RangeRepr.rtype_getitem: args_s[1] must be SomeInteger, got {other:?}"
                )));
            }
        };
        if !nonneg {
            return Err(TyperError::message(
                "RangeRepr.rtype_getitem: negative-index ll_rangeitem branch not yet ported",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::pairtype::ReprClassId;

    #[test]
    fn rangerepr_step1_lowleveltype_is_immutable_range_gcstruct() {
        let rr = RangeRepr::new(1).unwrap();
        assert_eq!(rr.repr_class_id(), ReprClassId::RangeRepr);
        match rr.lowleveltype() {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(st) => {
                    // RANGE = GcStruct("range", start, stop) — two Signed fields.
                    assert_eq!(st._names_without_voids(), vec!["start", "stop"]);
                }
                other => panic!("RangeRepr lltype TO not Struct: {other:?}"),
            },
            other => panic!("RangeRepr lltype not Ptr: {other:?}"),
        }
    }

    #[test]
    fn rangerepr_variable_step_lowleveltype_is_rangest_gcstruct() {
        // step == 0 → RANGEST with the extra `step` field.
        let rr = RangeRepr::new(0).unwrap();
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
        let ptr = RangeRepr::new(1).unwrap().lowleveltype().clone();
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
        let ptr = RangeRepr::new(1).unwrap().lowleveltype().clone();
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
    /// `RangeRepr` lowers to a `direct_call` of `ll_rangeitem_nonneg`
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
        let range_repr: Arc<RangeRepr> = Arc::new(RangeRepr::new(2).expect("RangeRepr::new(2)"));
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
        let range_repr: Arc<RangeRepr> = Arc::new(RangeRepr::new(1).expect("RangeRepr::new(1)"));
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
}
