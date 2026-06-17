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
}
