//! `rpython/rtyper/rtyper.py` — narrow pyre port of one of the rtyper's
//! responsibilities: rewriting opaque-pointer flowspace Constants into
//! their resolved post-rtype shape (`Constant(ll_ptr,
//! concretetype=lltype.Ptr(...))`).
//!
//! Upstream's full `RPythonTyper.specialize(annotator)` (rtyper.py:88-180)
//! does type inference + op-rewriting across an entire annotation;
//! constant resolution falls out as a side-effect of `specialize_block`
//! seeing each `Constant(value, concretetype)` and emitting the LL value
//! verbatim.  Pyre's flowspace shim carries pre-rtype `Opaque(Ref)`
//! constants instead (because pyre has no LL type system or rtyper),
//! and the canonical `flatten_graph` driver matches upstream's
//! `flatten.py:63` signature which takes no `lower_constant` closure
//! (the rtyper has already done the resolution).
//!
//! This module bridges the gap: a pre-flatten pass that walks every
//! `Opaque(Ref)` Constant in the graph and rewrites it to
//! `Signed(pointer)` with `Kind::Ref`, using a caller-supplied
//! `lower_pointer` closure that maps each opaque token back to the
//! runtime PyObject pointer.  After the pass runs,
//! `flatten_constant_operand` lowers `Signed(p) with Kind::Ref` to
//! `Operand::ConstRef(p)` and the canonical driver works on the
//! resolved graph.

use super::flatten::Kind;
use super::flow::{
    Constant, ConstantValue, ExitSwitch, ExitSwitchElement, FlowListOfKind, FlowValue,
    FunctionGraph, LinkRef, SpaceOperationArg,
};

/// Walk every flowspace Constant in `graph` and apply `lower_pointer`
/// to each `Constant { value: Opaque(_), kind: Some(Ref) }`, replacing
/// it with `Constant { value: Signed(pointer), kind: Some(Ref) }`.
///
/// This is the pyre-narrow analog of `rpython/rtyper/rtyper.py:88-180
/// RPythonTyper.specialize` — specifically the constant-resolution
/// substep that upstream interleaves with type inference.  Pyre runs
/// the pass standalone before calling `flatten_graph`.
///
/// `lower_pointer` is invoked once per opaque constant occurrence (no
/// deduplication — pyre's graph constants are `Clone` value types so
/// the same opaque-repr may appear multiple times).  Production
/// callers thread the same `lower_constant` closure they pass to
/// `flatten_graph_with_lowering`, projected onto the integer pointer.
pub fn rtype_opaque_constants<F>(graph: &FunctionGraph, mut lower_pointer: F)
where
    F: FnMut(&Constant) -> i64,
{
    for block in graph.iterblocks() {
        let mut block_mut = block.borrow_mut();
        for value in block_mut.inputargs.iter_mut() {
            rewrite_flow_value(value, &mut lower_pointer);
        }
        for op in block_mut.operations.iter_mut() {
            for arg in op.args.iter_mut() {
                rewrite_space_operation_arg(arg, &mut lower_pointer);
            }
            if let Some(result) = op.result.as_mut() {
                rewrite_flow_value(result, &mut lower_pointer);
            }
        }
        if let Some(exitswitch) = block_mut.exitswitch.as_mut() {
            rewrite_exit_switch(exitswitch, &mut lower_pointer);
        }
        // Block.exits are LinkRef (Rc<RefCell<Link>>); borrow each one
        // independently from the block.
        let exits = block_mut.exits.clone();
        drop(block_mut);
        for link in exits {
            rewrite_link(&link, &mut lower_pointer);
        }
    }
}

fn rewrite_constant<F>(constant: &mut Constant, lower_pointer: &mut F)
where
    F: FnMut(&Constant) -> i64,
{
    if let (ConstantValue::Opaque(_), Some(Kind::Ref)) = (&constant.value, constant.kind) {
        let pointer = lower_pointer(constant);
        constant.value = ConstantValue::Signed(pointer);
    }
}

fn rewrite_flow_value<F>(value: &mut FlowValue, lower_pointer: &mut F)
where
    F: FnMut(&Constant) -> i64,
{
    if let FlowValue::Constant(constant) = value {
        rewrite_constant(constant, lower_pointer);
    }
}

fn rewrite_flow_list_of_kind<F>(list: &mut FlowListOfKind, lower_pointer: &mut F)
where
    F: FnMut(&Constant) -> i64,
{
    for value in list.content.iter_mut() {
        rewrite_flow_value(value, lower_pointer);
    }
}

fn rewrite_space_operation_arg<F>(arg: &mut SpaceOperationArg, lower_pointer: &mut F)
where
    F: FnMut(&Constant) -> i64,
{
    match arg {
        SpaceOperationArg::Value(value) => rewrite_flow_value(value, lower_pointer),
        SpaceOperationArg::ListOfKind(list) => rewrite_flow_list_of_kind(list, lower_pointer),
        SpaceOperationArg::Descr(_) | SpaceOperationArg::IndirectCallTargets(_) => {}
    }
}

fn rewrite_exit_switch<F>(exitswitch: &mut ExitSwitch, lower_pointer: &mut F)
where
    F: FnMut(&Constant) -> i64,
{
    match exitswitch {
        ExitSwitch::Value(value) => rewrite_flow_value(value, lower_pointer),
        ExitSwitch::Tuple(elements) => {
            for element in elements.iter_mut() {
                if let ExitSwitchElement::Value(value) = element {
                    rewrite_flow_value(value, lower_pointer);
                }
            }
        }
    }
}

fn rewrite_link<F>(link: &LinkRef, lower_pointer: &mut F)
where
    F: FnMut(&Constant) -> i64,
{
    let mut link_mut = link.borrow_mut();
    for arg in link_mut.args.iter_mut() {
        if let Some(value) = arg.as_mut() {
            rewrite_flow_value(value, lower_pointer);
        }
    }
    if let Some(exitcase) = link_mut.exitcase.as_mut() {
        rewrite_flow_value(exitcase, lower_pointer);
    }
    if let Some(llexitcase) = link_mut.llexitcase.as_mut() {
        rewrite_flow_value(llexitcase, lower_pointer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::flow::{Block, FunctionGraph, Link, SpaceOperation};

    fn opaque_ref(repr: &str) -> Constant {
        Constant::opaque(repr, Some(Kind::Ref))
    }

    fn make_graph(start: super::super::flow::BlockRef) -> FunctionGraph {
        FunctionGraph::new("g", start, None)
    }

    #[test]
    fn rewrites_inputarg_opaque_ref_to_signed() {
        let block = Block::shared(vec![opaque_ref("pycode@1234").into()]);
        let graph = make_graph(block.clone());
        rtype_opaque_constants(&graph, |_| 0xdead);
        let block_borrow = block.borrow();
        match &block_borrow.inputargs[0] {
            FlowValue::Constant(Constant {
                value: ConstantValue::Signed(p),
                kind: Some(Kind::Ref),
            }) => assert_eq!(*p, 0xdead),
            other => panic!("expected Signed(0xdead) with Ref kind, got {other:?}"),
        }
    }

    #[test]
    fn rewrites_operation_arg_opaque_ref() {
        let block = Block::shared(Vec::new());
        let op = SpaceOperation::new(
            "load",
            vec![SpaceOperationArg::Value(opaque_ref("globals@4321").into())],
            None,
            0,
        );
        block.borrow_mut().operations.push(op);
        let graph = make_graph(block.clone());
        let mut call_count = 0;
        rtype_opaque_constants(&graph, |c| {
            call_count += 1;
            if let ConstantValue::Opaque(opaque) = &c.value {
                assert_eq!(opaque.repr(), "globals@4321");
            }
            0xfeed
        });
        assert_eq!(call_count, 1);
        let block_borrow = block.borrow();
        let arg = &block_borrow.operations[0].args[0];
        match arg {
            SpaceOperationArg::Value(FlowValue::Constant(Constant {
                value: ConstantValue::Signed(p),
                kind: Some(Kind::Ref),
            })) => assert_eq!(*p, 0xfeed),
            other => panic!("expected Signed Ref, got {other:?}"),
        }
    }

    #[test]
    fn rewrites_link_args_and_exitcase() {
        let start = Block::shared(Vec::new());
        let target = Block::shared(vec![Constant::none().into()]);
        let link = Link::new(
            vec![opaque_ref("OverflowError").into()],
            Some(target.clone()),
            Some(opaque_ref("default_case").into()),
        )
        .into_ref();
        start.borrow_mut().exits.push(link.clone());
        let graph = make_graph(start);
        rtype_opaque_constants(&graph, |c| match &c.value {
            ConstantValue::Opaque(opaque) if opaque.repr() == "OverflowError" => 0xaa,
            ConstantValue::Opaque(opaque) if opaque.repr() == "default_case" => 0xbb,
            other => panic!("unexpected opaque {other:?}"),
        });
        let link_borrow = link.borrow();
        match &link_borrow.args[0] {
            Some(FlowValue::Constant(Constant {
                value: ConstantValue::Signed(p),
                kind: Some(Kind::Ref),
            })) => assert_eq!(*p, 0xaa),
            other => panic!("expected link arg Signed(0xaa) Ref, got {other:?}"),
        }
        match &link_borrow.exitcase {
            Some(FlowValue::Constant(Constant {
                value: ConstantValue::Signed(p),
                kind: Some(Kind::Ref),
            })) => assert_eq!(*p, 0xbb),
            other => panic!("expected exitcase Signed(0xbb) Ref, got {other:?}"),
        }
    }

    #[test]
    fn leaves_non_opaque_constants_unchanged() {
        let block = Block::shared(vec![Constant::signed(42).into()]);
        let graph = make_graph(block.clone());
        rtype_opaque_constants(&graph, |_| panic!("lower_pointer should not be called"));
        match &block.borrow().inputargs[0] {
            FlowValue::Constant(Constant {
                value: ConstantValue::Signed(42),
                kind: Some(Kind::Int),
            }) => {}
            other => panic!("expected unchanged Signed(42) Int, got {other:?}"),
        }
    }
}
