//! A selected impl's primitive `TraitConst` is that impl's literal.
//!
//! `alloc_exception_nursery` is generic over `T: GcType`. Each concrete
//! call copies the body (`FunctionDesc.cachedgraph`) and the copy's
//! `T::SIZE` is the impl's `NamedConst`, folded by `const_eval_init_body`.
//! A zero-arg `__trait_const` call is not a registered graph.

use majit_charon_reader::Llbc;
use majit_translate::HostStaticAddrs;
use majit_translate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names;
use majit_translate::model::{CallTarget, OpKind};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

fn call_segments(target: &CallTarget) -> Option<&[String]> {
    match target {
        CallTarget::FunctionPath { segments, .. } => Some(segments),
        _ => None,
    }
}

#[test]
fn nursery_spec_folds_gc_type_size() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("pyre-object.ullbc is already extracted");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(&llbc),
        HostStaticAddrs::default(),
        &[],
        &[
            "alloc_exception_nursery",
            "w_exception_new_empty_impl",
            "w_exception_new_empty_extended_impl",
        ],
    )
    .expect("interp exception graphs lower");

    let mut sizes = Vec::new();
    for function in &program.functions {
        for op in function
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
        {
            let OpKind::Call { target, .. } = &op.kind else {
                continue;
            };
            let Some(segments) = call_segments(target) else {
                continue;
            };
            assert!(
                !(segments.len() == 1 && segments[0] == "__trait_const"),
                "{} still calls __trait_const",
                function.name
            );
        }
        if !function.name.contains("alloc_exception_nursery__spec_") {
            continue;
        }
        let call = function
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call { target, args, .. }
                    if call_segments(target)
                        .and_then(|segments| segments.last())
                        .is_some_and(|leaf| leaf == "try_gc_alloc_collecting_rooted") =>
                {
                    Some(args.clone())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("{} does not call the collecting allocator", function.name));
        let size_var = call
            .get(1)
            .and_then(|arg| arg.as_variable())
            .cloned()
            .unwrap_or_else(|| panic!("{} size argument is not a value", function.name));
        let size_op = function.graph.blocks.iter().find_map(|block| {
            block
                .operations
                .iter()
                .find_map(|op| (op.result.as_ref() == Some(&size_var)).then_some(op.kind.clone()))
        });
        let n = match size_op {
            Some(OpKind::ConstInt(n)) => n,
            Some(OpKind::ConstUInt(n)) => i64::try_from(n).expect("size fits i64"),
            other => panic!(
                "{} T::SIZE must be an integer constant, got {other:?}",
                function.name
            ),
        };
        assert!(n > 0, "{} folded a non-positive size {n}", function.name);
        sizes.push((function.name.clone(), n));
    }
    assert!(
        sizes.len() >= 2,
        "expected one spec copy per concrete exception layout, got {sizes:?}"
    );
    let distinct = sizes
        .iter()
        .map(|(_, n)| *n)
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        distinct.len() >= 2,
        "base and extended layouts must not share one size: {sizes:?}"
    );
}
