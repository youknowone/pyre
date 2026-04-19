//! Polymorphic Bound Callable (PBC) representation — lowers indirect
//! method calls on `dyn Trait` receivers.
//!
//! RPython equivalent:
//! `rpython/rtyper/rpbc.py:199-217 FunctionReprBase.call`. RPython's
//! rtyper materialises the funcptr Variable via
//! `convert_to_concrete_llfn` and then emits
//! `genop('indirect_call', [funcptr, *args, c_graphs])` carrying the
//! full row-of-graphs as the trailing Constant. Pyre's rtyper-equivalent
//! does the same shape over `OpKind::Call { target: CallTarget::Indirect
//! { trait_root, method_name }, .. }` sites: insert a `VtableMethodPtr`
//! to materialise the funcptr ValueId, then replace the original Call
//! with an `OpKind::IndirectCall` carrying funcptr + args + the family
//! `graphs` (full `c_graphs`, not yet filtered by JIT candidates —
//! that filtering happens later in `call.py::graphs_from(op)`).
//!
//! After this pass, no `CallTarget::Indirect` survives in the graph;
//! `jtransform` only sees `OpKind::IndirectCall` for indirect dispatch.

use crate::call::CallControl;
use crate::model::{BlockId, CallTarget, FunctionGraph, OpKind, SpaceOperation};
use crate::translator::rtyper::rclass;
// `TypeResolutionState` lives under `translate_legacy` until majit-rtyper
// (roadmap Phase 6) extracts it into a proper crate. Wiring bridge only.
use crate::translate_legacy::rtyper::rtyper::TypeResolutionState;

/// Walk every `OpKind::Call` whose target is `CallTarget::Indirect` and
/// rewrite it into the RPython-orthodox pair:
///
/// 1. `OpKind::VtableMethodPtr { receiver, trait_root, method_name }`
///    → produces `funcptr: ValueId` (Int kind)
/// 2. `OpKind::IndirectCall { funcptr, args, graphs, result_ty }`
///    → mirrors RPython `indirect_call(funcptr, *args, c_graphs)`
///
/// `args` remain the full call argument list, including the receiver.
/// RPython's `convert_to_concrete_llfn` materialises the funcptr but the
/// eventual `indirect_call` still receives `self, ...` as ordinary args.
pub fn lower_indirect_calls(
    graph: &mut FunctionGraph,
    type_state: &mut TypeResolutionState,
    call_control: &CallControl,
) {
    // Collect the (block, op_index) sites first so the rewrite below
    // can mutate the graph without aliasing the borrow.
    let sites: Vec<(usize, usize)> = graph
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(bid, block)| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(oi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    } => Some((bid, oi)),
                    _ => None,
                })
        })
        .collect();

    // Process in reverse so earlier indices stay valid as later sites
    // grow by 1 op (the inserted `VtableMethodPtr`).
    for (bid, oi) in sites.into_iter().rev() {
        let block_id = BlockId(bid);
        let op = graph.blocks[bid].operations[oi].clone();
        let (target, args, result_ty, result) = match op.kind {
            OpKind::Call {
                target,
                args,
                result_ty,
            } => (target, args, result_ty, op.result),
            _ => unreachable!("site filter mismatch"),
        };
        let (trait_root, method_name) = match target {
            CallTarget::Indirect {
                trait_root,
                method_name,
            } => (trait_root, method_name),
            _ => unreachable!("site filter mismatch"),
        };
        let receiver = *args
            .first()
            .expect("dyn-Trait method call must have a receiver arg");
        // RPython rclass.py:371-377 (condensed into a single op).
        let funcptr = rclass::class_get_method_ptr(
            graph,
            type_state,
            block_id,
            oi,
            receiver,
            trait_root.clone(),
            method_name.clone(),
        );
        // RPython rpbc.py:216 `c_graphs = row_of_graphs.values()` — full
        // family without JIT candidate filtering.
        //
        // `None` means "unknown family" — matches
        // `rpython/translator/backendopt/graphanalyze.py:117`, where
        // `graphs is None` short-circuits family analyzers to
        // `top_result()` (conservative: canraise, can_invalidate, etc.
        // default to True). `Some(vec![])` would instead be treated as
        // "empty family → No/false" by every analyzer, which is unsafe
        // when external/unregistered impls might still be called.
        // Pyre's source-only parser only sees `#[trait_method]`-
        // registered impls; if none are registered for a
        // `(trait, method)` family, we don't know whether that's
        // because the family is truly empty or because the impls live
        // outside the analyzed sources, so we take the conservative
        // `None` path.
        let family = call_control.all_impls_for_indirect(&trait_root, &method_name);
        let graphs = if family.is_empty() {
            None
        } else {
            Some(family)
        };
        // The original Call op is now at index `oi + 1` because
        // `class_get_method_ptr` inserted `VtableMethodPtr` at `oi`.
        graph.blocks[bid].operations[oi + 1] = SpaceOperation {
            result,
            kind: OpKind::IndirectCall {
                funcptr,
                args,
                graphs,
                result_ty,
            },
        };
    }
}

/// Debug-build invariant: after `lower_indirect_calls` runs, no
/// `CallTarget::Indirect` may survive in the graph. Callers can run
/// this assert to catch missed sites early.
#[cfg(debug_assertions)]
pub fn assert_no_indirect_call_targets(graph: &FunctionGraph) {
    for (bid, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            if let OpKind::Call {
                target: CallTarget::Indirect { .. },
                ..
            } = &op.kind
            {
                panic!(
                    "post-lowering invariant violation: \
                     CallTarget::Indirect survived at block {bid} op {oi}: {:?}",
                    op
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call::CallControl;
    use crate::model::{FunctionGraph, OpKind, Terminator, ValueType};
    use crate::translate_legacy::annotator::annrpython::annotate;
    use crate::translate_legacy::rtyper::rtyper::resolve_types;

    fn build_indirect_graph() -> FunctionGraph {
        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "h".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        graph
    }

    /// Boundary: pre-lowering graph has `CallTarget::Indirect`; post-lowering
    /// graph has `VtableMethodPtr + IndirectCall` and zero `Indirect`
    /// targets. Mirrors RPython rpbc.py:199-217 emit shape.
    #[test]
    fn lower_indirect_calls_eliminates_call_target_indirect() {
        let mut cc = CallControl::new();
        cc.register_trait_method("run", Some("Handler"), "A", FunctionGraph::new("A::run"));
        cc.register_trait_method("run", Some("Handler"), "B", FunctionGraph::new("B::run"));
        cc.find_all_graphs_for_tests();

        let mut graph = build_indirect_graph();

        // Pre-lowering: a CallTarget::Indirect exists.
        let pre_indirect = graph.blocks[graph.startblock.0]
            .operations
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(pre_indirect, 1, "expected 1 Indirect Call pre-lowering");

        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        // Post-lowering: invariant — zero Indirect targets.
        assert_no_indirect_call_targets(&graph);

        // Post-lowering: exactly one VtableMethodPtr and one IndirectCall.
        let ops = &graph.blocks[graph.startblock.0].operations;
        let vtable_ptr_count = ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::VtableMethodPtr { .. }))
            .count();
        let indirect_call_count = ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::IndirectCall { .. }))
            .count();
        assert_eq!(vtable_ptr_count, 1);
        assert_eq!(indirect_call_count, 1);
    }

    /// Regression: inherent (non-trait) method calls — `CallTarget::Method`
    /// targets — must pass through `lower_indirect_calls` unchanged.
    /// RPython rpbc.py:199-217 only rewrites the indirect-call dispatch
    /// (`s_pbc.callfamily`), inherent method calls are statically resolved
    /// upstream by the rtyper.
    #[test]
    fn inherent_method_unchanged_by_lowering() {
        let mut cc = CallControl::new();
        cc.register_function_graph(
            crate::parse::CallPath::from_segments(["Foo", "bar"]),
            FunctionGraph::new("Foo::bar"),
        );

        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "f".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::method("bar", Some("Foo".to_string())),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

        let pre_ops_len = graph.blocks[graph.startblock.0].operations.len();
        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        // Same op count; the Call op survives untouched.
        let ops = &graph.blocks[graph.startblock.0].operations;
        assert_eq!(ops.len(), pre_ops_len);
        let method_count = ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Method { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(method_count, 1, "Method target must survive lowering");
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::VtableMethodPtr { .. })),
            "inherent call must not produce VtableMethodPtr"
        );
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::IndirectCall { .. })),
            "inherent call must not produce IndirectCall"
        );
    }
}
