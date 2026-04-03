//! Graph inlining utility — substitute Call ops with callee body graphs.
//!
//! **Note:** RPython's codewriter does NOT inline callee bodies into callers.
//! Instead, `jtransform.py` rewrites `direct_call` to `inline_call_*`
//! (referencing the callee's JitCode) and the meta-interpreter descends
//! into callee JitCode at runtime.
//!
//! This module provides graph-level body splicing for use cases where
//! actual body expansion is needed (e.g. analysis, testing). It is NOT
//! part of the RPython-orthodox codewriter pipeline.

use std::collections::HashMap;

use crate::call::{CallControl, CallKind};
use crate::graph::{BasicBlockId, MajitGraph, Op, OpKind, Terminator, ValueId};

/// Inline all `Regular` calls in the graph, consulting `CallControl`
/// for the inline/residual decision.
///
/// RPython equivalent: flow space auto-inlining + `backendopt/inline.py`.
///
/// Returns the number of call sites inlined.
pub fn inline_graph(graph: &mut MajitGraph, call_control: &CallControl, max_depth: usize) -> usize {
    let mut total_inlined = 0;
    for _depth in 0..max_depth {
        let sites = find_inline_sites(graph, call_control);
        if sites.is_empty() {
            break;
        }
        // Process sites in reverse order so block indices remain valid
        // (later blocks are not affected by earlier inlining).
        for site in sites.into_iter().rev() {
            inline_call_site(graph, site);
            total_inlined += 1;
        }
    }
    total_inlined
}

/// A call site eligible for inlining.
struct InlineSite {
    block_id: BasicBlockId,
    op_index: usize,
    callee: MajitGraph,
}

/// Find all Call ops in the graph where `CallControl` says `Regular`.
fn find_inline_sites(graph: &MajitGraph, call_control: &CallControl) -> Vec<InlineSite> {
    let mut sites = Vec::new();
    for block in &graph.blocks {
        for (op_idx, op) in block.ops.iter().enumerate() {
            let target = match &op.kind {
                OpKind::Call { target, .. } => target,
                _ => continue,
            };
            if call_control.guess_call_kind(target) != CallKind::Regular {
                continue;
            }
            let callee = match call_control.graphs_from(target) {
                Some(g) => g.clone(),
                None => continue,
            };
            sites.push(InlineSite {
                block_id: block.id,
                op_index: op_idx,
                callee,
            });
        }
    }
    sites
}

/// Inline a single call site.
///
/// Algorithm:
/// 1. Split the caller block at the call site into "before" and "after"
/// 2. Remap all callee values/blocks to fresh IDs in the caller graph
/// 3. Connect: before → callee entry (passing call args)
/// 4. Connect: callee Return → merge block (passing return value)
/// 5. Merge block continues with the "after" ops
fn inline_call_site(graph: &mut MajitGraph, site: InlineSite) {
    let InlineSite {
        block_id,
        op_index,
        callee,
    } = site;

    // Extract the call op details
    let block = &graph.blocks[block_id.0];
    let call_op = &block.ops[op_index];
    let (call_args, call_result) = match &call_op.kind {
        OpKind::Call { args, .. } => (args.clone(), call_op.result),
        _ => unreachable!("InlineSite should point to a Call op"),
    };

    // Separate ops into before-call and after-call
    let after_ops: Vec<Op> = block.ops[op_index + 1..].to_vec();
    let after_terminator = block.terminator.clone();

    // Truncate the original block to before-call ops only
    graph.blocks[block_id.0].ops.truncate(op_index);

    // --- Remap callee values and blocks ---
    let value_map = remap_callee_values(graph, &callee);
    let block_map = remap_callee_blocks(graph, &callee);

    // --- Create merge block for after-call ops ---
    // If the call has a result, the merge block has one inputarg (Phi node)
    // that receives the callee's return value.
    let merge_block_id = if !after_ops.is_empty()
        || !matches!(after_terminator, Terminator::Unreachable)
        || call_result.is_some()
    {
        let (merge_id, merge_args) = if let Some(original_result) = call_result {
            let (id, args) = graph.create_block_with_args(1);
            // The merge block's inputarg replaces the original call result.
            // We need to remap all references to original_result → merge_args[0]
            // in the after_ops and after_terminator.
            let remapped_after_ops = remap_value_in_ops(&after_ops, original_result, args[0]);
            let remapped_after_term =
                remap_value_in_terminator(&after_terminator, original_result, args[0]);
            graph.blocks[id.0].ops = remapped_after_ops;
            graph.blocks[id.0].terminator = remapped_after_term;
            (id, args)
        } else {
            let id = graph.create_block();
            graph.blocks[id.0].ops = after_ops;
            graph.blocks[id.0].terminator = after_terminator;
            (id, vec![])
        };

        Some((merge_id, merge_args))
    } else {
        None
    };

    // --- Copy callee blocks into the graph ---
    let callee_entry = *block_map.get(&callee.entry).unwrap();

    for callee_block in &callee.blocks {
        let new_block_id = block_map[&callee_block.id];

        // Remap inputargs
        let new_inputargs: Vec<ValueId> = callee_block
            .inputargs
            .iter()
            .map(|v| value_map[v])
            .collect();
        graph.blocks[new_block_id.0].inputargs = new_inputargs;

        // Remap ops
        let new_ops: Vec<Op> = callee_block
            .ops
            .iter()
            .map(|op| remap_op(op, &value_map))
            .collect();
        graph.blocks[new_block_id.0].ops = new_ops;

        // Remap terminator, replacing Return with Goto to merge block
        let new_terminator = match &callee_block.terminator {
            Terminator::Return(Some(ret_val)) => {
                let remapped_ret = value_map[ret_val];
                if let Some((merge_id, ref merge_args)) = merge_block_id {
                    if merge_args.is_empty() {
                        Terminator::Goto {
                            target: merge_id,
                            args: vec![],
                        }
                    } else {
                        Terminator::Goto {
                            target: merge_id,
                            args: vec![remapped_ret],
                        }
                    }
                } else {
                    // No after ops — callee return becomes caller return
                    Terminator::Return(Some(remapped_ret))
                }
            }
            Terminator::Return(None) => {
                if let Some((merge_id, _)) = merge_block_id {
                    Terminator::Goto {
                        target: merge_id,
                        args: vec![],
                    }
                } else {
                    Terminator::Return(None)
                }
            }
            other => remap_terminator(other, &value_map, &block_map),
        };
        graph.blocks[new_block_id.0].terminator = new_terminator;
    }

    // --- Connect caller's before-block to callee entry ---
    // Map call arguments to callee's Input ops.
    // Callee's Input ops correspond to its entry block's first N ops.
    let callee_entry_block = &callee.blocks[callee.entry.0];
    let input_values: Vec<ValueId> = callee_entry_block
        .ops
        .iter()
        .filter(|op| matches!(&op.kind, OpKind::Input { .. }))
        .filter_map(|op| op.result)
        .collect();

    // Map call args to callee input values
    for (i, &callee_input) in input_values.iter().enumerate() {
        let remapped_input = value_map[&callee_input];
        if let Some(&call_arg) = call_args.get(i) {
            // Add alias: remapped callee input = call argument
            // We do this by prepending a "move" in the callee entry block
            // Actually, we set the entry block's inputargs and jump with call args
            // But callee entry doesn't have inputargs for Input ops...
            // Instead, remap all uses of remapped_input to call_arg in the callee blocks
            remap_value_in_graph(graph, &block_map, remapped_input, call_arg);
        }
    }

    // Set the before-block's terminator to jump to callee entry
    graph.blocks[block_id.0].terminator = Terminator::Goto {
        target: callee_entry,
        args: vec![],
    };
}

/// Allocate fresh ValueIds for all values in the callee graph.
fn remap_callee_values(graph: &mut MajitGraph, callee: &MajitGraph) -> HashMap<ValueId, ValueId> {
    let mut map = HashMap::new();
    // Collect all ValueIds used in the callee
    for block in &callee.blocks {
        for &v in &block.inputargs {
            map.entry(v).or_insert_with(|| graph.alloc_value());
        }
        for op in &block.ops {
            if let Some(result) = op.result {
                map.entry(result).or_insert_with(|| graph.alloc_value());
            }
            for v in op_value_refs(&op.kind) {
                map.entry(v).or_insert_with(|| graph.alloc_value());
            }
        }
        for v in terminator_value_refs(&block.terminator) {
            map.entry(v).or_insert_with(|| graph.alloc_value());
        }
    }
    map
}

/// Allocate fresh BasicBlockIds for all blocks in the callee graph.
fn remap_callee_blocks(
    graph: &mut MajitGraph,
    callee: &MajitGraph,
) -> HashMap<BasicBlockId, BasicBlockId> {
    let mut map = HashMap::new();
    for block in &callee.blocks {
        let new_id = graph.create_block();
        map.insert(block.id, new_id);
    }
    map
}

/// Remap a single Op's values.
fn remap_op(op: &Op, value_map: &HashMap<ValueId, ValueId>) -> Op {
    let remap = |v: &ValueId| *value_map.get(v).unwrap_or(v);
    let result = op.result.map(|v| remap(&v));
    let kind = remap_op_kind(&op.kind, &remap);
    Op { result, kind }
}

fn remap_op_kind(kind: &OpKind, remap: &impl Fn(&ValueId) -> ValueId) -> OpKind {
    match kind {
        OpKind::Input { name, ty } => OpKind::Input {
            name: name.clone(),
            ty: ty.clone(),
        },
        OpKind::ConstInt(v) => OpKind::ConstInt(*v),
        OpKind::FieldRead { base, field, ty } => OpKind::FieldRead {
            base: remap(base),
            field: field.clone(),
            ty: ty.clone(),
        },
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } => OpKind::FieldWrite {
            base: remap(base),
            field: field.clone(),
            value: remap(value),
            ty: ty.clone(),
        },
        OpKind::ArrayRead {
            base,
            index,
            item_ty,
        } => OpKind::ArrayRead {
            base: remap(base),
            index: remap(index),
            item_ty: item_ty.clone(),
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
        } => OpKind::ArrayWrite {
            base: remap(base),
            index: remap(index),
            value: remap(value),
            item_ty: item_ty.clone(),
        },
        OpKind::Call {
            target,
            args,
            result_ty,
        } => OpKind::Call {
            target: target.clone(),
            args: args.iter().map(remap).collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::GuardTrue { cond } => OpKind::GuardTrue { cond: remap(cond) },
        OpKind::GuardFalse { cond } => OpKind::GuardFalse { cond: remap(cond) },
        OpKind::GuardValue { value, kind_char } => OpKind::GuardValue {
            value: remap(value),
            kind_char: *kind_char,
        },
        OpKind::VableFieldRead { field_index, ty } => OpKind::VableFieldRead {
            field_index: *field_index,
            ty: ty.clone(),
        },
        OpKind::VableFieldWrite {
            field_index,
            value,
            ty,
        } => OpKind::VableFieldWrite {
            field_index: *field_index,
            value: remap(value),
            ty: ty.clone(),
        },
        OpKind::VableArrayRead {
            array_index,
            elem_index,
            item_ty,
        } => OpKind::VableArrayRead {
            array_index: *array_index,
            elem_index: remap(elem_index),
            item_ty: item_ty.clone(),
        },
        OpKind::VableArrayWrite {
            array_index,
            elem_index,
            value,
            item_ty,
        } => OpKind::VableArrayWrite {
            array_index: *array_index,
            elem_index: remap(elem_index),
            value: remap(value),
            item_ty: item_ty.clone(),
        },
        OpKind::BinOp {
            op,
            lhs,
            rhs,
            result_ty,
        } => OpKind::BinOp {
            op: op.clone(),
            lhs: remap(lhs),
            rhs: remap(rhs),
            result_ty: result_ty.clone(),
        },
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } => OpKind::UnaryOp {
            op: op.clone(),
            operand: remap(operand),
            result_ty: result_ty.clone(),
        },
        OpKind::VableForce => OpKind::VableForce,
        OpKind::Live => OpKind::Live,
        OpKind::CallElidable {
            descriptor,
            args,
            result_ty,
        } => OpKind::CallElidable {
            descriptor: descriptor.clone(),
            args: args.iter().map(remap).collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::CallResidual {
            descriptor,
            args,
            result_ty,
        } => OpKind::CallResidual {
            descriptor: descriptor.clone(),
            args: args.iter().map(remap).collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::CallMayForce {
            descriptor,
            args,
            result_ty,
        } => OpKind::CallMayForce {
            descriptor: descriptor.clone(),
            args: args.iter().map(remap).collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::InlineCall {
            jitcode_index,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::InlineCall {
            jitcode_index: *jitcode_index,
            args_i: args_i.iter().map(remap).collect(),
            args_r: args_r.iter().map(remap).collect(),
            args_f: args_f.iter().map(remap).collect(),
            result_kind: *result_kind,
        },
        OpKind::RecursiveCall {
            jd_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            result_kind,
        } => OpKind::RecursiveCall {
            jd_index: *jd_index,
            greens_i: greens_i.iter().map(remap).collect(),
            greens_r: greens_r.iter().map(remap).collect(),
            greens_f: greens_f.iter().map(remap).collect(),
            reds_i: reds_i.iter().map(remap).collect(),
            reds_r: reds_r.iter().map(remap).collect(),
            reds_f: reds_f.iter().map(remap).collect(),
            result_kind: *result_kind,
        },
        OpKind::Unknown { kind } => OpKind::Unknown { kind: kind.clone() },
    }
}

/// Remap a terminator's values and block targets.
fn remap_terminator(
    term: &Terminator,
    value_map: &HashMap<ValueId, ValueId>,
    block_map: &HashMap<BasicBlockId, BasicBlockId>,
) -> Terminator {
    let rv = |v: &ValueId| *value_map.get(v).unwrap_or(v);
    let rb = |b: &BasicBlockId| *block_map.get(b).unwrap_or(b);
    match term {
        Terminator::Goto { target, args } => Terminator::Goto {
            target: rb(target),
            args: args.iter().map(rv).collect(),
        },
        Terminator::Branch {
            cond,
            if_true,
            true_args,
            if_false,
            false_args,
        } => Terminator::Branch {
            cond: rv(cond),
            if_true: rb(if_true),
            true_args: true_args.iter().map(rv).collect(),
            if_false: rb(if_false),
            false_args: false_args.iter().map(rv).collect(),
        },
        Terminator::Return(v) => Terminator::Return(v.as_ref().map(rv)),
        Terminator::Abort { reason } => Terminator::Abort {
            reason: reason.clone(),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

/// Collect all ValueId references used in an OpKind (not including result).
pub fn op_value_refs(kind: &OpKind) -> Vec<ValueId> {
    match kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::VableForce
        | OpKind::Live
        | OpKind::Unknown { .. } => {
            vec![]
        }
        OpKind::FieldRead { base, .. } => vec![*base],
        OpKind::FieldWrite { base, value, .. } => vec![*base, *value],
        OpKind::ArrayRead { base, index, .. } => vec![*base, *index],
        OpKind::ArrayWrite {
            base, index, value, ..
        } => vec![*base, *index, *value],
        OpKind::Call { args, .. } => args.clone(),
        OpKind::GuardTrue { cond } | OpKind::GuardFalse { cond } => vec![*cond],
        OpKind::GuardValue { value, .. } => vec![*value],
        OpKind::VableFieldRead { .. } => vec![],
        OpKind::VableFieldWrite { value, .. } => vec![*value],
        OpKind::VableArrayRead { elem_index, .. } => vec![*elem_index],
        OpKind::VableArrayWrite {
            elem_index, value, ..
        } => vec![*elem_index, *value],
        OpKind::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
        OpKind::UnaryOp { operand, .. } => vec![*operand],
        OpKind::CallElidable { args, .. }
        | OpKind::CallResidual { args, .. }
        | OpKind::CallMayForce { args, .. } => args.clone(),
        OpKind::InlineCall {
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = args_i.clone();
            refs.extend(args_r);
            refs.extend(args_f);
            refs
        }
        OpKind::RecursiveCall {
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        } => {
            let mut refs = greens_i.clone();
            refs.extend(greens_r);
            refs.extend(greens_f);
            refs.extend(reds_i);
            refs.extend(reds_r);
            refs.extend(reds_f);
            refs
        }
    }
}

/// Collect all ValueId references in a Terminator.
fn terminator_value_refs(term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::Goto { args, .. } => args.clone(),
        Terminator::Branch {
            cond,
            true_args,
            false_args,
            ..
        } => {
            let mut refs = vec![*cond];
            refs.extend(true_args);
            refs.extend(false_args);
            refs
        }
        Terminator::Return(Some(v)) => vec![*v],
        Terminator::Return(None) | Terminator::Abort { .. } | Terminator::Unreachable => vec![],
    }
}

/// Replace all occurrences of `old` with `new` in ops within the specified blocks.
fn remap_value_in_graph(
    graph: &mut MajitGraph,
    block_map: &HashMap<BasicBlockId, BasicBlockId>,
    old: ValueId,
    new: ValueId,
) {
    let target_blocks: Vec<BasicBlockId> = block_map.values().copied().collect();
    for &bid in &target_blocks {
        let block = &mut graph.blocks[bid.0];
        block.ops = remap_value_in_ops(&block.ops, old, new);
        block.terminator = remap_value_in_terminator(&block.terminator, old, new);
    }
}

/// Replace all occurrences of `old` with `new` in a list of ops.
fn remap_value_in_ops(ops: &[Op], old: ValueId, new: ValueId) -> Vec<Op> {
    let remap = |v: &ValueId| if *v == old { new } else { *v };
    ops.iter()
        .map(|op| Op {
            result: op.result,
            kind: remap_op_kind(&op.kind, &remap),
        })
        .collect()
}

/// Replace all occurrences of `old` with `new` in a terminator.
fn remap_value_in_terminator(term: &Terminator, old: ValueId, new: ValueId) -> Terminator {
    let rv = |v: &ValueId| if *v == old { new } else { *v };
    match term {
        Terminator::Goto { target, args } => Terminator::Goto {
            target: *target,
            args: args.iter().map(rv).collect(),
        },
        Terminator::Branch {
            cond,
            if_true,
            true_args,
            if_false,
            false_args,
        } => Terminator::Branch {
            cond: rv(cond),
            if_true: *if_true,
            true_args: true_args.iter().map(rv).collect(),
            if_false: *if_false,
            false_args: false_args.iter().map(rv).collect(),
        },
        Terminator::Return(v) => Terminator::Return(v.as_ref().map(rv)),
        Terminator::Abort { reason } => Terminator::Abort {
            reason: reason.clone(),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call::CallControl;
    use crate::graph::{CallTarget, FieldDescriptor, MajitGraph, OpKind, Terminator, ValueType};
    use crate::parse::CallPath;

    fn make_simple_callee() -> MajitGraph {
        // Callee: fn callee(base) -> value { ArrayRead(base, const(0)) }
        let mut g = MajitGraph::new("callee");
        let entry = g.entry;
        let base = g.push_op(
            entry,
            OpKind::Input {
                name: "base".into(),
                ty: ValueType::Ref,
            },
            true,
        );
        let idx = g.push_op(entry, OpKind::ConstInt(0), true);
        let result = g.push_op(
            entry,
            OpKind::ArrayRead {
                base: base.unwrap(),
                index: idx.unwrap(),
                item_ty: ValueType::Ref,
            },
            true,
        );
        g.set_terminator(entry, Terminator::Return(result));
        g
    }

    #[test]
    fn inline_single_call() {
        // Caller: fn caller() { v = Call("callee", [base]); Return v }
        let mut caller = MajitGraph::new("caller");
        let entry = caller.entry;
        let base = caller.push_op(
            entry,
            OpKind::Input {
                name: "base".into(),
                ty: ValueType::Ref,
            },
            true,
        );
        let result = caller.push_op(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: vec![base.unwrap()],
                result_ty: ValueType::Ref,
            },
            true,
        );
        caller.set_terminator(entry, Terminator::Return(result));

        let callee = make_simple_callee();

        let mut cc = CallControl::new();
        cc.register_function_graph(CallPath::from_segments(["callee"]), callee);
        cc.find_all_graphs_for_tests();

        let count = inline_graph(&mut caller, &cc, 3);
        assert_eq!(count, 1);

        // After inlining: the graph should have ArrayRead from the callee
        let has_array_read = caller
            .blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|op| matches!(&op.kind, OpKind::ArrayRead { .. }));
        assert!(
            has_array_read,
            "inlined graph should contain ArrayRead from callee"
        );

        // Should NOT have the original Call op
        let has_call = caller
            .blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|op| matches!(&op.kind, OpKind::Call { .. }));
        assert!(!has_call, "Call op should be replaced by inlined body");
    }

    #[test]
    fn inline_preserves_residual_calls() {
        let mut caller = MajitGraph::new("caller");
        let entry = caller.entry;
        let result = caller.push_op(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["unknown_fn"]),
                args: vec![],
                result_ty: ValueType::Ref,
            },
            true,
        );
        caller.set_terminator(entry, Terminator::Return(result));

        let cc = CallControl::new(); // empty — no graphs registered
        let count = inline_graph(&mut caller, &cc, 3);
        assert_eq!(count, 0);

        // Call should still be there
        let has_call = caller
            .blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|op| matches!(&op.kind, OpKind::Call { .. }));
        assert!(has_call, "residual Call should be preserved");
    }

    #[test]
    fn inline_two_levels() {
        // inner: fn inner(x) -> ArrayRead(x, 0)
        let inner = make_simple_callee();

        // outer: fn outer(base) -> Call("callee", [base])
        let mut outer = MajitGraph::new("outer");
        let entry = outer.entry;
        let base = outer.push_op(
            entry,
            OpKind::Input {
                name: "base".into(),
                ty: ValueType::Ref,
            },
            true,
        );
        let result = outer.push_op(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: vec![base.unwrap()],
                result_ty: ValueType::Ref,
            },
            true,
        );
        outer.set_terminator(entry, Terminator::Return(result));

        // caller: fn caller(x) -> Call("outer", [x])
        let mut caller = MajitGraph::new("caller");
        let centry = caller.entry;
        let x = caller.push_op(
            centry,
            OpKind::Input {
                name: "x".into(),
                ty: ValueType::Ref,
            },
            true,
        );
        let result = caller.push_op(
            centry,
            OpKind::Call {
                target: CallTarget::function_path(["outer"]),
                args: vec![x.unwrap()],
                result_ty: ValueType::Ref,
            },
            true,
        );
        caller.set_terminator(centry, Terminator::Return(result));

        let mut cc = CallControl::new();
        cc.register_function_graph(CallPath::from_segments(["callee"]), inner);
        cc.register_function_graph(CallPath::from_segments(["outer"]), outer);
        cc.find_all_graphs_for_tests();

        let count = inline_graph(&mut caller, &cc, 3);
        assert!(count >= 2, "should inline at least 2 levels, got {count}");

        // After 2-level inlining: should have ArrayRead, no Call ops
        let has_array_read = caller
            .blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|op| matches!(&op.kind, OpKind::ArrayRead { .. }));
        assert!(
            has_array_read,
            "2-level inlined graph should have ArrayRead"
        );
    }
}
