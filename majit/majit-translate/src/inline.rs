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
use crate::model::{
    BlockId, CallFuncPtr, FunctionGraph, OpKind, SpaceOperation, remap_control_flow_metadata_var,
};

fn remap_call_funcptr<
    F: Fn(&crate::flowspace::model::Variable) -> crate::flowspace::model::Variable,
>(
    funcptr: &CallFuncPtr,
    remap_var: &F,
) -> CallFuncPtr {
    match funcptr {
        CallFuncPtr::Target(target) => CallFuncPtr::Target(target.clone()),
        CallFuncPtr::Value(var) => CallFuncPtr::Value(remap_var(var)),
    }
}

/// Inline all `Regular` calls in the graph, consulting `CallControl`
/// for the inline/residual decision.
///
/// RPython equivalent: flow space auto-inlining + `backendopt/inline.py`.
///
/// Returns the number of call sites inlined.
pub fn inline_graph(
    graph: &mut FunctionGraph,
    call_control: &CallControl,
    max_depth: usize,
) -> usize {
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
    block_id: BlockId,
    op_index: usize,
    callee: FunctionGraph,
}

/// Find all Call ops in the graph where `CallControl` says `Regular`.
fn find_inline_sites(graph: &FunctionGraph, call_control: &CallControl) -> Vec<InlineSite> {
    let mut sites = Vec::new();
    for block in &graph.blocks {
        for (op_idx, op) in block.operations.iter().enumerate() {
            // Inline only direct calls — RPython's inline pass skips
            // indirect family dispatch (each callee is resolved
            // dynamically at runtime, not statically at inline time).
            let target = match &op.kind {
                OpKind::Call { target, .. } => target,
                _ => continue,
            };
            if call_control.guess_call_kind(op) != CallKind::Regular {
                continue;
            }
            let callee = match call_control.direct_graph_for(target) {
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
fn inline_call_site(graph: &mut FunctionGraph, site: InlineSite) {
    let InlineSite {
        block_id,
        op_index,
        callee,
    } = site;

    // Extract the call op details
    let block = &graph.blocks[block_id.0];
    let call_op = &block.operations[op_index];
    let (call_args, call_result) = match &call_op.kind {
        OpKind::Call { args, .. } => (args.clone(), call_op.result.clone()),
        _ => unreachable!("InlineSite should point to a Call op"),
    };

    // Separate ops into before-call and after-call.  Upstream
    // `rpython/flowspace/model.py:171-180` treats `Block.exitswitch` +
    // `Block.exits` as the single CFG source of truth, so they move
    // together with the ops they guard into the merge block.
    let after_ops: Vec<SpaceOperation> = block.operations[op_index + 1..].to_vec();
    let after_exitswitch = block.exitswitch.clone();
    let after_exits = block.exits.clone();

    // Truncate the original block to before-call ops only
    graph.blocks[block_id.0].operations.truncate(op_index);

    // --- Remap callee values and blocks ---
    let value_map = remap_callee_values(graph, &callee);
    let block_map = remap_callee_blocks(graph, &callee);

    // --- Create merge block for after-call ops ---
    // Upstream `backendopt/inline.py:253-264` copies caller-block-after
    // ops + exits into a fresh afterblock whenever there is something
    // to preserve.  Pyre creates the merge block when (a) after-call
    // ops exist, (b) the caller block was already closed with exits or
    // an exitswitch, or (c) the call produced a result that downstream
    // code consumes.  (Order matches upstream's `(exits) or (stmts)`
    // guard.)
    let caller_was_closed = !after_exits.is_empty() || after_exitswitch.is_some();
    let merge_block_id = if !after_ops.is_empty() || caller_was_closed || call_result.is_some() {
        let (merge_id, merge_args) = if let Some(original_result_var) = call_result {
            let (id, arg_vars) = graph.create_block_with_arg_vars(1);
            // The merge block's inputarg replaces the original call result.
            // Remap every reference to `original_result_var` → `arg_vars[0]` in
            // after-call ops and exit metadata so the phi-node-style
            // merge carries the callee's return value forward.
            let merge_inputarg_var = arg_vars[0].clone();
            let remapped_after_ops =
                remap_value_in_ops(&after_ops, &original_result_var, &merge_inputarg_var);
            graph.blocks[id.0].operations = remapped_after_ops;
            let (remapped_switch, remapped_exits) = remap_control_flow_metadata_var(
                &after_exitswitch,
                &after_exits,
                |var| {
                    if var == &original_result_var {
                        merge_inputarg_var.clone()
                    } else {
                        var.clone()
                    }
                },
                |b| b,
            );
            graph.set_control_flow_metadata(id, remapped_switch, remapped_exits);
            (id, arg_vars)
        } else {
            let id = graph.create_block();
            graph.blocks[id.0].operations = after_ops;
            let (remapped_switch, remapped_exits) = remap_control_flow_metadata_var(
                &after_exitswitch,
                &after_exits,
                |var| var.clone(),
                |b| b,
            );
            graph.set_control_flow_metadata(id, remapped_switch, remapped_exits);
            (id, Vec::<crate::flowspace::model::Variable>::new())
        };

        Some((merge_id, merge_args))
    } else {
        None
    };

    // --- Copy callee blocks into the graph ---
    let callee_entry = *block_map.get(&callee.startblock).unwrap();

    for callee_block in &callee.blocks {
        let new_block_id = block_map[&callee_block.id];

        // Remap inputargs (callee Variables → caller-graph Variables
        // via value_map).
        let new_inputargs: Vec<crate::flowspace::model::Variable> = callee_block
            .inputargs
            .iter()
            .map(|var| value_map[var].clone())
            .collect();
        graph.blocks[new_block_id.0].inputargs = new_inputargs;

        // Remap ops
        let new_ops: Vec<SpaceOperation> = callee_block
            .operations
            .iter()
            .map(|op| remap_op(op, &value_map))
            .collect();
        graph.blocks[new_block_id.0].operations = new_ops;

        // Identify the callee's canonical returnblock by ID, matching
        // upstream `rpython/translator/backendopt/inline.py:289
        // rewire_returnblock` which reads `graph_to_inline.returnblock`
        // and rewires its exits to point at the caller's afterblock.
        let is_returnblock = callee_block.id == callee.returnblock;
        if is_returnblock {
            // Upstream `backendopt/inline.py:289-296`:
            //   copiedreturnblock = copy_block(self.graph_to_inline.returnblock)
            //   linkargs = ([copiedreturnblock.inputargs[0]] + passon_vars)
            //   linkfrominlined = Link(linkargs, afterblock)
            //   copiedreturnblock.exitswitch = None
            //   copiedreturnblock.recloseblock(linkfrominlined)
            // When there is no afterblock (the call was the caller
            // block's only statement and the block terminates unclosed),
            // forward to the caller graph's canonical returnblock so
            // the inlined function's return value becomes the caller's
            // return value — still a Goto into a final block, matching
            // upstream's `exits=[Link(..., returnblock)]` shape.
            let ret_val: Option<crate::flowspace::model::Variable> = callee_block
                .inputargs
                .first()
                .map(|var| value_map[var].clone());
            let caller_returnblock = graph.returnblock;
            let (target, arg_vars) = match (&merge_block_id, ret_val) {
                (Some((merge_id, merge_args)), Some(remapped_ret)) => {
                    if merge_args.is_empty() {
                        (*merge_id, vec![])
                    } else {
                        (*merge_id, vec![remapped_ret])
                    }
                }
                (Some((merge_id, _)), None) => (*merge_id, vec![]),
                (None, Some(remapped_ret)) => (caller_returnblock, vec![remapped_ret]),
                (None, None) => (caller_returnblock, vec![]),
            };
            graph.set_goto(new_block_id, target, arg_vars);
        } else {
            // Preserve the callee block's upstream CFG shape (single
            // goto, can-raise, typed-exception, bool-branch) with
            // renamed values and blocks.  `set_control_flow_metadata`
            // stamps `prevblock` on every link per
            // `flowspace/model.py:120`.
            let (exitswitch, exits) = remap_control_flow_metadata_var(
                &callee_block.exitswitch,
                &callee_block.exits,
                |var| value_map[var].clone(),
                |b| block_map[&b],
            );
            graph.set_control_flow_metadata(new_block_id, exitswitch, exits);
        }
    }

    // --- Connect caller's before-block to callee entry ---
    // Map call arguments to callee's Input ops.
    // Callee's Input ops correspond to its entry block's first N ops.
    let callee_entry_block = &callee.blocks[callee.startblock.0];
    let input_vars: Vec<crate::flowspace::model::Variable> = callee_entry_block
        .operations
        .iter()
        .filter(|op| matches!(&op.kind, OpKind::Input { .. }))
        .filter_map(|op| op.result.clone())
        .collect();

    // Map call args to callee input values
    for (i, callee_input_var) in input_vars.iter().enumerate() {
        let remapped_input_var = value_map[callee_input_var].clone();
        if let Some(call_arg_var) = call_args.get(i) {
            // Add alias: remapped callee input = call argument
            // We do this by prepending a "move" in the callee entry block
            // Actually, we set the entry block's inputargs and jump with call args
            // But callee entry doesn't have inputargs for Input ops...
            // Instead, remap all uses of remapped_input to call_arg in the callee blocks
            remap_value_in_graph(graph, &block_map, &remapped_input_var, call_arg_var);
        }
    }

    // Set the before-block's terminator to jump to callee entry.
    // `set_terminator` resyncs `exits`/`exitswitch` to a single Goto
    // link with `prevblock = block_id`, replacing the caller block's
    // original exits (which have already been moved to the merge block
    // above).
    //
    // RPython parity (`backendopt/inline.py:265-273
    // inline_function`): the caller-side Link supplies `call_args`
    // positionally so the callee entry block's `inputargs` Variables
    // receive the actual argument values.  Pyre's older `Input`-op
    // remap path (the `remap_value_in_graph(graph, &block_map,
    // remapped_input, call_arg)` calls above) rewrites every
    // reference to a callee Input result with the corresponding
    // call_arg directly — that path is sufficient when the cloned
    // callee was built via pyre's frontend (which leaves
    // `Block.inputargs` empty and threads parameters through
    // `OpKind::Input` ops).  For callees built via the upstream
    // Variable-keyed shape (where `startblock.inputargs` carries
    // the parameter Variables), the Link.args path below is the
    // required binding mechanism — without it the cloned inputarg
    // Variables arrive unbound and the link arity diverges from
    // the target's inputarg count.  Both shapes converge here: an
    // empty inputarg list yields empty `entry_args`, so the prior
    // behaviour is preserved bit-for-bit when the callee has no
    // inputargs.
    let inputarg_count = callee_entry_block.inputargs.len();
    let entry_arg_vars: Vec<crate::flowspace::model::Variable> =
        call_args.iter().take(inputarg_count).cloned().collect();
    graph.set_goto(block_id, callee_entry, entry_arg_vars);
}

/// Allocate fresh caller-graph Variables for every callee-graph
/// Variable referenced inside the callee.  Returns the callee-Variable
/// → caller-Variable rename map consumed by [`remap_op`] and the
/// cross-graph remap closures in [`inline_call_site`].
fn remap_callee_values(
    graph: &mut FunctionGraph,
    callee: &FunctionGraph,
) -> HashMap<crate::flowspace::model::Variable, crate::flowspace::model::Variable> {
    let mut map: HashMap<crate::flowspace::model::Variable, crate::flowspace::model::Variable> =
        HashMap::new();
    // Collect all Variables used in the callee
    for block in &callee.blocks {
        for var in block.inputargs.iter() {
            map.entry(var.clone())
                .or_insert_with(|| graph.alloc_value_var());
        }
        for op in &block.operations {
            if let Some(result) = op.result.as_ref() {
                map.entry(result.clone())
                    .or_insert_with(|| graph.alloc_value_var());
            }
            for var in op_variable_refs(&op.kind) {
                map.entry(var).or_insert_with(|| graph.alloc_value_var());
            }
        }
        // Upstream `rpython/flowspace/model.py:224-229 getvariables`
        // walks `link.args` for every exit in addition to ops.  The
        // exitswitch variable is always a block-local value referenced
        // by the raising op / branch condition, so it is already in
        // `op_variable_refs` — but the per-link args must be copied here.
        // Upstream `rpython/translator/backendopt/inline.py:268-269
        // copy_link` also renames `link.last_exception` and
        // `link.last_exc_value`, so the extravars must be present in
        // the value map before `remap_control_flow_metadata` runs.
        for link in &block.exits {
            for arg in &link.args {
                if let Some(var) = arg.as_variable() {
                    map.entry(var.clone())
                        .or_insert_with(|| graph.alloc_value_var());
                }
            }
            if let Some(arg) = &link.last_exception {
                if let Some(var) = arg.as_variable() {
                    map.entry(var.clone())
                        .or_insert_with(|| graph.alloc_value_var());
                }
            }
            if let Some(arg) = &link.last_exc_value {
                if let Some(var) = arg.as_variable() {
                    map.entry(var.clone())
                        .or_insert_with(|| graph.alloc_value_var());
                }
            }
        }
        if let Some(crate::model::ExitSwitch::Value(cond)) = &block.exitswitch {
            map.entry(cond.clone())
                .or_insert_with(|| graph.alloc_value_var());
        }
    }
    map
}

/// Allocate fresh BlockIds for all blocks in the callee graph.
fn remap_callee_blocks(
    graph: &mut FunctionGraph,
    callee: &FunctionGraph,
) -> HashMap<BlockId, BlockId> {
    let mut map = HashMap::new();
    for block in &callee.blocks {
        let new_id = graph.create_block();
        map.insert(block.id, new_id);
    }
    map
}

/// Remap a single Op's operand and result Variables across the
/// inline boundary.  `value_map` carries the callee Variable →
/// caller Variable rename allocated by [`remap_callee_values`];
/// every operand Variable is looked up directly (falling back to
/// the source Variable if it is not in the map — e.g. a Constant
/// or an external-graph reference).
fn remap_op(
    op: &SpaceOperation,
    value_map: &HashMap<crate::flowspace::model::Variable, crate::flowspace::model::Variable>,
) -> SpaceOperation {
    let remap_var = |var: &crate::flowspace::model::Variable| {
        value_map.get(var).cloned().unwrap_or_else(|| var.clone())
    };
    let result = op.result.as_ref().map(|v| remap_var(v));
    let kind = remap_op_kind(&op.kind, &remap_var);
    SpaceOperation { result, kind }
}

pub(crate) fn remap_op_kind(
    kind: &OpKind,
    remap_var: &impl Fn(&crate::flowspace::model::Variable) -> crate::flowspace::model::Variable,
) -> OpKind {
    match kind {
        OpKind::Input {
            name,
            ty,
            class_root,
        } => OpKind::Input {
            name: name.clone(),
            ty: ty.clone(),
            class_root: class_root.clone(),
        },
        OpKind::ConstInt(v) => OpKind::ConstInt(*v),
        OpKind::ConstBool(v) => OpKind::ConstBool(*v),
        OpKind::ConstSymbolic { tag, ty } => OpKind::ConstSymbolic {
            tag: *tag,
            ty: ty.clone(),
        },
        OpKind::ConstFloat(bits) => OpKind::ConstFloat(*bits),
        OpKind::ConstRef(obj) => OpKind::ConstRef(obj.clone()),
        OpKind::ConstRefNull => OpKind::ConstRefNull,
        OpKind::ConstRefAddr(addr) => OpKind::ConstRefAddr(*addr),
        OpKind::NewWithVtable { owner, vtable } => OpKind::NewWithVtable {
            owner: owner.clone(),
            vtable: *vtable,
        },
        OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        } => OpKind::FieldRead {
            base: remap_var(base),
            field: field.clone(),
            ty: ty.clone(),
            pure: *pure,
        },
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } => OpKind::FieldWrite {
            base: remap_var(base),
            field: field.clone(),
            value: value.map_value(remap_var),
            ty: ty.clone(),
        },
        OpKind::ArrayRead {
            base,
            index,
            item_ty,
            array_type_id,
            nolength,
        } => OpKind::ArrayRead {
            base: remap_var(base),
            index: remap_var(index),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
            array_type_id,
            nolength,
        } => OpKind::ArrayWrite {
            base: remap_var(base),
            index: remap_var(index),
            value: value.map_value(remap_var),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
        },
        OpKind::InteriorFieldRead {
            base,
            index,
            field,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldRead {
            base: remap_var(base),
            index: remap_var(index),
            field: field.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::InteriorFieldWrite {
            base,
            index,
            field,
            value,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldWrite {
            base: remap_var(base),
            index: remap_var(index),
            field: field.clone(),
            value: remap_var(value),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::Call {
            target,
            args,
            result_ty,
        } => OpKind::Call {
            target: target.clone(),
            args: args.iter().map(&remap_var).collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::GuardTrue { cond } => OpKind::GuardTrue {
            cond: remap_var(cond),
        },
        OpKind::GuardFalse { cond } => OpKind::GuardFalse {
            cond: remap_var(cond),
        },
        OpKind::GuardValue { value, kind_char } => OpKind::GuardValue {
            value: remap_var(value),
            kind_char: *kind_char,
        },
        OpKind::VtableMethodPtr {
            receiver,
            trait_root,
            method_name,
        } => OpKind::VtableMethodPtr {
            receiver: remap_var(receiver),
            trait_root: trait_root.clone(),
            method_name: method_name.clone(),
        },
        OpKind::IndirectCall {
            funcptr,
            args,
            graphs,
            result_ty,
        } => OpKind::IndirectCall {
            funcptr: remap_var(funcptr),
            args: args.iter().map(&remap_var).collect(),
            graphs: graphs.clone(),
            result_ty: result_ty.clone(),
        },
        OpKind::RecordQuasiImmutField {
            base,
            field,
            mutate_field,
        } => OpKind::RecordQuasiImmutField {
            base: remap_var(base),
            field: field.clone(),
            mutate_field: mutate_field.clone(),
        },
        OpKind::VableFieldRead {
            base,
            field_index,
            ty,
        } => OpKind::VableFieldRead {
            base: remap_var(base),
            field_index: *field_index,
            ty: ty.clone(),
        },
        OpKind::VableFieldWrite {
            base,
            field_index,
            value,
            ty,
        } => OpKind::VableFieldWrite {
            base: remap_var(base),
            field_index: *field_index,
            value: value.map_value(remap_var),
            ty: ty.clone(),
        },
        OpKind::VableArrayRead {
            base,
            array_index,
            elem_index,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayRead {
            base: remap_var(base),
            array_index: *array_index,
            elem_index: remap_var(elem_index),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::VableArrayWrite {
            base,
            array_index,
            elem_index,
            value,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayWrite {
            base: remap_var(base),
            array_index: *array_index,
            elem_index: remap_var(elem_index),
            value: remap_var(value),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::BinOp {
            op,
            lhs,
            rhs,
            result_ty,
        } => OpKind::BinOp {
            op: op.clone(),
            lhs: remap_var(lhs),
            rhs: remap_var(rhs),
            result_ty: result_ty.clone(),
        },
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } => OpKind::UnaryOp {
            op: op.clone(),
            operand: remap_var(operand),
            result_ty: result_ty.clone(),
        },
        OpKind::VableForce { base } => OpKind::VableForce {
            base: remap_var(base),
        },
        OpKind::Hint { value, kind } => OpKind::Hint {
            value: remap_var(value),
            kind: *kind,
        },
        OpKind::JitDebug { args } => OpKind::JitDebug {
            args: args.iter().map(&remap_var).collect(),
        },
        OpKind::RecordKnownResult {
            result_value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::RecordKnownResult {
            result_value: remap_var(result_value),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::AssertGreen { value, kind_char } => OpKind::AssertGreen {
            value: remap_var(value),
            kind_char: *kind_char,
        },
        OpKind::CurrentTraceLength => OpKind::CurrentTraceLength,
        OpKind::IsConstant { value, kind_char } => OpKind::IsConstant {
            value: remap_var(value),
            kind_char: *kind_char,
        },
        OpKind::IsVirtual { value, kind_char } => OpKind::IsVirtual {
            value: remap_var(value),
            kind_char: *kind_char,
        },
        OpKind::IsInstance {
            obj,
            class_carrier,
            result_ty,
        } => OpKind::IsInstance {
            obj: remap_var(obj),
            class_carrier: remap_var(class_carrier),
            result_ty: result_ty.clone(),
        },
        OpKind::Live => OpKind::Live,
        OpKind::JitMergePoint {
            jitdriver_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
        } => OpKind::JitMergePoint {
            jitdriver_index: *jitdriver_index,
            greens_i: greens_i.iter().map(&remap_var).collect(),
            greens_r: greens_r.iter().map(&remap_var).collect(),
            greens_f: greens_f.iter().map(&remap_var).collect(),
            reds_i: reds_i.iter().map(&remap_var).collect(),
            reds_r: reds_r.iter().map(&remap_var).collect(),
            reds_f: reds_f.iter().map(&remap_var).collect(),
        },
        OpKind::LoopHeader { jitdriver_index } => OpKind::LoopHeader {
            jitdriver_index: *jitdriver_index,
        },
        OpKind::CallElidable {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::CallElidable {
            funcptr: remap_call_funcptr(funcptr, remap_var),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::CallResidual {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            indirect_targets,
        } => OpKind::CallResidual {
            funcptr: remap_call_funcptr(funcptr, remap_var),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
            indirect_targets: indirect_targets.clone(),
        },
        OpKind::CallMayForce {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::CallMayForce {
            funcptr: remap_call_funcptr(funcptr, remap_var),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::InlineCall {
            jitcode,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::InlineCall {
            jitcode: jitcode.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
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
            greens_i: greens_i.iter().map(&remap_var).collect(),
            greens_r: greens_r.iter().map(&remap_var).collect(),
            greens_f: greens_f.iter().map(&remap_var).collect(),
            reds_i: reds_i.iter().map(&remap_var).collect(),
            reds_r: reds_r.iter().map(&remap_var).collect(),
            reds_f: reds_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::ConditionalCall {
            condition,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
        } => OpKind::ConditionalCall {
            condition: remap_var(condition),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
        },
        OpKind::ConditionalCallValue {
            value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::ConditionalCallValue {
            value: remap_var(value),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::Abort { kind } => OpKind::Abort { kind: kind.clone() },
        OpKind::NewTuple { args } => OpKind::NewTuple {
            args: args.iter().map(&remap_var).collect(),
        },
        OpKind::LoweredBlackholeOp { opname, args } => OpKind::LoweredBlackholeOp {
            opname: opname.clone(),
            args: args.iter().map(&remap_var).collect(),
        },
        OpKind::LoadStatic {
            segments,
            ty,
            value,
        } => OpKind::LoadStatic {
            segments: segments.clone(),
            ty: ty.clone(),
            value: value.clone(),
        },
    }
}

/// Variable-identity primary walker — clones every `Variable` operand
/// directly out of [`OpKind`]'s arm fields.
///
/// RPython parity — upstream `SpaceOperation.args` is already a
/// `Vec<Hlvalue>` where each `Hlvalue::Variable` carries the
/// authoritative operand identity (`flowspace/model.py:140`).  Pyre's
/// `OpKind` already stores `flowspace::model::Variable` per operand,
/// so this walker needs no graph round-trip.
pub fn op_variable_refs(kind: &OpKind) -> Vec<crate::flowspace::model::Variable> {
    let clone_var = |var: &crate::flowspace::model::Variable| var.clone();
    match kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstSymbolic { .. }
        | OpKind::ConstFloat(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstRefAddr(_)
        | OpKind::CurrentTraceLength
        | OpKind::Live
        | OpKind::LoopHeader { .. }
        | OpKind::Abort { .. }
        | OpKind::LoadStatic { .. }
        | OpKind::NewWithVtable { .. } => {
            vec![]
        }
        OpKind::NewTuple { args } => args.iter().map(clone_var).collect(),
        OpKind::LoweredBlackholeOp { args, .. } => args.iter().map(clone_var).collect(),
        OpKind::VableForce { base } => vec![clone_var(base)],
        OpKind::Hint { value, .. } => vec![clone_var(value)],
        OpKind::JitMergePoint {
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        } => {
            let mut v = Vec::with_capacity(
                greens_i.len()
                    + greens_r.len()
                    + greens_f.len()
                    + reds_i.len()
                    + reds_r.len()
                    + reds_f.len(),
            );
            v.extend(greens_i.iter().map(&clone_var));
            v.extend(greens_r.iter().map(&clone_var));
            v.extend(greens_f.iter().map(&clone_var));
            v.extend(reds_i.iter().map(&clone_var));
            v.extend(reds_r.iter().map(&clone_var));
            v.extend(reds_f.iter().map(&clone_var));
            v
        }
        OpKind::FieldRead { base, .. } => vec![clone_var(base)],
        OpKind::FieldWrite { base, value, .. } => {
            // `setfield_gc` stores either a register operand or an inline
            // constant; only a `Variable` value contributes an SSA
            // reference (a `Const` literal carries no defining op).
            let mut refs = vec![clone_var(base)];
            if let Some(var) = value.as_variable() {
                refs.push(clone_var(var));
            }
            refs
        }
        OpKind::ArrayRead { base, index, .. } => vec![clone_var(base), clone_var(index)],
        OpKind::ArrayWrite {
            base, index, value, ..
        } => {
            // `setarrayitem_gc` stores either a register operand or an
            // inline constant; only a `Variable` value contributes an SSA
            // reference (a `Const` literal carries no defining op).
            let mut refs = vec![clone_var(base), clone_var(index)];
            if let Some(var) = value.as_variable() {
                refs.push(clone_var(var));
            }
            refs
        }
        OpKind::InteriorFieldRead { base, index, .. } => vec![clone_var(base), clone_var(index)],
        OpKind::InteriorFieldWrite {
            base, index, value, ..
        } => vec![clone_var(base), clone_var(index), clone_var(value)],
        OpKind::Call { args, .. } => args.iter().map(&clone_var).collect(),
        OpKind::GuardTrue { cond } | OpKind::GuardFalse { cond } => vec![clone_var(cond)],
        OpKind::GuardValue { value, .. } => vec![clone_var(value)],
        OpKind::AssertGreen { value, .. }
        | OpKind::IsConstant { value, .. }
        | OpKind::IsVirtual { value, .. } => vec![clone_var(value)],
        OpKind::IsInstance {
            obj, class_carrier, ..
        } => {
            vec![clone_var(obj), clone_var(class_carrier)]
        }
        OpKind::VtableMethodPtr { receiver, .. } => vec![clone_var(receiver)],
        OpKind::IndirectCall { funcptr, args, .. } => {
            let mut v = vec![clone_var(funcptr)];
            v.extend(args.iter().map(&clone_var));
            v
        }
        OpKind::RecordQuasiImmutField { base, .. } => vec![clone_var(base)],
        OpKind::JitDebug { args, .. } => args.iter().map(&clone_var).collect(),
        OpKind::VableFieldRead { base, .. } => vec![clone_var(base)],
        OpKind::VableFieldWrite { base, value, .. } => {
            // Only a `Variable` value contributes an SSA reference; a
            // `setfield_vable_i` inline `Const` carries no defining op.
            let mut refs = vec![clone_var(base)];
            if let Some(var) = value.as_variable() {
                refs.push(clone_var(var));
            }
            refs
        }
        OpKind::VableArrayRead {
            base, elem_index, ..
        } => vec![clone_var(base), clone_var(elem_index)],
        OpKind::VableArrayWrite {
            base,
            elem_index,
            value,
            ..
        } => vec![clone_var(base), clone_var(elem_index), clone_var(value)],
        OpKind::BinOp { lhs, rhs, .. } => vec![clone_var(lhs), clone_var(rhs)],
        OpKind::UnaryOp { operand, .. } => vec![clone_var(operand)],
        OpKind::CallElidable {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallResidual {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallMayForce {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = match funcptr {
                CallFuncPtr::Target(_) => Vec::new(),
                CallFuncPtr::Value(var) => vec![clone_var(var)],
            };
            refs.extend(args_i.iter().map(&clone_var));
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::InlineCall {
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs: Vec<crate::flowspace::model::Variable> =
                args_i.iter().map(&clone_var).collect();
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::ConditionalCall {
            condition,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = vec![clone_var(condition)];
            refs.extend(args_i.iter().map(&clone_var));
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::ConditionalCallValue {
            value,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = vec![clone_var(value)];
            refs.extend(args_i.iter().map(&clone_var));
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::RecordKnownResult {
            result_value,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = vec![clone_var(result_value)];
            refs.extend(args_i.iter().map(&clone_var));
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
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
            let mut refs: Vec<crate::flowspace::model::Variable> =
                greens_i.iter().map(&clone_var).collect();
            refs.extend(greens_r.iter().map(&clone_var));
            refs.extend(greens_f.iter().map(&clone_var));
            refs.extend(reds_i.iter().map(&clone_var));
            refs.extend(reds_r.iter().map(&clone_var));
            refs.extend(reds_f.iter().map(&clone_var));
            refs
        }
    }
}

/// `true` iff `kind` is side-effect-free and may be removed from the
/// graph when its result has no readers.  Direct port of RPython
/// `simplify.py:405-417 CanRemove` set + the lltype-level
/// `lloperation.enum_ops_without_sideeffects()` extension that
/// `simplify.py:414-416` unions in.
///
/// Pure ops correspond to RPython:
///   - flowspace pure: `add sub mul div mod ... lt le eq ... bool len
///     hash getattr getitem`-family (`simplify.py:407-412`).
///   - lltype pure: `int_add int_lt ... getfield(_pure)
///     getarrayitem(_pure) getinteriorfield ... cast_*` from
///     `enum_ops_without_sideeffects`.
///
/// Side-effecting ops correspond to RPython `setfield setarrayitem
/// setinteriorfield` (writes), `direct_call indirect_call` (calls
/// without elidable EI), every `*guard*` opname (control-flow guards),
/// and the JIT marker family (`jit_marker`, `debug_merge_point`,
/// `loop_header`, `live`, `record_known_result`,
/// `record_quasiimmut_field`, `jit_debug`).
///
/// Used by `model::prune_dead_phis` to mirror PyPy
/// `simplify.py:441-445`'s split: pure ops route their args via
/// `dependencies[op.result] += op.args` (args become live only if
/// the result becomes live), while non-pure ops add their args
/// directly to `read_vars` (args always live).  Without the split,
/// a phi feeding only a dead pure op would be kept alive via the
/// pure op's args even though both should die together.
pub fn is_pure_op(kind: &OpKind) -> bool {
    match kind {
        // `new_with_vtable` heap-allocates a fresh boxed object: NOT pure.
        // CSE must never coalesce two allocations, or Python `is` object
        // identity would break.
        OpKind::NewWithVtable { .. } => false,
        // `OpKind::ConstInt` / `OpKind::ConstFloat` materialize a
        // `Variable` for a literal in pyre's IR.  There
        // is NO upstream `int_constant` op — RPython's `Constant` is
        // a value class (`flowmodel.py Constant(rfloat)`), not an
        // operation, so it appears inline in `op.args` rather than
        // as a standalone op in `block.operations`.  Pyre's
        // op-shaped representation is forced by the
        // `Block.operations: Vec<Op>` /
        // `Op.result: Option<Variable>` design: every value the
        // graph produces must be materialised through an op so a
        // backing `Variable` can be minted for it.  Returning `true` here
        // is the dataflow-equivalent of upstream's "Constant args
        // pin nothing" behaviour: the const op is removed by
        // `prune_dead_phis` Step 5 when its result is unread, which
        // mirrors upstream's "dead Constant simply doesn't appear in
        // any live op's args" outcome.
        //
        // `Input` is structurally pure: inputarg-shaped Input ops
        // are protected from `model::prune_dead_phis` Step 5 sweep
        // by their result `Variable` being pinned in `read_vars` (Step
        // 1+3+dependency-routing); naked Input ops (legacy frontend
        // fallback) are removed by Step 5 when their result is
        // dead.  Returned as `true` here for consistency with the
        // dependency-routing classification.
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstBool(_)
        // `_we_are_jitted` symbolic const — a pure `Constant`, removable
        // when its result is unread (folded to a real const by
        // `jtransform` before assembly).
        | OpKind::ConstSymbolic { .. }
        | OpKind::ConstFloat(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstRefAddr(_)
        // Pure reads — `getfield(_pure) getarrayitem(_pure)
        // getinteriorfield` in `enum_ops_without_sideeffects`.
        | OpKind::FieldRead { .. }
        | OpKind::ArrayRead { .. }
        | OpKind::InteriorFieldRead { .. }
        // Pure virtualizable reads — no heap mutation.
        | OpKind::VableFieldRead { .. }
        | OpKind::VableArrayRead { .. }
        // Pure vtable slot read — `cast_pointer + getfield` chain
        // collapsed into one op (see `OpKind::VtableMethodPtr` doc).
        | OpKind::VtableMethodPtr { .. }
        // `newtuple` is `PureOperation` (`operation.py:542-548`).
        | OpKind::NewTuple { .. }
        // `isinstance` lowers to `int_between` over `obj.typeptr`'s
        // subclass-range fields plus an optional null branch — all
        // pure reads, classified `canfold=True` upstream
        // (`lloperation.py instance_isinstance`).  Keeping dead
        // `IsInstance` results alive would block prune_dead_phis Step
        // 5 even though the predicate is side-effect-free.
        | OpKind::IsInstance { .. }
        // `LoadStatic` reads a `static` declaration's compile-time
        // address — equivalent to `LOAD_GLOBAL` → Constant lookup,
        // pure.
        | OpKind::LoadStatic { .. } => true,
        // Per-opname classification for `OpKind::BinOp` mirrors
        // `simplify.CanRemove` (`simplify.py:405-417`) +
        // `enum_ops_without_sideeffects()` for binary ops.  Pyre's
        // `OpKind::BinOp` carries the opname as a string field, so
        // the parity-correct classification is opname-keyed rather
        // than enum-blanket — short-circuit `&&` / `||` (Rust
        // logical-and / -or) emit `BinOp{op:"and"/"or"}` which have
        // no flowspace peer and must surface a fail-loud TyperError
        // at the rtyper rather than be silently DCE'd.
        OpKind::BinOp { op, .. } => is_pure_binop_opname(op),
        // Per-opname classification mirrors `enum_ops_without_sideeffects()`'s
        // `LL_OPERATIONS[opname].sideeffects` lookup
        // (`rpython/rtyper/lltypesystem/lloperation.py:128-134`).
        // Pyre's `OpKind::UnaryOp` carries the opname as a string
        // field, so the parity-correct classification is opname-keyed
        // rather than enum-blanket.
        OpKind::UnaryOp { op, .. } => is_pure_unary_opname(op),
        // Side-effecting writes / calls / guards / markers / aborts.
        // `direct_call`-family ops are routed here even when the
        // callee is elidable: `simplify.py:441-445`'s `canremove`
        // split treats `direct_call` as side-effecting (args go
        // straight to `read_vars`), and `simplify.py:500` performs a
        // separate elidable-graph removal that requires `translator`
        // to be supplied (pyre's call site passes `translator=None`,
        // so the removal arm is unreachable).  The post-jtransform
        // `OpKind::CallElidable` shape arrives after `prune_dead_phis`
        // has already run, so its classification here is moot for the
        // current pipeline ordering.
        OpKind::FieldWrite { .. }
        | OpKind::ArrayWrite { .. }
        | OpKind::InteriorFieldWrite { .. }
        | OpKind::VableFieldWrite { .. }
        | OpKind::VableArrayWrite { .. }
        | OpKind::Call { .. }
        | OpKind::IndirectCall { .. }
        | OpKind::CallResidual { .. }
        | OpKind::CallMayForce { .. }
        | OpKind::CallElidable { .. }
        | OpKind::InlineCall { .. }
        | OpKind::RecursiveCall { .. }
        | OpKind::ConditionalCall { .. }
        | OpKind::ConditionalCallValue { .. }
        | OpKind::RecordKnownResult { .. }
        | OpKind::RecordQuasiImmutField { .. }
        | OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::IsConstant { .. }
        | OpKind::IsVirtual { .. }
        | OpKind::CurrentTraceLength
        | OpKind::JitDebug { .. }
        | OpKind::JitMergePoint { .. }
        | OpKind::LoopHeader { .. }
        | OpKind::Live
        | OpKind::VableForce { .. }
        | OpKind::Hint { .. }
        | OpKind::Abort { .. } => false,

        // `LoweredBlackholeOp` carries register-shaped blackhole insns whose
        // side-effect class is opname-dependent — the same split `op_can_raise`
        // (`call.rs`) and upstream `lloperation.py:382-383` (sideeffects /
        // canfold) make.  The read family
        // (`strlen`/`unicodelen`/`strgetitem`/`unicodegetitem`) is
        // pure/removable; the alloc/store/copy family
        // (`newstr`/`newunicode`/`strsetitem`/`unicodesetitem`/
        // `copystrcontent`/`copyunicodecontent`) mutates.  Any unlisted opname
        // falls through to side-effecting, so the dead-op sweep never drops a
        // store or an allocation by default.
        OpKind::LoweredBlackholeOp { opname, .. } => matches!(
            opname.as_str(),
            "strlen" | "unicodelen" | "strgetitem" | "unicodegetitem"
        ),
    }
}

/// Whitelist of `OpKind::UnaryOp` opnames that are side-effect-free
/// upstream — direct port of the unary entries in
/// `simplify.CanRemove` (`rpython/translator/simplify.py:405-417`)
/// + `enum_ops_without_sideeffects()`
/// (`rpython/rtyper/lltypesystem/lloperation.py:128-134`).
///
/// Any opname not in this list is treated as side-effecting so the
/// dead-op DCE pass does not silently remove it.  Notably absent:
/// `not` — Python's `not` is control flow, `operation.py:465-474`
/// does not register it; pyre's adapter
/// (`translator/rtyper/flowspace_adapter.rs:344-353`) requires the
/// frontend to desugar `!x` away before reaching the rtyper, so DCE
/// must surface a live `not` op to the rtyper rather than silently
/// dropping a dead one.  `front::mir` collapses Rust deref `*x`
/// (no flowspace peer) when it lowers `UnaryOp`/`Deref`, so the
/// frontend never emits it; when the frontend stops emitting `not`,
/// this whitelist will only retain post-jtransform / rtyper-emitted
/// opnames.
/// Whitelist of `OpKind::BinOp` opnames that are side-effect-free
/// upstream — direct port of the binary entries in
/// `simplify.CanRemove` (`simplify.py:405-417`) +
/// `enum_ops_without_sideeffects()`.
///
/// Notable omissions:
///   - `and` / `or` (no trailing underscore): Rust `&&` / `||`
///     short-circuit operators; `operation.py:475-510` does not
///     register them as binary operators (they are control flow),
///     and `translator/rtyper/flowspace_adapter.rs:392-400` requires
///     the frontend to desugar them before reaching the rtyper.  DCE
///     must keep dead occurrences alive so the rtyper surfaces the
///     fail-loud TyperError instead of silently dropping them.  The
///     trailing-underscore canonical names `and_` / `or_` (PyPy's
///     bitwise AND/OR registered at `operation.py:485-486`) ARE
///     pure and listed below.
///   - `*_assign` (Rust compound assignments): `inplace_*` upstream;
///     `simplify.py:CanRemove` does not include `inplace_*`, and
///     compound assignments mutate their LHS so DCE shouldn't drop
///     a live-LHS write just because the result vid is unread.
///   - `unknown_binop`: pyre fallback for unsupported ops;
///     fail-loud territory.
///
/// CanRemove entries that do not surface as `OpKind::BinOp` or
/// `OpKind::UnaryOp` (handled elsewhere or shape-incompatible):
///   - `newtuple` / `newlist` / `newdict` — collection constructors,
///     surfaced by frontend lowering passes, not via BinOp/UnaryOp.
///   - `getattr` — variable arity (`getattr(o, name [, default])`),
///     pyre lowers as `OpKind::FieldRead` / Call rather than BinOp.
///   - `get` (`operation.py:514`) — 3-arg descriptor `__get__`; pyre
///     has no TernaryOp surface.  Drop is the call-DCE's concern.
fn is_pure_binop_opname(opname: &str) -> bool {
    if matches!(
        opname,
        // `simplify.py:405-417` CanRemove — every entry registered as
        // a 2-arg `add_operator(...)` at `flowspace/operation.py`.
        // Arithmetic — `operation.py:475-484`.
        "add"
        | "sub"
        | "mul"
        | "div"
        | "truediv"
        | "floordiv"
        | "mod"
        | "divmod"
        | "pow"
        | "lshift"
        | "rshift"
        // Canonical PyPy bitwise binops — `simplify.py:410-411`
        // `and_ or_ xor` (`operation.py:485-487`).  These are the
        // trailing-underscore forms that surface after
        // `flowspace_adapter.rs:379-381 normalize_binop_name` rewrites
        // pyre's `bitand`/`bitor`/`bitxor`; both forms are pure so the
        // DCE whitelist accepts pre- and post-normalize names.
        | "and_"
        | "or_"
        | "xor"
        | "bitand"
        | "bitor"
        | "bitxor"
        // Comparisons — `simplify.py:411 lt le eq ne gt ge`.
        | "lt"
        | "le"
        | "eq"
        | "ne"
        | "gt"
        | "ge"
        // Remaining 2-arg CanRemove entries:
        //   is_       operation.py:445  (identity test)
        //   issubtype operation.py:448  (issubclass for new-style classes)
        //   isinstance operation.py:449
        //   getitem   operation.py:457
        //   cmp       operation.py:511  (`simplify.py:411`)
        //   coerce    operation.py:512  (`simplify.py:411`)
        //   contains  operation.py:513  (`simplify.py:411`)
        | "is_"
        | "issubtype"
        | "isinstance"
        | "getitem"
        | "cmp"
        | "coerce"
        | "contains"
    ) {
        return true;
    }
    // Post-rtyper / lltype binops — `enum_ops_without_sideeffects()`
    // (`lloperation.py:128-134`) registers `int_add int_sub int_mul
    // int_lt ... uint_add ... float_add ...` with sideeffects=False.
    // Pyre's BinOp arrives here pre-rtyper (frontend names like
    // `add`); the post-rtyper shape is also accepted so a future
    // post-rtyper DCE call site requires no further widening.
    let prefix_match = opname
        .strip_prefix("int_")
        .or_else(|| opname.strip_prefix("uint_"))
        .or_else(|| opname.strip_prefix("float_"));
    if let Some(suffix) = prefix_match {
        return matches!(
            suffix,
            "add"
                | "sub"
                | "mul"
                | "floordiv"
                | "mod"
                | "lshift"
                | "rshift"
                | "and"
                | "or"
                | "xor"
                | "lt"
                | "le"
                | "eq"
                | "ne"
                | "gt"
                | "ge"
        );
    }
    false
}

fn is_pure_unary_opname(opname: &str) -> bool {
    matches!(
        opname,
        // `simplify.py:405-417` CanRemove — every entry registered as
        // a 1-arg `add_operator(...)` at `flowspace/operation.py`.
        //   id     operation.py:446
        //   type   operation.py:447
        //   repr   operation.py:450
        //   str    operation.py:451
        //   len    operation.py:453
        //   hash   operation.py:454
        //   pos    operation.py:465
        //   neg    operation.py:466
        //   bool   operation.py:467
        //   abs    operation.py:469
        //   hex    operation.py:470
        //   oct    operation.py:471
        //   ord    operation.py:473
        //   invert operation.py:474
        //   int    operation.py:488
        //   float  operation.py:490
        //   long   operation.py:491
        //   iter   operation.py:582 (Iter HLOperation, arity=1)
        "id"
        | "type"
        | "repr"
        | "str"
        | "len"
        | "hash"
        | "pos"
        | "neg"
        | "bool"
        | "abs"
        | "hex"
        | "oct"
        | "ord"
        | "invert"
        | "int"
        | "float"
        | "long"
        | "iter"
        // `same_as` — `jtransform.py:246-248 rewrite_op_same_as`
        // explicitly drops the op + aliases the result.
        | "same_as"
        // `cast_pointer cast_ptr_to_int cast_int_to_ptr ...` — pure
        // RPython `enum_ops_without_sideeffects` peers.
        | "cast_pointer"
        | "cast_ptr_to_int"
        | "cast_int_to_ptr"
        | "cast_opaque_ptr"
    )
}

/// Replace all occurrences of `old_var` with `new_var` in ops within the specified blocks.
fn remap_value_in_graph(
    graph: &mut FunctionGraph,
    block_map: &HashMap<BlockId, BlockId>,
    old_var: &crate::flowspace::model::Variable,
    new_var: &crate::flowspace::model::Variable,
) {
    let target_blocks: Vec<BlockId> = block_map.values().copied().collect();
    for &bid in &target_blocks {
        let (exitswitch, exits) = {
            let block = &graph.blocks[bid.0];
            remap_control_flow_metadata_var(
                &block.exitswitch,
                &block.exits,
                |var| {
                    if var == old_var {
                        new_var.clone()
                    } else {
                        var.clone()
                    }
                },
                |b| b,
            )
        };
        let new_ops = remap_value_in_ops(&graph.blocks[bid.0].operations, old_var, new_var);
        let block = &mut graph.blocks[bid.0];
        block.operations = new_ops;
        block.exitswitch = exitswitch;
        block.exits = exits;
    }
}

/// Replace all occurrences of `old_var` with `new_var` in a list of ops.
fn remap_value_in_ops(
    ops: &[SpaceOperation],
    old_var: &crate::flowspace::model::Variable,
    new_var: &crate::flowspace::model::Variable,
) -> Vec<SpaceOperation> {
    let remap_var = |var: &crate::flowspace::model::Variable| {
        if var == old_var {
            new_var.clone()
        } else {
            var.clone()
        }
    };
    ops.iter()
        .map(|op| SpaceOperation {
            result: op.result.clone(),
            kind: remap_op_kind(&op.kind, &remap_var),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call::CallControl;
    use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};
    use crate::parse::CallPath;

    #[test]
    fn lowered_blackhole_op_purity_is_opname_aware() {
        let pure = |name: &str| {
            is_pure_op(&OpKind::LoweredBlackholeOp {
                opname: name.to_string(),
                args: vec![],
            })
        };
        // Read family: pure/removable when the result is dead.
        for name in ["strlen", "unicodelen", "strgetitem", "unicodegetitem"] {
            assert!(pure(name), "{name} is a pure read");
        }
        // Alloc / store / copy family: side-effecting, must never be dropped.
        for name in [
            "newstr",
            "newunicode",
            "strsetitem",
            "unicodesetitem",
            "copystrcontent",
            "copyunicodecontent",
        ] {
            assert!(!pure(name), "{name} mutates and must be kept");
        }
    }

    fn make_simple_callee() -> FunctionGraph {
        // Callee: fn callee(base) -> value { ArrayRead(base, const(0)) }
        let mut g = FunctionGraph::new("callee");
        let entry = g.startblock;
        let base_var = g
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let idx_var = g.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let result = g.push_op_var(
            entry,
            OpKind::ArrayRead {
                base: base_var,
                index: idx_var,
                item_ty: ValueType::Ref(None),
                array_type_id: None,
                nolength: false,
            },
            true,
        );
        g.set_return(entry, result);
        g
    }

    #[test]
    fn inline_single_call() {
        // Caller: fn caller() { v = Call("callee", [base]); Return v }
        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let base_var = caller
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = caller.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: vec![base_var],
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(entry, result);

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
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::ArrayRead { .. }));
        assert!(
            has_array_read,
            "inlined graph should contain ArrayRead from callee"
        );

        // Should NOT have the original Call op
        let has_call = caller
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::Call { .. }));
        assert!(!has_call, "Call op should be replaced by inlined body");
    }

    #[test]
    fn inline_preserves_residual_calls() {
        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let result = caller.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["unknown_fn"]),
                args: vec![],
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(entry, result);

        let cc = CallControl::new(); // empty — no graphs registered
        let count = inline_graph(&mut caller, &cc, 3);
        assert_eq!(count, 0);

        // Call should still be there
        let has_call = caller
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::Call { .. }));
        assert!(has_call, "residual Call should be preserved");
    }

    #[test]
    fn inline_two_levels() {
        // inner: fn inner(x) -> ArrayRead(x, 0)
        let inner = make_simple_callee();

        // outer: fn outer(base) -> Call("callee", [base])
        let mut outer = FunctionGraph::new("outer");
        let entry = outer.startblock;
        let base_var = outer
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = outer.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: vec![base_var],
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        outer.set_return(entry, result);

        // caller: fn caller(x) -> Call("outer", [x])
        let mut caller = FunctionGraph::new("caller");
        let centry = caller.startblock;
        let x_var = caller
            .push_op_var(
                centry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = caller.push_op_var(
            centry,
            OpKind::Call {
                target: CallTarget::function_path(["outer"]),
                args: vec![x_var],
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(centry, result);

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
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::ArrayRead { .. }));
        assert!(
            has_array_read,
            "2-level inlined graph should have ArrayRead"
        );
    }

    /// Post-inline regression: every block whose terminator is a
    /// control-flow op (Goto/Branch) must carry matching `Block.exits`
    /// metadata, and every resulting Link must stamp `prevblock` with
    /// the block it exits.  RPython `flowspace/model.py:174` keeps
    /// `exitswitch`/`exits` as the single CFG source of truth, so pyre
    /// must not let the inline rewrite produce terminator/exits drift
    /// or `prevblock = None` links.
    #[test]
    fn inline_preserves_exits_and_prevblock_invariants() {
        let callee = make_simple_callee();

        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let base_var = caller
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = caller.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: vec![base_var],
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(entry, result);

        let mut cc = CallControl::new();
        cc.register_function_graph(CallPath::from_segments(["callee"]), callee);
        cc.find_all_graphs_for_tests();

        let count = inline_graph(&mut caller, &cc, 3);
        assert!(count >= 1, "callee should inline at least once");

        for block in &caller.blocks {
            // Upstream `flowspace/model.py:171-180` — a closed block (one
            // with `exitswitch.is_some()` or at least one exit) always
            // carries both: the exitswitch names the branch condition,
            // the exits hold every outgoing `Link`.  An unclosed block
            // (startblock before its first closeblock, or a freshly
            // created merge block with no afterblock payload) has
            // `exits=()` and `exitswitch=None`, matching the initial
            // state in `FunctionGraph.__init__`.
            assert!(
                block.is_closed() || !block.exits.is_empty() || block.exitswitch.is_none(),
                "block {:?} exits/exitswitch out of sync after inline",
                block.id
            );
            for link in &block.exits {
                assert_eq!(
                    link.prevblock,
                    Some(block.id),
                    "link in block {:?} targeting {:?} missing prevblock stamp",
                    block.id,
                    link.target
                );
            }
        }
    }
}
