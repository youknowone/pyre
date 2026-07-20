//! Iterator `next()` → `next` op + StopIteration handler lowering.
//!
//! ## Positioning
//!
//! Layer 3 of the iterator vertical (annotator iter/next ops = Layer 1,
//! rtyper `ListIteratorRepr` = Layer 2): the front-end lift that turns a
//! Rust `Iterator::next()` call and its `Option` match into the graph's
//! native `next` op with a StopIteration exception edge.
//!
//! Rust source models `for x in it` as
//! ```text
//!     opt = Iterator::next(&mut it);    // -> Option<T>
//!     match opt { Some(x) => body, None => break }
//! ```
//! lowered (in MIR) as a `__discriminant` read on `opt` plus a two-way
//! `switchInt` (None = 0, Some = 1).  RPython's `next` op returns the
//! element directly and raises `StopIteration` at exhaustion
//! (`rpython/flowspace/operation.py` `next`; the annotator's
//! `op.next.can_only_throw` is `[StopIteration, RuntimeError]`).  This
//! module rewrites the value-encoded Option diamond into that exception
//! representation — the mirror of [`crate::front::result_exc`]'s
//! `Result`/`?` rewrite.  The Option diamond is the same shape minus the
//! `branch()` indirection, and its exception edge targets the loop-break
//! arm (a local `StopIteration` catch) instead of `exceptblock` — the
//! `try: x = next(it) except StopIteration: break` shape.  The rewrite
//! gates on the iterator tracing back to an `iter` op
//! (`originates_from_iter_op`), so only a list iterator reaches it, and
//! `ListIteratorRepr::rtype_next` raises solely `StopIteration` — the
//! block carries no catch-all propagation edge (the `RuntimeError` half
//! of `next`'s conservative `can_only_throw` is dict-iterator mutation
//! detection that never fires here, and `flowin` drops the unhandled
//! remainder).
//!
//! ## The rewrite (`rewire_one_next_site`)
//!
//! Block A holds the `next()` residual call producing `opt`; block C is
//! its single successor — the discriminant switch.  The rewrite:
//! 1. replaces A's residual call with the native `next` op (the
//!    `[__iter_next]` marker the [`crate::translator::rtyper::flowspace_adapter`]
//!    maps to the raising flowspace `next` op), reusing `opt` as the
//!    element result;
//! 2. closes A with `LastException` exits — normal → the `Some` arm
//!    (`opt.__pos_0` collapses to the element), `StopIteration` → the
//!    `None` arm (loop break).
//!
//! The `None` arm carries no value in RPython (`ll_listnext` raises
//! `StopIteration` without a result), so any `Option` scrutinee the Rust MIR
//! threads through the break continuation is a value-encoded SSA merge with
//! no RPython counterpart.  `collect_transitive_dead_slots` follows that
//! thread across its whole forward chain and prunes every slot that is never
//! read, switched on, or escaped — a chain reaching a genuine use declines.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `next` callee
//! makes the rtyper census Skip the graph (no regression vs the legacy
//! walker).

use crate::flowspace::model::{ConstValue, Constant, Variable};
use crate::front::result_exc::{
    assert_block_pure_besides, assert_single_pred, back_substitute, collapse_pos0_read,
    follow_single_exit, op_operand_vars, split_diamond_exits,
};
use crate::model::{
    CallTarget, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
    ValueType,
};

/// The `[__iter_next]` FunctionPath marker the rewrite emits in place of
/// the residual `Iterator::next()` call.  `flowspace_adapter::translate_op`
/// maps it to the raising flowspace `next` op (`operation.rs` `OpKind::Next`).
pub(crate) fn next_op_segments() -> Vec<String> {
    vec!["__iter_next".to_string()]
}

/// `true` iff `segments` is the `[__iter_next]` marker.
pub(crate) fn is_iter_next_segments(segments: &[String]) -> bool {
    segments.len() == 1 && segments[0] == "__iter_next"
}

/// `true` iff the residual call target is an `Iterator::next()` — a
/// `next`-leaf method or FunctionPath.  Combined with an `Option` return
/// type at the recording site, this records a `next`-diamond candidate;
/// the rewrite itself validates the surrounding match shape.
pub(crate) fn is_iterator_next_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => name == "next",
        CallTarget::FunctionPath { segments } => segments.last().is_some_and(|s| s == "next"),
        _ => false,
    }
}

/// `true` iff `segments` is the `core::slice` `iter` constructor the
/// front-end lowers a slice/Vec/array iterator to.  Two spellings reach
/// here: `is_concrete_iter_constructor`'s `into_iter` rewrite collapses to
/// `["core", "slice", "iter"]`, while a direct `<[T]>::iter` method call
/// (e.g. `boxed_slice.iter()`) stays as the raw method path
/// `["core", "slice", "<Impl>", "iter"]`.  Both name the same constructor —
/// `flowspace_adapter::nonraising_core_bridge_opname` already maps both to
/// the `iter` flowspace op feeding `ListIteratorRepr` by keying on the
/// `slice` family + `iter` leaf, so mirror that family/leaf match here
/// rather than the exact-length form (which missed the method spelling and
/// declined every `boxed_slice.iter()` for-loop).
fn is_iter_op_segments(segments: &[String]) -> bool {
    segments.len() >= 3
        && segments[0] == "core"
        && segments[1] == "slice"
        && segments.last().is_some_and(|s| s == "iter")
}

/// `true` iff `var` originates — directly or through loop-carried block
/// inputargs — from an `iter` op (a list iterator).  A backward walk: a
/// var produced by an `iter` op is a list iterator; a var that is a block
/// inputarg is traced to each predecessor's link arg in the matching slot
/// (so the loop header's iterator phi resolves through its entry edge to
/// the pre-loop `iter` op, while the back edge re-threads an already-seen
/// var).  Conservative: a var produced by any other op (a reborrow, a
/// foreign iterator constructor) is not followed, so the walk returns
/// `true` only on a positively-confirmed `iter` source.
fn originates_from_iter_op(graph: &FunctionGraph, var: &Variable) -> bool {
    let mut visited: Vec<Variable> = Vec::new();
    let mut stack: Vec<Variable> = vec![var.clone()];
    while let Some(v) = stack.pop() {
        if visited.contains(&v) {
            continue;
        }
        visited.push(v.clone());
        // (1) produced directly by an iter op → confirmed list iterator.
        for b in &graph.blocks {
            for op in &b.operations {
                if op.result.as_ref() == Some(&v)
                    && let OpKind::Call {
                        target: CallTarget::FunctionPath { segments },
                        ..
                    } = &op.kind
                    && is_iter_op_segments(segments)
                {
                    return true;
                }
            }
        }
        // (2) a block inputarg → trace each predecessor's link arg in the
        // matching slot (the loop-carried iterator phi).
        for b in &graph.blocks {
            if let Some(pos) = b.inputargs.iter().position(|iv| iv == &v) {
                let target_id = b.id;
                for pb in &graph.blocks {
                    for link in &pb.exits {
                        if link.target == target_id
                            && let Some(LinkArg::Value(src)) = link.args.get(pos)
                        {
                            stack.push(src.clone());
                        }
                    }
                }
            }
        }
    }
    false
}

/// Transitive closure of dead forwarded inputarg slots starting from
/// `(start_block, start_slot)`.  Returns the full `(block, slot)` removal
/// set, or an abort reason if any node in the closure is read by an op,
/// tested by an exitswitch, or escapes to the return/except block.
/// Validate-before-mutate: callers mutate only on `Ok`.  RPython's
/// `ll_listnext` StopIteration exhaustion path carries no value, so a
/// forwarded-but-never-read `Option` thread is a Rust SSA-merge artifact
/// prunable along its whole forward chain; removing a never-read inputarg
/// slot drops only dead arguments from each predecessor link (the values
/// stay defined and used wherever else they are live).
fn collect_transitive_dead_slots(
    graph: &FunctionGraph,
    start_block: usize,
    start_slot: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let mut set: Vec<(usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(start_block, start_slot)];
    while let Some((bi, si)) = stack.pop() {
        if set.contains(&(bi, si)) {
            continue; // cycle / already in the removal set
        }
        // The return/except block inputargs are observable (return value /
        // exception payload), so a forward into them escapes — not removable.
        if bi == graph.returnblock.0 || bi == graph.exceptblock.0 {
            return Err(format!("escapes to return/except block {bi}"));
        }
        let Some(v) = graph.blocks[bi].inputargs.get(si).cloned() else {
            return Err(format!("block {bi} lacks inputarg slot {si}"));
        };
        // Genuinely observed → not removable.
        if graph.blocks[bi]
            .operations
            .iter()
            .any(|op| op_operand_vars(&op.kind).contains(&v))
        {
            return Err(format!("slot {si} of block {bi} read by an op"));
        }
        match &graph.blocks[bi].exitswitch {
            Some(ExitSwitch::Value(x)) if *x == v => {
                return Err(format!("slot {si} of block {bi} is the exitswitch"));
            }
            Some(ExitSwitch::Fused { args, .. }) if args.contains(&v) => {
                return Err(format!(
                    "slot {si} of block {bi} is a fused exitswitch operand"
                ));
            }
            _ => {}
        }
        set.push((bi, si));
        // Follow every onward forward of `v` to its target slot.
        for l in &graph.blocks[bi].exits {
            for (j, a) in l.args.iter().enumerate() {
                if matches!(a, LinkArg::Value(x) if *x == v) {
                    stack.push((l.target.0, j));
                }
            }
        }
    }
    Ok(set)
}

/// The typed `StopIteration` exitcase the `next` block's break link
/// carries — the handler analogue of [`crate::model::exception_exitcase`]
/// (`Exception` catch-all), narrowed to the single exception the loop
/// catches.  `ConstValue::builtin` resolves the class to a `HostObject`,
/// the `Constant(HostObject(class))` shape `annrpython::flowin` matches.
fn stopiteration_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("StopIteration"))
}

fn int_const(i: i64) -> LinkArg {
    LinkArg::Const(Constant::new(ConstValue::Int(i)))
}

/// Rewrite every recorded `next()` call site into the `next` op +
/// StopIteration handler shape.  Fail-safe: a site whose surrounding
/// `Option` match does not fit the for-loop shape is left as the residual
/// call (Skip), so a mismatch never regresses a graph the legacy walker
/// already handled.  Returns the number of sites rewritten.
pub(crate) fn rewire_next_call_sites(graph: &mut FunctionGraph, sites: &[Variable]) -> usize {
    let mut rewritten = 0;
    for opt in sites {
        match rewire_one_next_site(graph, opt) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `next` call; the unregistered callee
                // makes the rtyper census Skip this graph (no regression).
            }
        }
    }
    rewritten
}

/// `true` iff `segments` is a `__pyre_cast_instance` narrow — the front-end
/// pointer-downcast marker (`front::mir` `Rvalue::Cast` #298 arm) the MIR
/// emits when an opaque `Ref` result is reinterpreted as a registered struct
/// root.  It lowers to `cast_pointer` (a pure alias), so it carries no
/// side effect and its result is bit-identical to its operand.
fn is_recast_narrow(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if args.len() == 1 && segments.first().is_some_and(|s| s == "__pyre_cast_instance")
    )
}

/// An unregistered `next()` returns an opaque `Ref`; the MIR immediately
/// narrows it to the concrete `Option<T>` with one or more pure
/// `__pyre_cast_instance` recasts (the front-end analogue of
/// `ListIteratorRepr::rtype_next`'s `recast`, `rpython/rtyper/rlist.py:448`).
/// Peel that trailing chain: starting from the raw `next` result at
/// `next_idx`, follow each contiguous recast whose sole operand is the prior
/// result, and return the chain's final result — the value the block forwards
/// as the `Option` scrutinee.  Every op from `next_idx+1` to the block end
/// must be part of the chain (so the native `next` op, replacing the whole
/// chain, becomes the block's last / raising op), and no chain intermediate
/// may be read by anything but the next recast (else collapsing the chain to
/// a single value would drop a live use).  With no recast (`next` registered,
/// or a list element that needs no narrow) the chain is length 0 and this
/// returns `opt` unchanged.  Any other shape declines (fail-safe).
fn peel_recast_chain(
    graph: &FunctionGraph,
    a: usize,
    next_idx: usize,
    opt: &Variable,
) -> Result<Variable, String> {
    let name = &graph.name;
    let ops = &graph.blocks[a].operations;
    let mut cur = opt.clone();
    let mut idx = next_idx;
    while idx + 1 < ops.len() {
        let follow = &ops[idx + 1];
        // The next op must be a recast whose single operand is `cur`.
        let OpKind::Call { args, .. } = &follow.kind else {
            return Err(format!(
                "{name}: op after next() in block {a} is not the block terminator \
                 nor a recast narrow"
            ));
        };
        if !is_recast_narrow(&follow.kind) || args[0] != cur {
            return Err(format!(
                "{name}: op after next() in block {a} is not a recast of the prior result"
            ));
        }
        // `cur` (a non-final chain intermediate) must be dead except for
        // this recast: no other op reads it AND it does not escape block A
        // on any exit link arg.  Otherwise truncating the chain to the
        // native `next` result would strand a live reference.
        let other_reads = ops
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx + 1)
            .any(|(_, op)| op_operand_vars(&op.kind).contains(&cur));
        if other_reads {
            return Err(format!(
                "{name}: recast operand in block {a} has a second reader — cannot peel"
            ));
        }
        let escapes = graph.blocks[a].exits.iter().any(|l| {
            l.args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if *v == cur))
        });
        if escapes {
            return Err(format!(
                "{name}: recast operand in block {a} escapes on an exit link — cannot peel"
            ));
        }
        cur = follow
            .result
            .clone()
            .ok_or_else(|| format!("{name}: recast narrow in block {a} has no result"))?;
        idx += 1;
    }
    Ok(cur)
}

fn rewire_one_next_site(graph: &mut FunctionGraph, opt: &Variable) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the block whose op produces `opt` — the residual `next()`
    // call, closed by lower_call with a single forwarding exit.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(opt))
        })
        .ok_or_else(|| format!("{name}: next() result var has no producer block"))?;
    let next_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(opt))
        .ok_or_else(|| format!("{name}: next() producer op vanished from block {a}"))?;

    // Capture the iterator operand (the `next` op's single argument) from
    // the raw call, before peeling any recast narrows off its result.
    let iter_arg = match &graph.blocks[a].operations[next_idx].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: next() producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    // `lower_call` closes the block right after the raising call, so the
    // `next()` call is normally A's last op.  An UNREGISTERED `next()`
    // returns an opaque `Ref` that the MIR immediately recasts to the
    // concrete `Option<T>` via one or more pure `__pyre_cast_instance`
    // narrows (the front-end analogue of `ListIteratorRepr::rtype_next`'s
    // `recast`, `rpython/rtyper/rlist.py:448` `self.r_list.recast`).  Such a
    // narrow is a `cast_pointer` alias, so the native `next` op subsumes it:
    // peel the trailing chain and scrutinise its final result as the Option.
    let effective_opt = peel_recast_chain(graph, a, next_idx, opt)?;
    let opt = &effective_opt;

    // The 2-exit StopIteration-only shape is faithful ONLY for a list
    // iterator (`ListIteratorRepr::rtype_next` / `ll_listnext` raises
    // solely `StopIteration`).  A non-list iterator's `next` annotates
    // `OpKind::Next` over a non-list tag (poison / mis-tag panic) and its
    // `can_only_throw` carries the live `RuntimeError` half the shape
    // drops.  `is_iterator_next_target` alone cannot tell them apart, so
    // gate on the iterator tracing back to an `iter` op — the front-end's
    // slice/Vec/array iterator constructor (`front::mir`
    // `is_concrete_iter_constructor` + the `.iter()` bridge, both lowered
    // to `["core", "slice", "iter"]`).  Anything else declines (the
    // residual call keeps the rtyper Skip), so a non-list iterator never
    // reaches the rewrite.
    if !originates_from_iter_op(graph, &iter_arg) {
        return Err(format!(
            "{name}: next() iterator operand does not originate from an iter op — \
             not a list-iterator for-loop"
        ));
    }

    // A's single exit → C (the Option discriminant switch).  Unlike the
    // Result `?` diamond there is no intervening `branch()` block.
    let (c, opt_c) = follow_single_exit(graph, a, opt)
        .map_err(|e| format!("{name}: next call block exit: {e}"))?;
    assert_single_pred(graph, c, &name)?;

    // Block C: `d = opt.__discriminant`; `switch d { 0 → None, 1 → Some }`.
    let (disc_idx, disc_var) = graph.blocks[c]
        .operations
        .iter()
        .enumerate()
        .find_map(|(i, op)| match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if *base == opt_c && field.name == "__discriminant" =>
            {
                op.result.clone().map(|r| (i, r))
            }
            _ => None,
        })
        .ok_or_else(|| format!("{name}: block {c} lacks the Option __discriminant read"))?;
    match &graph.blocks[c].exitswitch {
        Some(ExitSwitch::Value(v)) if *v == disc_var => {}
        other => {
            return Err(format!(
                "{name}: block {c} exitswitch {other:?} is not the Option discriminant switch"
            ));
        }
    }
    // Block C is bypassed; only the discriminant read may carry an effect.
    assert_block_pure_besides(graph, c, &[disc_idx], "discriminant", &name)?;

    // Option discriminant: None = 0, Some = 1.  `split_diamond_exits`
    // returns `(case 0, case 1)` = `(None arm, Some arm)`.
    let (none_link, some_link) = split_diamond_exits(&graph.blocks[c].exits, &name)?;
    let some_target = some_link.target;
    let none_target = none_link.target;

    // Some arm (normal exit): the `next` op result IS the element.  Map
    // the Some-link args back to A scope; the forwarded Option value
    // becomes the element result, the threaded discriminant the constant 1.
    let mut normal_args: Vec<LinkArg> = Vec::with_capacity(some_link.args.len());
    let mut payload_positions: Vec<usize> = Vec::new();
    for (i, arg) in some_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(c0) => normal_args.push(LinkArg::Const(c0.clone())),
            LinkArg::Value(v) => {
                if *v == opt_c {
                    normal_args.push(LinkArg::Value(opt.clone()));
                    payload_positions.push(i);
                } else if *v == disc_var {
                    normal_args.push(int_const(1));
                } else {
                    let v_a = back_substitute(graph, &[(a, c)], v, &name)?;
                    normal_args.push(LinkArg::Value(v_a));
                }
            }
        }
    }

    // The payload collapse below (`collapse_pos0_read` per position) is
    // the only fallible mutation; it mutates the Some target on success
    // but can still `Err` on a later position.  With at most one position
    // the collapse is the first mutation and itself atomic (it errs before
    // writing), so a decline leaves the graph byte-identical.  Two or more
    // positions (the same Option threaded into several Some-link slots)
    // could half-collapse before a later `Err`, handing the legacy walker
    // a partially-rewritten graph — decline that unusual shape up front to
    // keep the "validate-before-mutate" fail-safe contract airtight.
    if payload_positions.len() > 1 {
        return Err(format!(
            "{name}: Option value threaded into {} Some-arm slots — multi-slot \
             payload collapse is not fail-safe",
            payload_positions.len()
        ));
    }

    // None arm (StopIteration exit): the loop-break continuation.  RPython's
    // `ll_listnext` raises `StopIteration` with NO value on the exhaustion
    // path (`rpython/rtyper/lltypesystem/rlist.py:476` ll_listnext;
    // `rpython/rtyper/rlist.py:444` rtype_next — the iteration value is
    // defined solely on the normal path), so a forwarded `Option` scrutinee
    // `opt_c` is a Rust value-encoded SSA merge-thread with no RPython
    // counterpart.  Prune it and its whole transitive forward chain iff every
    // slot in the chain is dead — never read by an op, tested by an
    // exitswitch, or escaping to the return/except block.  A chain reaching a
    // genuine use declines (the residual call keeps the rtyper Skip).
    // `opt_c` can be forwarded into MORE THAN ONE slot of `none_link` (the
    // same Option merge-thread duplicated across slots); each occurrence roots
    // its own transitive chain.  Collect every such slot, not just the first,
    // so a surviving duplicate is not back-substituted onto the StopIteration
    // edge below (where the native `next` result is not produced).  If any
    // chain reaches a genuine use the whole rewrite declines.
    let opt_none_positions: Vec<usize> = none_link
        .args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, LinkArg::Value(v) if *v == opt_c).then_some(i))
        .collect();
    let mut dead: std::collections::BTreeMap<usize, std::collections::BTreeSet<usize>> =
        std::collections::BTreeMap::new();
    for opt_none_pos in opt_none_positions {
        let set = collect_transitive_dead_slots(graph, none_target.0, opt_none_pos).map_err(
            |reason| format!("{name}: None arm forwards a live Option value — {reason}"),
        )?;
        for (b, s) in set {
            dead.entry(b).or_default().insert(s);
        }
    }

    // Generalized arity guard: removing slot `s` from a block also drops arg
    // `s` from every predecessor link, so every such link must carry the
    // block's full pre-removal inputarg arity.  Enforce it for EVERY block in
    // the removal set — a loop-header re-entry can pull several blocks, even
    // several slots of `none_target`, into the chain — not just `none_target`.
    // Block A's pre-rewrite exit targets C (not a removal-set block), so A is
    // not yet among these; its new StopIteration link is built post-removal.
    for &b in dead.keys() {
        let arity = graph.blocks[b].inputargs.len();
        for blk in &graph.blocks {
            for l in &blk.exits {
                if l.target.0 == b && l.args.len() != arity {
                    return Err(format!(
                        "{name}: predecessor link to block {b} has arity {} != {arity} — \
                         unsafe to prune the transitive dead-Option chain",
                        l.args.len()
                    ));
                }
            }
        }
    }

    // A's StopIteration link to `none_target`: the None-arm link args with
    // EVERY `none_target` slot in the removal set dropped (not only the
    // directly-forwarded `opt_c` slot — a loop re-entry can add a second
    // `none_target` slot), the discriminant threaded as the const 0, and
    // surviving values back-substituted from C scope into A scope.
    let none_target_dead = dead.get(&none_target.0).cloned().unwrap_or_default();
    let mut none_args: Vec<LinkArg> = Vec::with_capacity(none_link.args.len());
    for (p, arg) in none_link.args.iter().enumerate() {
        if none_target_dead.contains(&p) {
            continue; // pruned slot
        }
        match arg {
            LinkArg::Const(c0) => none_args.push(LinkArg::Const(c0.clone())),
            LinkArg::Value(v) if *v == disc_var => none_args.push(int_const(0)),
            LinkArg::Value(v) => {
                let v_a = back_substitute(graph, &[(a, c)], v, &name)?;
                none_args.push(LinkArg::Value(v_a));
            }
        }
    }

    // --- All structural validation passed; mutate the graph. ---

    // The Some target reads the payload via `opt.__pos_0`; with the `next`
    // result flowing directly, that read collapses to the carried value.
    for pos in payload_positions {
        collapse_pos0_read(graph, some_target, pos, &name)?;
    }

    // Drop every dead slot in the transitive chain from its block's inputargs
    // and from every predecessor link feeding that block (descending indices
    // so earlier positions stay valid).  Each link's args are reduced only for
    // its target block's removed slots, keeping link arity == target inputarg
    // arity across the whole graph.  `none_args` already omits `none_target`'s
    // dead slots, and A's new StopIteration link is added below, so the
    // reduced arity stays consistent across all predecessors.
    for (&b, slots) in dead.iter().rev() {
        for &s in slots.iter().rev() {
            graph.blocks[b].inputargs.remove(s);
            for blk in &mut graph.blocks {
                for l in &mut blk.exits {
                    if l.target.0 == b {
                        l.args.remove(s);
                    }
                }
            }
        }
    }

    // Replace A's residual `next()` call with the native `next` op: the
    // `[__iter_next]` marker, the iterator as its single operand, `opt`
    // reused as the element.  Any pure recast narrows that followed the raw
    // call (peeled above) are dropped so the native `next` op — which
    // produces the scrutinised `opt` directly — is A's last op and thus the
    // block's `raising_op` under the `LastException` exitswitch below.
    graph.blocks[a].operations.truncate(next_idx + 1);
    graph.blocks[a].operations[next_idx] = SpaceOperation {
        result: Some(opt.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: next_op_segments(),
            },
            args: vec![iter_arg],
            result_ty: ValueType::Ref(None),
        },
    };

    // Rewire A: `LastException` exits.
    //   normal        → Some arm (element)
    //   StopIteration → None arm (loop break)
    // `OpKind::Next.can_only_throw` is the conservative `[StopIteration,
    // RuntimeError]` default, but the `originates_from_iter_op` gate above
    // admits only a list iterator, and `ListIteratorRepr::rtype_next`
    // (`ll_listnext`) raises solely `StopIteration` — the `RuntimeError`
    // half is dict-iterator mutation-detection that never materialises
    // here.  The annotator's
    // `flowin` drops the unhandled-exception remainder, so no catch-all
    // propagation edge to `exceptblock` is synthesised (preserving the
    // front graph's "exceptblock edges == MIR unwind terminators"
    // invariant).
    let stop_etype = graph.alloc_value_var();
    let stop_evalue = graph.alloc_value_var();
    let mut stopiter_link = Link::new_mixed(none_args, none_target, Some(stopiteration_exitcase()));
    stopiter_link.last_exception = Some(LinkArg::Value(stop_etype));
    stopiter_link.last_exc_value = Some(LinkArg::Value(stop_evalue));

    let block_a = &mut graph.blocks[a];
    block_a.exitswitch = Some(ExitSwitch::LastException);
    block_a.exits = vec![
        Link::new_mixed(normal_args, some_target, None),
        stopiter_link,
    ];
    Ok(())
}
