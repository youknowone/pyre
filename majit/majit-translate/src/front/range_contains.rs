//! `(a..=b).contains(&x)` → `int_between(a, x, b + 1)`.
//!
//! ## Positioning
//!
//! `RangeInclusive::new(a, b)` has an Opaque body in the LLBC, so the
//! front emits a residual `FunctionPath` call whose owner-qualified
//! segments end `["range", "RangeInclusive", "new"]` — an unregistered
//! callee the rtyper census Skips, dropping the enclosing graph to the
//! legacy walker.  `RangeInclusive::contains(&self, &x)` routes as an
//! owner-qualified `FunctionPath` too (the opaque foreign receiver has no
//! extracted body to route a `CallTarget::Method` through), ending
//! `["range", "RangeInclusive", "contains"]` with `args[0]` the
//! `&RangeInclusive` receiver and `args[1]` the `&x` value.  `new` is the
//! wall: it is emitted before `contains`, so the translation loop's
//! failure surfaces on `new` first.
//!
//! `int_between(n, m, p)` is the predicate `n <= m < p` (it assumes
//! `n <= p`).  `RangeInclusive` is `a <= x <= b`, so the exclusive upper
//! bound is `b + 1`:
//!
//! ```text
//!     r = RangeInclusive::new(a, b)          // residual `new` call (block N)
//!     ...
//!     t = r.contains(&x)                     // residual `contains` call (block C)
//! becomes
//!     upper = b + 1                          // constant when `b` is
//!     t     = int_between(a, x, upper)      // a <= x < b + 1
//! ```
//!
//! A constant `b` of `i64::MAX` has no representable successor, and a
//! non-constant `b` may be `i64::MAX` at run time; both keep the two
//! comparisons `bitand(le(a, x), ge(b, x))`.  Exclusive `Range` (`a..b`, which
//! is already `a <= x < b`) is not rewritten here.
//!
//! ## Cross-block shape
//!
//! Each `TermKind::Call` closes its block with a `set_goto`, so `new`
//! (block N) and `contains` (block C) sit in adjacent blocks, C the
//! single successor of N's Call edge.  The `new` result rides N → C on a
//! goto link arg; the front's liveness-forwarding keeps the same
//! `Variable` identity across the edge (and onward toward the value's
//! drop), so the `contains` receiver may be that same Variable or a
//! fresh C inputarg fed positionally from it (framestate SSA).  The fold
//! traces the receiver back through the single-predecessor link-arg
//! chain to its `new` site, then threads the bounds `a` / `b` into block
//! C via [`FunctionGraph::ensure_variable_at_block`] (the same
//! single-predecessor carry-through) before splicing the compares.
//!
//! ## Why `new` must be removed explicitly
//!
//! No sweep reclaims a dead `OpKind::Call { FunctionPath }`: `is_pure_op`
//! returns `false` for `Call`, so `prune_dead_phis` never removes it, and
//! `remove_dead_aggregates` only handles `SyntheticTransparentCtor` +
//! `Box::new_uninit`.  A fold that emits the compares at `contains` and
//! lets `new` "die" is UNSAFE — the orphaned `new` still walls the graph.
//! So the fold removes the `new` call op itself; the now-dead threaded
//! range value / link args are reclaimed by `prune_dead_phis` (the caller
//! runs it after any rewrite).
//!
//! ## Scope + fail-safe
//!
//! Int-only.  Float ranges (`complex_pow`), exclusive `Range`, and the
//! pyre `W_Range` Python object are out of scope.  Iteration-form ranges
//! (`for i in a..=b`) are excluded by the consumer-shape gate: an
//! iterator reads the range a second time (its stateful `exhausted`
//! field) as an OP operand, so a range value read by any op other than
//! the single `contains` being folded declines.  Link-arg threading of
//! the range value is liveness forwarding, not consumption, so it does
//! not count.  Every mismatch declines (leaves BOTH residual calls,
//! census Skip) — no regression.

use crate::flowspace::model::Variable;
use crate::model::{
    BlockId, CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// A recognized `RangeInclusive::new(lo, hi)` call site captured during
/// body lowering (`front::mir`).  Carries the result var (the range) and
/// the two int bounds so the fold can thread them into the `contains`
/// block.
#[derive(Clone)]
pub(crate) struct RangeInclusiveNewSite {
    /// The `new` call result (the `RangeInclusive` value) — locates the
    /// producer op and matches the `contains` receiver.
    pub result_var: Variable,
    /// Lower bound `a` — `int_between`'s first operand.
    pub lo: Variable,
    /// Inclusive upper bound `b`.  The op's exclusive limit is `b + 1`.
    pub hi: Variable,
}

/// A recognized `RangeInclusive::contains(&self, &x)` call site captured
/// during body lowering (`front::mir`).  Carries the result var (the
/// membership bool) — the fold reuses it for `int_between`.
#[derive(Clone)]
pub(crate) struct RangeContainsSite {
    /// The `contains` call result (the membership `bool`) — locates the
    /// producer op; reused as the `bitand` result.
    pub result_var: Variable,
}

/// Rewrite every recorded `(a..=b).contains(&x)` call site into
/// `int_between(a, x, b + 1)`.  Fail-safe: a site
/// that does not match the expected cross-block `new` → `contains` shape
/// is left untouched (both residual calls survive, census Skip).  Returns
/// the number of sites rewritten.
pub(crate) fn rewire_range_contains_call_sites(
    graph: &mut FunctionGraph,
    new_sites: &[RangeInclusiveNewSite],
    contains_sites: &[RangeContainsSite],
) -> usize {
    let mut rewritten = 0;
    for site in contains_sites {
        match rewire_one_range_contains_site(graph, site, new_sites) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave both residual calls; the unregistered `new`
                // callee keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_range_contains_site(
    graph: &mut FunctionGraph,
    site: &RangeContainsSite,
    new_sites: &[RangeInclusiveNewSite],
) -> Result<(), String> {
    let name = graph.name.clone();

    // 1. Locate the `contains` producer op + its block C.
    let (c_idx, call_idx) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| op.result.as_ref() == Some(&site.result_var))
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: range contains result var has no producer op"))?;

    // 2. The producer must be the 2-arg `contains` call: `args[0]` the
    //    `&RangeInclusive` receiver, `args[1]` the `&x` value.  The
    //    receiver's opaque foreign ADT routes it as an owner-qualified
    //    `FunctionPath` ending `["range", "RangeInclusive", "contains"]`
    //    (not a `CallTarget::Method`).
    let (range_v, x) = match &graph.blocks[c_idx].operations[call_idx].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if segments_end_with(segments, &["range", "RangeInclusive", "contains"])
            && args.len() == 2 =>
        {
            (args[0].clone(), args[1].clone())
        }
        other => {
            return Err(format!(
                "{name}: range contains producer op is not the 2-arg `contains` FunctionPath: {other:?}"
            ));
        }
    };

    // 3. Match the `RangeInclusiveNewSite` whose `result_var` threads to
    //    block C.  `new` is in block N (a single predecessor of C) and
    //    its result rides a goto link-arg that becomes a C inputarg, so
    //    `range_v` (the C-local receiver) is a distinct Variable.  A
    //    site matches when its `result_var` is defined in a block
    //    reachable from C through single-predecessor edges only AND the
    //    receiver `range_v` is either that same Variable (rare, when
    //    identity is preserved) or a C inputarg fed exclusively from that
    //    definition.  The single-consumer gate below is what makes this
    //    sound; here we only need to pick the producing site.
    let c_block = graph.blocks[c_idx].id;
    let new_site = new_sites
        .iter()
        .find(|s| range_matches_new_site(graph, c_block, &range_v, &s.result_var))
        .ok_or_else(|| {
            format!("{name}: range contains receiver traces to no RangeInclusive::new site")
        })?;

    // 4. Consumer-shape gate (excludes iteration-form ranges, whose
    //    range carries a stateful `exhausted` field read a second time by
    //    an iterator's `into_iter` / `next`).  The range value must be
    //    read as an OP operand by exactly the `contains` op being folded;
    //    dead framestate `Link.args` forwarding past the `contains` block
    //    is reclaimed by `prune_dead_phis` and does not count.
    if !range_value_single_op_consumer(graph, &new_site.result_var, &site.result_var) {
        return Err(format!(
            "{name}: range value has a second consumer — declining (iteration form?)"
        ));
    }

    // 5. Thread `lo` / `hi` into block C as inputargs (single-pred carry-
    //    through).  Decline if either cannot be threaded (orphan / multi-
    //    predecessor block) rather than panicking.
    if !can_thread_to_block(graph, c_block, &new_site.lo)
        || !can_thread_to_block(graph, c_block, &new_site.hi)
    {
        return Err(format!(
            "{name}: RangeInclusive::new bounds cannot thread to the contains block"
        ));
    }
    let lo_in_c = new_site.lo.clone();
    let hi_in_c = new_site.hi.clone();
    let ok = graph.ensure_variable_at_block(c_block, &lo_in_c)
        && graph.ensure_variable_at_block(c_block, &hi_in_c);
    if !ok {
        return Err(format!(
            "{name}: RangeInclusive::new bounds threading failed at the contains block"
        ));
    }

    // 6. Splice `int_between` into block C, replacing the `contains` op
    //    in place and reusing its result Variable.
    let inserts =
        build_range_contains_compares(graph, &site.result_var, lo_in_c, hi_in_c, x.into_variable());
    let ops = &mut graph.blocks[c_idx].operations;
    ops.remove(call_idx);
    for (offset, op) in inserts.into_iter().enumerate() {
        ops.insert(call_idx + offset, op);
    }

    // 7. Remove the now-unread `new` call op in its block N.  After
    //    removal `lo` / `hi` stay live (threaded to C); the dead threaded
    //    range inputarg / link args are reclaimed by `prune_dead_phis`.
    remove_op_by_result(graph, &new_site.result_var);

    Ok(())
}

/// `true` when the block-C receiver `range_v` is fed exactly by the
/// `new` result `new_result` — either the same Variable (identity
/// preserved across the boundary) or a C inputarg threaded from
/// `new_result` through a single-predecessor link-arg chain.
///
/// The trace is positional and exact so that a body containing two
/// distinct `a..=b` ranges never pairs a `contains` with the wrong
/// `new` (which would splice the wrong bounds): at each level the
/// receiver must be an inputarg whose SINGLE predecessor supplies the
/// value at the matching link-arg index, recursing on that supplied
/// Variable until it either reaches `new_result` (match) or a block that
/// defines it via an op (must then equal `new_result`).
fn range_matches_new_site(
    graph: &FunctionGraph,
    c_block: BlockId,
    range_v: &Variable,
    new_result: &Variable,
) -> bool {
    let mut cur_block = c_block;
    let mut cur_var = range_v.clone();
    let mut hops = 0usize;
    loop {
        if &cur_var == new_result {
            return true;
        }
        hops += 1;
        if hops > graph.blocks.len() {
            return false;
        }
        // `cur_var` must be a block inputarg to trace across the edge;
        // if it is defined by an op in `cur_block` and is not
        // `new_result`, it is a different value — no match.
        let Some(arg_idx) = graph
            .block(cur_block)
            .inputargs
            .iter()
            .position(|v| v == &cur_var)
        else {
            return false;
        };
        // Exactly one predecessor edge, carrying the source at `arg_idx`.
        let pred_edges: Vec<(BlockId, usize)> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                let bid = b.id;
                b.exits
                    .iter()
                    .enumerate()
                    .filter(move |(_, e)| e.target == cur_block)
                    .map(move |(i, _)| (bid, i))
            })
            .collect();
        if pred_edges.len() != 1 {
            return false;
        }
        let (pred_block, exit_idx) = pred_edges[0];
        let Some(LinkArg::Value(src)) = graph.block(pred_block).exits[exit_idx].args.get(arg_idx)
        else {
            return false;
        };
        cur_var = src.clone();
        cur_block = pred_block;
    }
}

/// `var` is defined in a block reachable from `block` by walking
/// single-predecessor edges only — the precondition under which
/// [`FunctionGraph::ensure_variable_at_block`] threads cleanly.  Mirrors
/// the private helper of the same name in `model::thread_undefined_op_operands`.
fn defined_via_single_pred_chain(graph: &FunctionGraph, block: BlockId, var: &Variable) -> bool {
    let mut cur = block;
    let mut hops = 0usize;
    loop {
        hops += 1;
        if hops > graph.blocks.len() {
            return false;
        }
        let preds = graph.predecessors(cur);
        if preds.len() != 1 {
            return false;
        }
        let p = preds[0];
        if graph.variable_defined_in_block(p, var) {
            return true;
        }
        cur = p;
    }
}

/// A `var` can be threaded to `block` iff it is already defined there or
/// reachable through a single-predecessor chain.  Same predicate
/// `ensure_variable_at_block` succeeds on; checked first so a
/// multi-predecessor / orphan case declines instead of panicking.
fn can_thread_to_block(graph: &FunctionGraph, block: BlockId, var: &Variable) -> bool {
    graph.variable_defined_in_block(block, var) || defined_via_single_pred_chain(graph, block, var)
}

/// `true` when the range value is CONSUMED by exactly the one
/// `contains` op being folded — the consumer-shape gate that excludes
/// iteration-form ranges (whose range carries a stateful `exhausted`
/// field an iterator's `into_iter` / `next` reads a second time).
///
/// The genuine soundness requirement is that no operation other than the
/// `contains` being folded READS the range value.  A range that merely
/// rides framestate `Link.args` past the `contains` block without being
/// read is dead once `new`/`contains` are removed — `prune_dead_phis`
/// (run by the caller after any rewrite) reclaims the dead inputarg /
/// link-arg forwards.  So link-arg forwarding is NOT a second consumer;
/// only an op operand, an `exitswitch` discriminator, or an exception
/// payload read is.
///
/// Framestate SSA can rename the range value into a fresh inputarg at
/// every forwarding block, so a purely backward positional trace cannot
/// see a read that happens under a renamed alias.  This computes the
/// FORWARD alias closure from the `new` result — the fixpoint set of
/// every Variable the range value flows into through `Link.args` — and
/// then rejects if any op/exitswitch/exception payload (other than the
/// folded `contains`) reads a Variable in that closure.  An iteration-form
/// range's `into_iter`/`next` reads the range as an op operand, so it is
/// caught even after a rename; a dead forward is not.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn range_value_single_op_consumer(
    graph: &FunctionGraph,
    new_result: &Variable,
    contains_result: &Variable,
) -> bool {
    use std::collections::HashSet;
    // Forward alias closure: every Variable the `new` result reaches
    // through positional `Link.args` forwarding.  Seed with the producer
    // result, then iterate to a fixpoint — a link arg in the closure adds
    // the target block's inputarg at the matching position (the renamed
    // alias in the successor).
    let mut closure: HashSet<Variable> = HashSet::new();
    closure.insert(new_result.clone());
    loop {
        let mut grew = false;
        for block in &graph.blocks {
            for link in &block.exits {
                let Some(target_idx) = graph.blocks.iter().position(|b| b.id == link.target) else {
                    continue;
                };
                let target_inputargs = &graph.blocks[target_idx].inputargs;
                for (arg_idx, arg) in link.args.iter().enumerate() {
                    let LinkArg::Value(var) = arg else { continue };
                    if closure.contains(var)
                        && let Some(iarg) = target_inputargs.get(arg_idx)
                        && closure.insert(iarg.clone())
                    {
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            break;
        }
    }

    let is_range_value = |v: &Variable| closure.contains(v);
    for block in &graph.blocks {
        for op in &block.operations {
            let reads_range = crate::inline::op_variable_refs(&op.kind)
                .iter()
                .any(is_range_value);
            if reads_range && op.result.as_ref() != Some(contains_result) {
                // Some op other than the `contains` being folded reads
                // the range value — a second consumer (iteration form?).
                return false;
            }
        }
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(v)) if is_range_value(v) => return false,
            Some(crate::model::ExitSwitch::Fused { args, .. })
                if args.iter().any(is_range_value) =>
            {
                return false;
            }
            _ => {}
        }
        for link in &block.exits {
            if link
                .last_exception
                .as_ref()
                .and_then(LinkArg::as_variable)
                .is_some_and(is_range_value)
                || link
                    .last_exc_value
                    .as_ref()
                    .and_then(LinkArg::as_variable)
                    .is_some_and(is_range_value)
            {
                return false;
            }
        }
    }
    true
}

/// Match a `FunctionPath`'s trailing segments against `tail` so a
/// crate-qualified spelling and the crate-stripped front-end spelling
/// both resolve — the same contract as `front::mir::fmt_path_ends_with`.
fn segments_end_with(segments: &[String], tail: &[&str]) -> bool {
    segments.len() >= tail.len()
        && segments[segments.len() - tail.len()..]
            .iter()
            .zip(tail)
            .all(|(s, t)| s.as_str() == *t)
}

/// Remove the (single) op in the graph whose result is `result_var`.
fn remove_op_by_result(graph: &mut FunctionGraph, result_var: &Variable) {
    for block in &mut graph.blocks {
        if let Some(i) = block
            .operations
            .iter()
            .position(|op| op.result.as_ref() == Some(result_var))
        {
            block.operations.remove(i);
            return;
        }
    }
}

/// The signed constant `var` is produced by, if its single producer is
/// `ConstInt`.
fn const_int_of(graph: &FunctionGraph, var: &Variable) -> Option<i64> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find_map(|op| match (&op.result, &op.kind) {
                (Some(result), OpKind::ConstInt(value)) if result == var => Some(*value),
                _ => None,
            })
    })
}

/// `int_between(lo, x, upper)` bound to `result_var`.  The result is a
/// 0/1 integer in the signed register bank.
fn int_between_op(
    result_var: &Variable,
    lo: Variable,
    x: Variable,
    upper: Variable,
) -> SpaceOperation {
    FunctionGraph::set_concretetype_of_inline(result_var, crate::model::ConcreteType::Signed);
    SpaceOperation {
        result: Some(result_var.clone()),
        kind: OpKind::LoweredBlackholeOp {
            opname: "int_between".to_string(),
            args: vec![lo, x, upper],
        },
    }
}

/// `lo <= x <= hi` when `hi + 1` does not fit in `i64`:
/// `bitand(le(lo, x), ge(hi, x))`.
fn build_range_contains_compare_pair(
    graph: &mut FunctionGraph,
    result_var: &Variable,
    lo: Variable,
    hi: Variable,
    x: Variable,
) -> Vec<SpaceOperation> {
    let lo_le = graph.alloc_value_var();
    let hi_ge = graph.alloc_value_var();
    vec![
        SpaceOperation {
            result: Some(lo_le.clone()),
            kind: OpKind::BinOp {
                op: "le".to_string(),
                lhs: lo,
                rhs: x.clone(),
                result_ty: ValueType::Int,
            },
        },
        SpaceOperation {
            result: Some(hi_ge.clone()),
            kind: OpKind::BinOp {
                op: "ge".to_string(),
                lhs: hi,
                rhs: x,
                result_ty: ValueType::Int,
            },
        },
        SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::BinOp {
                op: "bitand".to_string(),
                lhs: lo_le,
                rhs: hi_ge,
                result_ty: ValueType::Int,
            },
        },
    ]
}

/// Replace `contains` with `int_between(lo, x, hi + 1)`.
///
/// `int_between(n, m, p)` is `n <= m < p`.  Inclusive `hi` therefore
/// needs a successor.  A constant `i64::MAX` has none, and that case
/// keeps the two-comparison form, as does a non-constant `hi`, whose
/// run-time value may be `i64::MAX`.  Any other constant is materialised
/// as `ConstInt(hi + 1)`.
fn build_range_contains_compares(
    graph: &mut FunctionGraph,
    result_var: &Variable,
    lo: Variable,
    hi: Variable,
    x: Variable,
) -> Vec<SpaceOperation> {
    let exclusive = const_int_of(graph, &hi).map(|value| value.checked_add(1));
    match exclusive {
        Some(None) => build_range_contains_compare_pair(graph, result_var, lo, hi, x),
        Some(Some(upper)) => {
            let upper_v = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
            vec![
                SpaceOperation {
                    result: Some(upper_v.clone()),
                    kind: OpKind::ConstInt(upper),
                },
                int_between_op(result_var, lo, x, upper_v),
            ]
        }
        // A non-constant `hi` may be `i64::MAX` at run time, where `hi + 1`
        // wraps and `int_between` answers false for every `x`; the two
        // comparisons are exact for every value.
        None => build_range_contains_compare_pair(graph, result_var, lo, hi, x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ops", "range", "RangeInclusive", "new"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn contains_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ops", "range", "RangeInclusive", "contains"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    /// Count `FunctionPath` calls whose segments end with `tail`.
    fn functionpath_calls_ending(g: &FunctionGraph, tail: &[&str]) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments_end_with(segments, tail)
                )
            })
            .count()
    }

    /// `(result, args)` of every `int_between` in the graph.
    fn int_between_ops(g: &FunctionGraph) -> Vec<(Variable, Vec<Variable>)> {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|sop| match &sop.kind {
                OpKind::LoweredBlackholeOp { opname, args } if opname == "int_between" => Some((
                    sop.result.clone().expect("int_between result"),
                    args.clone(),
                )),
                _ => None,
            })
            .collect()
    }

    /// `int_between(lo, x, exclusive)` bound to `result`, and no leftover
    /// compare pair.
    fn assert_int_between(
        g: &FunctionGraph,
        result: &Variable,
        lo: &Variable,
        x: &Variable,
        exclusive: i64,
    ) {
        let sites = int_between_ops(g);
        assert_eq!(sites.len(), 1, "one int_between, got {sites:?}");
        let (bound, args) = &sites[0];
        assert_eq!(bound, result, "int_between reuses the contains result");
        assert_eq!(args.len(), 3);
        assert_eq!(&args[0], lo, "lower bound is lo");
        assert_eq!(&args[1], x, "middle value is x");
        assert_eq!(
            const_int_of(g, &args[2]),
            Some(exclusive),
            "exclusive upper bound is hi + 1",
        );
        assert!(binop_results(g, "le").is_empty(), "no leftover le");
        assert!(binop_results(g, "ge").is_empty(), "no leftover ge");
        assert!(binop_results(g, "bitand").is_empty(), "no leftover bitand");
    }

    /// Result Variables of every `BinOp` with the given opname.
    fn binop_results(g: &FunctionGraph, op_name: &str) -> Vec<Variable> {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|sop| match &sop.kind {
                OpKind::BinOp { op, .. } if op == op_name => sop.result.clone(),
                _ => None,
            })
            .collect()
    }

    /// Build a two-block `new` → `contains` graph and assert the rewrite
    /// drops both residual FunctionPath calls (`new` + `contains`) and
    /// emits `int_between(lo, x, hi + 1)` bound to the original
    /// `contains` result var.
    #[test]
    fn rewrite_folds_new_contains_to_compares() {
        let mut g = FunctionGraph::new("test_range_contains");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        // Block C: one inputarg = the threaded range receiver.
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c.clone(), x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range.clone(),
                lo: a.clone(),
                hi: b.clone(),
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "the contains site must be rewritten");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            0,
            "residual new removed",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            0,
            "residual contains removed",
        );
        // `(0..=255).contains` → `int_between(0, x, 256)`.
        assert_int_between(&g, &contains, &a, &x, 256);
    }

    /// A second consumer of the range result (an iterator's stateful
    /// read, modeled here as a dummy op reading the threaded receiver)
    /// declines the fold.
    #[test]
    fn rewrite_declines_when_range_has_second_consumer() {
        let mut g = FunctionGraph::new("test_range_contains_second_consumer");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        // Second consumer: another op reads the `new` result directly.
        let _second = g.push_op_var(
            n,
            OpKind::UnaryOp {
                op: "invert".to_string(),
                operand: range.clone(),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c.clone(), x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range.clone(),
                lo: a,
                hi: b,
            }],
            &[RangeContainsSite {
                result_var: contains,
            }],
        );
        assert_eq!(rewritten, 0, "a second range consumer declines the fold");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            1,
            "residual new survives",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            1,
            "residual contains survives",
        );
    }

    /// Framestate threading gives the successor a third identity for the
    /// range.  Carrying that alias beyond the middle `contains` block must
    /// keep both residual calls: deleting `new` would leave the successor's
    /// input undefined.
    #[test]
    fn rewrite_declines_when_successor_carries_fresh_range_alias() {
        let mut g = FunctionGraph::new("test_range_contains_successor_alias");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();

        // Middle block M consumes one alias in `contains`, then threads it
        // onward.  The successor S receives a fresh framestate inputarg and
        // consumes that alias independently.
        let (m, m_args) = g.create_block_with_arg_vars(1);
        let range_in_m = m_args[0].clone();
        let x = g.push_op_var(m, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                m,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_m.clone(), x]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (s, s_args) = g.create_block_with_arg_vars(1);
        let range_in_s = s_args[0].clone();
        let later_use = g
            .push_op_var(
                s,
                OpKind::UnaryOp {
                    op: "invert".to_string(),
                    operand: range_in_s,
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(s, Some(later_use));
        g.set_goto(m, s, vec![range_in_m]);
        g.set_goto(n, m, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
            }],
            &[RangeContainsSite {
                result_var: contains,
            }],
        );
        assert_eq!(rewritten, 0, "a successor range alias declines the fold");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            1,
            "residual new survives",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            1,
            "residual contains survives",
        );
    }

    /// A producer that is not the 2-arg `contains` FunctionPath (here a
    /// 1-arg call) declines (fail-safe).
    #[test]
    fn rewrite_declines_when_producer_not_contains_method() {
        let mut g = FunctionGraph::new("test_range_contains_wrong_producer");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        // A 1-arg call — not the 2-arg `contains` FunctionPath shape.
        let result = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(result.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
            }],
            &[RangeContainsSite { result_var: result }],
        );
        assert_eq!(rewritten, 0, "a non-2-arg `contains` producer declines");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            1,
            "residual new survives",
        );
    }

    /// The range value rides framestate `Link.args` past the `contains`
    /// block into a successor that only forwards it onward (never reads it
    /// as an op operand) — the shape the real `setitem_bytearray`
    /// `(0..=255).contains(&v)` lowers to, where the loop-carried `v`
    /// framestate slot threads the dead range temporary through the whole
    /// tail CFG.  This is a DEAD forward, not a second consumer: after the
    /// fold removes `new`/`contains`, `prune_dead_phis` reclaims the dead
    /// inputarg / link args.  It must FOLD (contrast
    /// `rewrite_declines_when_successor_carries_fresh_range_alias`, where the
    /// successor READS the alias via an op).
    #[test]
    fn rewrite_folds_when_dead_range_alias_forwarded_past_contains() {
        let mut g = FunctionGraph::new("test_range_contains_dead_forward");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();

        // Block C: identity-preserved delivery of the range to `contains`.
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c.clone(), x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        // Successor S receives the range on a framestate link-arg but never
        // reads it — a dead forward.  Its only live op reads an unrelated
        // value (the `contains` result), and it returns that, not the range.
        let (s, s_args) = g.create_block_with_arg_vars(2);
        let range_in_s = s_args[0].clone();
        let other_in_s = s_args[1].clone();
        let ret = g
            .push_op_var(
                s,
                OpKind::UnaryOp {
                    op: "invert".to_string(),
                    operand: other_in_s,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(s, Some(ret));
        // C forwards the dead range plus the `contains` result onward.
        let _ = range_in_s;
        g.set_goto(c, s, vec![range_in_c.clone(), contains.clone()]);
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(
            rewritten, 1,
            "a dead range forward past contains must still fold"
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            0,
            "residual new removed",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            0,
            "residual contains removed",
        );
        assert_int_between(&g, &contains, &a, &x, 256);
    }

    /// A constant inclusive upper bound of `i64::MAX` has no `hi + 1`.
    /// Keep the two comparisons rather than wrapping the limit.
    #[test]
    fn rewrite_keeps_compares_when_upper_bound_is_i64_max() {
        let mut g = FunctionGraph::new("test_range_contains_max");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(i64::MAX), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c, x]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "the overflow case still folds");
        assert!(
            int_between_ops(&g).is_empty(),
            "i64::MAX must not wrap into int_between"
        );
        assert_eq!(binop_results(&g, "le").len(), 1);
        assert_eq!(binop_results(&g, "ge").len(), 1);
        assert_eq!(binop_results(&g, "bitand"), vec![contains]);
    }

    /// A non-constant inclusive upper bound may be `i64::MAX` at run
    /// time, so it keeps the two comparisons instead of `int_between`.
    #[test]
    fn rewrite_keeps_compares_for_a_nonconstant_upper_bound() {
        let mut g = FunctionGraph::new("test_range_contains_var_hi");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let hi_src = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let b = g
            .push_op_var(
                n,
                OpKind::UnaryOp {
                    op: "neg".to_string(),
                    operand: hi_src,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c, x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            int_between_ops(&g).is_empty(),
            "a run-time hi must not be wrapped into int_between"
        );
        assert_eq!(binop_results(&g, "le").len(), 1);
        assert_eq!(binop_results(&g, "ge").len(), 1);
        assert_eq!(binop_results(&g, "bitand"), vec![contains]);
        let _ = (a, b, x);
    }
}
