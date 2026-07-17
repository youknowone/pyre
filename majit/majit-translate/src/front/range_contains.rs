//! `(a..=b).contains(&x)` → `bitand(le(a, x), ge(b, x))`.
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
//! The membership test `a <= x && b >= x` has a native lowering — two
//! integer comparisons joined by a native bitwise-and, exactly the
//! `a <= x <= b` shape the rtyper lowers.  This pass reproduces it:
//!
//! ```text
//!     r = RangeInclusive::new(a, b)          // residual `new` call (block N)
//!     ...
//!     t = r.contains(&x)                     // residual `contains` call (block C)
//! becomes
//!     lo_le  = le(a, x)                      // a <= x
//!     hi_ge  = ge(b, x)                      // b >= x
//!     t      = bitand(lo_le, hi_ge)          // (block C)
//! ```
//!
//! `le` / `ge` / `bitand` are all pure native binops (CSE/DCE-safe);
//! `bitand`, not bare `and`, is emitted — bare `and`/`or` are reserved
//! for short-circuit control flow and rejected downstream.
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
    /// Lower bound `a` — the `le(a, x)` left operand.
    pub lo: Variable,
    /// Upper bound `b` — the `ge(b, x)` left operand.
    pub hi: Variable,
}

/// A recognized `RangeInclusive::contains(&self, &x)` call site captured
/// during body lowering (`front::mir`).  Carries the result var (the
/// membership bool) — the fold reuses it for the final `bitand`.
#[derive(Clone)]
pub(crate) struct RangeContainsSite {
    /// The `contains` call result (the membership `bool`) — locates the
    /// producer op; reused as the `bitand` result.
    pub result_var: Variable,
}

/// Rewrite every recorded `(a..=b).contains(&x)` call site into the
/// native `bitand(le(a, x), ge(b, x))` compare pair.  Fail-safe: a site
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
            target: CallTarget::FunctionPath { segments },
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
    let (new_site, range_trace) = new_sites
        .iter()
        .find_map(|s| {
            range_matches_new_site(graph, c_block, &range_v, &s.result_var).map(|trace| (s, trace))
        })
        .ok_or_else(|| {
            format!("{name}: range contains receiver traces to no RangeInclusive::new site")
        })?;

    // 4. Consumer-shape gate (excludes iteration-form ranges, whose
    //    range carries a stateful `exhausted` field read a second time by
    //    an iterator's `into_iter` / `next`).  The range value must be
    //    read as an OP operand by exactly the `contains` op being folded.
    //    Only the positional producer-to-consumer Links recorded by the
    //    trace are permitted; any other Link carrying any traced alias means
    //    the range survives past (or branches away from) `contains`.
    if !range_value_single_op_consumer(graph, &range_trace, &site.result_var) {
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

    // 6. Splice the compares into block C, replacing the `contains` op in
    //    place and reusing its result Variable for the final `bitand`.
    let lo_le = graph.alloc_value_var();
    let hi_ge = graph.alloc_value_var();
    let inserts =
        build_range_contains_compares(&site.result_var, lo_in_c, hi_in_c, x, lo_le, hi_ge);
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
struct RangeAliasTrace {
    aliases: Vec<Variable>,
    /// `(predecessor block, exit index, link-argument index)` for each
    /// positional forwarding edge on the producer-to-`contains` trace;
    /// RPython's `Link.args` carries these Variables positionally
    /// (`rpython/flowspace/model.py:140-149`).
    threaded_links: Vec<(BlockId, usize, usize)>,
}

fn range_matches_new_site(
    graph: &FunctionGraph,
    c_block: BlockId,
    range_v: &Variable,
    new_result: &Variable,
) -> Option<RangeAliasTrace> {
    let mut cur_block = c_block;
    let mut cur_var = range_v.clone();
    let mut aliases = vec![cur_var.clone()];
    let mut threaded_links = Vec::new();
    let mut hops = 0usize;
    loop {
        if &cur_var == new_result {
            if !aliases.contains(new_result) {
                aliases.push(new_result.clone());
            }
            return Some(RangeAliasTrace {
                aliases,
                threaded_links,
            });
        }
        hops += 1;
        if hops > graph.blocks.len() {
            return None;
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
            return None;
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
            return None;
        }
        let (pred_block, exit_idx) = pred_edges[0];
        let Some(LinkArg::Value(src)) = graph.block(pred_block).exits[exit_idx].args.get(arg_idx)
        else {
            return None;
        };
        threaded_links.push((pred_block, exit_idx, arg_idx));
        cur_var = src.clone();
        if !aliases.contains(&cur_var) {
            aliases.push(cur_var.clone());
        }
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
/// Framestate SSA can mint a fresh inputarg at every forwarding block, so the
/// gate checks every Variable in the positional producer trace rather than
/// only its endpoints.  The trace's own Link slots are the only permitted
/// forwarding reads.  Any other op, exitswitch, Link arg, exception Link
/// payload, return, iterator, second `contains`, or clone carrying an alias is
/// a second consumer and declines before either residual call is removed.
fn range_value_single_op_consumer(
    graph: &FunctionGraph,
    trace: &RangeAliasTrace,
    contains_result: &Variable,
) -> bool {
    let is_range_value = |v: &Variable| trace.aliases.contains(v);
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
        for (exit_idx, link) in block.exits.iter().enumerate() {
            for (arg_idx, arg) in link.args.iter().enumerate() {
                let LinkArg::Value(var) = arg else { continue };
                if is_range_value(var)
                    && !trace
                        .threaded_links
                        .contains(&(block.id, exit_idx, arg_idx))
                {
                    return false;
                }
            }
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

/// Build the three native compare ops that replace `contains`:
/// `lo_le = le(lo, x)`, `hi_ge = ge(hi, x)`, `result = bitand(lo_le, hi_ge)`.
/// All `ValueType::Int` (bools fold to the int kind), reusing
/// `result_var` for the final `bitand` so downstream reads are unchanged.
fn build_range_contains_compares(
    result_var: &Variable,
    lo: Variable,
    hi: Variable,
    x: Variable,
    lo_le: Variable,
    hi_ge: Variable,
) -> [SpaceOperation; 3] {
    [
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

#[cfg(test)]
mod tests {
    use super::*;

    fn new_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ops", "range", "RangeInclusive", "new"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    fn contains_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ops", "range", "RangeInclusive", "contains"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
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
                        target: CallTarget::FunctionPath { segments },
                        ..
                    } if segments_end_with(segments, tail)
                )
            })
            .count()
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
    /// emits `le` / `ge` / `bitand` with the `bitand` bound to the
    /// original `contains` result var.
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
                    args: vec![a.clone(), b.clone()],
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
                    args: vec![range_in_c.clone(), x.clone()],
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
        assert_eq!(binop_results(&g, "le").len(), 1, "one `le` compare emitted");
        assert_eq!(binop_results(&g, "ge").len(), 1, "one `ge` compare emitted");
        let bitands = binop_results(&g, "bitand");
        assert_eq!(bitands.len(), 1, "one `bitand` emitted");
        assert_eq!(
            bitands[0], contains,
            "the bitand reuses the original contains result var",
        );
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
                    args: vec![a.clone(), b.clone()],
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
                    args: vec![range_in_c.clone(), x.clone()],
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
                    args: vec![a.clone(), b.clone()],
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
                    args: vec![range_in_m.clone(), x],
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
                    args: vec![a.clone(), b.clone()],
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
                    args: vec![range_in_c.clone()],
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
}
