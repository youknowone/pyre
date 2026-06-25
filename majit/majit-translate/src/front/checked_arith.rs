//! `i64::checked_{add,sub,mul}()` → `*_ovf` op + OverflowError handler.
//!
//! ## Positioning
//!
//! The front-end lift that turns a Rust `i64::checked_add(x, y)` call and
//! its `Option<i64>` match into the graph's native `add_ovf` op with an
//! OverflowError exception edge — the integer-arithmetic analogue of
//! [`crate::front::iter_next`]'s `next()` → `next`-op rewrite, and the
//! direct mirror of `simplify.py:70-108 transform_ovfcheck`.
//!
//! Rust source models the int fast path
//! ```text
//!     match va.checked_add(vb) {
//!         Some(r) => Ok(w_int_new(r)),   // no overflow → box int
//!         None    => Ok(w_long_new(bigint_add(..))),  // overflow → BigInt
//!     }
//! ```
//! lowered (in MIR) as a `checked_add(va, vb) -> Option<i64>` call, a
//! `__discriminant` read on the result, and a two-way `switchInt`
//! (None = 0, Some = 1).  RPython spells the same idiom as
//! ```text
//!     try:    z = ovfcheck(x + y)
//!     except OverflowError:  return _make_long(..)
//!     return wrapint(space, z)
//! ```
//! where `ovfcheck(x + y)` is the `add_ovf` op (`operation.py:466
//! add_operator('add', ..., ovf=True)` → its `_ovf` twin), which returns
//! the sum directly and raises `OverflowError` on overflow
//! (`operation.py:760-761 _add_except_ovf`; `OpKind::AddOvf.canraise() =
//! [OverflowError]`).  This module rewrites the value-encoded `Option`
//! diamond into that exception representation: the `Some` arm becomes the
//! op's normal continuation (the int payload `r`) and the `None` arm
//! becomes the OverflowError handler (the BigInt path).
//!
//! ## The rewrite (`rewire_one_checked_arith_site`)
//!
//! Block A holds the `checked_add` residual call producing `opt`; block C
//! is its single successor — the discriminant switch.  The rewrite:
//! 1. replaces A's residual call with the native `add_ovf` op (a `BinOp`
//!    the [`crate::translator::rtyper::flowspace_adapter`] maps to the
//!    raising flowspace `add_ovf` op), reusing `opt` as the sum result;
//! 2. closes A with `LastException` exits — normal → the `Some` arm
//!    (`opt.__pos_0` collapses to the sum), `OverflowError` → the `None`
//!    arm (the BigInt path).
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `checked_add`
//! callee makes the rtyper census Skip the graph (no regression vs the
//! legacy walker).

use crate::flowspace::model::{ConstValue, Constant, Variable};
use crate::front::result_exc::{
    assert_block_pure_besides, assert_single_pred, back_substitute, collapse_pos0_read,
    follow_single_exit, split_diamond_exits,
};
use crate::model::{
    CallTarget, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType,
};

/// The `core::num::<Impl>::checked_{add,sub,mul}` leaf the rewrite
/// recognises, paired with the `_ovf` BinOp opname it lowers to.
/// `checked_div` / `checked_rem` are intentionally absent: their overflow
/// idiom is `floordiv_ovf` / `mod_ovf` (a ZeroDivisionError-carrying
/// helper, not a bare `_ovf` twin), and the live integer fast paths spell
/// those with explicit value guards (`int_floordiv` / `int_mod`), so they
/// never reach this shape.
fn checked_arith_ovf_opname(leaf: &str) -> Option<&'static str> {
    match leaf {
        "checked_add" => Some("add_ovf"),
        "checked_sub" => Some("sub_ovf"),
        "checked_mul" => Some("mul_ovf"),
        _ => None,
    }
}

/// `true` iff the residual call target is a
/// `core::num::<Impl>::checked_{add,sub,mul}` — the `[.., "num",
/// "<Impl>", "checked_*"]` FunctionPath shape Charon emits for the
/// inherent `i64` method (core fn bodies are Opaque in the LLBC, so the
/// call is permanently unliftable).  Combined with an `Option` return
/// type at the recording site, this records a checked-arith candidate;
/// the rewrite itself validates the surrounding match shape.
pub(crate) fn is_checked_arith_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments } = target else {
        return false;
    };
    let [first, .., module, impl_seg, leaf] = segments.as_slice() else {
        return false;
    };
    first == "core"
        && module == "num"
        && impl_seg == "<Impl>"
        && checked_arith_ovf_opname(leaf).is_some()
}

/// The typed `OverflowError` exitcase the `_ovf` block's overflow link
/// carries — the handler analogue of [`crate::front::iter_next`]'s
/// `StopIteration` exitcase, narrowed to the single exception `add_ovf` /
/// `sub_ovf` / `mul_ovf` raises (`OpKind::AddOvf.canraise() =
/// [OverflowError]`).  `ConstValue::builtin` resolves the class to a
/// `HostObject`, the `Constant(HostObject(class))` shape
/// `annrpython::flowin` matches.
fn overflowerror_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("OverflowError"))
}

fn int_const(i: i64) -> LinkArg {
    LinkArg::Const(Constant::new(ConstValue::Int(i)))
}

/// `true` iff `var` is read — as an op operand or an `ExitSwitch::Value`
/// discriminator — by any block reachable from `start` (inclusive).
/// Forwarding `var` in a `Link.args` slot is NOT a read: framestate
/// threading reuses the same `Variable` identity for a loop-/branch-carried
/// local, so a value that is only *forwarded* through the subgraph is
/// dead-threaded and may be safely substituted.  Used to admit a dead
/// `opt` thread through the overflow arm (`rewire_one_checked_arith_site`).
fn var_read_in_reachable(
    graph: &FunctionGraph,
    start: crate::model::BlockId,
    var: &Variable,
) -> bool {
    let mut visited = vec![false; graph.blocks.len()];
    let mut stack = vec![start.0];
    while let Some(bi) = stack.pop() {
        if bi >= graph.blocks.len() || visited[bi] {
            continue;
        }
        visited[bi] = true;
        let b = &graph.blocks[bi];
        for op in &b.operations {
            if crate::front::result_exc::op_operand_vars(&op.kind).contains(var) {
                return true;
            }
        }
        if let Some(ExitSwitch::Value(v)) = &b.exitswitch
            && v == var
        {
            return true;
        }
        for link in &b.exits {
            stack.push(link.target.0);
        }
    }
    false
}

/// Rewrite every recorded `checked_{add,sub,mul}()` call site into the
/// `*_ovf` op + OverflowError handler shape.  Fail-safe: a site whose
/// surrounding `Option` match does not fit the overflow-fallback shape is
/// left as the residual call (Skip), so a mismatch never regresses a graph
/// the legacy walker already handled.  Returns the number of sites
/// rewritten.
pub(crate) fn rewire_checked_arith_call_sites(
    graph: &mut FunctionGraph,
    sites: &[Variable],
) -> usize {
    let mut rewritten = 0;
    for opt in sites {
        match rewire_one_checked_arith_site(graph, opt) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                if std::env::var_os("PYRE_MIR_FRONTEND_DEBUG").is_some() {
                    eprintln!(
                        "[checked_arith] {} decline at {opt:?}: {_decline}",
                        graph.name
                    );
                }
                // Leave the residual `checked_*` call; the unregistered
                // callee makes the rtyper census Skip this graph (no
                // regression).
            }
        }
    }
    rewritten
}

fn rewire_one_checked_arith_site(graph: &mut FunctionGraph, opt: &Variable) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `checked_*()` residual call producing `opt`, closed by
    // lower_call with a single forwarding exit.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(opt))
        })
        .ok_or_else(|| format!("{name}: checked_* result var has no producer block"))?;

    // The call must be A's last op (lower_call closes the block right
    // after pushing it) so it becomes the block's `raising_op`.
    let call_idx = graph.blocks[a].operations.len() - 1;
    let last_is_call = graph.blocks[a].operations[call_idx].result.as_ref() == Some(opt);
    if !last_is_call {
        return Err(format!(
            "{name}: checked_* call is not the last op of block {a}"
        ));
    }
    // Capture the two operands (the `_ovf` op's args) and resolve the
    // `_ovf` opname from the callee leaf.
    let (lhs, rhs, ovf_opname) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if args.len() == 2 => {
            let leaf = segments
                .last()
                .ok_or_else(|| format!("{name}: checked_* call path is empty"))?;
            let ovf = checked_arith_ovf_opname(leaf)
                .ok_or_else(|| format!("{name}: call leaf {leaf} is not a checked_* arith op"))?;
            (args[0].clone(), args[1].clone(), ovf)
        }
        other => {
            return Err(format!(
                "{name}: checked_* producer op is not a 2-arg FunctionPath call: {other:?}"
            ));
        }
    };

    // A's single exit → C (the Option discriminant switch).  Unlike the
    // Result `?` diamond there is no intervening `branch()` block.
    let (c, opt_c) = follow_single_exit(graph, a, opt)
        .map_err(|e| format!("{name}: checked_* call block exit: {e}"))?;
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

    // Some arm (normal exit): the `_ovf` op result IS the sum.  Map the
    // Some-link args back to A scope; the forwarded Option value becomes
    // the sum result, the threaded discriminant the constant 1.
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

    // None arm (OverflowError exit): the BigInt-fallback continuation.  It
    // recomputes from the operands (`BigInt::from(va)` …) and never reads
    // the overflowed Option.  Framestate threading may still keep `opt`
    // live across the match and forward it into the overflow arm as a
    // dead phi thread; that is admissible only if no block reachable from
    // the overflow target actually reads it (the value is genuinely
    // undefined on the raise edge — the `_ovf` op never produced it).  A
    // forward into a *read* site is a real consumer the rewrite cannot
    // honour, so decline.
    //
    // The overflow link is the op's `c_last_exception` edge, so it must
    // not carry the raising op's result `opt` (checkgraph:
    // "raising operation result cannot flow into exception link" —
    // `simplify.py` graph invariant; the result is undefined when the op
    // raises).  Substitute the dead slot with `lhs`, a live operand
    // defined before the op: it satisfies the link's defined-arg
    // requirement, is never the raising result, and is dropped unread in
    // the overflow subgraph.
    let opt_dead_in_overflow_arm = !var_read_in_reachable(graph, none_link.target, &opt_c);
    let mut none_args: Vec<LinkArg> = Vec::with_capacity(none_link.args.len());
    for arg in &none_link.args {
        match arg {
            LinkArg::Const(c0) => none_args.push(LinkArg::Const(c0.clone())),
            LinkArg::Value(v) => {
                if *v == opt_c {
                    if !opt_dead_in_overflow_arm {
                        return Err(format!(
                            "{name}: None arm of block {c} reads the Option value — unsupported"
                        ));
                    }
                    none_args.push(LinkArg::Value(lhs.clone()));
                } else if *v == disc_var {
                    none_args.push(int_const(0));
                } else {
                    let v_a = back_substitute(graph, &[(a, c)], v, &name)?;
                    none_args.push(LinkArg::Value(v_a));
                }
            }
        }
    }

    // --- All structural validation passed; mutate the graph. ---

    // The Some target reads the payload via `opt.__pos_0`; with the `_ovf`
    // result flowing directly, that read collapses to the carried value.
    for pos in payload_positions {
        collapse_pos0_read(graph, some_target, pos, &name)?;
    }

    // Replace A's residual `checked_*()` call with the native `_ovf`
    // BinOp: the two operands, `opt` reused as the sum.  The
    // `LastException` exitswitch below makes the block a `canraise` block
    // whose `raising_op` is this op.
    graph.blocks[a].operations[call_idx].kind = OpKind::BinOp {
        op: ovf_opname.to_string(),
        lhs,
        rhs,
        result_ty: ValueType::Int,
    };

    // Rewire A: `LastException` exits.
    //   normal        → Some arm (sum)
    //   OverflowError → None arm (BigInt fallback)
    // `OpKind::AddOvf.canraise()` is exactly `[OverflowError]`
    // (`operation.py:760-761 _add_except_ovf`), so no catch-all
    // propagation edge to `exceptblock` is synthesised (preserving the
    // front graph's "exceptblock edges == MIR unwind terminators"
    // invariant).
    let ovf_etype = graph.alloc_value_var();
    let ovf_evalue = graph.alloc_value_var();
    let mut overflow_link = Link::new_mixed(none_args, none_target, Some(overflowerror_exitcase()));
    overflow_link.last_exception = Some(LinkArg::Value(ovf_etype));
    overflow_link.last_exc_value = Some(LinkArg::Value(ovf_evalue));

    let block_a = &mut graph.blocks[a];
    block_a.exitswitch = Some(ExitSwitch::LastException);
    block_a.exits = vec![
        Link::new_mixed(normal_args, some_target, None),
        overflow_link,
    ];
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FieldDescriptor;

    /// Build the minimal `checked_add` overflow-fallback diamond and assert
    /// the rewrite lowers it to `add_ovf` + a `LastException` block whose
    /// overflow exit carries the `OverflowError` exitcase.  Mirrors the
    /// `int_add` MIR shape: block A = `checked_add(va, vb) -> opt`; block C
    /// = `__discriminant` switch; Some arm reads `opt.__pos_0`; None arm
    /// (overflow) dead-threads `opt`.
    fn checked_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "num", "<Impl>", "checked_add"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    #[test]
    fn rewrite_lifts_checked_add_to_add_ovf_with_overflow_edge() {
        let mut g = FunctionGraph::new("test_checked_add");
        let a = g.startblock;
        let va = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let vb = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        // Block A's last op = the residual `checked_add(va, vb)`.
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: vec![va.clone(), vb.clone()],
                    result_ty: ValueType::Ref(Some("core::option::Option".into())),
                },
                true,
            )
            .unwrap();

        // Block C — the discriminant switch — with `opt` rebound as its
        // inputarg.
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();

        // Some target: reads `opt.__pos_0` then `w_int_new`.
        let (some_t, some_args) = g.create_block_with_arg_vars(1);
        let some_opt = some_args[0].clone();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: some_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        g.set_return(some_t, None);

        // None target (overflow): a leaf block that never reads `opt`.
        let none_t = g.create_block();
        g.set_return(none_t, None);

        // Close A → C, threading `opt`.
        g.set_goto(a, c, vec![opt.clone()]);
        // Close C with the discriminant switch: case 0 → None, case 1 → Some.
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        let rewritten = rewire_checked_arith_call_sites(&mut g, &[opt.clone()]);
        assert_eq!(rewritten, 1, "the checked_add site must be rewritten");

        // Block A's last op is now `add_ovf(va, vb)`, reusing `opt`.
        let last = g.blocks[a.0].operations.last().unwrap();
        match &last.kind {
            OpKind::BinOp { op, lhs, rhs, .. } => {
                assert_eq!(op, "add_ovf");
                assert_eq!(lhs, &va);
                assert_eq!(rhs, &vb);
                assert_eq!(last.result.as_ref(), Some(&opt));
            }
            other => panic!("expected add_ovf BinOp, got {other:?}"),
        }

        // Block A closes with LastException: normal → Some, OverflowError →
        // None.
        assert!(matches!(
            g.blocks[a.0].exitswitch,
            Some(ExitSwitch::LastException)
        ));
        let exits = &g.blocks[a.0].exits;
        assert_eq!(exits.len(), 2);
        assert!(
            exits[0].exitcase.is_none(),
            "normal arm carries no exitcase"
        );
        assert_eq!(exits[0].target, some_t);
        assert_eq!(exits[1].exitcase, Some(overflowerror_exitcase()));
        assert_eq!(exits[1].target, none_t);
        // The overflow link must not carry the raising op result `opt`
        // (checkgraph invariant) — the dead `opt` thread is substituted.
        assert!(
            !exits[1]
                .args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if *v == opt)),
            "overflow link must not carry the raising op result"
        );
        assert!(
            exits[1].last_exception.is_some() && exits[1].last_exc_value.is_some(),
            "overflow link carries the last_exception / last_exc_value pair"
        );
    }

    #[test]
    fn rewrite_declines_when_none_arm_reads_option() {
        // A None arm that READS `opt` (not a dead thread) is a real
        // consumer the rewrite cannot honour — it must decline, leaving the
        // residual call.
        let mut g = FunctionGraph::new("test_reads_opt");
        let a = g.startblock;
        let va = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let vb = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(),
                    args: vec![va, vb],
                    result_ty: ValueType::Ref(Some("core::option::Option".into())),
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();

        let (some_t, _) = g.create_block_with_arg_vars(1);
        g.set_return(some_t, None);
        // None target READS opt via a FieldRead — disqualifies the rewrite.
        // Model framestate reuse: the target's inputarg IS `opt_c` (the
        // same identity the None link forwards), as the production
        // framestate threading produces, so `var_read_in_reachable` sees
        // the read.
        let none_t = g.create_block();
        g.push_inputarg_var(none_t, opt_c.clone());
        g.push_op_var(
            none_t,
            OpKind::FieldRead {
                base: opt_c.clone(),
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        g.set_return(none_t, None);

        g.set_goto(a, c, vec![opt.clone()]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        let rewritten = rewire_checked_arith_call_sites(&mut g, &[opt.clone()]);
        assert_eq!(rewritten, 0, "a None arm that reads opt must decline");
        // The residual call survives untouched.
        let last = g.blocks[a.0].operations.last().unwrap();
        assert!(matches!(
            &last.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { .. },
                ..
            }
        ));
    }
}
