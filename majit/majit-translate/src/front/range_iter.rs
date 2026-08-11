//! `for _ in a..b` (exclusive int `Range`) → `range(a, b)` + list-iter.
//!
//! ## Positioning
//!
//! A Rust `for _ in a..b` desugars to `IntoIterator::into_iter(a..b)` then a
//! `Iterator::next()` / `Option` match loop.  `a..b` lowers as an
//! `Rvalue::Aggregate(Adt(core::ops::range::Range), [a, b])`, which
//! `front::mir` emits as a `SyntheticTransparentCtor` + `FieldWrite("start")`
//! / `FieldWrite("end")` chain — i.e. `SomeInstance(classdef=
//! 'core.ops.range.Range')`.  The struct arm attaches no per-instantiation
//! suffix, so every `Range<T>` collapses onto one classdef and the `start` /
//! `end` fields union across `T` (`int ∪ <other>`), while the loop's
//! reflexive `into_iter` (`impl<I: Iterator> IntoIterator for I`) is an
//! identity alias that plants no `iter` op, so `front::iter_next`'s
//! `originates_from_iter_op` gate declines and the residual `next()` blocks
//! as `getattr(range, "next")`.
//!
//! RPython models `range(a, b)` as a `SomeList` carrying `range_step`
//! (`builtin.builtin_range`), whose `RangeRepr` `make_iterator_repr` yields
//! the integer-induction `RangeIteratorRepr` — the orthodox path.  This pass
//! reroutes the exclusive-int-`Range` for-loop onto it:
//!
//! ```text
//!     r = Range{start=a, end=b}          // ctor + FieldWrite(start/end)
//!     ... for _ in r (reflexive into_iter alias) ... next(r)
//! becomes
//!     t = range(a, b)                    // `SomeList` (range_step = 1)
//!     r = iter(t)                        // `["core","slice","iter"]` → SomeIterator
//!     ... next(r) folded by front::iter_next into the native `next` op
//! ```
//!
//! `range` is emitted under the reserved `["__pyre_range"]` spelling (see
//! [`range_builtin_call`]) and resolves through
//! `HOST_ENV.lookup_builtin("range")` →
//! `SomeBuiltin("range")` → `builtin_range` (`SomeList` with `range_step`);
//! `["core","slice","iter"]` is the same `iter`-op bridge the slice/Vec
//! for-loop uses (`is_concrete_iter_constructor`), and on a range-`SomeList`
//! receiver it still selects `RangeIteratorRepr`.  Rewriting the aggregate to
//! that pair makes `originates_from_iter_op` see the `iter` op, so the
//! ALREADY-CAPTURED `next` site folds through the existing diamond — no new
//! iteration machinery, no `is_iter_op_segments` change.
//!
//! ## Consumer gate + fail-safe
//!
//! Only a range whose value feeds EXACTLY its own construction
//! (`FieldWrite`) and this for-loop's `next` is diverted; a range read by any
//! other op (`v[a..b]` slice-index via `Index::index`, a `.start` field read,
//! a stored range) keeps the ordinary ADT ctor path — turning it into a
//! `SomeList`/iterator would break that consumer.  Every structural mismatch
//! declines (leaves the ctor + FieldWrite chain, census Skip), so a decline
//! never regresses a graph the legacy walker already handled.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized exclusive int-`Range { start, end }` aggregate captured
/// during body lowering (`front::mir`'s `Rvalue::Aggregate` arm).  Carries
/// the ctor result (the range value the reflexive `into_iter` aliases into
/// the loop iterator) and the two int bounds threaded into the `range()`
/// builtin call.
#[derive(Clone)]
pub(crate) struct RangeNewSite {
    /// The `SyntheticTransparentCtor` result (the `Range` value) — locates
    /// the ctor op and the `next` iterator's backward-traced origin.
    pub result_var: Variable,
    /// Lower bound `a` — the `range(a, b)` start argument.
    pub start: Variable,
    /// Upper bound `b` — the `range(a, b)` stop argument.
    pub end: Variable,
}

/// Rewrite every captured exclusive-int-`Range` aggregate that feeds a
/// for-loop `next` into the `range(a, b)` builtin + list-iter pair, so the
/// `front::iter_next` diamond (which runs immediately after) folds the loop.
/// Fail-safe: a site that does not match the expected for-loop shape is left
/// as the ordinary ctor + FieldWrite chain (census Skip).  Returns the number
/// of range aggregates rewritten.
pub(crate) fn rewire_range_iter_sites(
    graph: &mut FunctionGraph,
    range_sites: &[RangeNewSite],
    next_results: &[Variable],
) -> usize {
    let mut rewritten = 0;
    for next_opt in next_results {
        match rewire_one_range_iter_site(graph, next_opt, range_sites) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the ctor + FieldWrite chain; the shared Range
                // classdef keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_range_iter_site(
    graph: &mut FunctionGraph,
    next_opt: &Variable,
    range_sites: &[RangeNewSite],
) -> Result<(), String> {
    let name = graph.name.clone();

    // 1. Locate the `next()` producer op and its iterator operand (args[0]).
    let (na, ni) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| op.result.as_ref() == Some(next_opt))
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: next() result var has no producer op"))?;
    let iter_arg = match &graph.blocks[na].operations[ni].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: next() producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    // 2. Backward-trace the iterator operand to a captured int-`Range`
    //    aggregate (the reflexive `into_iter` aliases the range value into
    //    the loop iterator, so the trace reaches the ctor result var).
    let site = range_sites
        .iter()
        .find(|s| range_iter_origin_matches(graph, &iter_arg, &s.result_var))
        .ok_or_else(|| {
            format!(
                "{name}: next() iterator does not originate from a captured int-Range aggregate"
            )
        })?;
    let res = site.result_var.clone();

    // 3. Consumer gate: the range value must feed ONLY its own construction
    //    (`FieldWrite`) and this for-loop's `next`.  A `v[a..b]` slice-index,
    //    a `.start` read, or a stored range reads it as a different op
    //    operand — diverting it to a `SomeList`/iterator would break that
    //    consumer, so decline.
    if !range_feeds_only_forloop(graph, &res, next_opt) {
        return Err(format!(
            "{name}: range value has a non-iteration consumer — declining"
        ));
    }

    // 4. The bounds must be defined so the `range()` call is well-formed.
    if !graph_defines(graph, &site.start) || !graph_defines(graph, &site.end) {
        return Err(format!("{name}: range bounds are not defined in the graph"));
    }
    let start = site.start.clone();
    let end = site.end.clone();

    // 5. Remove every `start` / `end` `FieldWrite` on the range value (any
    //    block): after the divert `res` is a `SomeIterator`, so a surviving
    //    `setattr("start")` would wall on it.
    for b in &mut graph.blocks {
        b.operations
            .retain(|op| !matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == &res));
    }

    // 6. Replace the `SyntheticTransparentCtor` (still the producer of `res`
    //    after the FieldWrite sweep) with `t = range(a, b)` and bind `res`
    //    to `iter(t)` so the reflexive-into_iter alias and the captured
    //    `next` site see an iter-op-produced iterator.
    let (cb, ctor_idx) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| {
                    op.result.as_ref() == Some(&res)
                        && matches!(
                            &op.kind,
                            OpKind::Call {
                                target: CallTarget::SyntheticTransparentCtor { .. },
                                ..
                            }
                        )
                })
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: range aggregate ctor op vanished"))?;
    let tmp = graph.alloc_value_var();
    graph.blocks[cb].operations[ctor_idx] = range_builtin_call(tmp.clone(), start, end);
    graph.blocks[cb]
        .operations
        .insert(ctor_idx + 1, slice_iter_call(res, tmp));
    Ok(())
}

/// `true` iff `iter_arg` traces — directly or through loop-carried block
/// inputargs — to the `Range` aggregate result `range_result`.  Mirrors
/// `iter_next::originates_from_iter_op`'s backward walk (a var that is a
/// block inputarg is traced to each predecessor's link arg in the matching
/// slot, so the loop-header iterator phi resolves through its entry edge to
/// the pre-loop range value), but keys on reaching the captured range value
/// instead of an `iter` op.
fn range_iter_origin_matches(
    graph: &FunctionGraph,
    iter_arg: &Variable,
    range_result: &Variable,
) -> bool {
    let mut visited: Vec<Variable> = Vec::new();
    let mut stack: Vec<Variable> = vec![iter_arg.clone()];
    while let Some(v) = stack.pop() {
        if visited.contains(&v) {
            continue;
        }
        visited.push(v.clone());
        if &v == range_result {
            return true;
        }
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

/// `true` when the range value is consumed by EXACTLY its own construction
/// (`FieldWrite` writes) and this for-loop's `next` op — the gate that keeps
/// a range read by any other op (slice-index, `.start`, a stored range) on
/// the ordinary ADT ctor path.  Computes the forward alias closure of the
/// range value (every Variable it flows into through positional `Link.args`,
/// including the renamed loop-iterator phi) and rejects if any op / exitswitch
/// / exception payload other than the construction writes or the `next` op
/// reads a closure member.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn range_feeds_only_forloop(
    graph: &FunctionGraph,
    range_result: &Variable,
    next_opt: &Variable,
) -> bool {
    use std::collections::HashSet;
    let mut closure: HashSet<Variable> = HashSet::new();
    closure.insert(range_result.clone());
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

    let in_closure = |v: &Variable| closure.contains(v);
    for block in &graph.blocks {
        for op in &block.operations {
            let reads_range = crate::inline::op_variable_refs(&op.kind)
                .iter()
                .any(&in_closure);
            if !reads_range {
                continue;
            }
            // A construction `FieldWrite` (base in the closure) writes the
            // range's `start` / `end`; the `next` op reads the iterator.
            // Both are removed / folded by the rewrite; anything else is a
            // genuine second consumer.
            let is_construction_write =
                matches!(&op.kind, OpKind::FieldWrite { base, .. } if in_closure(base));
            let is_next = op.result.as_ref() == Some(next_opt);
            if !(is_construction_write || is_next) {
                return false;
            }
        }
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(v)) if in_closure(v) => return false,
            Some(crate::model::ExitSwitch::Fused { args, .. }) if args.iter().any(&in_closure) => {
                return false;
            }
            _ => {}
        }
        for link in &block.exits {
            if link
                .last_exception
                .as_ref()
                .and_then(LinkArg::as_variable)
                .is_some_and(&in_closure)
                || link
                    .last_exc_value
                    .as_ref()
                    .and_then(LinkArg::as_variable)
                    .is_some_and(&in_closure)
            {
                return false;
            }
        }
    }
    true
}

/// `true` when `var` is produced by some op or is a block inputarg — i.e. it
/// is a well-defined value the rewritten `range()` call can reference.
fn graph_defines(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|b| {
        b.inputargs.iter().any(|iv| iv == var)
            || b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(var))
    })
}

/// `t = range(start, end)` — the `range` builtin call the adapter resolves
/// through `HOST_ENV.lookup_builtin("range")` → `SomeBuiltin("range")` →
/// `builtin_range` (a `SomeList` carrying `range_step`).
///
/// Spelled with the reserved `__pyre_range` segment rather than a bare
/// `["range"]`: the adapter resolves a single-segment path against the
/// `PyreCallRegistry` before `HOST_ENV`, and pyre's registry is one flat
/// namespace keyed by leaf-only `CallPath`s, so a pyre free function named
/// `range` (`pyre_interpreter::module::_ast::convert::range`) registers as
/// `["range"]` and would capture this call. The reserved spelling has a
/// dedicated `translate_op` arm that binds the `HOST_ENV` singleton, keeping
/// the Arc identity `BUILTIN_TYPER` keys `rtype_builtin_range` on.
fn range_builtin_call(result: Variable, start: Variable, end: Variable) -> SpaceOperation {
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__pyre_range".to_string()],
            },
            args: vec![start, end],
            result_ty: ValueType::Ref(None),
        },
    }
}

/// `r = iter(t)` — the `["core","slice","iter"]` bridge op (the same
/// constructor the slice/Vec for-loop uses); on the range-`SomeList`
/// receiver `Repr::rtype_iter` selects `RangeIteratorRepr`.
fn slice_iter_call(result: Variable, container: Variable) -> SpaceOperation {
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["core".to_string(), "slice".to_string(), "iter".to_string()],
            },
            args: vec![container],
            result_ty: ValueType::Ref(None),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range_ctor_target() -> CallTarget {
        CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["core".to_string(), "ops".to_string(), "range".to_string()],
            "Range".to_string(),
        )
    }

    fn field_write(base: &Variable, name: &str, value: &Variable) -> OpKind {
        OpKind::FieldWrite {
            base: base.clone(),
            field: crate::model::FieldDescriptor {
                name: name.to_string(),
                owner_root: Some("core::ops::range::Range".to_string()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        }
    }

    fn count_calls_ending(g: &FunctionGraph, tail: &[&str]) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(&op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                        if segments.len() >= tail.len()
                            && segments[segments.len() - tail.len()..]
                                .iter()
                                .zip(tail)
                                .all(|(s, t)| s == t))
            })
            .count()
    }

    fn count_range_ctors(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { .. },
                        ..
                    }
                )
            })
            .count()
    }

    /// Build a `for _ in a..b` shape — the range ctor + FieldWrites in the
    /// entry block, the reflexive-into_iter alias delivering the range to a
    /// loop header, `next(range)` in the header — and assert the rewrite
    /// drops the ctor + both FieldWrites and emits `range(a,b)` + `iter`.
    #[test]
    fn rewrite_diverts_int_range_forloop_to_range_plus_iter() {
        let mut g = FunctionGraph::new("test_range_iter");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: range_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::Range".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: field_write(&range, "start", &a),
        });
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: field_write(&range, "end", &b),
        });

        // Loop header H: inputarg = the iterator (entry-threaded range).
        let (h, h_args) = g.create_block_with_arg_vars(1);
        let it = h_args[0].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: CallTarget::method("next", Some("Range".to_string())),
                    args: vec![it.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(h, Some(opt.clone()));
        g.set_goto(n, h, vec![range.clone()]);

        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range.clone(),
                start: a,
                end: b,
            }],
            &[opt],
        );
        assert_eq!(rewritten, 1, "the range for-loop must be diverted");
        assert_eq!(count_range_ctors(&g), 0, "range ctor removed");
        assert_eq!(
            count_calls_ending(&g, &["__pyre_range"]),
            1,
            "one range() builtin emitted"
        );
        assert_eq!(
            count_calls_ending(&g, &["core", "slice", "iter"]),
            1,
            "one iter op emitted"
        );
    }

    /// A range whose value is also read by a non-iteration op (here a
    /// `.start` field read modeled as a unary op on the range) keeps the
    /// ordinary ctor path (fail-safe): diverting it to an iterator would
    /// break that reader.
    #[test]
    fn rewrite_declines_when_range_has_second_consumer() {
        let mut g = FunctionGraph::new("test_range_iter_second_consumer");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: range_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::Range".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: field_write(&range, "start", &a),
        });
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: field_write(&range, "end", &b),
        });
        // Second consumer: an op reads the range value directly.
        let _second = g.push_op_var(
            n,
            OpKind::UnaryOp {
                op: "invert".to_string(),
                operand: range.clone(),
                result_ty: ValueType::Ref(None),
            },
            true,
        );

        let (h, h_args) = g.create_block_with_arg_vars(1);
        let it = h_args[0].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: CallTarget::method("next", Some("Range".to_string())),
                    args: vec![it.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(h, Some(opt.clone()));
        g.set_goto(n, h, vec![range.clone()]);

        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range.clone(),
                start: a,
                end: b,
            }],
            &[opt],
        );
        assert_eq!(rewritten, 0, "a second range consumer declines the divert");
        assert_eq!(count_range_ctors(&g), 1, "range ctor survives");
        assert_eq!(
            count_calls_ending(&g, &["__pyre_range"]),
            0,
            "no range() builtin emitted"
        );
    }
}
