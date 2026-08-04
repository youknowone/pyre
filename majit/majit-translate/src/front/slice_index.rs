//! `&s[k..]` / `&s[..k]` sub-slice indexes → orthodox `getslice` copies.
//!
//! ## Status: RangeTo and RangeFrom dormant
//!
//! These recognizers are CAPSTONES, not independently wired primitives. The
//! RangeFrom arm is built and unit-tested but is not captured or invoked from
//! `front::mir`: its rewrite plants a Void `ConstNone` stop, which had no
//! regalloc coloring when the rewritten
//! `pyre_interpreter::module::_io::buffered_rwpair::<Impl>::readinto` graph
//! dropped to the legacy walker.
//!
//! RangeTo's primary blocker is semantic: Rust's `&s[..end]` panics when
//! `end > s.len()`, but `build_ll_listslice_startstop_helper_graph` clamps an
//! oversized stop to the list length and returns the whole slice
//! (`translator/rtyper/rlist.rs:3563-3576`). Until lowering can prove
//! `end <= len`, emitting `getslice(s, 0, end)` would turn an interpreter
//! failure into success. Its secondary blocker is non-negativity: activating
//! RangeTo previously emitted bare `getslice/rii>r` from
//! `split_builtin_kwargs` and `do_warn_explicit`; both stops are
//! `args.len() - 1`, statically unsigned but not constant. An unsigned gate
//! would admit those measured failures, while the constant-nonnegative gate
//! declines them. The upper-bound proof is still missing, so both capture
//! arms remain dormant and leave the original residual ctor + call chain
//! untouched.
#![allow(dead_code)]
//!
//! ## Positioning
//!
//! `&s[k..]` on a `&[T]` slice desugars to
//! `<[T] as Index<RangeFrom<usize>>>::index(s, RangeFrom { start: k })`.
//! Its sibling `&s[..k]` uses `RangeTo { end: k }` and rewrites to
//! `getslice(s, 0, k)` (`SliceKind::StartStop`).
//! `lower_call` keeps `core::slice::index::<Impl>::index` as an unregistered
//! residual (the gh#346 census wall), and the `RangeFrom { start }` operand is
//! built by `front::mir`'s `Rvalue::Aggregate` arm as a
//! `SyntheticTransparentCtor("RangeFrom")` + `FieldWrite("start")` chain —
//! i.e. `SomeInstance(classdef='core.ops.range.RangeFrom')`, the same shape
//! `front::range_iter` sees for a `Range { start, end }` for-loop.
//!
//! RPython models `l[start:]` as a **copy**: the annotator's `getslice`
//! handler (`unaryop.py:420-423`) returns a fresh `listdef.offspring`, and the
//! rtyper lowers it through `AbstractBaseListRepr.rtype_getslice`
//! (`rlist.py:409-414`) to a `gendirectcall` of `ll_listslice_startonly`.
//! There is no borrowed-view concept upstream. This pass reroutes the residual
//! `slice::index` onto that orthodox path:
//!
//! ```text
//!     r = RangeFrom { start = k }        // ctor + FieldWrite(start)
//!     v = slice::index(s, r)             // residual index call
//! becomes
//!     v = getslice(s, k, None)           // StartOnly: stop is a const None
//! ```
//!
//! The `getslice` operand contract (`decompose_slice_args`,
//! `rtyper.rs:2787-2846`) selects `SliceKind::StartOnly` when args[2] (`stop`)
//! annotates as a constant `None`; `k` (a proven-nonneg / const start) becomes
//! args[1]. The stop operand is a fresh [`OpKind::ConstNone`] — the annotator
//! constant `None` (`Void` `concretetype`), which folds to `SomeValue::None_`
//! so the `stop_is_none` branch fires and the `len-k` copy runs at runtime.
//!
//! ## Consumer gate + fail-safe (the UNWIRED hazard)
//!
//! `getslice` is UNWIRED at the assembler — a bare `getslice` opname has no
//! blackhole handler. In a graph that fully rtypes, `rtype_getslice` replaces
//! the op with a `direct_call` before assembly, so no bare primitive survives.
//! But a graph that plants `getslice` and then DROPS to the legacy walker
//! (blocked by a co-wall the recognizer does not touch) would carry the raw op
//! to the assembler default arm and break
//! `default_bh_builder_unwired_set_matches_task_85_snapshot` — the same hazard
//! the vec!/`NewList` recognizer has (`design-346-foreign-lib-cluster-epic.md`).
//! Unlike `slice_first`, whose planted ops (`ArrayRead`/`ConstInt`/`Some`/
//! `None`) are all wired, this pass is only safe once the rewritten graph
//! rtypes. The rewrite is therefore fail-safe by construction: a range whose
//! value feeds anything other than exactly its own construction
//! (`FieldWrite`) and this one `slice::index` call is left as the residual
//! ADT ctor + call chain (census Skip), so a decline never regresses a graph
//! the legacy walker already handled, and the snapshot test is the empirical
//! gate that no dropped graph carries a bare `getslice`.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation};

/// A recognized `RangeFrom { start }` aggregate feeding a residual
/// `core::slice::index::<Impl>::index(slice, range)` call, captured during
/// body lowering (`front::mir`'s `Rvalue::Aggregate` arm). Carries the ctor
/// result (the `RangeFrom` value the index call consumes as its `range`
/// operand) and the `start` bound threaded into the `getslice` op.
#[derive(Clone)]
pub(crate) struct SliceIndexRangeFromSite {
    /// The `SyntheticTransparentCtor` result (the `RangeFrom` value) — locates
    /// the ctor op and the consuming `slice::index` call's `range` operand.
    pub range_result: Variable,
    /// The `start` bound `k` — the `getslice` `start` argument (args[1]).
    pub start: Variable,
}

/// A recognized `RangeTo { end }` aggregate feeding a residual
/// `core::slice::index::<Impl>::index(slice, range)` call.
#[derive(Clone)]
pub(crate) struct SliceIndexRangeToSite {
    pub range_result: Variable,
    pub end: Variable,
}

/// Rewrite every captured `RangeFrom` aggregate that feeds exactly one
/// residual `slice::index` call into an orthodox `getslice(slice, start,
/// None)` op. Fail-safe: a site that does not match the expected single-
/// consumer shape is left as the ordinary ctor + FieldWrite + residual call
/// chain (census Skip). Returns the number of sites rewritten.
pub(crate) fn rewire_slice_index_rangefrom_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceIndexRangeFromSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_index_site(
            graph,
            &site.range_result,
            SliceIndexBounds::RangeFrom { start: &site.start },
        ) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the ctor + FieldWrite + residual `slice::index` call;
                // the unregistered callee keeps the rtyper census Skip for
                // this graph.
            }
        }
    }
    rewritten
}

/// Rewrite captured `RangeTo { end }` indexes into
/// `getslice(slice, 0, end)` (`SliceKind::StartStop`). Production capture must
/// remain dormant until it can prove `end <= slice.len()`; the StartStop
/// helper clamps instead of preserving Rust's out-of-bounds panic.
pub(crate) fn rewire_slice_index_rangeto_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceIndexRangeToSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_index_site(
            graph,
            &site.range_result,
            SliceIndexBounds::RangeTo { end: &site.end },
        ) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual chain intact so legacy keeps its prior
                // census Skip and never sees a newly planted `getslice`.
            }
        }
    }
    rewritten
}

#[derive(Clone, Copy)]
enum SliceIndexBounds<'a> {
    RangeFrom { start: &'a Variable },
    RangeTo { end: &'a Variable },
}

fn rewire_one_slice_index_site(
    graph: &mut FunctionGraph,
    range_result: &Variable,
    bounds: SliceIndexBounds<'_>,
) -> Result<(), String> {
    let name = graph.name.clone();
    let range = range_result.clone();

    // 1. Locate the residual `slice::index` call consuming the `RangeFrom`
    //    value as its `range` operand (args[1]). Capture its `slice` operand
    //    (args[0]) and result; the position is re-found after the sweeps
    //    below, since removing the ctor / FieldWrite shifts op indices.
    let (slice, index_result) = graph
        .blocks
        .iter()
        .find_map(|b| {
            b.operations.iter().find_map(|op| {
                if !is_slice_range_index_call(&op.kind) {
                    return None;
                }
                let OpKind::Call { args, .. } = &op.kind else {
                    return None;
                };
                if args.len() != 2 || args[1] != range {
                    return None;
                }
                let result = op.result.as_ref()?;
                Some((args[0].clone(), result.clone()))
            })
        })
        .ok_or_else(|| format!("{name}: no slice::index call consumes the RangeFrom value"))?;

    // 2. Consumer gate: the `RangeFrom` value must feed EXACTLY its own
    //    construction (`FieldWrite`) and this one index call. A range read by
    //    a `.start` field read, a second index, or a stored range keeps the
    //    ordinary ADT ctor path — rewriting it would break that consumer.
    if !range_feeds_only_index(graph, &range, &index_result) {
        return Err(format!(
            "{name}: RangeFrom value has a non-index consumer — declining"
        ));
    }

    // 3. Validate the selected bounds before mutating. Both bounds must be a
    // NON-NEGATIVE CONSTANT. `decompose_slice_args`
    //     (rtyper.rs:2809-2816) raises "slice start must be proved
    //     non-negative" for a `nonneg==false` runtime `SomeInteger` start, and
    //     a `usize` literal decodes to a bare `OpKind::ConstInt(k)` with no
    //     `Unsigned`/nonneg annotation (only its value is ≥ 0). A runtime
    //     start could annotate `nonneg==false`, making the getslice fail to
    //     rtype → the graph drops to legacy carrying a bare unwired `getslice`
    //     → the unwired-opname snapshot breaks. Restricting to a const k ≥ 0
    //     (the `&s[1..]` / `&s[k..]` literal shape, which
    //     `immutablevalue(ConstInt(k))` annotates `nonneg = k>=0`) keeps every
    //     rewritten graph lift-able; a computed start declines cleanly
    //     (residual, census Skip). RangeTo's constant-nonnegative gate is only
    //     the secondary blocker: activating it admitted bare unwired
    //     `getslice/rii>r` from `split_builtin_kwargs` and
    //     `do_warn_explicit`; both measured ends were `args.len() - 1`,
    //     statically unsigned but not constant, so an unsigned-only gate would
    //     not decline them. Even a non-negative constant must not be wired in
    //     production without the primary `end <= slice.len()` proof: the
    //     StartStop helper clamps an oversized stop (`rlist.rs:3563-3576`),
    //     whereas Rust indexing panics.
    match bounds {
        SliceIndexBounds::RangeFrom { start } => {
            if !graph_defines(graph, start) {
                return Err(format!(
                    "{name}: RangeFrom start is not defined in the graph"
                ));
            }
            if !bound_is_const_nonneg(graph, start) {
                return Err(format!(
                    "{name}: RangeFrom start is not a non-negative constant — declining"
                ));
            }
        }
        SliceIndexBounds::RangeTo { end } => {
            if !graph_defines(graph, end) {
                return Err(format!("{name}: RangeTo end is not defined in the graph"));
            }
            if !bound_is_const_nonneg(graph, end) {
                return Err(format!(
                    "{name}: RangeTo end is not a non-negative constant — declining"
                ));
            }
        }
    }

    // --- All structural validation passed; mutate the graph. ---

    // 4. Remove every `start` `FieldWrite` on the `RangeFrom` value and the
    //    `SyntheticTransparentCtor` producing it. After the rewrite `range` is
    //    dead (the gate proved the index call is its only non-construction
    //    consumer), so a surviving `setattr("start")` or ctor would wall on a
    //    value nothing reads.
    for b in &mut graph.blocks {
        b.operations.retain(|op| {
            let is_field_write =
                matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == &range);
            let is_ctor = op.result.as_ref() == Some(&range)
                && matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { .. },
                        ..
                    }
                );
            !(is_field_write || is_ctor)
        });
    }

    // 5. Replace the residual `slice::index` call (re-located by result
    //    identity — the sweeps above shifted op indices) with
    //    `getslice(slice, start, None)`. The `None` stop is a fresh
    //    `ConstNone` op inserted immediately before the getslice so its result
    //    Variable is defined in the same block.
    let (rb, ri) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| {
                    op.result.as_ref() == Some(&index_result) && is_slice_range_index_call(&op.kind)
                })
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: slice::index op vanished before rewrite"))?;
    let synthetic_bound = graph.alloc_value_var();
    let (args, synthetic_kind) = match bounds {
        SliceIndexBounds::RangeFrom { start } => (
            vec![slice, start.clone(), synthetic_bound.clone()],
            OpKind::ConstNone,
        ),
        SliceIndexBounds::RangeTo { end } => (
            vec![slice, synthetic_bound.clone(), end.clone()],
            OpKind::ConstInt(0),
        ),
    };
    graph.blocks[rb].operations[ri] = SpaceOperation {
        result: Some(index_result),
        kind: OpKind::GetSlice { args },
    };
    graph.blocks[rb].operations.insert(
        ri,
        SpaceOperation {
            result: Some(synthetic_bound),
            kind: synthetic_kind,
        },
    );
    Ok(())
}

/// `true` when `kind` is a residual `core::slice::index::<Impl>::index` call —
/// the `&s[range]` sub-slice residual whose `range` operand this pass folds.
/// The scalar `Vec`/`Constants` integer index is intercepted earlier in
/// `lower_call` (`is_vec_index_regular` / `constants_call_leaf`), so only the
/// range-indexed slice residual reaches here.
fn is_slice_range_index_call(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            ..
        } if slice_index_segments_match(segments)
    )
}

/// The residual callee spelling for `<[T]>::index` with a range index.
/// Charon runs `monomorphize:false`, so the callee keeps the inherent-impl
/// path `core::slice::index::<Impl>::index`.
fn slice_index_segments_match(segments: &[String]) -> bool {
    let n = segments.len();
    n >= 3
        && segments[n - 1] == "index"
        && segments[n - 2] == "<Impl>"
        && segments[n - 3] == "index"
        && segments.first().map(String::as_str) == Some("core")
}

/// `true` when the `RangeFrom` value is consumed by EXACTLY its own
/// construction (`FieldWrite`) and the one `slice::index` call whose result is
/// `index_result`. Computes the forward alias closure of the range value
/// (every Variable it flows into through positional `Link.args`) and rejects
/// if any op / exitswitch / exception payload other than the construction
/// writes or the index op reads a closure member. Mirrors
/// `range_iter::range_feeds_only_forloop`.
fn range_feeds_only_index(
    graph: &FunctionGraph,
    range_result: &Variable,
    index_result: &Variable,
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
            // A construction `FieldWrite` on the `RangeFrom` value writes its
            // `start`; the `slice::index` op reads the range as its `range`
            // operand. Both are removed / rewritten; anything else is a genuine
            // second consumer. The base must be the range value ITSELF, not any
            // aliased inputarg in the closure: `rewire_one_slice_index_site`'s
            // removal sweep drops only a `FieldWrite` with `base == range`, so
            // accepting an aliased-base write here would leave that write
            // behind after its ctor is removed — an undefined-operand shape
            // `flowspace_adapter` rejects. `front::mir` emits the ctor and the
            // `start` write in one block (`base == range`), so this loses no
            // real site; a range threaded across a block edge declines cleanly.
            let is_construction_write =
                matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == range_result);
            let is_index = op.result.as_ref() == Some(index_result);
            if !(is_construction_write || is_index) {
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
/// is a well-defined value the rewritten `getslice` op can reference.
fn graph_defines(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|b| {
        b.inputargs.iter().any(|iv| iv == var)
            || b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(var))
    })
}

/// `true` when `var` is produced by an `OpKind::ConstInt(k)` op with `k >= 0`
/// — the `&s[k..]` literal start `decompose_slice_args` selects `StartOnly`
/// for. A `usize` slice literal decodes to `ConstInt(k)` (`decode_literal`
/// collapses `Usize` into `DecodedConst::Int`), which
/// `immutablevalue(ConstInt(k))` annotates as `SomeInteger` with
/// `nonneg = k>=0`, so a const k ≥ 0 satisfies the getslice nonneg contract
/// without a computed-stop hazard.
fn bound_is_const_nonneg(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|b| {
        b.operations.iter().any(|op| {
            op.result.as_ref() == Some(var) && matches!(&op.kind, OpKind::ConstInt(k) if *k >= 0)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ValueType;

    fn rangefrom_ctor_target() -> CallTarget {
        CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["core".to_string(), "ops".to_string(), "range".to_string()],
            "RangeFrom".to_string(),
        )
    }

    fn rangeto_ctor_target() -> CallTarget {
        CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["core".to_string(), "ops".to_string(), "range".to_string()],
            "RangeTo".to_string(),
        )
    }

    fn start_field_write(base: &Variable, value: &Variable) -> OpKind {
        OpKind::FieldWrite {
            base: base.clone(),
            field: crate::model::FieldDescriptor {
                name: "start".to_string(),
                owner_root: Some("core::ops::range::RangeFrom".to_string()),
                owner_id: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        }
    }

    fn end_field_write(base: &Variable, value: &Variable) -> OpKind {
        OpKind::FieldWrite {
            base: base.clone(),
            field: crate::model::FieldDescriptor {
                name: "end".to_string(),
                owner_root: Some("core::ops::range::RangeTo".to_string()),
                owner_id: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        }
    }

    fn slice_index_call_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: vec![
                "core".into(),
                "slice".into(),
                "index".into(),
                "<Impl>".into(),
                "index".into(),
            ],
        }
    }

    /// Build the minimal `&s[k..]` shape — a `ConstInt(k)` start, a
    /// `RangeFrom` ctor + `start` FieldWrite, and a residual
    /// `slice::index(slice, range)` — and assert the rewrite drops the ctor +
    /// FieldWrite + residual call and emits `getslice(slice, k, None)`.
    #[test]
    fn rewrite_lifts_rangefrom_index_to_getslice() {
        let mut g = FunctionGraph::new("test_slice_index");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let k = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangefrom_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeFrom".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: start_field_write(&range, &k),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: vec![slice.clone(), range.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub.clone()]);

        let site = SliceIndexRangeFromSite {
            range_result: range.clone(),
            start: k.clone(),
        };
        let rewritten = rewire_slice_index_rangefrom_sites(&mut g, &[site]);
        assert_eq!(rewritten, 1, "the slice::index RangeFrom site is rewritten");

        // The residual `slice::index` call, the RangeFrom ctor, and the
        // `start` FieldWrite are all gone.
        let has_index = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .any(|op| is_slice_range_index_call(&op.kind));
        assert!(!has_index, "residual slice::index call removed");
        let has_ctor = g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )
        });
        assert!(!has_ctor, "RangeFrom ctor removed");
        let has_start_write = g.blocks.iter().flat_map(|blk| &blk.operations).any(
            |op| matches!(&op.kind, OpKind::FieldWrite { field, .. } if field.name == "start"),
        );
        assert!(!has_start_write, "start FieldWrite removed");

        // A single `getslice` op with 3 args (slice, k, None) is emitted, its
        // stop operand produced by a `ConstNone` op.
        let getslice_ops: Vec<_> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::GetSlice { .. }))
            .collect();
        assert_eq!(getslice_ops.len(), 1, "exactly one getslice op");
        let OpKind::GetSlice { args } = &getslice_ops[0].kind else {
            unreachable!()
        };
        assert_eq!(args.len(), 3, "getslice has [slice, start, stop]");
        assert_eq!(args[0], slice, "arg0 = slice receiver");
        assert_eq!(args[1], k, "arg1 = const start k");
        let stop = &args[2];
        let stop_is_const_none =
            g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
                op.result.as_ref() == Some(stop) && matches!(&op.kind, OpKind::ConstNone)
            });
        assert!(stop_is_const_none, "arg2 = ConstNone (StartOnly stop)");
    }

    /// `&s[..k]` uses the existing StartStop contract: a synthetic constant
    /// zero start and the aggregate's real `end` value as stop.
    #[test]
    fn rewrite_lifts_rangeto_index_to_getslice() {
        let mut g = FunctionGraph::new("test_slice_index_rangeto");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let end = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: vec![slice.clone(), range.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // Model the production failure shape: the now-dead RangeTo value is
        // also threaded into a successor phi beside the live slice result.
        let (b, b_args) = g.create_block_with_arg_vars(2);
        g.set_return(b, Some(b_args[0].clone()));
        g.set_goto(a, b, vec![sub, range.clone()]);

        let site = SliceIndexRangeToSite {
            range_result: range.clone(),
            end: end.clone(),
        };
        assert_eq!(rewire_slice_index_rangeto_sites(&mut g, &[site]), 1);
        crate::model::prune_dead_phis(&mut g);

        assert_eq!(g.block(b).inputargs, vec![b_args[0].clone()]);
        assert_eq!(g.block(a).exits[0].args.len(), 1);
        assert_ne!(
            g.block(a).exits[0].args[0].as_variable(),
            Some(&range),
            "orphan RangeTo link arg must be removed"
        );

        let getslice = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .find_map(|op| match &op.kind {
                OpKind::GetSlice { args } => Some(args),
                _ => None,
            })
            .expect("RangeTo index becomes getslice");
        assert_eq!(getslice[0], slice);
        assert_eq!(getslice[2], end);
        let start = &getslice[1];
        assert!(g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            op.result.as_ref() == Some(start) && matches!(&op.kind, OpKind::ConstInt(0))
        }));
        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| is_slice_range_index_call(&op.kind))
        );
    }

    /// A RangeTo whose end is computed at runtime declines even when its Rust
    /// type is unsigned. The measured `args.len() - 1` sites have this shape
    /// and produced bare unwired `getslice/rii>r` when admitted.
    #[test]
    fn rewrite_declines_runtime_end() {
        let mut g = FunctionGraph::new("test_slice_index_rangeto_runtime");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let len = g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        let one = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let end = g
            .push_op_var(
                a,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: len,
                    rhs: one,
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: vec![slice, range.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub]);

        let site = SliceIndexRangeToSite {
            range_result: range,
            end,
        };
        assert_eq!(
            rewire_slice_index_rangeto_sites(&mut g, &[site]),
            0,
            "runtime RangeTo end declines"
        );
        assert!(
            g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| is_slice_range_index_call(&op.kind)),
            "residual slice::index call remains"
        );
        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| matches!(&op.kind, OpKind::GetSlice { .. })),
            "no getslice is planted on decline"
        );
    }

    /// A RangeFrom whose start is a RUNTIME value (not a const) declines: a
    /// non-nonneg runtime start would fail `decompose_slice_args` and drop the
    /// graph to legacy carrying a bare unwired `getslice`.
    #[test]
    fn rewrite_declines_runtime_start() {
        let mut g = FunctionGraph::new("test_slice_index_runtime");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        // A runtime start: the result of a BinOp, not a ConstInt.
        let x = g.push_op_var(a, OpKind::ConstInt(5), true).unwrap();
        let start = g
            .push_op_var(
                a,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: x.clone(),
                    rhs: x.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangefrom_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeFrom".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: start_field_write(&range, &start),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: vec![slice.clone(), range.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub.clone()]);

        let site = SliceIndexRangeFromSite {
            range_result: range.clone(),
            start: start.clone(),
        };
        let rewritten = rewire_slice_index_rangefrom_sites(&mut g, &[site]);
        assert_eq!(rewritten, 0, "runtime start declines");
        // The residual call and ctor survive untouched (census Skip).
        assert!(
            g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| is_slice_range_index_call(&op.kind)),
            "residual slice::index call left in place"
        );
        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| matches!(&op.kind, OpKind::GetSlice { .. })),
            "no getslice planted on decline"
        );
    }

    /// A RangeFrom read by a SECOND consumer (a `.start` field read, here
    /// modeled as another op reading the range value) declines — rewriting it
    /// would break that consumer.
    #[test]
    fn rewrite_declines_multi_consumer_range() {
        let mut g = FunctionGraph::new("test_slice_index_multi");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let k = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangefrom_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeFrom".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: start_field_write(&range, &k),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: vec![slice.clone(), range.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // A second consumer: a FieldRead of `.start` on the range value.
        g.push_op_var(
            a,
            OpKind::FieldRead {
                base: range.clone(),
                field: crate::model::FieldDescriptor {
                    name: "start".to_string(),
                    owner_root: Some("core::ops::range::RangeFrom".to_string()),
                    owner_id: None,
                },
                ty: ValueType::Int,
                pure: true,
            },
            true,
        )
        .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub.clone()]);

        let site = SliceIndexRangeFromSite {
            range_result: range.clone(),
            start: k.clone(),
        };
        let rewritten = rewire_slice_index_rangefrom_sites(&mut g, &[site]);
        assert_eq!(rewritten, 0, "a second range consumer declines the rewrite");
        assert!(
            g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| is_slice_range_index_call(&op.kind)),
            "residual slice::index call left in place"
        );
    }
}
