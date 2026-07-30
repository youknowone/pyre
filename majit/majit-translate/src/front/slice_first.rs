//! `<[T]>::first(slice)` → length-checked `Option<&T>` diamond.
//!
//! ## Positioning
//!
//! `core::slice::<Impl>::first` is a foreign leaf whose body is Opaque in the
//! LLBC (Charon cannot extract `core`), so the caller emits a residual `first`
//! call — an unregistered callee the rtyper census Skips.  Its `Self` is the
//! primitive slice `[T]` (not an ADT), so `lower_call` keeps the raw
//! `FunctionPath` segments `["core","slice","<Impl>","first"]` (same shape as
//! `bool::then`), receiver in `args[0]`.  Unlike `bool::then`, the condition is
//! not an operand — `first` returns `Some(&slice[0])` iff `slice` is non-empty,
//! so this pass *synthesizes* the guard `len(slice) > 0`:
//!
//! ```text
//!     opt = first(slice)                 // residual `first` call
//! becomes
//!     if len(slice) > 0 { opt = Some(&slice[0]) } else { opt = None }
//! ```
//!
//! The branch is mandatory — the element read must not run when `slice` is
//! empty (`slice[0]` would be an out-of-bounds read), so a single-block
//! always-read encoding is unsound.  This is exactly why an *inline* fold (the
//! `emit_tagged_pair_aggregate` path, which writes `__pos_0` unconditionally in
//! the call's own block before the consumer's discriminant switch) cannot
//! express `first`; a post-pass that splits block A into a guarded Some/None
//! diamond can.
//!
//! ## Payload representation
//!
//! `first` returns `Option<&T>`, but the `Some` payload is materialised by
//! `OpKind::ArrayRead` (element 0), which yields the element VALUE, not a
//! pointer-to-slot.  In the list model a `&T` and a `T` are the same one GC
//! pointer word, and no consumer derefs a slot-pointer (there is no `copied`
//! pass that would; the front has no pointer-to-slot op at all).  So the
//! `&T`-vs-`T` distinction collapses harmlessly, and the value payload is both
//! the only representable and the correct choice.  The receiver is a
//! `SomeList`-modelled slice (its `Input` op carries the list-container
//! `class_root`), so `__len` and the element read repr-dispatch to
//! `arraylen_gc` / `getarrayitem` on the underlying length-prefixed GcArray.
//!
//! ## The rewrite (`rewire_one_slice_first_site`)
//!
//! Block A holds the residual `first` call producing `opt`.  Because
//! `Option<&PyObjectRef>` triggers `option_residual_narrow_root`, `lower_call`
//! appends a trailing `__pyre_cast_instance` after the call, so the call is NOT
//! A's last op — the block-A skeleton absorbs that optional cast, exactly as
//! [`crate::front::option_map_or`] does.  The rewrite:
//! 1. drops the `first` call (+ absorbed cast) and closes A with a
//!    `bool(len(slice) > 0)` branch to two fresh arms;
//! 2. the `then_bb` arm reads `slice[0]` and wraps it in `Some`
//!    (`__discriminant = 1` / `__pos_0 = elem`);
//! 3. the `else_bb` arm builds `None` (`__discriminant = 0`);
//! 4. both arms re-apply the absorbed narrowing and forward to B, reproducing
//!    A's original exit args with the `opt` slot sourced from the arm's
//!    `Some`/`None` value and every other live value threaded through.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `first` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::front::option_map_or::emit_narrow;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `<[T]>::first(slice)` call site captured during body lowering
/// (`front::mir` `recognize_slice_first_site`).  The owner strings are resolved
/// at the recording site where the destination `Option<&T>` type is in hand;
/// the post-pass only needs them to spell the `Some`/`None` aggregates.
#[derive(Clone)]
pub(crate) struct SliceFirstSite {
    /// The `first` call result (the `Option<&T>` value) — locates block A.  The
    /// slice operand is read from that located call's `args[0]`.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` (per-instantiation, suffixed) — the
    /// ctor owner for the `Some`/`None` aggregates.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub some_owner: String,
    /// The `Option`'s payload `&T` projected to a [`ValueType`] — the
    /// `Some::__pos_0` field kind (`Ref(None)` for `Option<&PyObjectRef>`).
    pub payload_ty: ValueType,
}

/// Rewrite every recorded `<[T]>::first` call site into the length-checked
/// `Option` diamond.  Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip), so a mismatch never regresses
/// a graph the legacy walker already handled.  Returns the number of sites
/// rewritten.
pub(crate) fn rewire_slice_first_call_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceFirstSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_first_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `first` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_slice_first_site(
    graph: &mut FunctionGraph,
    site: &SliceFirstSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `first` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: slice::first result var has no producer block"))?;

    // Locate the `first` call op by its result (not assuming it is the block
    // tail — the trailing `__pyre_cast_instance` may follow, see below).
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: slice::first call op not found in block {a}"))?;
    let ops_len = graph.blocks[a].operations.len();

    // `Option<&PyObjectRef>` gains a trailing `__pyre_cast_instance` narrowing
    // op (`result_narrow_root`) whose output is what the block forwards on.
    // Absorb that optional cast: `flow_result` is the value B consumes,
    // `narrow_root` re-applies the narrowing per arm, `remove_upto` bounds the
    // ops to drop.  Any other trailing shape declines (fail-safe).
    let (flow_result, narrow_root, remove_upto) = if ci + 1 == ops_len {
        (site.result_var.clone(), None, ci)
    } else if ci + 2 == ops_len {
        let cast = &graph.blocks[a].operations[ci + 1];
        match (&cast.kind, cast.result.as_ref()) {
            (
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    args,
                    ..
                },
                Some(narrowed),
            ) if segments.len() == 2
                && segments[0] == "__pyre_cast_instance"
                && args.len() == 1
                && args[0] == site.result_var =>
            {
                (narrowed.clone(), Some(segments[1].clone()), ci + 1)
            }
            _ => {
                return Err(format!(
                    "{name}: slice::first call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: slice::first call is not the last op of block {a}"
        ));
    };

    // Capture the slice receiver operand.
    let slice = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: slice::first producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the Option).  Must be a
    // plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: slice::first call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: slice::first call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // Option itself (`flow_result`); each must be threaded through the diamond
    // arms to reach B (a fresh block cannot see A-scope Variables directly).
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // --- All structural validation passed; mutate the graph. ---

    // `then_bb` (`Some`) carries `carried` plus `slice` (the base for the
    // element read); `else_bb` (`None`) carries only `carried`.  The
    // source-var lists double as the branch link args.
    let mut then_sources = carried.clone();
    if !then_sources.contains(&slice) {
        then_sources.push(slice.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(carried.len());

    // `then_bb`: elem = slice[0]; opt = Some(elem).
    let slice_in_then = map_source(&then_sources, &then_inputs, &slice)
        .ok_or_else(|| format!("{name}: slice not threaded into Some arm"))?;
    let idx0 = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(idx0.clone()),
        kind: OpKind::ConstInt(0),
    });
    let elem = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(elem.clone()),
        kind: OpKind::ArrayRead {
            base: slice_in_then,
            // The element read and the `Some::__pos_0` field write below
            // consume the same `elem`, so the read's declared element type
            // must be the payload type the field carries (`site.payload_ty`,
            // `Ref(None)` for `Option<&PyObjectRef>`) — a hardcoded `Ref(None)`
            // would disagree for any instantiation whose payload projects to
            // another `ValueType`.
            item_ty: site.payload_ty.clone(),
            index: idx0,
            array_type_id: None,
            nolength: false,
            pure: false,
        },
    });
    let some_var = emit_option_variant(
        graph,
        then_bb,
        &site.option_owner,
        1,
        Some((&site.some_owner, elem, site.payload_ty.clone())),
    );
    let then_result = emit_narrow(graph, then_bb, some_var, &narrow_root);
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &then_result,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    // `else_bb`: opt = None.
    let none_var = emit_option_variant(graph, else_bb, &site.option_owner, 0, None);
    let else_result = emit_narrow(graph, else_bb, none_var, &narrow_root);
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &else_result,
        &carried,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    // A: drop the residual `first` call (+ absorbed cast), synthesize the guard
    // `len(slice) > 0`, branch on it.  `__len` on the `SomeList`-modelled slice
    // routes through the rtyper `len` op → `AbstractBaseListRepr.rtype_len` →
    // `arraylen_gc`; the `gt` of two `Int` operands lowers to `int_gt`.
    // `set_branch` appends the idempotent `bool(cond)` hop and installs the
    // Bool(false)/Bool(true) arm links, so the true (non-empty) arm is `then_bb`.
    let a_id = graph.blocks[a].id;
    for _ in ci..=remove_upto {
        graph.blocks[a].operations.remove(ci);
    }
    let len = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(len.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__len".to_string()],
            },
            args: vec![slice.clone()],
            result_ty: ValueType::Int,
        },
    });
    let zero = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(zero.clone()),
        kind: OpKind::ConstInt(0),
    });
    let cond = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(cond.clone()),
        kind: OpKind::BinOp {
            op: "gt".to_string(),
            lhs: len,
            rhs: zero,
            result_ty: ValueType::Int,
        },
    });
    graph.set_branch(a_id, cond, then_bb, then_sources, else_bb, carried);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn slice_first_site(result_var: Variable) -> SliceFirstSite {
        SliceFirstSite {
            result_var,
            option_owner: "core::option::Option".into(),
            some_owner: "core::option::Option::Some".into(),
            payload_ty: ValueType::Ref(None),
        }
    }

    /// Build the minimal `opt = first(slice)` shape — block A = the residual
    /// call closed by a single goto to B (which consumes the Option) — and
    /// assert the rewrite drops the call, synthesizes `len(slice) > 0`, and
    /// branches to a `Some` arm (`slice[0]` → `Some`) and a `None` arm, both
    /// merging to B.
    #[test]
    fn rewrite_lifts_first_to_length_checked_option() {
        let mut g = FunctionGraph::new("test_slice_first");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "first".into(),
                        ],
                    },
                    args: vec![slice.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        // B: the continuation consuming the first() result.
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let rewritten = rewire_slice_first_call_sites(&mut g, &[slice_first_site(opt.clone())]);
        assert_eq!(rewritten, 1, "the slice::first site must be rewritten");

        // The residual `first` call is gone from block A.
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().map(String::as_str) == Some("first")
            )),
            "residual first call removed from A"
        );
        // A synthesizes the `__len` guard and a `gt` compare, then branches.
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.first().map(String::as_str) == Some("__len")
            )),
            "A synthesizes the __len guard"
        );
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "gt")),
            "A compares len > 0"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        // Exactly one arm reads an array element (the Some payload).
        let elem_reads = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ArrayRead { .. }))
            .count();
        assert_eq!(elem_reads, 1, "the Some arm reads slice[0]");
        // Both arms write an Option `__discriminant` (Some=1, None=0).
        let disc_writes = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldWrite { field, .. } if field.name == "__discriminant")
            })
            .count();
        assert_eq!(disc_writes, 2, "both arms write a discriminant");
    }

    /// The production shape: an `Option<&RegisteredStruct>` result appends a
    /// trailing `__pyre_cast_instance` narrowing op, so the call is NOT the
    /// block tail.  The rewrite absorbs the cast, fires, and re-applies the
    /// narrowing in both arms.
    #[test]
    fn rewrite_absorbs_trailing_narrow_cast() {
        let mut g = FunctionGraph::new("test_slice_first_narrow");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "first".into(),
                        ],
                    },
                    args: vec![slice.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let narrowed = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__pyre_cast_instance".into(), "PyObject".into()],
                    },
                    args: vec![opt.clone()],
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();

        // B consumes the NARROWED value (what the block actually forwards).
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![narrowed.clone()]);

        // Site records the CALL result; the rewrite discovers the trailing cast.
        let rewritten = rewire_slice_first_call_sites(&mut g, &[slice_first_site(opt.clone())]);
        assert_eq!(rewritten, 1, "the slice::first site must be rewritten");

        // The residual call and the trailing cast are gone from block A.
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().map(String::as_str) == Some("first")
            )),
            "residual first call removed from A"
        );
        // Two arms each re-emit a `__pyre_cast_instance` narrowing.
        let narrow_casts = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                        if segments.first().map(String::as_str) == Some("__pyre_cast_instance")
                )
            })
            .count();
        assert_eq!(narrow_casts, 2, "each diamond arm re-applies the narrowing");
    }

    /// A call block whose trailing shape is neither the bare call nor a single
    /// absorbed cast declines (fail-safe): the residual call survives untouched.
    #[test]
    fn rewrite_declines_on_unexpected_trailing_shape() {
        let mut g = FunctionGraph::new("test_slice_first_decline");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "first".into(),
                        ],
                    },
                    args: vec![slice],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // Two trailing ops break both the "call is last" and "single cast" shapes.
        g.push_op_var(a, OpKind::ConstInt(9), true).unwrap();
        g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        g.set_return(a, None);

        let rewritten = rewire_slice_first_call_sites(&mut g, &[slice_first_site(opt)]);
        assert_eq!(rewritten, 0, "an unexpected trailing shape declines");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().map(String::as_str) == Some("first")
            )),
            "residual call survives on decline"
        );
    }
}
