//! `<[T]>::get(slice, i)` → bounds-checked `Option<&T>` diamond.
//!
//! ## Positioning
//!
//! `core::slice::<Impl>::get` is a foreign leaf whose body is Opaque in the
//! LLBC (Charon cannot extract `core`), so the caller emits a residual `get`
//! call — an unregistered callee the rtyper census Skips, which Skips the
//! CALLING graph with it.  Its `Self` is the primitive slice `[T]` (not an
//! ADT), so `lower_call` keeps the raw `FunctionPath` segments
//! `["core","slice","<Impl>","get"]`, receiver in `args[0]` and index in
//! `args[1]`.  `get` returns `Some(&slice[i])` iff `i` is in bounds, so this
//! pass *synthesizes* the guard `i < len(slice)`:
//!
//! ```text
//!     opt = get(slice, i)                // residual `get` call
//! becomes
//!     if i < len(slice) { opt = Some(&slice[i]) } else { opt = None }
//! ```
//!
//! The branch is mandatory — the element read must not run when `i` is out of
//! range (`slice[i]` would read past the block), so a single-block always-read
//! encoding is unsound.  This is exactly why an *inline* fold (the
//! `emit_tagged_pair_aggregate` path, which writes `__pos_0` unconditionally in
//! the call's own block before the consumer's discriminant switch) cannot
//! express `get`; a post-pass that splits block A into a guarded Some/None
//! diamond can.  It is the same "diamond is unavoidable" shape as
//! [`crate::front::slice_first`], of which `get` is the general case: `first`
//! is `get` at a fixed index 0, where `i < len` collapses to `len > 0`.
//!
//! Only the upper bound is tested.  The index is a `usize` — the scalar
//! `SliceIndex` instantiation is the only one this pass fires for (see below)
//! — so it is non-negative by type and needs no lower-bound compare.
//!
//! ## The `SliceIndex` instantiation gate
//!
//! `<[T]>::get` is generic over `SliceIndex`, and only its scalar
//! instantiation has the shape this diamond encodes.  `get(0..2)` returns
//! `Option<&[T]>` — a sub-slice, not an element — and an `ArrayRead` at the
//! range's start would hand the consumer a `T` where a `[T]` is expected.
//! `front::mir` therefore pins the instantiation through
//! `is_slice_get_scalar_call`: a local index uses the operand's declared
//! `usize`, while a literal uses the method's `I` generic.  Every range form
//! falls through to the range lowering instead of being mistaken for an item
//! read.
//!
//! ## Payload representation
//!
//! `get` returns `Option<&T>`, but the `Some` payload is materialised by
//! `OpKind::ArrayRead`, which yields the element VALUE, not a pointer-to-slot.
//! In the list model a `&T` and a `T` use the same item register — a GC pointer
//! for object items and the scalar bank for `u8`/`i64` items — and no consumer
//! derefs a slot-pointer.  The site's payload type therefore comes from the
//! destination `Option` consumer shape rather than being hard-coded to a
//! reference.  The receiver is a `SomeList`-modelled slice (its
//! `Input` op carries the list-container `class_root`), so `__len` and the
//! element read repr-dispatch to `arraylen_gc` / `getarrayitem` on the
//! underlying length-prefixed GcArray.
//!
//! ## The rewrite (`rewire_one_slice_get_site`)
//!
//! Block A holds the residual `get` call producing `opt`.  Because
//! `Option<&PyObjectRef>` triggers `option_residual_narrow_root`, `lower_call`
//! appends a trailing `__cast_instance_intrinsic` after the call, so the call is NOT
//! A's last op — the block-A skeleton absorbs that optional cast, exactly as
//! [`crate::front::option_map_or`] does.  The rewrite:
//! 1. drops the `get` call (+ absorbed cast) and closes A with a
//!    `bool(i < len(slice))` branch to two fresh arms;
//! 2. the `then_bb` arm reads `slice[i]` and wraps it in `Some`
//!    (`__discriminant = 1` / `__pos_0 = elem`);
//! 3. the `else_bb` arm builds `None` (`__discriminant = 0`);
//! 4. both arms re-apply the absorbed narrowing and forward to B, reproducing
//!    A's original exit args with the `opt` slot sourced from the arm's
//!    `Some`/`None` value and every other live value threaded through.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `get` callee keeps
//! the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::front::option_map_or::emit_narrow;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `<[T]>::get(slice, i)` call site captured during body lowering
/// (`front::mir` `recognize_slice_get_site`).  The owner strings are resolved
/// at the recording site where the destination `Option<&T>` type is in hand;
/// the post-pass only needs them to spell the `Some`/`None` aggregates.
#[derive(Clone)]
pub(crate) struct SliceGetSite {
    /// The `get` call result (the `Option<&T>` value) — locates block A.  The
    /// slice and index operands are read from that located call's
    /// `args[0]`/`args[1]`.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` (per-instantiation, suffixed) — the
    /// ctor owner for the `Some`/`None` aggregates.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub some_owner: String,
    /// The slice element `T` projected to a [`ValueType`] — the devirtualized
    /// `ArrayRead` and `Some::__pos_0` field kind. Rust spells the source
    /// result `Option<&T>`, but RPython's getarrayitem yields the value `T`.
    pub payload_ty: ValueType,
    /// Concrete ARRAY identity carrying the scalar/RPython item spelling to
    /// the descr, or `None` only for a proven thin-pointer element whose
    /// identity-less descr already has the correct one-word stride.
    pub array_type_id: Option<String>,
}

/// Rewrite every recorded `<[T]>::get` call site into the bounds-checked
/// `Option` diamond.  Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip), so a mismatch never regresses
/// a graph the legacy walker already handled.  Returns the number of sites
/// rewritten.
pub(crate) fn rewire_slice_get_call_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceGetSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_get_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `get` call; the unregistered callee keeps
                // the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_slice_get_site(graph: &mut FunctionGraph, site: &SliceGetSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `get` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: slice::get result var has no producer block"))?;

    // Locate the `get` call op by its result (not assuming it is the block
    // tail — the trailing `__cast_instance_intrinsic` may follow, see below).
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: slice::get call op not found in block {a}"))?;
    let ops_len = graph.blocks[a].operations.len();

    // `Option<&PyObjectRef>` gains a trailing `__cast_instance_intrinsic` narrowing
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
                && segments[0] == crate::runtime_names::shims::CAST_INSTANCE
                && args.len() == 1
                && args[0] == site.result_var =>
            {
                (narrowed.clone(), Some(segments[1].clone()), ci + 1)
            }
            _ => {
                return Err(format!(
                    "{name}: slice::get call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: slice::get call is not the last op of block {a}"
        ));
    };

    // Capture the slice receiver and the index operand.
    let (slice, index) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: slice::get producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the Option).  Must be a
    // plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: slice::get call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: slice::get call block {a} exit is not a plain goto"
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

    // `then_bb` (`Some`) carries `carried` plus `slice` and `index` (the base
    // and subscript of the element read); `else_bb` (`None`) carries only
    // `carried`.  The source-var lists double as the branch link args.
    let mut then_sources = carried.clone();
    for v in [&slice, &index] {
        if !then_sources.contains(v) {
            then_sources.push(v.clone());
        }
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(carried.len());

    // `then_bb`: elem = slice[i]; opt = Some(elem).
    let slice_in_then = map_source(&then_sources, &then_inputs, &slice)
        .ok_or_else(|| format!("{name}: slice not threaded into Some arm"))?;
    let index_in_then = map_source(&then_sources, &then_inputs, &index)
        .ok_or_else(|| format!("{name}: index not threaded into Some arm"))?;
    let elem = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(elem.clone()),
        kind: OpKind::ArrayRead {
            base: slice_in_then,
            // The element read and the `Some::__pos_0` field write below
            // consume the same `elem`, so the read's declared element type
            // must be the value type the devirtualized `T` carries
            // (`site.payload_ty`, `Ref(None)` for a GC-pointer element) — a
            // hardcoded `Ref(None)` would disagree for scalar elements.
            item_ty: site.payload_ty.clone(),
            index: index_in_then,
            array_type_id: site.array_type_id.clone(),
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

    // A: drop the residual `get` call (+ absorbed cast), synthesize the guard
    // `i < len(slice)`, branch on it.  `__len` on the `SomeList`-modelled slice
    // routes through the rtyper `len` op → `AbstractBaseListRepr.rtype_len` →
    // `arraylen_gc`.  The compare's result is a Bool, so it declares `Int` (as
    // `range_contains` and `slice_first` do) and never the `Unsigned` that
    // would wrap the flag in `r_uint`; the annotator derives the `lt` llop from
    // the OPERAND annotations, where the `usize` index and the non-negative
    // `arraylen_gc` both admit the unsigned compare.  `set_branch` appends the
    // idempotent `bool(cond)` hop and installs the Bool(false)/Bool(true) arm
    // links, so the true (in-bounds) arm is `then_bb`.
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
            args: vec![slice],
            result_ty: ValueType::Int,
        },
    });
    let cond = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(cond.clone()),
        kind: OpKind::BinOp {
            op: "lt".to_string(),
            lhs: index,
            rhs: len,
            result_ty: ValueType::Int,
        },
    });
    graph.set_branch(a_id, cond, then_bb, then_sources, else_bb, carried);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice_get_site(result_var: Variable) -> SliceGetSite {
        SliceGetSite {
            result_var,
            option_owner: "core::option::Option".into(),
            some_owner: "core::option::Option::Some".into(),
            payload_ty: ValueType::Ref(None),
            array_type_id: None,
        }
    }

    fn emit_call(g: &mut FunctionGraph, a: crate::model::BlockId, args: Vec<Variable>) -> Variable {
        g.push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "slice".into(), "<Impl>".into(), "get".into()],
                },
                args,
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap()
    }

    fn residual_get_survives(g: &FunctionGraph, a: crate::model::BlockId) -> bool {
        g.blocks[a.0].operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().map(String::as_str) == Some("get")
            )
        })
    }

    /// Build the minimal `opt = get(slice, i)` shape — block A = the residual
    /// call closed by a single goto to B (which consumes the Option) — and
    /// assert the rewrite drops the call, synthesizes `i < len(slice)`, and
    /// branches to a `Some` arm (`slice[i]` → `Some`) and a `None` arm, both
    /// merging to B.
    #[test]
    fn rewrite_lifts_get_to_bounds_checked_option() {
        let mut g = FunctionGraph::new("test_slice_get");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice.clone(), index.clone()]);

        // B: the continuation consuming the get() result.
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let rewritten = rewire_slice_get_call_sites(&mut g, &[slice_get_site(opt)]);
        assert_eq!(rewritten, 1, "the slice::get site must be rewritten");

        // The residual `get` call is gone from block A.
        assert!(
            !residual_get_survives(&g, a),
            "residual get call removed from A"
        );
        // A synthesizes the `__len` guard and an `lt` compare, then branches.
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
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lt")),
            "A compares i < len"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        // Exactly one arm reads an array element (the Some payload), and it
        // subscripts with the arm's threaded index — NOT a synthesized 0, which
        // is what would make this pass a mis-indexed `slice_first`.
        let elem_reads: Vec<&Variable> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::ArrayRead { index, .. } => Some(index),
                _ => None,
            })
            .collect();
        assert_eq!(elem_reads.len(), 1, "the Some arm reads slice[i]");
        let then_inputs = &g.block(g.blocks[a.0].exits[1].target).inputargs;
        assert!(
            then_inputs.contains(elem_reads[0]),
            "the element read subscripts the arm's threaded index"
        );
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

    /// A scalar slice item uses the integer register bank throughout the
    /// guarded diamond.  This is the ordinary RPython `getitem` result shape,
    /// not a pointer-to-slot or an object-pointer-only special case.
    #[test]
    fn rewrite_preserves_unsigned_item_payload() {
        let mut g = FunctionGraph::new("test_slice_get_u8");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice, index]);
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let mut site = slice_get_site(opt);
        site.payload_ty = ValueType::Unsigned;
        assert_eq!(rewire_slice_get_call_sites(&mut g, &[site]), 1);
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(
                    &op.kind,
                    OpKind::ArrayRead {
                        item_ty: ValueType::Unsigned,
                        ..
                    }
                ))
        );
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(
                    &op.kind,
                    OpKind::FieldWrite {
                        field,
                        ty: ValueType::Unsigned,
                        ..
                    } if field.name == "__pos_0"
                ))
        );
    }

    /// The production shape: an `Option<&RegisteredStruct>` result appends a
    /// trailing `__cast_instance_intrinsic` narrowing op, so the call is NOT the
    /// block tail.  The rewrite absorbs the cast, fires, and re-applies the
    /// narrowing in both arms.
    #[test]
    fn rewrite_absorbs_trailing_narrow_cast() {
        let mut g = FunctionGraph::new("test_slice_get_narrow");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice, index]);
        let narrowed = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::runtime_names::shims::CAST_INSTANCE.into(),
                            "PyObject".into(),
                        ],
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
        g.set_goto(a, b, vec![narrowed]);

        // Site records the CALL result; the rewrite discovers the trailing cast.
        let rewritten = rewire_slice_get_call_sites(&mut g, &[slice_get_site(opt)]);
        assert_eq!(rewritten, 1, "the slice::get site must be rewritten");

        // The residual call and the trailing cast are gone from block A.
        assert!(
            !residual_get_survives(&g, a),
            "residual get call removed from A"
        );
        // Two arms each re-emit a `__cast_instance_intrinsic` narrowing.
        let narrow_casts = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                        if segments.first().map(String::as_str) == Some(crate::runtime_names::shims::CAST_INSTANCE)
                )
            })
            .count();
        assert_eq!(narrow_casts, 2, "each diamond arm re-applies the narrowing");
    }

    /// A call block whose trailing shape is neither the bare call nor a single
    /// absorbed cast declines (fail-safe): the residual call survives untouched.
    #[test]
    fn rewrite_declines_on_unexpected_trailing_shape() {
        let mut g = FunctionGraph::new("test_slice_get_decline");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice, index]);
        // Two trailing ops break both the "call is last" and "single cast" shapes.
        g.push_op_var(a, OpKind::ConstInt(9), true).unwrap();
        g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        g.set_return(a, None);

        let rewritten = rewire_slice_get_call_sites(&mut g, &[slice_get_site(opt)]);
        assert_eq!(rewritten, 0, "an unexpected trailing shape declines");
        assert!(
            residual_get_survives(&g, a),
            "residual call survives on decline"
        );
    }

    /// The arity guard: `get` is recorded with two operands, so a located
    /// producer that is a 1-arg call (the `first` shape) cannot supply an
    /// index.  It declines rather than reading `slice[?]`.
    #[test]
    fn rewrite_declines_on_one_arg_producer() {
        let mut g = FunctionGraph::new("test_slice_get_arity");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice]);

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let rewritten = rewire_slice_get_call_sites(&mut g, &[slice_get_site(opt)]);
        assert_eq!(rewritten, 0, "a 1-arg producer declines");
        assert!(
            residual_get_survives(&g, a),
            "residual call survives on decline"
        );
    }
}
