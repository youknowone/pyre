//! `Option::unwrap_or(opt, default)` / `Result::unwrap_or(res, default)` →
//! discriminant value-select.
//!
//! ## Positioning
//!
//! `core::{option,result}::<Impl>::unwrap_or` is a foreign combinator whose
//! body is Opaque in the LLBC (Charon cannot extract `core`), so the caller
//! emits a residual `unwrap_or` call — an unregistered callee the rtyper
//! census Skips.  Like [`crate::front::bool_then`] and unlike
//! [`crate::front::checked_arith`], the combinator's match lives inside the
//! opaque body: at the call site there is no discriminant switch to rewrite,
//! only `result = unwrap_or(recv, default)` flowing on.  This pass *creates*
//! the two-way select the combinator's semantics imply:
//!
//! ```text
//!     result = opt.unwrap_or(default)     // residual `unwrap_or` call
//! becomes
//!     if opt.__discriminant == 1 { result = opt.__pos_0 } else { result = default }
//! ```
//!
//! Both candidates are already-computed values with no side effects, so a
//! single-block value-select would be sound — but the graph has no
//! conditional-move primitive, so the select is spelled as a two-arm diamond
//! (the payload variant reads `recv.__pos_0`, the other arm forwards
//! `default`).  The front models every enum uniformly with an explicit
//! `__discriminant` field (Rust niche optimisation is a codegen detail below
//! the IR).  The payload variant's discriminant differs by enum: `Option::Some
//! = 1` (`None = 0`), `Result::Ok = 0` (`Err = 1`), so `UnwrapOrSite`'s
//! `payload_on_disc_true` records which `bool(disc)` arm reads `__pos_0`.
//!
//! ## The rewrite (`rewire_one_unwrap_or_site`)
//!
//! Block A holds the residual `unwrap_or` call producing `result` as its
//! last op, closed by `lower_call` with a single forwarding exit to block B
//! (the continuation consuming `result`).  The rewrite:
//! 1. drops the `unwrap_or` call, reads `disc = recv.__discriminant`, and
//!    closes A with a `bool(disc)` branch to two fresh arms;
//! 2. the payload arm reads `recv.__pos_0`, the other arm forwards `default`;
//!    `payload_on_disc_true` picks which is the `bool(disc)`-true arm;
//! 3. both arms forward to B, reproducing A's original exit args with the
//!    `result` slot sourced from the arm's payload / default value and every
//!    other live value threaded through the arm's inputargs.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `unwrap_or`
//! callee keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{
    CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// A recognized `Option::unwrap_or(opt, default)` / `Result::unwrap_or(res,
/// default)` call site captured during body lowering (`front::mir`
/// `recognize_unwrap_or_site`).  The owner strings are resolved at the
/// recording site where the receiver enum type is in hand; the post-pass only
/// needs them to spell the `__discriminant` / `__pos_0` field reads in the
/// synthesized arms.
#[derive(Clone)]
pub(crate) struct UnwrapOrSite {
    /// The `unwrap_or` call result (the payload `T` value) — locates block A.
    pub result_var: Variable,
    /// The enum root `name_path` (`Option` / `Result`) — the `__discriminant`
    /// field owner.
    pub enum_owner: String,
    /// The payload variant `name_path` (`Option::Some` / `Result::Ok`) — the
    /// `__pos_0` payload field owner (matching the variant-qualified
    /// `resolve_adt_field` read owner).
    pub payload_owner: String,
    /// The payload `T` projected to a [`ValueType`] — the `__pos_0` field kind
    /// and the select result kind.
    pub payload_ty: ValueType,
    /// True when the payload variant sits on discriminant 1 (`Option::Some`),
    /// false when on discriminant 0 (`Result::Ok`) — selects which `bool(disc)`
    /// arm reads `__pos_0` versus forwards the `default`.
    pub payload_on_disc_true: bool,
}

/// Rewrite every recorded `Option::unwrap_or` call site into the
/// discriminant value-select diamond.  Fail-safe: a site whose block does
/// not fit the residual-call shape is left untouched (Skip), so a mismatch
/// never regresses a graph the legacy walker already handled.  Returns the
/// number of sites rewritten.
pub(crate) fn rewire_unwrap_or_call_sites(
    graph: &mut FunctionGraph,
    sites: &[UnwrapOrSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_unwrap_or_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `unwrap_or` call; the unregistered
                // callee keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_unwrap_or_site(graph: &mut FunctionGraph, site: &UnwrapOrSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `unwrap_or` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: unwrap_or result var has no producer block"))?;

    // Index of the `unwrap_or` call op within block A.
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .expect("result var producer resolved to block A above");
    let last_idx = graph.blocks[a].operations.len() - 1;
    // The call sits at the block tail, optionally followed by a single
    // `__pyre_cast_instance(result)` narrowing op.  `lower_call` appends that
    // cast when the payload is a `*mut <registered ADT>` (a raw-pointer
    // `Option` payload) and reassigns the destination local to the narrowed
    // var, so the continuation consumes `narrowed`, not the raw call result.
    // The cast is jitcode-identity (`cast_pointer` → `same_as`); carry it into
    // each arm so B keeps receiving a narrowed value.  `out_var` is the value
    // block B actually consumes for the select result.
    let (cast, out_var): (Option<(Vec<String>, ValueType)>, Variable) = if call_idx == last_idx {
        (None, site.result_var.clone())
    } else if call_idx == last_idx - 1 {
        let tail = &graph.blocks[a].operations[last_idx];
        match (&tail.kind, tail.result.clone()) {
            (
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    args,
                    result_ty,
                },
                Some(narrowed),
            ) if segments.first().map(String::as_str) == Some("__pyre_cast_instance")
                && args.as_slice() == std::slice::from_ref(&site.result_var) =>
            {
                (Some((segments.clone(), result_ty.clone())), narrowed)
            }
            _ => {
                return Err(format!(
                    "{name}: unwrap_or call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: unwrap_or call is not the last op of block {a}"
        ));
    };
    // Capture the receiver `Option` + default operands.
    let (opt, default) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: unwrap_or producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the payload).  Must be
    // a plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: unwrap_or call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: unwrap_or call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // select output itself; each must be threaded through the diamond arms to
    // reach B (a fresh block cannot see A-scope Variables directly).
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != out_var
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // --- All structural validation passed; mutate the graph. ---

    // The payload arm carries `carried` plus `recv` (the base for the
    // `__pos_0` read); the default arm carries `carried` plus `default` (the
    // forwarded fallback).  The source-var lists double as the branch link
    // args.  Which arm is the `bool(disc)`-true (`then`) target is decided
    // below from `payload_on_disc_true`.
    let mut payload_sources = carried.clone();
    if !payload_sources.contains(&opt) {
        payload_sources.push(opt.clone());
    }
    let mut default_sources = carried.clone();
    if !default_sources.contains(&default) {
        default_sources.push(default.clone());
    }
    let (payload_bb, payload_inputs) = graph.create_block_with_arg_vars(payload_sources.len());
    let (default_bb, default_inputs) = graph.create_block_with_arg_vars(default_sources.len());

    // Payload arm: value = recv.__pos_0 (narrowed if the call carried a cast).
    let recv_in_payload = map_source(&payload_sources, &payload_inputs, &opt)
        .ok_or_else(|| format!("{name}: receiver not threaded into payload arm"))?;
    let payload = graph.alloc_value_var();
    graph.block_mut(payload_bb).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: recv_in_payload,
            field: FieldDescriptor {
                name: "__pos_0".to_string(),
                owner_root: Some(site.payload_owner.clone()),
                owner_id: None,
            },
            ty: site.payload_ty.clone(),
            pure: true,
        },
    });
    let payload_value = emit_narrow(graph, payload_bb, &cast, payload);
    let payload_link_args = reproduce_exit_args(
        &saved_exit,
        &out_var,
        &payload_value,
        &payload_sources,
        &payload_inputs,
        &name,
    )?;
    close_goto_mixed(graph, payload_bb, b_target, payload_link_args);

    // Default arm: forward `default` (narrowed if the call carried a cast).
    let default_in_arm = map_source(&default_sources, &default_inputs, &default)
        .ok_or_else(|| format!("{name}: default value not threaded into default arm"))?;
    let default_value = emit_narrow(graph, default_bb, &cast, default_in_arm);
    let default_link_args = reproduce_exit_args(
        &saved_exit,
        &out_var,
        &default_value,
        &default_sources,
        &default_inputs,
        &name,
    )?;
    close_goto_mixed(graph, default_bb, b_target, default_link_args);

    // A: drop the residual `unwrap_or` call, read the discriminant, branch on
    // it.  `set_branch` appends the `bool(disc)` hop and installs the
    // Bool(false)/Bool(true) arm links.  The payload variant's discriminant
    // differs by enum (`Option::Some = 1`, `Result::Ok = 0`), so
    // `payload_on_disc_true` picks whether the payload arm is the true (`then`)
    // target.  The receiver/default construction ops stay as A's tail.
    let a_id = graph.blocks[a].id;
    // Drop the trailing narrowing cast (if any) first so `call_idx` stays valid,
    // then the `unwrap_or` call — both are subsumed by the diamond (the cast is
    // re-emitted per arm).
    if cast.is_some() {
        graph.blocks[a].operations.remove(last_idx);
    }
    graph.blocks[a].operations.remove(call_idx);
    let disc = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(disc.clone()),
        kind: OpKind::FieldRead {
            base: opt.clone(),
            field: FieldDescriptor {
                name: "__discriminant".to_string(),
                owner_root: Some(site.enum_owner.clone()),
                owner_id: None,
            },
            ty: ValueType::Int,
            pure: true,
        },
    });
    let ((then_bb, then_sources), (else_bb, else_sources)) = if site.payload_on_disc_true {
        ((payload_bb, payload_sources), (default_bb, default_sources))
    } else {
        ((default_bb, default_sources), (payload_bb, payload_sources))
    };
    graph.set_branch(a_id, disc, then_bb, then_sources, else_bb, else_sources);
    Ok(())
}

/// Re-emit the `__pyre_cast_instance` narrowing the residual call carried into
/// an arm's `raw` payload value.  `None` (no cast on the original result) is a
/// pass-through: the raw value flows on unchanged.  The cast is jitcode-identity
/// (`cast_pointer` → `same_as`), so duplicating it per arm is sound.
fn emit_narrow(
    graph: &mut FunctionGraph,
    bb: crate::model::BlockId,
    cast: &Option<(Vec<String>, ValueType)>,
    raw: Variable,
) -> Variable {
    let Some((segments, result_ty)) = cast else {
        return raw;
    };
    let narrowed = graph.alloc_value_var();
    graph.block_mut(bb).operations.push(SpaceOperation {
        result: Some(narrowed.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: segments.clone(),
            },
            args: vec![raw],
            result_ty: result_ty.clone(),
        },
    });
    narrowed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn unwrap_or_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "option", "<Impl>", "unwrap_or"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    fn option_site(result_var: Variable) -> UnwrapOrSite {
        UnwrapOrSite {
            result_var,
            enum_owner: "core::option::Option".into(),
            payload_owner: "core::option::Option::Some".into(),
            payload_ty: ValueType::Int,
            payload_on_disc_true: true,
        }
    }

    fn result_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "result", "<Impl>", "unwrap_or"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    fn result_site(result_var: Variable) -> UnwrapOrSite {
        UnwrapOrSite {
            result_var,
            enum_owner: "core::result::Result".into(),
            payload_owner: "core::result::Result::Ok".into(),
            payload_ty: ValueType::Int,
            // `Result::Ok = 0`, so the payload arm is the `bool(disc)`-false arm.
            payload_on_disc_true: false,
        }
    }

    /// Build the minimal `result = unwrap_or(opt, default)` shape — block A =
    /// the residual call closed by a single goto to B (which consumes the
    /// result) — and assert the rewrite drops the call, reads
    /// `opt.__discriminant`, and branches to a `Some` arm (`opt.__pos_0`) and
    /// a `None` arm (the `default`), both merging to B.
    #[test]
    fn rewrite_lifts_unwrap_or_to_discriminant_select() {
        let mut g = FunctionGraph::new("test_unwrap_or");
        let a = g.startblock;
        // `opt` (the receiver Option) and `default`, both defined before the
        // call; a `ConstInt` placeholder stands in for the Option value.
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(42), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: vec![opt.clone(), default.clone()],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        // B: the continuation consuming the unwrapped payload.
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[option_site(result.clone())]);
        assert_eq!(rewritten, 1, "the unwrap_or site must be rewritten");

        // The residual `unwrap_or` call is gone from the whole graph.
        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| matches!(&op.kind, OpKind::Call { .. })),
            "residual unwrap_or call removed"
        );
        // Block A reads `opt.__discriminant` exactly once.
        let disc_reads = g.blocks[a.0]
            .operations
            .iter()
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
            })
            .count();
        assert_eq!(disc_reads, 1, "A reads the Option discriminant once");
        // Block A now branches two ways (Some / None arms).
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        // Exactly one arm reads `opt.__pos_0` (the Some payload).
        let pos0_reads = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(
                |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0"),
            )
            .count();
        assert_eq!(pos0_reads, 1, "the Some arm reads __pos_0");
        // Both arms merge to B.
        let arms_to_b = g.blocks[a.0]
            .exits
            .iter()
            .filter(|link| g.blocks[link.target.0].exits.iter().any(|l| l.target == b))
            .count();
        assert_eq!(
            arms_to_b, 2,
            "both diamond arms forward to the continuation"
        );
    }

    /// A raw-pointer payload (`Option<*mut PyObject>`) makes `lower_call`
    /// append a `__pyre_cast_instance(result)` narrowing op after the
    /// `unwrap_or` call, so the call is the block's second-to-last op and the
    /// continuation consumes the *narrowed* var.  The rewrite must tolerate
    /// that trailing identity cast: lift to the discriminant select and apply
    /// the same narrowing to each arm's payload so B still receives a narrowed
    /// value.
    #[test]
    fn rewrite_lifts_unwrap_or_with_trailing_narrowing_cast() {
        let mut g = FunctionGraph::new("test_unwrap_or_cast");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(42), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: vec![opt.clone(), default.clone()],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // The narrowing cast `lower_call` appends for a `*mut <registered ADT>`
        // result; the continuation consumes its `narrowed` var, not `result`.
        let narrowed = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__pyre_cast_instance".into(), "PyObject".into()],
                    },
                    args: vec![result.clone()],
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![narrowed.clone()]);

        let mut site = option_site(result.clone());
        site.payload_ty = ValueType::Ref(None);
        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[site]);
        assert_eq!(
            rewritten, 1,
            "the trailing-cast unwrap_or site is rewritten"
        );

        // No residual `unwrap_or` call survives (the two arm casts remain).
        assert!(
            !g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().map(String::as_str) == Some("unwrap_or"))
            }),
            "residual unwrap_or call removed"
        );
        // Block A branches two ways after reading the discriminant.
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        // Each arm applies the narrowing cast to its payload (Some: __pos_0,
        // None: default), so two casts remain in the arms.
        let arm_casts = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.first().map(String::as_str) == Some("__pyre_cast_instance"))
            })
            .count();
        assert_eq!(arm_casts, 2, "each diamond arm narrows its payload");
    }

    /// `Result::unwrap_or` inverts the payload polarity: `Result::Ok = 0`, so
    /// the payload (`__pos_0`) arm is reached via the `bool(disc)`-false exit
    /// and the `default` via the `bool(disc)`-true exit — the mirror of the
    /// `Option` layout.
    #[test]
    fn rewrite_lifts_result_unwrap_or_payload_on_disc_false() {
        use crate::model::ExitCase;
        let mut g = FunctionGraph::new("test_result_unwrap_or");
        let a = g.startblock;
        let res = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: result_target(),
                    args: vec![res.clone(), default.clone()],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[result_site(result.clone())]);
        assert_eq!(rewritten, 1, "the Result unwrap_or site is rewritten");

        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Ok/Err arms");
        let reads_pos0 = |bb: usize| {
            g.blocks[bb].operations.iter().any(
                |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0"),
            )
        };
        for link in &g.blocks[a.0].exits {
            let tgt = link.target.0;
            match &link.exitcase {
                Some(ExitCase::Bool(false)) => {
                    assert!(reads_pos0(tgt), "Ok arm (disc==0) reads __pos_0")
                }
                Some(ExitCase::Bool(true)) => {
                    assert!(
                        !reads_pos0(tgt),
                        "Err arm (disc==1) forwards default, no __pos_0"
                    )
                }
                other => panic!("unexpected branch exitcase {other:?}"),
            }
        }
    }

    /// A call block whose last op is not the recorded result declines
    /// (fail-safe): the residual call survives untouched.
    #[test]
    fn rewrite_declines_when_call_not_last_op() {
        let mut g = FunctionGraph::new("test_unwrap_or_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_or_target(),
                    args: vec![opt, default],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // A trailing op after the call breaks the "call is A's last op" shape.
        g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        g.set_return(a, None);

        let rewritten = rewire_unwrap_or_call_sites(&mut g, &[option_site(result)]);
        assert_eq!(rewritten, 0, "a non-last-op call declines");
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::Call { .. })),
            "residual call survives on decline"
        );
    }
}
