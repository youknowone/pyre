//! `Option::unwrap(opt)` / `Result::unwrap(res)` → discriminant guard +
//! payload extraction.
//!
//! ## Positioning
//!
//! `core::{option,result}::<Impl>::unwrap` is a foreign combinator whose body
//! is Opaque in the LLBC (Charon cannot extract `core`), so the caller emits a
//! residual `unwrap` call — an unregistered callee the rtyper census Skips. Like
//! [`crate::front::option_unwrap_or`] the combinator's match lives inside the
//! opaque body: at the call site there is only `result = unwrap(opt)` flowing
//! on.  This pass creates the guard the combinator's semantics imply:
//!
//! ```text
//!     result = recv.unwrap()              // residual `unwrap` call
//! becomes
//!     if recv is the payload variant { result = recv.__pos_0 } else { <panic> }
//! ```
//!
//! Unlike `unwrap_or` there is no default: `unwrap` panics on the non-payload
//! variant.  That arm is therefore not a value path but an implicit-`AssertionError`
//! raise ([`FunctionGraph::set_raise_implicit`]) — the "shouldn't occur at
//! run-time" shape [`crate::model::remove_assertion_errors`] prunes, matching
//! `unwrap`'s successful-path contract.  The payload arm reads
//! `recv.__pos_0` exactly as `unwrap_or`'s does. `Option`'s payload tag is
//! `Some = 1`; `Result`'s is `Ok = 0`, so the recorded polarity selects the
//! correct `bool(disc)` arm.
//!
//! ## The rewrite (`rewire_one_unwrap_site`)
//!
//! Block A holds the residual `unwrap` call producing `result` as its last op,
//! closed by `lower_call` with a single forwarding exit to block B (the
//! continuation consuming `result`).  The rewrite:
//! 1. drops the `unwrap` call, reads `disc = recv.__discriminant`, and closes A
//!    with a `bool(disc)` branch to two fresh arms;
//! 2. the payload arm reads `recv.__pos_0` and forwards it to B as
//!    the `result` slot, threading every other live value through;
//! 3. the non-payload arm raises the implicit `AssertionError` (no edge to B).
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `unwrap` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `Option::unwrap(opt)` / `Result::unwrap(res)` call site
/// captured during body lowering
/// (`front::mir` `recognize_unwrap_site`).  The owner strings are resolved at
/// the recording site where the receiver enum type is in hand; the
/// post-pass only needs them to spell the `__discriminant` / `__pos_0` field
/// reads in the synthesized payload arm.
#[derive(Clone)]
pub(crate) struct UnwrapSite {
    /// The `unwrap` call result (the payload `T` value) — locates block A.
    pub result_var: Variable,
    /// The enum root `name_path` — the `__discriminant` field owner.
    pub enum_owner: String,
    /// The payload variant `name_path` (`Option::Some` / `Result::Ok`) — the
    /// `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub payload_owner: String,
    /// The enum's payload `T` projected to a [`ValueType`] — the `__pos_0`
    /// field kind and the extracted result kind.
    pub payload_ty: ValueType,
    /// True for `Option::Some = 1`; false for `Result::Ok = 0`.
    pub payload_on_disc_true: bool,
    /// True when the receiver is a niche `Option<NonNull<_>>` — a one-word
    /// pointer with no aggregate `__discriminant` / `__pos_0`.  The `Some`-arm
    /// discriminant is then `opt != null` and the payload is the base pointer
    /// itself (identity), not a `__pos_0` field read.
    pub niche: bool,
}

/// Rewrite every recorded `Option::unwrap` / `Result::unwrap` call site into
/// the discriminant
/// guard.  Fail-safe: a site whose block does not fit the residual-call shape
/// is left untouched (Skip), so a mismatch never regresses a graph the legacy
/// walker already handled.  Returns the number of sites rewritten.
pub(crate) fn rewire_unwrap_call_sites(graph: &mut FunctionGraph, sites: &[UnwrapSite]) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_unwrap_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `unwrap` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_unwrap_site(graph: &mut FunctionGraph, site: &UnwrapSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `unwrap` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: unwrap result var has no producer block"))?;

    // The call must be A's last op (lower_call closes the block right after
    // pushing it) so removing it leaves the receiver construction as the tail.
    let call_idx = graph.blocks[a].operations.len() - 1;
    if graph.blocks[a].operations[call_idx].result.as_ref() != Some(&site.result_var) {
        return Err(format!(
            "{name}: unwrap call is not the last op of block {a}"
        ));
    }
    // Capture the receiver enum operand (the sole argument).
    let recv = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: unwrap producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the payload).  Must be a
    // plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: unwrap call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: unwrap call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // payload itself; each must be threaded through the payload arm to reach B.
    // The non-payload arm raises and never reaches B, so it carries nothing.
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != site.result_var
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // --- All structural validation passed; mutate the graph. ---

    // `payload_bb` carries `carried` plus `recv` (the base for the `__pos_0`
    // read); `failure_bb` has no inputs — it raises. The source-var list also
    // serves as the branch link args.
    let mut payload_sources = carried.clone();
    if !payload_sources.contains(&recv) {
        payload_sources.push(recv.clone());
    }
    let (payload_bb, payload_inputs) = graph.create_block_with_arg_vars(payload_sources.len());
    let (failure_bb, _failure_inputs) = graph.create_block_with_arg_vars(0);

    // `then_bb`: payload = opt.__pos_0.  A niche `Option<NonNull>` has no
    // aggregate `__pos_0`; the payload IS the base pointer (identity).
    let recv_in_payload = map_source(&payload_sources, &payload_inputs, &recv)
        .ok_or_else(|| format!("{name}: enum value not threaded into payload arm"))?;
    let payload = if site.niche {
        recv_in_payload
    } else {
        let payload = graph.alloc_value_var();
        graph.block_mut(payload_bb).operations.push(SpaceOperation {
            result: Some(payload.clone()),
            kind: OpKind::FieldRead {
                base: recv_in_payload,
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(site.payload_owner.clone()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                },
                ty: site.payload_ty.clone(),
                pure: true,
            },
        });
        payload
    };
    let payload_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &payload,
        &payload_sources,
        &payload_inputs,
        &name,
    )?;
    close_goto_mixed(graph, payload_bb, b_target, payload_link_args);

    // The non-payload arm raises: `unwrap` panics on `None` / `Err`.
    graph.set_raise_implicit(failure_bb, "unwrap on non-payload variant");

    // A: drop the residual `unwrap` call, read the discriminant, branch on it.
    // The receiver construction ops stay as A's tail. `Option::Some = 1`, so
    // its payload is the true arm; `Result::Ok = 0`, so its payload is false.
    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.remove(call_idx);
    let disc = graph.alloc_value_var();
    if site.niche {
        // Niche `Option<NonNull>`: discriminant = `opt != null` (`None` = null
        // = 0, `Some` = non-null = 1) — a `ne` on two `Ref` operands lowers to
        // `ptr_ne` with an `Int` result matching the aggregate read.  The null
        // is a repr-adaptive `null_mut()` call, not a fixed-GCREF
        // `ConstRefNull`, so `ptr_ne` sees the receiver's `InstanceRepr`.
        let nullc = graph.push_null_mut_ptr(a_id);
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(disc.clone()),
            kind: OpKind::BinOp {
                op: "ne".to_string(),
                lhs: recv.clone(),
                rhs: nullc,
                result_ty: ValueType::Int,
            },
        });
    } else {
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(disc.clone()),
            kind: OpKind::FieldRead {
                base: recv.clone(),
                field: FieldDescriptor {
                    name: "__discriminant".to_string(),
                    owner_root: Some(site.enum_owner.clone()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                },
                ty: ValueType::Int,
                pure: true,
            },
        });
    }
    if site.payload_on_disc_true {
        graph.set_branch(
            a_id,
            disc,
            payload_bb,
            payload_sources,
            failure_bb,
            Vec::new(),
        );
    } else {
        graph.set_branch(
            a_id,
            disc,
            failure_bb,
            Vec::new(),
            payload_bb,
            payload_sources,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn unwrap_target() -> CallTarget {
        CallTarget::method("unwrap", None)
    }

    /// Build `result = opt.unwrap()` closed by a goto to a continuation block
    /// consuming `result`, and assert the rewrite drops the `unwrap` call,
    /// reads `opt.__discriminant`, branches, extracts `opt.__pos_0` in the
    /// `Some` arm, and raises in the `None` arm.
    #[test]
    fn rewrite_lifts_unwrap_to_discriminant_guard() {
        let mut g = FunctionGraph::new("test_option_unwrap");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_target(),
                    args: vec![opt.clone()],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_unwrap_call_sites(
            &mut g,
            &[UnwrapSite {
                result_var: result.clone(),
                enum_owner: "core::option::Option".to_string(),
                payload_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
                payload_on_disc_true: true,
                niche: false,
            }],
        );
        assert_eq!(rewritten, 1, "the unwrap site must be rewritten");

        // The residual `unwrap` call is gone.
        let has_unwrap_call = g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "unwrap"
            )
        });
        assert!(!has_unwrap_call, "residual unwrap call removed");

        // A `__discriminant` read and a `__pos_0` read exist.
        let field_reads: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. } => Some(field.name.clone()),
                _ => None,
            })
            .collect();
        assert!(
            field_reads.iter().any(|n| n == "__discriminant"),
            "discriminant read emitted"
        );
        assert!(
            field_reads.iter().any(|n| n == "__pos_0"),
            "Some-arm payload read emitted"
        );

        // Exactly one arm raises to the exceptblock (the `None` arm).
        let raises = g
            .blocks
            .iter()
            .filter(|blk| blk.exits.iter().any(|link| link.target == g.exceptblock))
            .count();
        assert_eq!(raises, 1, "the None arm raises to exceptblock");
    }

    /// `Result::Ok = 0`, so its payload arm is the `bool(disc)`-false exit;
    /// the true (`Err`) exit raises. This is the mirror of `Option::Some = 1`.
    #[test]
    fn rewrite_lifts_result_unwrap_payload_on_disc_false() {
        use crate::model::ExitCase;

        let mut g = FunctionGraph::new("test_result_unwrap");
        let a = g.startblock;
        let recv = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_target(),
                    args: vec![recv],
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_unwrap_call_sites(
            &mut g,
            &[UnwrapSite {
                result_var: result,
                enum_owner: "core::result::Result<usize,TryFromIntError>".into(),
                payload_owner: "core::result::Result<usize,TryFromIntError>::Ok".into(),
                payload_ty: ValueType::Unsigned,
                payload_on_disc_true: false,
                niche: false,
            }],
        );
        assert_eq!(rewritten, 1, "the Result unwrap site must be rewritten");

        let reads_pos0 = |bb: usize| {
            g.blocks[bb].operations.iter().any(
                |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0"),
            )
        };
        for link in &g.blocks[a.0].exits {
            match &link.exitcase {
                Some(ExitCase::Bool(false)) => {
                    assert!(reads_pos0(link.target.0), "Ok arm reads the payload");
                    assert!(
                        g.blocks[link.target.0]
                            .exits
                            .iter()
                            .any(|arm| arm.target == b),
                        "Ok arm reaches the continuation"
                    );
                }
                Some(ExitCase::Bool(true)) => assert!(
                    g.blocks[link.target.0]
                        .exits
                        .iter()
                        .any(|arm| arm.target == g.exceptblock),
                    "Err arm raises"
                ),
                other => panic!("unexpected branch exitcase {other:?}"),
            }
        }
    }

    /// A producer op that is not a 1-arg call declines (fail-safe).
    #[test]
    fn rewrite_declines_when_producer_not_unary_call() {
        let mut g = FunctionGraph::new("test_option_unwrap_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let extra = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: unwrap_target(),
                    args: vec![opt, extra],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(a, None);

        let rewritten = rewire_unwrap_call_sites(
            &mut g,
            &[UnwrapSite {
                result_var: result,
                enum_owner: "core::option::Option".to_string(),
                payload_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
                payload_on_disc_true: true,
                niche: false,
            }],
        );
        assert_eq!(rewritten, 0, "a non-unary producer declines");
    }
}
