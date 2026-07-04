//! `Option::unwrap(opt)` → discriminant guard + payload extraction.
//!
//! ## Positioning
//!
//! `core::option::<Impl>::unwrap` is a foreign combinator whose body is Opaque
//! in the LLBC (Charon cannot extract `core`), so the caller emits a residual
//! `unwrap` call — an unregistered callee the rtyper census Skips.  Like
//! [`crate::front::option_unwrap_or`] the combinator's match lives inside the
//! opaque body: at the call site there is only `result = unwrap(opt)` flowing
//! on.  This pass creates the guard the combinator's semantics imply:
//!
//! ```text
//!     result = opt.unwrap()               // residual `unwrap` call
//! becomes
//!     if opt.__discriminant == 1 { result = opt.__pos_0 } else { <panic> }
//! ```
//!
//! Unlike `unwrap_or` there is no default: `Option::unwrap` panics on `None`.
//! The `None` arm is therefore not a value path but an implicit-`AssertionError`
//! raise ([`FunctionGraph::set_raise_implicit`]) — the "shouldn't occur at
//! run-time" shape [`crate::model::remove_assertion_errors`] prunes, matching
//! `unwrap`'s never-`None` contract.  The `Some` arm reads `opt.__pos_0` exactly
//! as `unwrap_or`'s does.  `Option`'s tags are `None = 0` / `Some = 1`, so
//! branching on `bool(disc)` selects the `Some` arm.
//!
//! ## The rewrite (`rewire_one_unwrap_site`)
//!
//! Block A holds the residual `unwrap` call producing `result` as its last op,
//! closed by `lower_call` with a single forwarding exit to block B (the
//! continuation consuming `result`).  The rewrite:
//! 1. drops the `unwrap` call, reads `disc = opt.__discriminant`, and closes A
//!    with a `bool(disc)` branch to two fresh arms;
//! 2. the `then_bb` (`Some`) arm reads `opt.__pos_0` and forwards it to B as
//!    the `result` slot, threading every other live value through;
//! 3. the `else_bb` (`None`) arm raises the implicit `AssertionError` (no edge
//!    to B — the `None` path never produces a `result`).
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `unwrap` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `Option::unwrap(opt)` call site captured during body lowering
/// (`front::mir` `recognize_unwrap_site`).  The owner strings are resolved at
/// the recording site where the receiver `Option` type is in hand; the
/// post-pass only needs them to spell the `__discriminant` / `__pos_0` field
/// reads in the synthesized `Some` arm.
#[derive(Clone)]
pub(crate) struct UnwrapSite {
    /// The `unwrap` call result (the payload `T` value) — locates block A.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub some_owner: String,
    /// The `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `Some::__pos_0` field kind and the extracted result kind.
    pub payload_ty: ValueType,
}

/// Rewrite every recorded `Option::unwrap` call site into the discriminant
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
    // Capture the receiver `Option` operand (the sole argument).
    let opt = match &graph.blocks[a].operations[call_idx].kind {
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
    // payload itself; each must be threaded through the `Some` arm to reach B.
    // The `None` arm raises and never reaches B, so it carries nothing.
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

    // `then_bb` (`Some`) carries `carried` plus `opt` (the base for the
    // `__pos_0` read); `else_bb` (`None`) has no inputs — it raises.  The
    // source-var lists double as the branch link args.
    let mut then_sources = carried.clone();
    if !then_sources.contains(&opt) {
        then_sources.push(opt.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, _else_inputs) = graph.create_block_with_arg_vars(0);

    // `then_bb`: payload = opt.__pos_0.
    let opt_in_then = map_source(&then_sources, &then_inputs, &opt)
        .ok_or_else(|| format!("{name}: Option value not threaded into Some arm"))?;
    let payload = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: opt_in_then,
            field: FieldDescriptor {
                name: "__pos_0".to_string(),
                owner_root: Some(site.some_owner.clone()),
                owner_id: None,
            },
            ty: site.payload_ty.clone(),
            pure: true,
        },
    });
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &payload,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    // `else_bb` (`None`): raise the implicit `AssertionError` — `unwrap` panics
    // on `None`, and the never-`None` contract makes this the "shouldn't occur"
    // shape `remove_assertion_errors` prunes.
    graph.set_raise_implicit(else_bb, "Option::unwrap on None value");

    // A: drop the residual `unwrap` call, read the discriminant, branch on it.
    // `Option` tags None=0 / Some=1, so `bool(disc)` selects the `Some` (then)
    // arm.  The receiver construction ops stay as A's tail.
    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.remove(call_idx);
    let disc = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(disc.clone()),
        kind: OpKind::FieldRead {
            base: opt.clone(),
            field: FieldDescriptor {
                name: "__discriminant".to_string(),
                owner_root: Some(site.option_owner.clone()),
                owner_id: None,
            },
            ty: ValueType::Int,
            pure: true,
        },
    });
    graph.set_branch(a_id, disc, then_bb, then_sources, else_bb, Vec::new());
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
                option_owner: "core::option::Option".to_string(),
                some_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
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
                option_owner: "core::option::Option".to_string(),
                some_owner: "core::option::Option::Some".to_string(),
                payload_ty: ValueType::Int,
            }],
        );
        assert_eq!(rewritten, 0, "a non-unary producer declines");
    }
}
