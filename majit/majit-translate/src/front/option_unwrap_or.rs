//! `Option::unwrap_or(opt, default)` → discriminant value-select.
//!
//! ## Positioning
//!
//! `core::option::<Impl>::unwrap_or` is a foreign combinator whose body is
//! Opaque in the LLBC (Charon cannot extract `core`), so the caller emits a
//! residual `unwrap_or` call — an unregistered callee the rtyper census
//! Skips.  Like [`crate::front::bool_then`] and unlike
//! [`crate::front::checked_arith`], the combinator's match lives inside the
//! opaque body: at the call site there is no discriminant switch to rewrite,
//! only `result = unwrap_or(opt, default)` flowing on.  This pass *creates*
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
//! (the `Some` arm reads `opt.__pos_0`, the `None` arm forwards `default`).
//! `Option`'s tags are `None = 0` / `Some = 1`, and the front models every
//! `Option<T>` uniformly with an explicit `__discriminant` field (Rust niche
//! optimisation is a codegen detail below the IR), so branching on the
//! discriminant read as `bool(disc)` selects the `Some` arm exactly.
//!
//! ## The rewrite (`rewire_one_unwrap_or_site`)
//!
//! Block A holds the residual `unwrap_or` call producing `result` as its
//! last op, closed by `lower_call` with a single forwarding exit to block B
//! (the continuation consuming `result`).  The rewrite:
//! 1. drops the `unwrap_or` call, reads `disc = opt.__discriminant`, and
//!    closes A with a `bool(disc)` branch to two fresh arms;
//! 2. the `then_bb` (`Some`) arm reads `opt.__pos_0` as the payload;
//! 3. the `else_bb` (`None`) arm forwards `default`;
//! 4. both arms forward to B, reproducing A's original exit args with the
//!    `result` slot sourced from the arm's payload / default value and every
//!    other live value threaded through the arm's inputargs.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `unwrap_or`
//! callee keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `Option::unwrap_or(opt, default)` call site captured during
/// body lowering (`front::mir` `recognize_unwrap_or_site`).  The owner
/// strings are resolved at the recording site where the receiver `Option`
/// type is in hand; the post-pass only needs them to spell the
/// `__discriminant` / `__pos_0` field reads in the synthesized arms.
#[derive(Clone)]
pub(crate) struct UnwrapOrSite {
    /// The `unwrap_or` call result (the payload `T` value) — locates block A.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub some_owner: String,
    /// The `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `Some::__pos_0` field kind and the select result kind.
    pub payload_ty: ValueType,
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

    // The call must be A's last op (lower_call closes the block right after
    // pushing it) so removing it leaves the receiver/default construction as
    // the block tail.
    let call_idx = graph.blocks[a].operations.len() - 1;
    if graph.blocks[a].operations[call_idx].result.as_ref() != Some(&site.result_var) {
        return Err(format!(
            "{name}: unwrap_or call is not the last op of block {a}"
        ));
    }
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
    // payload itself; each must be threaded through the diamond arms to reach
    // B (a fresh block cannot see A-scope Variables directly).
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
    // `__pos_0` read); `else_bb` (`None`) carries `carried` plus `default`
    // (the forwarded fallback).  The source-var lists double as the branch
    // link args.
    let mut then_sources = carried.clone();
    if !then_sources.contains(&opt) {
        then_sources.push(opt.clone());
    }
    let mut else_sources = carried.clone();
    if !else_sources.contains(&default) {
        else_sources.push(default.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

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

    // `else_bb`: forward `default` as the result.
    let default_in_else = map_source(&else_sources, &else_inputs, &default)
        .ok_or_else(|| format!("{name}: default value not threaded into None arm"))?;
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &default_in_else,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    // A: drop the residual `unwrap_or` call, read the discriminant, branch on
    // it.  `set_branch` appends the `bool(disc)` hop and installs the
    // Bool(false)/Bool(true) arm links; `Option` tags None=0 / Some=1, so
    // `bool(disc)` selects the `Some` (then) arm.  The receiver/default
    // construction ops stay as A's tail.
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
    graph.set_branch(a_id, disc, then_bb, then_sources, else_bb, else_sources);
    Ok(())
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
            option_owner: "core::option::Option".into(),
            some_owner: "core::option::Option::Some".into(),
            payload_ty: ValueType::Int,
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
