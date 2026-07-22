//! `Option::map` / `Option::and_then` / `Option::unwrap_or_else` →
//! discriminant closure-select.
//!
//! ## Positioning
//!
//! These three `core::option::<Impl>` combinators are foreign (Opaque body in
//! the LLBC) with the `Option` ADT receiver, so `first_is_self` routes each to
//! a two-arg `CallTarget::Method` (receiver in `args[0]`, closure env in
//! `args[1]`).  Like [`crate::front::option_map_or`] they carry a single closure
//! and imply a two-way select on the receiver's tag, which this pass *creates*.
//! They differ only in how each arm produces its result:
//!
//! ```text
//!     map(opt, f):            Some(x) => Some(f(x))   None => None
//!     and_then(opt, f):       Some(x) => f(x)         None => None
//!     unwrap_or_else(opt, f): Some(x) => x            None => f()
//!     or_else(opt, f):        Some(x) => Some(x)      None => f()
//! ```
//!
//! `map`/`and_then` run the closure on the `Some` payload; `unwrap_or_else` and
//! `or_else` run their niladic closure on the `None` arm instead.  On `Some`,
//! `unwrap_or_else` forwards the payload directly while `or_else` forwards the
//! whole receiver `Option` unchanged (no `__pos_0` read).  `map` wraps the
//! closure result back into `Some`, and both `map`/`and_then` build a fresh
//! `None` on the empty arm; `unwrap_or_else` returns a bare `T` and `or_else`
//! an `Option<T>`.  Everything else — locating the residual call, absorbing
//! the trailing `*mut <registered ADT>` narrowing cast, threading the live
//! values through the arms, and closing the `bool(disc)` branch — is the
//! [`crate::front::option_map_or`] skeleton.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller leaves
//! the residual call untouched, and the unregistered callee keeps the rtyper
//! census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::front::option_map_or::emit_narrow;
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// Which `Option` closure combinator a recorded site is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ClosureCombinator {
    /// `Some(x) => Some(f(x))`, `None => None`.
    Map,
    /// `Some(x) => f(x)`, `None => None`.
    AndThen,
    /// `Some(x) => x`, `None => f()`.
    UnwrapOrElse,
    /// `Some(x) => Some(x)`, `None => f()` where `f() -> Option<T>`.
    OrElse,
}

/// A recognized `Option::map`/`and_then`/`unwrap_or_else(opt, closure_env)`
/// call site captured during body lowering (`front::mir`
/// `recognize_closure_select_site`).  The owner strings are resolved at the
/// recording site where the receiver `Option` and the closure env type are in
/// hand; the post-pass only needs them to spell the field reads, the closure
/// `call_once`, and any `Some`/`None` it builds.
#[derive(Clone)]
pub(crate) struct ClosureSelectSite {
    /// Which combinator — selects the per-arm result construction.
    pub kind: ClosureCombinator,
    /// The combinator call result — locates block A.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner
    /// (both the receiver read and any built `Some`/`None`).
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (the receiver read and any built `Some`).
    pub some_owner: String,
    /// The closure env ADT `name_path` — the `call_once` inherent-method owner.
    pub call_once_owner: String,
    /// The receiver `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `Some::__pos_0` read kind and the `(x,)` args-tuple element.
    pub payload_ty: ValueType,
    /// The type the closure's `call_once` returns, projected to a
    /// [`ValueType`]: `U` for `map`, `Option<U>` for `and_then`, `T` for
    /// `unwrap_or_else`, `Option<T>` for `or_else`.
    pub call_result_ty: ValueType,
    /// The `<X>` suffix for the closure `Args` tuple `(payload,)` — the same
    /// `Tuple<X>` leaf the extracted `call_once` reads its `.0` under, derived
    /// from the receiver `Option<X>`'s payload node so the synthesized write
    /// and the `resolve_place` read key under one classdef.  "" (bare `Tuple`)
    /// for a unit payload.
    pub args_tuple_suffix: String,
}

/// Rewrite every recorded closure-select call site into the discriminant
/// diamond.  Fail-safe: a site whose block does not fit the residual-call shape
/// is left untouched (Skip).  Returns the number of sites rewritten.
pub(crate) fn rewire_closure_select_call_sites(
    graph: &mut FunctionGraph,
    sites: &[ClosureSelectSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        if rewire_one_closure_select_site(graph, site).is_ok() {
            rewritten += 1;
        }
    }
    rewritten
}

fn rewire_one_closure_select_site(
    graph: &mut FunctionGraph,
    site: &ClosureSelectSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: closure-select result var has no producer block"))?;
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: closure-select call op not found in block {a}"))?;
    let ops_len = graph.blocks[a].operations.len();

    // Absorb the optional trailing `__pyre_cast_instance` narrowing cast a
    // `*mut <registered ADT>` result gains (see `option_map_or`).
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
                    "{name}: closure-select call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: closure-select call is not the last op of block {a}"
        ));
    };

    // Capture the receiver `Option` + closure env operands.
    let (opt, env) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: closure-select producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B.  Must be a plain goto.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: closure-select call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: closure-select call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // Distinct live Values A forwards to B other than the result.
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // `map`/`and_then` run the closure on the `Some` arm (so need `env` there),
    // while `unwrap_or_else`/`or_else` run their closure on the `None` arm
    // instead.  The `Some` arm always needs `opt` — to read `opt.__pos_0`
    // (`map`/`and_then`/`unwrap_or_else`) or to forward the receiver unchanged
    // (`or_else`).  Thread each arm exactly the sources it consumes.
    let closure_on_some = !matches!(
        site.kind,
        ClosureCombinator::UnwrapOrElse | ClosureCombinator::OrElse
    );
    let mut then_sources = carried.clone();
    if !then_sources.contains(&opt) {
        then_sources.push(opt.clone());
    }
    if closure_on_some && !then_sources.contains(&env) {
        then_sources.push(env.clone());
    }
    let mut else_sources = carried.clone();
    if !closure_on_some && !else_sources.contains(&env) {
        else_sources.push(env.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

    // --- `then_bb` (`Some`) ---
    let opt_in_then = map_source(&then_sources, &then_inputs, &opt)
        .ok_or_else(|| format!("{name}: Option value not threaded into Some arm"))?;
    let then_value = match site.kind {
        // `or_else` forwards the whole receiver `Option` unchanged — no `__pos_0`.
        ClosureCombinator::OrElse => opt_in_then,
        // `unwrap_or_else` forwards the bare payload.
        ClosureCombinator::UnwrapOrElse => read_some_payload(graph, then_bb, opt_in_then, site),
        // `map`/`and_then` run the closure on the payload.
        ClosureCombinator::Map | ClosureCombinator::AndThen => {
            let payload = read_some_payload(graph, then_bb, opt_in_then, site);
            let env_in_then = map_source(&then_sources, &then_inputs, &env)
                .ok_or_else(|| format!("{name}: closure env not threaded into Some arm"))?;
            let call_result = emit_call_once(
                graph,
                then_bb,
                env_in_then,
                Some((payload, site.payload_ty.clone())),
                &site.call_once_owner,
                site.call_result_ty.clone(),
                &site.args_tuple_suffix,
            );
            match site.kind {
                // `map` wraps the closure result back into `Some(U)`.
                ClosureCombinator::Map => emit_option_variant(
                    graph,
                    then_bb,
                    &site.option_owner,
                    1,
                    Some((&site.some_owner, call_result, site.call_result_ty.clone())),
                ),
                // `and_then`'s closure already returns `Option<U>`.
                _ => call_result,
            }
        }
    };
    let then_result = emit_narrow(graph, then_bb, then_value, &narrow_root);
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &then_result,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    // --- `else_bb` (`None`) ---
    let else_value = match site.kind {
        // `map`/`and_then` build a fresh `None`.
        ClosureCombinator::Map | ClosureCombinator::AndThen => {
            emit_option_variant(graph, else_bb, &site.option_owner, 0, None)
        }
        // `unwrap_or_else`/`or_else` run their niladic closure and forward its
        // result (`T` for `unwrap_or_else`, `Option<T>` for `or_else`).
        ClosureCombinator::UnwrapOrElse | ClosureCombinator::OrElse => {
            let env_in_else = map_source(&else_sources, &else_inputs, &env)
                .ok_or_else(|| format!("{name}: closure env not threaded into None arm"))?;
            emit_call_once(
                graph,
                else_bb,
                env_in_else,
                None,
                &site.call_once_owner,
                site.call_result_ty.clone(),
                &site.args_tuple_suffix,
            )
        }
    };
    let else_result = emit_narrow(graph, else_bb, else_value, &narrow_root);
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &else_result,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    // A: drop the residual call (+ absorbed cast), read the discriminant, branch
    // on `bool(disc)` (Option tags None=0 / Some=1 → the Some arm is `then`).
    let a_id = graph.blocks[a].id;
    for _ in ci..=remove_upto {
        graph.blocks[a].operations.remove(ci);
    }
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

/// Read `opt.__pos_0` (the `Some` payload) in `block`, returning the payload
/// value.  Used by every combinator that consumes the payload on the `Some`
/// arm (`map`/`and_then`/`unwrap_or_else`); `or_else` forwards the whole
/// receiver instead and skips this.
fn read_some_payload(
    graph: &mut FunctionGraph,
    block: BlockId,
    opt: Variable,
    site: &ClosureSelectSite,
) -> Variable {
    let payload = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: opt,
            field: FieldDescriptor {
                name: "__pos_0".to_string(),
                owner_root: Some(site.some_owner.clone()),
                owner_id: None,
            },
            ty: site.payload_ty.clone(),
            pure: true,
        },
    });
    payload
}

/// Emit `call_once(env, args)` in `block`, returning the call result.  `arg` is
/// the single closure argument (a `(x,)` tuple) or `None` for a niladic closure
/// (an empty tuple the opaque body ignores) — the same `Args`-tuple shape
/// `Rvalue::Aggregate` emits.
fn emit_call_once(
    graph: &mut FunctionGraph,
    block: BlockId,
    env: Variable,
    arg: Option<(Variable, ValueType)>,
    call_once_owner: &str,
    result_ty: ValueType,
    args_tuple_suffix: &str,
) -> Variable {
    // The `(payload,)` Args tuple keys under the per-shape `Tuple<X>` leaf the
    // extracted `call_once` reads `.0` from at `resolve_place`; a niladic
    // closure's `()` tuple has no `.0` and stays bare `Tuple` (empty types →
    // "" on both sides), so the suffix applies only when an argument is written.
    let tuple_owner = if arg.is_some() {
        format!("Tuple{args_tuple_suffix}")
    } else {
        "Tuple".to_string()
    };
    let args_tuple = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(args_tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor(&tuple_owner),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(tuple_owner.clone())),
        },
    });
    if let Some((value, _value_ty)) = arg {
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: args_tuple.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(tuple_owner.clone()),
                    owner_id: None,
                },
                value: LinkArg::Value(value),
                ty: ValueType::Ref(None),
            },
        });
    }
    let call_result = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(call_result.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("call_once", Some(call_once_owner.to_string())),
            args: vec![env, args_tuple],
            result_ty,
        },
    });
    call_result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(kind: ClosureCombinator, result_var: Variable) -> ClosureSelectSite {
        ClosureSelectSite {
            kind,
            result_var,
            option_owner: "core::option::Option".into(),
            some_owner: "core::option::Option::Some".into(),
            call_once_owner: "test::closure".into(),
            payload_ty: ValueType::Int,
            call_result_ty: ValueType::Int,
            args_tuple_suffix: String::new(),
        }
    }

    /// Build `result = <combinator>(opt, env)` — block A = the residual call
    /// closed by a single goto to B — and rewrite it.
    fn build_and_rewrite(kind: ClosureCombinator, method: &str) -> (FunctionGraph, usize) {
        let mut g = FunctionGraph::new("test_closure_select");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method(method, Some("core::option::Option".into())),
                    args: vec![opt, env],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);
        let rewritten = rewire_closure_select_call_sites(&mut g, &[site(kind, result)]);
        assert_eq!(rewritten, 1, "the {method} site must be rewritten");
        (g, a.0)
    }

    fn count_calls(g: &FunctionGraph, pred: impl Fn(&CallTarget) -> bool) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::Call { target, .. } if pred(target)))
            .count()
    }

    fn residual_gone(g: &FunctionGraph, method: &str) -> bool {
        count_calls(
            g,
            |t| matches!(t, CallTarget::Method { name, .. } if name == method),
        ) == 0
    }

    #[test]
    fn map_selects_some_call_wrapped_and_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::Map, "map");
        assert!(residual_gone(&g, "map"), "residual map call removed");
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the Some arm calls the closure once"
        );
        assert_eq!(g.blocks[a].exits.len(), 2, "A branches to Some/None arms");
        // Two `Option` ctors: Some(f(x)) in the then arm, None in the else arm.
        let ctors = count_calls(
            &g,
            |t| matches!(t, CallTarget::SyntheticTransparentCtor { name, .. } if name == "Option"),
        );
        assert_eq!(ctors, 2, "map builds Some(U) and None");
    }

    #[test]
    fn and_then_forwards_some_call_and_builds_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::AndThen, "and_then");
        assert!(
            residual_gone(&g, "and_then"),
            "residual and_then call removed"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the Some arm calls the closure once"
        );
        // Only the None arm builds an Option; the Some arm forwards the call.
        let ctors = count_calls(
            &g,
            |t| matches!(t, CallTarget::SyntheticTransparentCtor { name, .. } if name == "Option"),
        );
        assert_eq!(ctors, 1, "and_then builds only None");
        assert_eq!(g.blocks[a].exits.len(), 2);
    }

    #[test]
    fn unwrap_or_else_forwards_payload_and_calls_on_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::UnwrapOrElse, "unwrap_or_else");
        assert!(
            residual_gone(&g, "unwrap_or_else"),
            "residual unwrap_or_else call removed"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the None arm calls the niladic closure once"
        );
        // No Option is built — result is a bare T.
        let ctors = count_calls(
            &g,
            |t| matches!(t, CallTarget::SyntheticTransparentCtor { name, .. } if name == "Option"),
        );
        assert_eq!(ctors, 0, "unwrap_or_else returns a bare value");
        assert_eq!(g.blocks[a].exits.len(), 2);
    }

    fn count_field_reads(g: &FunctionGraph, field_name: &str) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == field_name))
            .count()
    }

    #[test]
    fn or_else_forwards_receiver_and_calls_on_none() {
        let (g, a) = build_and_rewrite(ClosureCombinator::OrElse, "or_else");
        assert!(
            residual_gone(&g, "or_else"),
            "residual or_else call removed"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_once")
            ),
            1,
            "the None arm calls the niladic closure once"
        );
        // The Some arm forwards the WHOLE receiver Option — it must not read the
        // `Some` payload (`__pos_0`); only the branch discriminant is read.
        assert_eq!(
            count_field_reads(&g, "__pos_0"),
            0,
            "or_else forwards the receiver, never reading the Some payload"
        );
        assert_eq!(
            count_field_reads(&g, "__discriminant"),
            1,
            "only the branch discriminant is read"
        );
        // Neither arm builds an Option: Some forwards the receiver, None returns
        // the closure's own `Option<T>`.
        let ctors = count_calls(
            &g,
            |t| matches!(t, CallTarget::SyntheticTransparentCtor { name, .. } if name == "Option"),
        );
        assert_eq!(ctors, 0, "or_else builds no Option");
        assert_eq!(g.blocks[a].exits.len(), 2, "A branches to Some/None arms");
    }

    #[test]
    fn declines_when_call_not_last_op() {
        let mut g = FunctionGraph::new("test_closure_select_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let env = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::method("map", Some("core::option::Option".into())),
                    args: vec![opt, env],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.push_op_var(a, OpKind::ConstInt(9), true).unwrap();
        g.set_return(a, None);
        let rewritten =
            rewire_closure_select_call_sites(&mut g, &[site(ClosureCombinator::Map, result)]);
        assert_eq!(rewritten, 0, "a non-last-op call declines");
        assert!(
            !residual_gone(&g, "map"),
            "residual call survives on decline"
        );
    }
}
