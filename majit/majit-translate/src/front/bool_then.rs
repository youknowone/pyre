//! `bool::then(cond, closure)` → short-circuit `Option` diamond.
//!
//! ## Positioning
//!
//! `core::bool::<Impl>::then` is a foreign combinator whose body is
//! Opaque in the LLBC (Charon cannot extract core), so the caller emits a
//! residual `then` call — an unregistered callee the rtyper census Skips.
//! Unlike the `Result` `?` and iterator `next` diamonds, which *rewrite* a
//! Charon-emitted `Option`/`ControlFlow` match into the graph's native
//! exception shape, `then` has no diamond in the source MIR at all: the
//! closure is built, `then` is called, and the `Option` result flows on.
//! This pass *creates* the short-circuit diamond the combinator's
//! semantics imply:
//!
//! ```text
//!     opt = cond.then(|| body)          // residual `then` call
//! becomes
//!     if cond { opt = Some(closure()) } else { opt = None }
//! ```
//!
//! The branch is mandatory — the closure must not run when `cond` is
//! false (`pyframe.rs` `(!self.debugdata.is_null()).then(|| unsafe {
//! &*self.debugdata })` would deref null on the false arm), so a
//! single-block always-compute-payload encoding is unsound.  The closure
//! body reaches the graph as the closure type's transparent `call_once`
//! inherent method (the spike confirmed Charon extracts it with a body);
//! the `then` arm calls it directly.
//!
//! ## The rewrite (`rewire_one_bool_then_site`)
//!
//! Block A holds the residual `then` call producing `opt` as its last op,
//! closed by `lower_call` with a single forwarding exit to block B (the
//! continuation consuming `opt`).  The rewrite:
//! 1. drops the `then` call and closes A with a `bool(cond)` branch to two
//!    fresh arms;
//! 2. the `then_bb` arm calls the closure's `call_once` and wraps the
//!    result in `Some` (the `__discriminant = 1` / `__pos_0 = payload`
//!    aggregate the front aggregate path emits);
//! 3. the `else_bb` arm builds `None` (`__discriminant = 0`);
//! 4. both arms forward to B, reproducing A's original exit args with the
//!    `opt` slot sourced from the arm's `Some`/`None` value and every
//!    other live value threaded through the arm's inputargs.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `then` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
    ValueType,
};

/// A recognized `bool::then(cond, closure_env)` call site captured during
/// body lowering (`front::mir` `recognize_bool_then_site`).  The owner
/// strings are resolved at the recording site where the destination
/// `Option` type and the closure env type are in hand; the post-pass only
/// needs them to spell the ctor / method targets in the synthesized arms.
#[derive(Clone)]
pub(crate) struct BoolThenSite {
    /// The `then` call result (the `Option<T>` value) — locates block A.
    pub result_var: Variable,
    /// The closure env ADT `name_path` — the inherent-method owner for the
    /// `call_once` call the `then` arm emits.
    pub call_once_owner: String,
    /// The `Option` enum root `name_path` — the ctor owner for the
    /// `Some`/`None` aggregates.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload
    /// field owner (matching the variant-qualified `resolve_adt_field`
    /// read owner).
    pub some_owner: String,
    /// The `Option`'s payload `T` projected to a [`ValueType`] — the
    /// `call_once` result kind and the `Some::__pos_0` field kind.
    pub payload_ty: ValueType,
}

/// Rewrite every recorded `bool::then` call site into the short-circuit
/// `Option` diamond.  Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip), so a mismatch never
/// regresses a graph the legacy walker already handled.  Returns the
/// number of sites rewritten.
pub(crate) fn rewire_bool_then_call_sites(
    graph: &mut FunctionGraph,
    sites: &[BoolThenSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_bool_then_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `then` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_bool_then_site(graph: &mut FunctionGraph, site: &BoolThenSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `then` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: bool::then result var has no producer block"))?;

    // The call must be A's last op (lower_call closes the block right
    // after pushing it) so removing it leaves the closure-env construction
    // ops as the block tail.
    let call_idx = graph.blocks[a].operations.len() - 1;
    if graph.blocks[a].operations[call_idx].result.as_ref() != Some(&site.result_var) {
        return Err(format!(
            "{name}: bool::then call is not the last op of block {a}"
        ));
    }
    // Capture the condition + closure env operands.
    let (cond, env) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: bool::then producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // A's single exit → B (the continuation consuming the Option).  Must be
    // a plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: bool::then call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: bool::then call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // Option itself; each must be threaded through the diamond arms to
    // reach B (a fresh block cannot see A-scope Variables directly).
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

    // `then_bb` carries `carried` plus `env` (the receiver for `call_once`;
    // it may already be among the carried set); `else_bb` carries only
    // `carried`.  The source-var lists double as the branch link args.
    let mut then_sources = carried.clone();
    if !then_sources.contains(&env) {
        then_sources.push(env.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(carried.len());

    // `then_bb`: payload = call_once(env, ()); opt = Some(payload).
    let env_in_then = map_source(&then_sources, &then_inputs, &env)
        .ok_or_else(|| format!("{name}: closure env not threaded into then arm"))?;
    // The closure's `Args` tuple; `then`'s closure is niladic, so the body
    // ignores it — a synthetic empty tuple satisfies the `call_once`
    // arity.
    let unit = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(unit.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple"),
            args: Vec::new(),
            result_ty: ValueType::Ref(None),
        },
    });
    let payload = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("call_once", Some(site.call_once_owner.clone())),
            args: vec![env_in_then, unit],
            result_ty: site.payload_ty.clone(),
        },
    });
    let some_var = emit_option_variant(
        graph,
        then_bb,
        &site.option_owner,
        1,
        Some((&site.some_owner, payload, site.payload_ty.clone())),
    );
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &some_var,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    // `else_bb`: opt = None.
    let none_var = emit_option_variant(graph, else_bb, &site.option_owner, 0, None);
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &none_var,
        &carried,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    // A: drop the residual `then` call, branch on `cond`.  `set_branch`
    // appends the `bool(cond)` hop and installs the Bool(false)/Bool(true)
    // arm links; the closure-env construction ops stay as A's tail.
    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.remove(call_idx);
    graph.set_branch(a_id, cond, then_bb, then_sources, else_bb, carried);
    Ok(())
}

/// The arm inputarg `v` binds to, by position in the arm's source list.
pub(crate) fn map_source(
    sources: &[Variable],
    inputs: &[Variable],
    v: &Variable,
) -> Option<Variable> {
    sources
        .iter()
        .position(|s| s == v)
        .map(|i| inputs[i].clone())
}

/// Reproduce block A's original exit args for a diamond arm: the `opt`
/// slot is sourced from the arm's `Some`/`None` value, every other live
/// Value is re-sourced from the arm's threaded inputarg, and constants
/// pass through.  `Err` if a forwarded Value was not threaded into the arm
/// (an unexpected live set) — the caller declines and keeps the residual.
pub(crate) fn reproduce_exit_args(
    saved: &Link,
    result_var: &Variable,
    option_val: &Variable,
    sources: &[Variable],
    inputs: &[Variable],
    name: &str,
) -> Result<Vec<LinkArg>, String> {
    let mut out = Vec::with_capacity(saved.args.len());
    for arg in &saved.args {
        match arg {
            LinkArg::Const(c) => out.push(LinkArg::Const(c.clone())),
            LinkArg::Value(v) if v == result_var => out.push(LinkArg::Value(option_val.clone())),
            LinkArg::Value(v) => {
                let mapped = map_source(sources, inputs, v).ok_or_else(|| {
                    format!("{name}: exit arg not threaded into bool::then diamond arm")
                })?;
                out.push(LinkArg::Value(mapped));
            }
        }
    }
    Ok(out)
}

/// Build an `Option` variant aggregate in `block` and return its value —
/// the enum-root ctor + `__discriminant` write (+ `__pos_0` payload write
/// for `Some`), the same transparent-ctor + `FieldWrite` chain
/// `Rvalue::Aggregate` emits (`front::mir` `emit_tagged_pair_aggregate`).
/// `disc` is the variant tag (`Some` = 1, `None` = 0); `payload` is
/// `Some((some_owner, value, value_ty))` for `Some`, `None` for `None`.
fn emit_option_variant(
    graph: &mut FunctionGraph,
    block: BlockId,
    option_owner: &str,
    disc: i64,
    payload: Option<(&str, Variable, ValueType)>,
) -> Variable {
    let mut owner_path: Vec<String> = option_owner.split("::").map(str::to_string).collect();
    let ctor_name = owner_path.pop().unwrap_or_default();
    let ctor_target = if owner_path.is_empty() {
        CallTarget::synthetic_transparent_ctor(ctor_name)
    } else {
        CallTarget::synthetic_transparent_ctor_with_owner(owner_path, ctor_name)
    };
    let res = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::Call {
            target: ctor_target,
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(option_owner.to_string())),
        },
    });
    // `__discriminant` keys the enum root (tag offset 0 of every variant);
    // materialize the tag as a `ConstInt` value, matching the aggregate
    // path's `FieldWrite { value: Value(..) }` shape.
    let disc_var = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(disc_var.clone()),
        kind: OpKind::ConstInt(disc),
    });
    graph.block_mut(block).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: res.clone(),
            field: FieldDescriptor {
                name: "__discriminant".to_string(),
                owner_root: Some(option_owner.to_string()),
                owner_id: None,
            },
            value: LinkArg::Value(disc_var),
            ty: ValueType::Int,
        },
    });
    // `__pos_0` keys the `Some` variant so its offset matches the
    // variant-qualified read owner.
    if let Some((some_owner, value, value_ty)) = payload {
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: res.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(some_owner.to_string()),
                    owner_id: None,
                },
                value: LinkArg::Value(value),
                ty: value_ty,
            },
        });
    }
    res
}

/// Close `block` with a single plain-goto exit carrying mixed
/// Value/Const args.  `set_goto` accepts only `Variable` args; the diamond
/// arms forward A's original exit args, which may include constants, so go
/// through `Link::new_mixed` + `set_control_flow_metadata`.  Arity matches
/// B's inputargs because the args are derived from A's original B-bound
/// exit.
pub(crate) fn close_goto_mixed(
    graph: &mut FunctionGraph,
    block: BlockId,
    target: BlockId,
    args: Vec<LinkArg>,
) {
    let link = Link::new_mixed(args, target, None);
    graph.set_control_flow_metadata(block, None, vec![link]);
}
