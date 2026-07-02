//! `Result<T, PyError>` → exception-link lowering.
//!
//! ## Positioning
//!
//! `front/mod.rs`'s charter mandates that "`?` / `PyResult` must be
//! lowered to exceptional successor edges of the existing
//! `Terminator`, matching `rpython/translator/exceptiontransform.py` +
//! `rpython/jit/codewriter/jtransform.py:rewrite_op_direct_call`".
//! This module is that lowering.  RPython's exception transformer is
//! the same bridge run in the opposite direction (exception links →
//! value encodings for the C backend); Rust source arrives
//! value-encoded, so pyre runs the inverse: the value-encoded
//! `Result` idiom becomes the graph's native exception representation
//! (`ExitSwitch::LastException` exits + `exceptblock` links), the same
//! way `simplify.py:transform_ovfcheck` converts the value-encoded
//! `ovfcheck()` idiom into an op with an implicit exception link.
//!
//! The residual-call ABI already performs this erasure at every host
//! boundary (`pyre-interpreter/src/opcode_ops.rs:265
//! bh_execute_store_subscr`: `Ok` → value, `Err` →
//! `BH_LAST_EXC_VALUE`), so jitcode-inlined graphs were the only
//! consumers still seeing `Result` shells — built by niladic
//! `SyntheticTransparentCtor` residuals that can never execute (a
//! synthetic ctor has no host symbol) and switched on a
//! `__discriminant` field read the walker cannot make concrete.
//!
//! ## The two rules
//!
//! - **Callee rule** ([`lower_result_exc_returns`]): a scoped graph
//!   whose declared return is `Result<T, PyError>` stops building
//!   `Ok`/`Err` shells.  `return Ok(v)` links `returnblock` with `v`;
//!   `return Err(e)` materialises the runtime exception object
//!   (`PyError::to_exc_object` — the trace-level exception value
//!   domain is the `W_BaseException` ref, the same value
//!   `BH_LAST_EXC_VALUE` carries) and closes the block towards
//!   `exceptblock` with `(op.type(exc), exc)`, exactly the
//!   `lower_exc_from_raise` tail shape (`flowcontext.py:600`).
//!
//! - **Caller rule** ([`rewire_result_exc_call_sites`]): a `?` on a
//!   call to a scoped callee lowers in MIR as a
//!   `Try::branch`-diamond — `cf = branch(r)` →
//!   `switch(cf.__discriminant)` → `{0: continue with cf.__pos_0,
//!   1: from_residual(cf.__pos_0) → return}`.  The rewrite gives the
//!   call block `ExitSwitch::LastException` with the normal exit
//!   jumping straight to the continue arm (the call result *is* `T`
//!   once the callee raises) and the exception exit propagating to
//!   `exceptblock` via the `last_exception` / `last_exc_value` link
//!   pair — RPython's default exception link (`flowspace/model.py`
//!   `Link.last_exception`), which `flatten.rs` already turns into
//!   `catch_exception` / rethrow shapes.
//!
//! ## Scope discipline
//!
//! Both rules must apply together per callee: a transformed callee
//! returns `T` and raises, so an untransformed caller-side discriminant
//! switch would read garbage.  Every `Result<T, PyError>` callee is
//! transformed uniformly (`exceptiontransform.py:212` `transform_completely`,
//! no allowlist); the callee-side type gate [`tyref_is_result_of_pyerror`]
//! is the only filter.  Every call site of such a callee either matches
//! the `?`-diamond, tail-forwards inside an enclosing transformed graph,
//! or — for hand-written `match` consumers (`eval_loop`, the
//! `eval_loop_jit*` portals whose `match step_result` merges seven
//! predecessors) — gets the [`catch_and_rewrap`] treatment: `LastException`
//! exits on the call block whose arms locally re-encode the `Result`
//! (`Ok(raw)` / `Err(PyError::from_exc_object(last_exc_value))`), leaving
//! the downstream destructuring untouched.  A call shape neither rule
//! recognises declines — the graph degrades to a residual call, no
//! miscompile.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::TyRef;

use crate::flowspace::model::Variable;
use crate::model::{
    CallFuncPtr, CallTarget, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType,
};

/// Resolve the JSON body behind a generics slot — `{"Deduplicated":
/// id}` indirections through the dedup table, `{"HashConsedValue":
/// [id, body]}` inline pairs, anything else as-is.
fn ty_json_body<'l>(v: &'l serde_json::Value, llbc: &'l Llbc) -> Option<&'l serde_json::Value> {
    if let Some(id) = v.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc.dedup_body(id);
    }
    if let Some(arr) = v
        .get("HashConsedValue")
        .and_then(serde_json::Value::as_array)
    {
        return arr.get(1);
    }
    Some(v)
}

/// `{"Adt": {"id": {"Adt": <id>}, …}}` → the TypeDecl's full name path.
fn adt_path_of(v: &serde_json::Value, llbc: &Llbc) -> Option<String> {
    let id = v.get("Adt")?.get("id")?.get("Adt")?.as_u64()?;
    Some(llbc.type_by_id(id)?.item_meta.name_path())
}

/// True when `ty` is `core::result::Result<T, E>` with `E` resolving
/// to the interpreter's `PyError` exception carrier.
pub(crate) fn tyref_is_result_of_pyerror(ty: &TyRef, llbc: &Llbc) -> bool {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    if adt_path_of(body, llbc).as_deref() != Some("core::result::Result") {
        return false;
    }
    let Some(err_slot) = body
        .get("Adt")
        .and_then(|a| a.get("generics"))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.get(1))
    else {
        return false;
    };
    let Some(err_body) = ty_json_body(err_slot, llbc) else {
        return false;
    };
    adt_path_of(err_body, llbc).is_some_and(|p| p == "pyre_interpreter::error::PyError")
}

/// True when `ty` is `core::option::Option<T>` — the return type of an
/// `Iterator::next()` call, recognised by [`crate::front::iter_next`] to
/// record the `next`-diamond rewrite site.
pub(crate) fn tyref_is_option(ty: &TyRef, llbc: &Llbc) -> bool {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    adt_path_of(body, llbc).as_deref() == Some("core::option::Option")
}

/// The per-instantiation `<…>` suffix for a scoped callee's
/// `Result<T, PyError>` return type, or `None` when the instantiation is
/// not Ref-shaped (bool/int payloads stay on the bare `Result::Ok`
/// classdef).  The suffix keys the rebuilt shell's ClassDef per
/// instantiation — `Result<StepResult,PyError>::Ok` distinct from
/// `Result<Tuple,PyError>::Ok` — matching the suffix the front aggregate
/// path (`resolve_aggregate_adt`) computes for the same instantiation, so
/// both writers agree on one ClassDef and the `__pos_0` payload no longer
/// unions across instantiations.  Both the `Ok` and `Err` shells of one
/// callee share this suffix so the two variants share one base ClassDef.
pub(crate) fn tyref_result_instantiation_suffix(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    let adt = body.get("Adt")?.as_object()?;
    crate::front::mir::adt_head_instantiation_suffix(adt, llbc)
}

/// True when `ty` is `Result<(), PyError>` — the Ok payload is the unit
/// type.  Such a callee returns void after the exception-link lowering
/// (`exceptiontransform.py` widens the value-encoded result to the inner
/// type, which is `Void` for the unit case), so its return must be
/// widened to a genuine void return rather than forwarding the unit
/// `()` value as a `Ref`-typed shell — see [`widen_unit_return_to_void`].
pub(crate) fn tyref_result_ok_is_unit(ty: &TyRef, llbc: &Llbc) -> bool {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    if adt_path_of(body, llbc).as_deref() != Some("core::result::Result") {
        return false;
    }
    let Some(ok_slot) = body
        .get("Adt")
        .and_then(|a| a.get("generics"))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.get(0))
    else {
        return false;
    };
    crate::front::mir::charon_type_value_to_ast_string(ok_slot, llbc, 0) == "()"
}

/// Collapse a scoped callee's returnblock to a genuine void return.
///
/// A `Result<(), PyError>` callee carries the unit `Ok` payload as a
/// `Ref`-typed value: the callee rule forwards the `()` aggregate (a
/// niladic transparent ctor, `front::mir` types every aggregate as
/// `Ref`), and a tail-forwarded `f(...)?` carries the inner callee's
/// `Ref` call result.  Either way the returnblock's return variable
/// colours `GcRef`, so `graph_result_kind` (and thus the call
/// descriptor's `FUNC.RESULT`) reads `r` for a function that
/// `exceptiontransform.py` would return `Void`.  Drop the return
/// variable and every arg on the exits feeding it; an empty returnblock
/// `inputargs` is the void-return shape `graph_result_kind` maps to
/// `v`.  The now-dead unit producers are swept by the `prune_dead_phis`
/// pass the codewriter runs immediately after this widen.
pub(crate) fn widen_unit_return_to_void(graph: &mut FunctionGraph) {
    let returnblock = graph.returnblock;
    for block in &mut graph.blocks {
        for link in &mut block.exits {
            if link.target == returnblock {
                link.args.clear();
            }
        }
    }
    graph.blocks[returnblock.0].inputargs.clear();
}

/// Is `target` the `Result::Ok` / `Result::Err` transparent ctor?
///
/// The owner's final segment may carry a per-instantiation `<…>` suffix
/// (`Result<Tuple,PyError>`) minted by the generic-ADT projection; strip
/// it before the bare-path compare so suffixed and bare Result ctors are
/// recognised alike.
fn result_ctor_kind(target: &CallTarget) -> Option<bool> {
    let CallTarget::SyntheticTransparentCtor { name, owner_path } = target else {
        return None;
    };
    let [head @ .., tail] = owner_path.as_slice() else {
        return None;
    };
    let tail_base = tail.split_once('<').map_or(tail.as_str(), |(b, _)| b);
    if head != ["core".to_string(), "result".to_string()] || tail_base != "Result" {
        return None;
    }
    match name.as_str() {
        "Ok" => Some(false),
        "Err" => Some(true),
        _ => None,
    }
}

/// Callee rule.  Rewrites every `Result::Ok` / `Result::Err` shell
/// construction that flows into `returnblock` into a plain value
/// return / a raise link.  Returns the number of rewritten returns.
/// `tail_forwarded_returns` counts the returns the caller rule already
/// disposed of (a `return f(...)` of another scoped callee builds no
/// shell of its own) — a body whose every return is such a forward
/// legitimately has nothing left to rewrite here.
///
/// Fail-loud on any shape outside the known construction pattern —
/// a scoped callee with an unrecognised return shape must break the
/// build, not silently keep its shell.
pub(crate) fn lower_result_exc_returns(
    graph: &mut FunctionGraph,
    tail_forwarded_returns: usize,
) -> Result<usize, String> {
    let nblocks = graph.blocks.len();
    let mut rewritten = 0usize;
    for bi in 0..nblocks {
        let block_id = crate::model::BlockId(bi);
        // Locate a Result ctor in this block.
        let mut ctor: Option<(usize, Variable, bool)> = None;
        for (i, op) in graph.blocks[bi].operations.iter().enumerate() {
            if let OpKind::Call { target, args, .. } = &op.kind
                && let Some(is_err) = result_ctor_kind(target)
            {
                if !args.is_empty() {
                    return Err(format!(
                        "{}: block {bi} Result ctor with non-empty args — \
                         operand-carrying ctor shape not expected from front::mir",
                        graph.name
                    ));
                }
                if ctor.is_some() {
                    return Err(format!(
                        "{}: block {bi} has two Result ctors — unsupported shape",
                        graph.name
                    ));
                }
                let Some(v) = op.result.clone() else {
                    return Err(format!(
                        "{}: block {bi} Result ctor without result var",
                        graph.name
                    ));
                };
                ctor = Some((i, v, is_err));
            }
        }
        let Some((ctor_idx, ctor_var, is_err)) = ctor else {
            continue;
        };
        // Payload FieldWrite (__pos_0).  Required: every scoped callee
        // returns a payload-carrying Result (unit payloads would lower
        // with no FieldWrite and need a Void widening).
        let mut fieldwrite_idx: Option<(usize, Variable)> = None;
        for (i, op) in graph.blocks[bi]
            .operations
            .iter()
            .enumerate()
            .skip(ctor_idx + 1)
        {
            if let OpKind::FieldWrite {
                base, field, value, ..
            } = &op.kind
                && *base == ctor_var
            {
                if field.name != "__pos_0" || fieldwrite_idx.is_some() {
                    return Err(format!(
                        "{}: block {bi} Result ctor with unexpected FieldWrite \
                         {} — only a single __pos_0 payload is supported",
                        graph.name, field.name
                    ));
                }
                // The `__pos_0` payload flows on to a `to_exc_object`
                // call / forwarding exit as an SSA operand, so an
                // exception-carrying Result writes the ref-kind evalue
                // `Variable`.  Since the int-kind `FieldWrite` widening
                // (`LinkArg::Value`→`LinkArg::Const`) a payload write may
                // instead carry an inline constant; that is never the
                // exception-bridge shape (a const evalue cannot reach
                // `to_exc_object`), so skip the rewrite rather than panic.
                let Some(payload_var) = value.as_variable() else {
                    return Err(format!(
                        "{}: block {bi} Result __pos_0 payload write stores an \
                         inline constant, not the evalue Variable threaded to \
                         to_exc_object — not an exception-carrying Result shape",
                        graph.name
                    ));
                };
                fieldwrite_idx = Some((i, payload_var.clone()));
            }
        }
        let Some((fw_idx, payload)) = fieldwrite_idx else {
            return Err(format!(
                "{}: block {bi} Result ctor without a __pos_0 payload write",
                graph.name
            ));
        };
        // The shell's only op use is the `__pos_0` payload FieldWrite
        // base.  Its link uses are forwarding exit args: the monotonic
        // lowering forwards the shell once, but the framestate-threaded
        // lowering can carry the same value in several `mergeable` slots
        // (a value occupying both a locals and a stack cell appears once
        // per slot in `getoutputargs`), so `link_uses` may exceed 1.
        // Every forwarding slot reaches the returnblock (verified below);
        // the `Ok` rewrite replaces every occurrence with the payload and
        // the `Err` rewrite discards the exit wholesale (`set_raise_values`
        // → `set_goto`), so multiple forwarding slots lower soundly.
        let consumers = count_var_uses(graph, &ctor_var);
        let well_formed_return = consumers.op_uses == 1 && consumers.link_uses >= 1;
        // The shell must flow out through this block's single
        // unconditional exit.  A conditional exit is acceptable only when
        // the ctor is a consumed intermediate, not a return value: the
        // `__new__` wrapper builds `Ok(obj)` and immediately `match`es it
        // (a `v.__discriminant` switch in the same block) to thread the
        // freshly-built object through its post-construction subclass
        // fix-up.  Such a shell is read more than once (its `__pos_0`
        // write plus the `__discriminant` / `__pos_0` match reads) so
        // `well_formed_return` is false; skip it — left materialised, the
        // `match` reads it as an ordinary ADT and the constant
        // discriminant folds to the `Ok` arm in `simplify_lowered_graph`,
        // exactly like the consumed-intermediate handling below.  A
        // conditional exit on a *well-formed* return shell is an ambiguous
        // shape this rewrite cannot lower soundly (an `Err` rewrite's
        // `set_raise_values` → `set_goto` would discard the other arm), so
        // decline it to a residual call.
        if graph.blocks[bi].exits.len() != 1 || graph.blocks[bi].exitswitch.is_some() {
            if well_formed_return {
                return Err(format!(
                    "{}: block {bi} Result shell block has a conditional exit — \
                     unsupported shape",
                    graph.name
                ));
            }
            // A non-well-formed conditional shell is skipped as a consumed
            // intermediate (the `__new__` in-block `match`: the shell is read
            // by its `__discriminant` / `__pos_0` match, never forwarded to
            // `returnblock` — only the extracted payload is).  But a
            // conditional shell that DOES reach `returnblock` on some arm is a
            // genuine return this rewrite cannot lower cleanly; skipping it
            // while another return in the same callee rewrites cleanly keeps
            // `rewritten > 0`, so callers are rewired to the unwrapped
            // `T`/exception yet this path still returns a materialised
            // `Result`.  Decline the whole callee (fail-safe → residual call),
            // mirroring the unconditional guard below.
            if shell_reaches_returnblock(graph, bi, &ctor_var) {
                return Err(format!(
                    "{}: conditional Result return shell in block {bi} reaches \
                     returnblock but cannot be lowered cleanly — declining the \
                     callee to avoid a partial rewrite",
                    graph.name
                ));
            }
            continue;
        }
        // A ctor is one of this graph's return values only if its value
        // flows purely to `returnblock`.  A ctor whose value is consumed
        // inside the graph — an inlined callee's return that this graph
        // then `match`es on or passes to a call — is an intermediate
        // `Result`, not a return shell.  Leave it materialised (the
        // consuming `match` / call reads it as an ordinary ADT) and skip
        // it, rather than failing the whole callee.  The intermediate
        // never reaches `returnblock`, and the graph's genuine returns are
        // distinct ctors, so the surviving return type stays uniform.
        // The `Ok` rewrite only edits the producer's exit link args, so the
        // payload threads through any intervening block untouched to
        // `returnblock` — the generalized forward stays sound.  The `Err`
        // rewrite below calls `set_raise_values`, which *replaces* the
        // producer block's exit with a jump to `exceptblock`, bypassing
        // every intervening block; operations carried by such a block would
        // be dropped and the JIT would raise earlier than the interpreter.
        // Require the strict pure-forwarder property (empty, unconditional
        // intervening blocks only) for `Err` shells; decline to a residual
        // call otherwise.
        let forwards_ok = if is_err {
            forwards_to_returnblock(graph, bi, &ctor_var)
        } else {
            verify_forwards_to_returnblock_general(graph, bi, &ctor_var).is_ok()
        };
        if !well_formed_return || !forwards_ok {
            // Leaving the ctor materialised is sound only when it is a
            // consumed intermediate that never returns — its value must NOT
            // reach `returnblock`.  If it DOES reach `returnblock` on some
            // path (a genuine return shell this rewrite cannot lower cleanly:
            // an extra shell use, an intervening read, a non-pure `Err`
            // forward), skipping it while another return in the same callee
            // rewrites cleanly would keep `rewritten > 0` — the callee is
            // reported transformed and its callers are rewired to receive the
            // unwrapped `T`/exception, yet this path still returns a `Result`
            // object.  Decline the whole callee (fail-safe → residual call)
            // rather than emit that partial, unsound rewrite.
            if shell_reaches_returnblock(graph, bi, &ctor_var) {
                return Err(format!(
                    "{}: Result return shell in block {bi} reaches returnblock \
                     but cannot be lowered cleanly (non-pure forward / extra \
                     shell use) — declining the callee to avoid a partial rewrite",
                    graph.name
                ));
            }
            continue;
        }

        // Drop the ctor + FieldWrite (higher index first).
        {
            let ops = &mut graph.blocks[bi].operations;
            debug_assert!(fw_idx > ctor_idx);
            ops.remove(fw_idx);
            ops.remove(ctor_idx);
        }
        if is_err {
            // `return Err(e)` → materialise the runtime exception
            // object and raise.  The trace-level exception value is
            // the `W_BaseException` ref (`BH_LAST_EXC_VALUE`'s
            // domain; the trait leg reads `ob_header.ob_type` off it),
            // so `PyError::to_exc_object` runs at the raise site.
            let v_exc = graph
                .push_op_var(
                    block_id,
                    OpKind::Call {
                        target: CallTarget::method("to_exc_object", Some("PyError".to_string())),
                        args: vec![payload],
                        result_ty: ValueType::Ref(None),
                    },
                    true,
                )
                .expect("to_exc_object call must produce a value");
            // `op.type(evalue)` — the `lower_exc_from_raise` tail
            // (`flowcontext.py:600` `w_type = op.type(w_value)`).
            let v_type = graph
                .push_op_var(
                    block_id,
                    OpKind::Call {
                        target: CallTarget::function_path(["type"]),
                        args: vec![v_exc.clone()],
                        result_ty: ValueType::Ref(None),
                    },
                    true,
                )
                .expect("op.type(evalue) must produce a value");
            graph.set_raise_values(block_id, v_type, v_exc);
        } else {
            // `return Ok(v)` → forward the payload itself.
            for link in &mut graph.blocks[bi].exits {
                for arg in &mut link.args {
                    if matches!(arg, LinkArg::Value(v) if *v == ctor_var) {
                        *arg = LinkArg::Value(payload.clone());
                    }
                }
            }
        }
        rewritten += 1;
    }
    if rewritten == 0 && tail_forwarded_returns == 0 {
        // A scoped callee whose body is `return f(...)?` where the
        // caller rule never recorded `f`'s `?`-site — `f`'s return is not
        // `Result<T, PyError>` (e.g. a trait method such as
        // `OpcodeStepExecutor::return_value`): the optimised MIR folds the
        // `?`-diamond into a direct forward of the callee's `Result` to
        // `returnblock`, leaving no `Ok`/`Err` shell to rewrite, so
        // `tail_forwarded_returns` is 0.
        // This is the same disposition as a scoped tail-forward
        // (`SiteOutcome::TailForward`): the residual-call ABI erases the
        // shell (`Ok` → value, `Err` → `BH_LAST_EXC_VALUE`) and the
        // codewriter re-derives `guard_no_exception` op-locally, so the
        // forward already carries `T` and the raise propagates
        // implicitly — no rewrite is needed.
        if has_tail_forwarded_call_result(graph) {
            return Ok(0);
        }
        return Err(format!(
            "{}: scoped Result-of-PyError callee with no rewritable returns",
            graph.name
        ));
    }
    Ok(rewritten)
}

/// True when some block's `Call` result flows straight to `returnblock`
/// through pure positional forwarding — the `return f(...)?` tail-forward
/// shape.  Used by [`lower_result_exc_returns`] to accept scoped callees
/// that forward an unscoped callee's `Result` directly (the caller-rule
/// capture only records scoped callees, so such sites never reach
/// `rewire_result_exc_call_sites`).
fn has_tail_forwarded_call_result(graph: &FunctionGraph) -> bool {
    for bi in 0..graph.blocks.len() {
        for op in &graph.blocks[bi].operations {
            if matches!(op.kind, OpKind::Call { .. })
                && let Some(r) = &op.result
                && forwards_to_returnblock(graph, bi, r)
            {
                return true;
            }
        }
    }
    false
}

struct UseCounts {
    op_uses: usize,
    link_uses: usize,
}

/// Count uses of `var` as an op operand and as a link arg across the
/// whole graph (producer `op.result` slots are not uses).
fn count_var_uses(graph: &FunctionGraph, var: &Variable) -> UseCounts {
    let mut op_uses = 0usize;
    let mut link_uses = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            op_uses += op_operand_vars(&op.kind)
                .iter()
                .filter(|v| *v == var)
                .count();
        }
        for link in &block.exits {
            link_uses += link
                .args
                .iter()
                .filter(|a| matches!(a, LinkArg::Value(v) if v == var))
                .count();
        }
    }
    UseCounts { op_uses, link_uses }
}

/// Every `Variable` operand of an op kind.
///
/// `count_var_uses` and the carrier-unused check in `collapse_pos0_read`
/// rely on this being exhaustive: a missed operand-bearing variant makes
/// a live `Result`-shell consumer invisible, so the rewrite could still
/// delete the shell or collapse `__pos_0`.  The match has no wildcard —
/// a new `OpKind` variant is a compile error here until its operands are
/// declared, keeping the pass fail-closed.  Producer / constant / marker
/// kinds carry no operand `Variable` and return empty.
pub(crate) fn op_operand_vars(kind: &OpKind) -> Vec<Variable> {
    let extend_all = |dst: &mut Vec<Variable>, lists: &[&Vec<Variable>]| {
        for list in lists {
            dst.extend(list.iter().cloned());
        }
    };
    match kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstSymbolic { .. }
        | OpKind::ConstFloat(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstRefAddr(_)
        | OpKind::CurrentTraceLength
        | OpKind::Live
        | OpKind::LoopHeader { .. }
        | OpKind::Abort { .. }
        | OpKind::LoadStatic { .. }
        | OpKind::NewWithVtable { .. } => Vec::new(),

        OpKind::FieldRead { base, .. }
        | OpKind::VableFieldRead { base, .. }
        | OpKind::VableForce { base }
        | OpKind::RecordQuasiImmutField { base, .. } => vec![base.clone()],
        OpKind::Hint { value, .. } => vec![value.clone()],
        OpKind::FieldWrite { base, value, .. } => {
            // Only a `Variable` value contributes an SSA reference; a
            // `setfield_gc` inline `Const` carries no defining op.
            let mut refs = vec![base.clone()];
            if let Some(var) = value.as_variable() {
                refs.push(var.clone());
            }
            refs
        }
        OpKind::VableFieldWrite { base, value, .. } => {
            // Only a `Variable` value contributes an SSA reference; a
            // `setfield_vable_i` inline `Const` carries no defining op.
            let mut refs = vec![base.clone()];
            if let Some(var) = value.as_variable() {
                refs.push(var.clone());
            }
            refs
        }
        OpKind::ArrayLen { base, .. } => vec![base.clone()],
        OpKind::ArrayRead { base, index, .. } | OpKind::InteriorFieldRead { base, index, .. } => {
            vec![base.clone(), index.clone()]
        }
        OpKind::ArrayWrite {
            base, index, value, ..
        } => {
            // An inline-const value references no Variable.
            let mut refs = vec![base.clone(), index.clone()];
            if let Some(v) = value.as_variable() {
                refs.push(v.clone());
            }
            refs
        }
        OpKind::InteriorFieldWrite {
            base, index, value, ..
        } => vec![base.clone(), index.clone(), value.clone()],
        OpKind::VableArrayRead {
            base, elem_index, ..
        } => vec![base.clone(), elem_index.clone()],
        OpKind::VableArrayWrite {
            base,
            elem_index,
            value,
            ..
        } => vec![base.clone(), elem_index.clone(), value.clone()],
        OpKind::Call { args, .. }
        | OpKind::JitDebug { args }
        | OpKind::NewTuple { args }
        | OpKind::NewList { args }
        | OpKind::LoweredBlackholeOp { args, .. } => args.clone(),
        OpKind::GuardTrue { cond } | OpKind::GuardFalse { cond } => vec![cond.clone()],
        OpKind::GuardValue { value, .. }
        | OpKind::AssertGreen { value, .. }
        | OpKind::IsConstant { value, .. }
        | OpKind::IsVirtual { value, .. } => vec![value.clone()],
        OpKind::VtableMethodPtr { receiver, .. } => vec![receiver.clone()],
        OpKind::IsInstance {
            obj, class_carrier, ..
        } => vec![obj.clone(), class_carrier.clone()],
        OpKind::BinOp { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
        OpKind::UnaryOp { operand, .. } => vec![operand.clone()],
        OpKind::IndirectCall { funcptr, args, .. } => {
            let mut v = vec![funcptr.clone()];
            v.extend(args.iter().cloned());
            v
        }
        OpKind::CallElidable {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallResidual {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallMayForce {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut v = Vec::new();
            if let CallFuncPtr::Value(var) = funcptr {
                v.push(var.clone());
            }
            extend_all(&mut v, &[args_i, args_r, args_f]);
            v
        }
        OpKind::InlineCall {
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut v = Vec::new();
            extend_all(&mut v, &[args_i, args_r, args_f]);
            v
        }
        OpKind::ConditionalCall {
            condition: gate,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::ConditionalCallValue {
            value: gate,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::RecordKnownResult {
            result_value: gate,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut v = vec![gate.clone()];
            extend_all(&mut v, &[args_i, args_r, args_f]);
            v
        }
        OpKind::RecursiveCall {
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        }
        | OpKind::JitMergePoint {
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        } => {
            let mut v = Vec::new();
            extend_all(
                &mut v,
                &[greens_i, greens_r, greens_f, reds_i, reds_r, reds_f],
            );
            v
        }
    }
}

/// Verify `var`, produced in `from_block`, reaches `returnblock` purely
/// as a forwarding link arg — never read by an operation, never used as
/// an `exitswitch` operand — along every path it takes.
///
/// Intermediate blocks may carry unrelated operations and conditional
/// exits: as long as none of them touch the tracked value, it is threaded
/// untouched through inputarg aliases to `returnblock`, so the producer's
/// `Ok`-payload substitution / `Err` raise stays sound regardless of the
/// surrounding control flow (the `Ok` rewrite only edits the producer's
/// exit, and `set_raise_values` redirects the whole producer block).
///
/// Worklist over `(block, alias)` states — `alias` is the inputarg the
/// tracked value binds to on entry to `block`.  Bounded by the number of
/// distinct `(block, inputarg)` pairs, so it always terminates.  A value
/// occupying several `mergeable` slots reaches a block under more than one
/// alias; each is a distinct state and is followed independently.
fn verify_forwards_to_returnblock_general(
    graph: &FunctionGraph,
    from_block: usize,
    var: &Variable,
) -> Result<(), String> {
    let mut seen: std::collections::HashSet<(usize, Variable)> = std::collections::HashSet::new();
    let mut work: Vec<(usize, Variable)> = vec![(from_block, var.clone())];
    let mut reached_return = false;
    while let Some((cur, v)) = work.pop() {
        if !seen.insert((cur, v.clone())) {
            continue;
        }
        let block = &graph.blocks[cur];
        // The producer block keeps the ctor + `__pos_0` write that
        // legitimately read `var`; every other block must thread the alias
        // untouched — an operation reading it would inspect or re-derive
        // the value en route, so deleting the shell would be unsound.
        if cur != from_block {
            for op in &block.operations {
                if op_operand_vars(&op.kind).iter().any(|o| o == &v) {
                    return Err(format!(
                        "{}: Result shell alias is read by an operation in \
                         block {cur} on the forwarding path — not a pure forward",
                        graph.name
                    ));
                }
            }
        }
        // The value steering control flow is the call-site discriminant
        // switch shape (handled by the caller rule), not a return forward.
        if let Some(ExitSwitch::Value(sw)) = &block.exitswitch
            && *sw == v
        {
            return Err(format!(
                "{}: Result shell alias drives the exitswitch in block {cur} — \
                 not a return-forwarding shape",
                graph.name
            ));
        }
        // Follow every exit that carries the alias.
        let mut carried = false;
        for link in &block.exits {
            let positions: Vec<usize> = link
                .args
                .iter()
                .enumerate()
                .filter_map(|(i, a)| match a {
                    LinkArg::Value(x) if *x == v => Some(i),
                    _ => None,
                })
                .collect();
            if positions.is_empty() {
                continue;
            }
            carried = true;
            if link.target == graph.returnblock {
                reached_return = true;
                continue;
            }
            let target = &graph.blocks[link.target.0];
            for pos in positions {
                let Some(next) = target.inputargs.get(pos) else {
                    return Err(format!(
                        "{}: forwarding target block {} has no inputarg at \
                         position {pos}",
                        graph.name, link.target.0
                    ));
                };
                work.push((link.target.0, next.clone()));
            }
        }
        if !carried {
            return Err(format!(
                "{}: Result shell alias lost at block {cur} (carried by no exit)",
                graph.name
            ));
        }
    }
    if !reached_return {
        return Err(format!(
            "{}: Result-return forwarding chain did not reach returnblock",
            graph.name
        ));
    }
    Ok(())
}

/// Whether the Result shell `var` produced in `from_block` reaches
/// `returnblock` along any forwarding path.  Purity-agnostic companion to
/// [`verify_forwards_to_returnblock_general`]: it answers reachability
/// only — it does not reject an intervening read or an exitswitch use — so
/// the caller can tell a genuine (but un-lowerable) return shell apart from
/// a consumed intermediate that never returns.  Follows the same
/// link-arg-position → target-inputarg aliasing as the verifier.
fn shell_reaches_returnblock(graph: &FunctionGraph, from_block: usize, var: &Variable) -> bool {
    let mut seen: std::collections::HashSet<(usize, Variable)> = std::collections::HashSet::new();
    let mut work: Vec<(usize, Variable)> = vec![(from_block, var.clone())];
    while let Some((cur, v)) = work.pop() {
        if !seen.insert((cur, v.clone())) {
            continue;
        }
        for link in &graph.blocks[cur].exits {
            let positions: Vec<usize> = link
                .args
                .iter()
                .enumerate()
                .filter_map(|(i, a)| match a {
                    LinkArg::Value(x) if *x == v => Some(i),
                    _ => None,
                })
                .collect();
            if positions.is_empty() {
                continue;
            }
            if link.target == graph.returnblock {
                return true;
            }
            let target = &graph.blocks[link.target.0];
            for pos in positions {
                if let Some(next) = target.inputargs.get(pos) {
                    work.push((link.target.0, next.clone()));
                }
            }
        }
    }
    false
}

/// What [`rewire_one_call_site`] found at a scoped call site.
pub(crate) struct RewireOutcome {
    /// `?`-diamond sites rewired into `LastException` exits.
    pub diamonds: usize,
    /// Tail-forwarded sites (`return f(...)` — the callee's `Result`
    /// IS this graph's return value).  Once the callee is transformed
    /// the forward already carries `T` and the raise propagates
    /// implicitly, so no rewrite is needed — but only inside a graph
    /// that is itself a scoped callee; an unscoped enclosing graph
    /// would hand `T` to callers still switching on a discriminant.
    pub tail_forwards: usize,
    /// Custom-match sites handled by [`catch_and_rewrap`]: the call
    /// gets `LastException` exits and the two arms locally rebuild the
    /// value-encoded `Result` the untouched downstream `match` keeps
    /// consuming.
    pub rewrapped: usize,
}

/// Per-site disposition for [`rewire_one_call_site`].
enum SiteOutcome {
    Diamond,
    TailForward,
    Rewrapped,
}

/// Caller rule.  `results` are the result `Variable`s of calls to
/// scoped callees (captured during lowering).  Each site is either a
/// `?`-diamond — rewired into `ExitSwitch::LastException` exits — or a
/// tail-forward to `returnblock` inside a scoped enclosing graph, or a
/// custom `match` consumer that gets the catch-and-rewrap treatment.
pub(crate) fn rewire_result_exc_call_sites(
    graph: &mut FunctionGraph,
    results: &[(Variable, Option<String>)],
    enclosing_scoped: bool,
) -> Result<RewireOutcome, String> {
    let mut outcome = RewireOutcome {
        diamonds: 0,
        tail_forwards: 0,
        rewrapped: 0,
    };
    for (r, suffix) in results {
        match rewire_one_call_site(graph, r, suffix.as_deref().unwrap_or(""), enclosing_scoped)? {
            SiteOutcome::Diamond => outcome.diamonds += 1,
            SiteOutcome::TailForward => outcome.tail_forwards += 1,
            SiteOutcome::Rewrapped => outcome.rewrapped += 1,
        }
    }
    Ok(outcome)
}

fn rewire_one_call_site(
    graph: &mut FunctionGraph,
    r: &Variable,
    suffix: &str,
    enclosing_scoped: bool,
) -> Result<SiteOutcome, String> {
    let name = graph.name.clone();
    // Block A: contains the call producing `r`; closed by lower_call
    // with a single forwarding exit.
    let a = graph
        .blocks
        .iter()
        .position(|b| b.operations.iter().any(|op| op.result.as_ref() == Some(r)))
        .ok_or_else(|| format!("{name}: scoped call result var has no producer block"))?;
    // Tail forward: the callee's Result flows straight to returnblock.
    if forwards_to_returnblock(graph, a, r) {
        if !enclosing_scoped {
            return Err(format!(
                "{name}: tail-forwards a scoped callee's Result out of a \
                 non-`Result<T, PyError>` graph — the callers' discriminant \
                 switches would read garbage"
            ));
        }
        return Ok(SiteOutcome::TailForward);
    }
    let (b, r_b) =
        follow_single_exit(graph, a, r).map_err(|e| format!("{name}: call block exit: {e}"))?;
    // Block B: `cf = Result::branch(r)`.  A block without the branch
    // call is a custom-match consumer (hand-written `match` on the
    // Result, possibly behind a multi-predecessor merge) — handled by
    // the local catch-and-rewrap arm instead of the diamond rewire.
    let branch_op_idx = graph.blocks[b].operations.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::Call { target: CallTarget::Method { name, .. }, args, .. }
                if name == "branch" && args.as_slice() == std::slice::from_ref(&r_b)
        )
    });
    let Some(branch_op_idx) = branch_op_idx else {
        catch_and_rewrap(graph, a, r, suffix)?;
        return Ok(SiteOutcome::Rewrapped);
    };
    assert_single_pred(graph, b, &name)?;
    // Block B is bypassed by the rewrite (A exits straight to the
    // continue target); only the `branch` destructuring may carry an
    // effect, so any other side-effecting op here is unsupported.
    assert_block_pure_besides(graph, b, &[branch_op_idx], "branch", &name)?;
    let cf = graph.blocks[b].operations[branch_op_idx]
        .result
        .clone()
        .ok_or_else(|| format!("{name}: branch() without result var"))?;
    let (c, cf_c) =
        follow_single_exit(graph, b, &cf).map_err(|e| format!("{name}: branch block exit: {e}"))?;
    assert_single_pred(graph, c, &name)?;
    // Block C: `d = cf.__discriminant`; switch d {0 → continue, 1 → break}.
    let (disc_idx, disc_var) = graph.blocks[c]
        .operations
        .iter()
        .enumerate()
        .find_map(|(i, op)| match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if *base == cf_c && field.name == "__discriminant" =>
            {
                op.result.clone().map(|r| (i, r))
            }
            _ => None,
        })
        .ok_or_else(|| format!("{name}: block {c} lacks the ControlFlow __discriminant read"))?;
    match &graph.blocks[c].exitswitch {
        Some(ExitSwitch::Value(v)) if *v == disc_var => {}
        other => {
            return Err(format!(
                "{name}: block {c} exitswitch {other:?} is not the \
                 ControlFlow discriminant switch"
            ));
        }
    }
    // Block C is bypassed too; only the discriminant read may carry an
    // effect.  Reject any extra side-effecting op the switch would drop.
    assert_block_pure_besides(graph, c, &[disc_idx], "discriminant", &name)?;
    let (continue_link, break_link) = split_diamond_exits(&graph.blocks[c].exits, &name)?;
    // The break arm must be the pure `?` re-raise tail
    // (`__pos_0` read + `from_residual` + return).  A custom handler
    // arm must not be silently disconnected.
    verify_break_arm_is_reraise(graph, &break_link, &cf_c, &name)?;

    // Map each continue-arm link arg back to A-scope variables: the
    // A→B→C chain is pure positional forwarding.
    let mut normal_args: Vec<LinkArg> = Vec::with_capacity(continue_link.args.len());
    let mut payload_positions: Vec<usize> = Vec::new();
    for (i, arg) in continue_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(c) => normal_args.push(LinkArg::Const(c.clone())),
            LinkArg::Value(v) => {
                if *v == cf_c {
                    // The ControlFlow value at the continue edge is the
                    // unwrapped payload once the callee raises: the
                    // call result itself flows in its place.
                    normal_args.push(LinkArg::Value(r.clone()));
                    payload_positions.push(i);
                } else if *v == disc_var {
                    // The framestate-threaded lowering carries the
                    // ControlFlow `__discriminant` temporary forward on
                    // the continue edge (the monotonic lowering does
                    // not).  Block C — its only reader, the discriminant
                    // switch — is removed by this rewrite, so the value
                    // has no A-scope origin; but the continue arm is the
                    // `Continue` case (`split_diamond_exits` keys it to
                    // discriminant `0`), so the value here is the
                    // constant `0`.  Carry that constant; the now-dead
                    // downstream threading is left to the post-rewrite
                    // `simplify_lowered_graph` dead-variable sweep.
                    normal_args.push(LinkArg::Const(crate::flowspace::model::Constant::new(
                        crate::flowspace::model::ConstValue::Int(0),
                    )));
                } else {
                    let v_a = back_substitute(graph, &[(a, b), (b, c)], v, &name)?;
                    normal_args.push(LinkArg::Value(v_a));
                }
            }
        }
    }
    // `collapse_pos0_read` below is the only fallible mutation; it mutates
    // the continue target on success but can still `Err` on a later
    // position.  With at most one position the collapse is the first
    // mutation and itself atomic (it errs before writing), so a decline
    // leaves the graph byte-identical.  Two or more positions (the same
    // Result threaded into several continue-arm slots) could half-collapse
    // before a later `Err`, handing the legacy walker a partially-rewritten
    // graph — decline that unusual shape up front to keep the
    // "validate-before-mutate" fail-safe contract airtight (mirrors
    // `iter_next::rewire_one_next_site`).
    if payload_positions.len() > 1 {
        return Err(format!(
            "{name}: Result value threaded into {} continue-arm slots — multi-slot \
             payload collapse is not fail-safe",
            payload_positions.len()
        ));
    }

    // The continue target reads the payload via `cf.__pos_0`; with the
    // call result flowing directly, that read collapses to the carried
    // value itself.
    let continue_target = continue_link.target;
    for pos in payload_positions {
        collapse_pos0_read(graph, continue_target, pos, &name)?;
    }

    // Rewire A: LastException exits — normal → continue arm,
    // exception → exceptblock via the default exception link
    // (`flowspace/model.py` `Link.last_exception` pair; `flatten.rs`
    // turns the `[last_exception, last_exc_value]` propagation shape
    // into the rethrow tail).
    let va = graph.alloc_value_var();
    let vb = graph.alloc_value_var();
    let exceptblock = graph.exceptblock;
    // `exception_exitcase()` marks the link catch-all
    // (`Link::catches_all_exceptions`), the propagation shape
    // `flatten.rs` rethrows without a `goto_if_exception_mismatch`.
    let mut exc_link = Link::new_mixed(
        vec![LinkArg::Value(va.clone()), LinkArg::Value(vb.clone())],
        exceptblock,
        Some(crate::model::exception_exitcase()),
    );
    exc_link.last_exception = Some(LinkArg::Value(va));
    exc_link.last_exc_value = Some(LinkArg::Value(vb));
    let block_a = &mut graph.blocks[a];
    block_a.exitswitch = Some(ExitSwitch::LastException);
    block_a.exits = vec![
        Link::new_mixed(normal_args, continue_target, None),
        exc_link,
    ];
    // Blocks B, C and the break arm are now unreachable; the dead-op
    // sweep leaves them to the reachability-walking consumers.
    Ok(SiteOutcome::Diamond)
}

/// Custom-match fallback: the call's `Result` is consumed by a
/// hand-written `match` (eval.rs `eval_loop` dispatches `StepResult` +
/// the error handler) — possibly behind a multi-predecessor merge
/// shared with sibling shells (`eval_loop_jit`'s `match step_result`
/// merges 7 predecessors).  Rewiring that destructuring in place would
/// need per-predecessor jump threading, so instead the rewrite stays
/// local to the call edge: the call block gets `LastException` exits
/// whose two arms REBUILD the value-encoded `Result` the untouched
/// downstream keeps consuming — the normal arm wraps the raw return in
/// an `Ok` shell, the exception arm binds the caught `W_BaseException`
/// back into the `PyError` domain (`PyError::from_exc_object`, the
/// inverse of the callee rule's `to_exc_object`) and wraps it in an
/// `Err` shell.  This is the same erasure boundary the residual-call
/// ABI implements at host calls (`Ok` → value, `Err` →
/// `BH_LAST_EXC_VALUE`), value-encoded again one block later.  The
/// rebuilt shells sit in the CALLER's graph next to the sibling shells
/// the caller already builds for its own returns, so no new shell
/// exposure is introduced on walked paths.
fn catch_and_rewrap(
    graph: &mut FunctionGraph,
    a: usize,
    r: &Variable,
    suffix: &str,
) -> Result<(), String> {
    use crate::model::BlockId;
    let name = graph.name.clone();
    if graph.blocks[a].exitswitch.is_some() {
        return Err(format!(
            "{name}: rewrap call block {a} has an exitswitch — expected the \
             single forwarding exit lower_call installs"
        ));
    }
    let [orig] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: rewrap call block {a} must have a single exit"
        ));
    };
    let orig = orig.clone();
    if orig.exitcase.is_some() {
        return Err(format!(
            "{name}: rewrap call block {a} exit carries an exitcase"
        ));
    }
    let is_r = |arg: &LinkArg| matches!(arg, LinkArg::Value(v) if v == r);
    let has_r = orig.args.iter().any(|a| is_r(a));

    // --- Normal arm N: receive every Value arg, rebuild `Ok(r)`.
    let value_args: Vec<LinkArg> = orig
        .args
        .iter()
        .filter(|a| matches!(a, LinkArg::Value(_)))
        .cloned()
        .collect();
    let (n_id, n_inputs) = graph.create_block_with_arg_vars(value_args.len());
    let n_shell: Option<Variable> = if has_r {
        let r_value_idx = value_args
            .iter()
            .position(|a| is_r(a))
            .expect("has_r implies a Value position");
        let payload = n_inputs[r_value_idx].clone();
        Some(build_shell(graph, n_id, "Ok", payload, suffix))
    } else {
        None
    };
    let mut vi = 0usize;
    let n_exit_args: Vec<LinkArg> = orig
        .args
        .iter()
        .map(|arg| match arg {
            LinkArg::Const(c) => LinkArg::Const(c.clone()),
            LinkArg::Value(_) => {
                let v = n_inputs[vi].clone();
                vi += 1;
                if is_r(arg) {
                    LinkArg::Value(n_shell.clone().expect("shell built when r flows"))
                } else {
                    LinkArg::Value(v)
                }
            }
        })
        .collect();
    graph.set_control_flow_metadata(
        n_id,
        None,
        vec![Link::new_mixed(n_exit_args, orig.target, None)],
    );

    // --- Exception arm E: receive the non-`r` Value args plus the
    // caught `[exc_type, exc_value]` pair, rebuild `Err(from_exc_object
    // (exc_value))`.
    let nonr_args: Vec<LinkArg> = orig
        .args
        .iter()
        .filter(|a| matches!(a, LinkArg::Value(_)) && !is_r(a))
        .cloned()
        .collect();
    let (e_id, e_inputs) = graph.create_block_with_arg_vars(nonr_args.len() + 2);
    let e_exc_value_in = e_inputs[nonr_args.len() + 1].clone();
    let e_shell: Option<Variable> = if has_r {
        // `PyError::from_exc_object(exc_value)` is an associated fn (no
        // `self`), yet it is spelled as a `Method` target with `receiver_root
        // = "PyError"` and the caught exception value as `args[0]`.  The
        // codewriter resolves the recorded call statically, not by a runtime
        // dispatch on the exception object: `call.rs::resolve_method` keys the
        // `getfunctionptr(graph)` identity on
        // `CallPath::for_impl_method(receiver_root, name)` →
        // `for_impl_method("PyError", "from_exc_object")` and never on the
        // runtime receiver's class, so `exc_value` flows positionally into
        // the callee's first param (`obj`) — the same way the inherent
        // `&self` `to_exc_object` above threads its receiver positionally.
        // (The rtyper's annotator-facing lowering in `flowspace_adapter.rs`
        // does turn every `Method` into `getattr(args[0], leaf) →
        // simple_call`; that view drives type inference only — the
        // codewriter's static `resolve_method` above mints the actual call.)
        // A `FunctionPath(["PyError", "from_exc_object"])` would instead
        // resolve to that bare two-segment path, which misses the
        // module-qualified impl-method registration and falls back to a
        // symbolic address.
        let v_err = graph
            .push_op_var(
                e_id,
                OpKind::Call {
                    target: CallTarget::method("from_exc_object", Some("PyError".to_string())),
                    args: vec![e_exc_value_in],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .expect("from_exc_object must produce a value");
        Some(build_shell(graph, e_id, "Err", v_err, suffix))
    } else {
        None
    };
    let mut ei = 0usize;
    let e_exit_args: Vec<LinkArg> = orig
        .args
        .iter()
        .map(|arg| match arg {
            LinkArg::Const(c) => LinkArg::Const(c.clone()),
            LinkArg::Value(_) if is_r(arg) => {
                LinkArg::Value(e_shell.clone().expect("shell built when r flows"))
            }
            LinkArg::Value(_) => {
                let v = e_inputs[ei].clone();
                ei += 1;
                LinkArg::Value(v)
            }
        })
        .collect();
    graph.set_control_flow_metadata(
        e_id,
        None,
        vec![Link::new_mixed(e_exit_args, orig.target, None)],
    );

    // --- Rewire A: LastException exits — normal → N, exception → E.
    let va = graph.alloc_value_var();
    let vb = graph.alloc_value_var();
    let a_to_e_args: Vec<LinkArg> = nonr_args
        .into_iter()
        .chain([LinkArg::Value(va.clone()), LinkArg::Value(vb.clone())])
        .collect();
    let mut exc_link = Link::new_mixed(a_to_e_args, e_id, Some(crate::model::exception_exitcase()));
    exc_link.last_exception = Some(LinkArg::Value(va));
    exc_link.last_exc_value = Some(LinkArg::Value(vb));
    graph.set_control_flow_metadata(
        BlockId(a),
        Some(ExitSwitch::LastException),
        vec![Link::new_mixed(value_args, n_id, None), exc_link],
    );
    Ok(())
}

/// Emit `shell = Result::<variant>(); shell.__pos_0 = payload` into
/// `block`, mirroring the front lowering's Aggregate shape
/// (`front/mir.rs` `Rvalue::Aggregate`: niladic transparent ctor + one
/// FieldWrite per operand, `result: None` on the write).
fn build_shell(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    variant: &str,
    payload: Variable,
    suffix: &str,
) -> Variable {
    use crate::model::FieldDescriptor;
    // `suffix` (`<Tuple,PyError>` or empty) keys the shell's ClassDef per
    // instantiation, matching the front aggregate path; both the `Ok` and
    // `Err` shells of one callee carry it so the variants share one base.
    let owner = format!("core::result::Result{suffix}::{variant}");
    let shell = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor_with_owner(
                    vec![
                        "core".to_string(),
                        "result".to_string(),
                        format!("Result{suffix}"),
                    ],
                    variant,
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner.clone())),
            },
            true,
        )
        .expect("Result ctor must produce a value");
    graph.blocks[block.0]
        .operations
        .push(crate::model::SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: shell.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(owner),
                    owner_id: None,
                },
                value: crate::model::LinkArg::Value(payload),
                ty: ValueType::Ref(None),
            },
        });
    shell
}

/// Probe: does `var` flow from `block`'s exit through pure positional
/// forwarding into `returnblock`?  The non-erroring twin of
/// [`verify_forwards_to_returnblock`] — any non-conforming hop means
/// "not a tail forward" rather than a build failure (the site is then
/// matched as a diamond, whose own checks fail loud).
fn forwards_to_returnblock(graph: &FunctionGraph, block: usize, var: &Variable) -> bool {
    let mut current = block;
    let mut tracked = var.clone();
    for _ in 0..graph.blocks.len() {
        // A pure tail forward only crosses empty, unconditional blocks
        // after the producer `block`; an intermediate block with
        // operations or a conditional exit inspects the value and is not
        // a tail forward.
        if current != block {
            let b = &graph.blocks[current];
            if !b.operations.is_empty() || b.exitswitch.is_some() {
                return false;
            }
        }
        let [link] = graph.blocks[current].exits.as_slice() else {
            return false;
        };
        let Some(pos) = link
            .args
            .iter()
            .position(|a| matches!(a, LinkArg::Value(v) if *v == tracked))
        else {
            return false;
        };
        if link.target == graph.returnblock {
            return true;
        }
        let target = link.target.0;
        let Some(next_var) = graph.blocks[target].inputargs.get(pos) else {
            return false;
        };
        tracked = next_var.clone();
        current = target;
    }
    false
}

/// Map a continue-arm link variable back to its A-scope origin
/// through the diamond's pure positional forwarding chain.  `chain`
/// is the ordered `(pred, succ)` edge list from the call block; a
/// variable that is `succ`'s inputarg maps through `pred`'s single
/// exit, a variable defined inside an intermediate block cannot flow
/// back and fails loud.
pub(crate) fn back_substitute(
    graph: &FunctionGraph,
    chain: &[(usize, usize)],
    var: &Variable,
    name: &str,
) -> Result<Variable, String> {
    let mut current = var.clone();
    for &(pred, succ) in chain.iter().rev() {
        let Some(pos) = graph.blocks[succ]
            .inputargs
            .iter()
            .position(|v| *v == current)
        else {
            return Err(format!(
                "{name}: continue-arm value is defined inside diamond block \
                 {succ} and cannot be carried across the rewired call edge"
            ));
        };
        let [link] = graph.blocks[pred].exits.as_slice() else {
            return Err(format!(
                "{name}: diamond forwarding block {pred} has multiple exits"
            ));
        };
        match link.args.get(pos) {
            Some(LinkArg::Value(v)) => current = v.clone(),
            other => {
                return Err(format!(
                    "{name}: diamond forwarding arg at position {pos} is \
                     {other:?}, expected a Value"
                ));
            }
        }
    }
    Ok(current)
}

/// `block`'s single exit must carry `var`; returns the target block
/// index and the inputarg `var` binds to there.
pub(crate) fn follow_single_exit(
    graph: &FunctionGraph,
    block: usize,
    var: &Variable,
) -> Result<(usize, Variable), String> {
    let [link] = graph.blocks[block].exits.as_slice() else {
        return Err(format!(
            "block {block} has {} exits, expected 1",
            graph.blocks[block].exits.len()
        ));
    };
    let Some(pos) = link
        .args
        .iter()
        .position(|a| matches!(a, LinkArg::Value(v) if v == var))
    else {
        return Err(format!(
            "block {block}'s exit does not carry the tracked value"
        ));
    };
    let target = link.target.0;
    let bound = graph.blocks[target]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("block {target} lacks inputarg {pos}"))?;
    Ok((target, bound))
}

/// The diamond's intermediate blocks must have exactly one
/// predecessor — the chain we arrived through.
pub(crate) fn assert_single_pred(
    graph: &FunctionGraph,
    block: usize,
    name: &str,
) -> Result<(), String> {
    let preds = graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter())
        .filter(|l| l.target.0 == block)
        .count();
    if preds != 1 {
        return Err(format!(
            "{name}: diamond block {block} has {preds} predecessors, expected 1"
        ));
    }
    Ok(())
}

/// Split a discriminant switch's exits into (continue = case 0,
/// break = case 1).  MIR lowers a two-variant discriminant switch
/// as one explicit case plus a `default` arm covering the
/// complementary discriminant (mir.rs `SwitchTargets::SwitchInt`),
/// so a `default` link stands in for whichever of 0/1 is absent.
pub(crate) fn split_diamond_exits(exits: &[Link], name: &str) -> Result<(Link, Link), String> {
    use crate::flowspace::model::ConstValue;
    use crate::model::ExitCase;
    if exits.len() != 2 {
        return Err(format!(
            "{name}: ControlFlow switch has {} exits, expected 2",
            exits.len()
        ));
    }
    let mut cont: Option<Link> = None;
    let mut brk: Option<Link> = None;
    let mut default: Option<Link> = None;
    for l in exits {
        match &l.exitcase {
            Some(ExitCase::Const(ConstValue::Int(0))) => cont = Some(l.clone()),
            Some(ExitCase::Const(ConstValue::Int(1))) => brk = Some(l.clone()),
            Some(ExitCase::Const(ConstValue::UniStr(s))) if s == "default" => {
                default = Some(l.clone())
            }
            _ => {
                return Err(format!(
                    "{name}: ControlFlow switch has a non-0/1 exit case {:?}",
                    l.exitcase
                ));
            }
        }
    }
    match (cont, brk, default) {
        (Some(c), Some(b), None) => Ok((c, b)),
        (Some(c), None, Some(d)) => Ok((c, d)),
        (None, Some(b), Some(d)) => Ok((d, b)),
        _ => Err(format!(
            "{name}: ControlFlow switch lacks the 0/1 case pair"
        )),
    }
}

/// The break arm must be exactly `e = cf.__pos_0; from_residual(e);
/// → returnblock` — the `?` re-raise tail that the exception link
/// replaces.  Anything else is a custom handler and must fail loud.
/// Assert every operation in `block` other than the `recognized`
/// indices is side-effect-free.  The `?`-diamond rewrite disconnects the
/// branch / discriminant / break-arm blocks, so an unrecognised
/// side-effecting op in any of them would be silently bypassed; RPython
/// exception links are equivalent only when the removed shape is pure
/// control / unwrap / reraise plumbing.  Pure extras (constants, reads)
/// are harmless to bypass and are allowed.
pub(crate) fn assert_block_pure_besides(
    graph: &FunctionGraph,
    block: usize,
    recognized: &[usize],
    role: &str,
    name: &str,
) -> Result<(), String> {
    for (i, op) in graph.blocks[block].operations.iter().enumerate() {
        if recognized.contains(&i) {
            continue;
        }
        if !crate::inline::is_pure_op(&op.kind) {
            return Err(format!(
                "{name}: {role} block {block} carries a side-effecting operation \
                 the `?`-diamond rewrite would silently bypass — unsupported shape"
            ));
        }
    }
    Ok(())
}

fn verify_break_arm_is_reraise(
    graph: &FunctionGraph,
    break_link: &Link,
    cf_c: &Variable,
    name: &str,
) -> Result<(), String> {
    let pos = break_link
        .args
        .iter()
        .position(|a| matches!(a, LinkArg::Value(v) if v == cf_c))
        .ok_or_else(|| format!("{name}: break arm does not carry the ControlFlow value"))?;
    let e_block = break_link.target.0;
    let cf_e = graph.blocks[e_block]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("{name}: break arm target lacks inputarg {pos}"))?;
    let ops = &graph.blocks[e_block].operations;
    let payload = ops.iter().enumerate().find_map(|(i, op)| match &op.kind {
        OpKind::FieldRead { base, field, .. } if *base == cf_e && field.name == "__pos_0" => {
            op.result.clone().map(|r| (i, r))
        }
        _ => None,
    });
    let Some((pos0_idx, payload_var)) = payload else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the __pos_0 residual read — \
             custom `?` handler shapes are not supported yet"
        ));
    };
    let residual = ops.iter().enumerate().find_map(|(i, op)| match &op.kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            args,
            ..
        } if m == "from_residual" && args.as_slice() == std::slice::from_ref(&payload_var) => {
            op.result.clone().map(|r| (i, r))
        }
        _ => None,
    });
    let Some((from_residual_idx, residual_result)) = residual else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the from_residual call — \
             custom `?` handler shapes are not supported yet"
        ));
    };
    // Only the `__pos_0` read and the `from_residual` call may carry an
    // effect; any other side-effecting op would be dropped by the rewrite.
    assert_block_pure_besides(
        graph,
        e_block,
        &[pos0_idx, from_residual_idx],
        "break arm",
        name,
    )?;
    verify_forwards_to_returnblock_general(graph, e_block, &residual_result)
}

/// In the continue-arm target, the `__pos_0` read off the inputarg at
/// `pos` collapses: the inherited value already *is* the payload.
/// Deletes the read and renames its result to the inputarg.
pub(crate) fn collapse_pos0_read(
    graph: &mut FunctionGraph,
    target: crate::model::BlockId,
    pos: usize,
    name: &str,
) -> Result<(), String> {
    let ti = target.0;
    let carrier = graph.blocks[ti]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("{name}: continue target lacks inputarg {pos}"))?;
    let read_idx = graph.blocks[ti].operations.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { base, field, .. }
                if *base == carrier && field.name == "__pos_0"
        )
    });
    let Some(read_idx) = read_idx else {
        // The continue arm may legitimately discard the payload
        // (`let _ = f()?;` or `f()?;` on a non-void T).  Nothing reads
        // the carrier — but verify so a moved read does not survive
        // unrewired.
        let reads = graph.blocks[ti]
            .operations
            .iter()
            .filter(|op| op_operand_vars(&op.kind).contains(&carrier))
            .count();
        if reads != 0 {
            return Err(format!(
                "{name}: continue target block {ti} uses the ControlFlow \
                 carrier outside a __pos_0 read — unsupported shape"
            ));
        }
        return Ok(());
    };
    let read_result = graph.blocks[ti].operations[read_idx]
        .result
        .clone()
        .ok_or_else(|| format!("{name}: __pos_0 read without result"))?;
    graph.blocks[ti].operations.remove(read_idx);
    // Rename the read's result to the carrier across the block's
    // remaining ops, exitswitch, and exits.
    let rename = |v: &Variable| -> Variable {
        if *v == read_result {
            carrier.clone()
        } else {
            v.clone()
        }
    };
    let block = &mut graph.blocks[ti];
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &rename);
    }
    let (sw, exits) = crate::model::remap_control_flow_metadata_var(
        &block.exitswitch,
        &block.exits,
        rename,
        |b| b,
    );
    block.exitswitch = sw;
    block.exits = exits;
    Ok(())
}
