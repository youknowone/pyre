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
//!   domain is the `W_ExceptionObject` ref, the same value
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
//! returns `T` and raises, so an untransformed caller-side
//! discriminant switch would read garbage.  Until the whole-program
//! conformance scan lands, [`RESULT_EXC_LOWERING_SCOPE`] plus the
//! `pyopcode::execute_*` wrapper family pin the callee set; every call
//! site of a scoped callee either matches the `?`-diamond, tail-forwards
//! inside a scoped enclosing graph, or — for hand-written `match`
//! consumers (`eval_loop`, the `eval_loop_jit*` portals whose
//! `match step_result` merges seven predecessors) — gets the
//! [`catch_and_rewrap`] treatment: `LastException` exits on the call
//! block whose arms locally re-encode the `Result` (`Ok(raw)` /
//! `Err(PyError::from_exc_object(last_exc_value))`), leaving the
//! downstream destructuring untouched.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::TyRef;

use crate::flowspace::model::Variable;
use crate::model::{
    CallFuncPtr, CallTarget, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType,
};

/// Callees whose `Result<T, PyError>` surface lowers to raise links.
/// Grown deliberately, one fail-loud pipeline convergence at a time;
/// replaced by a whole-program conformance scan once every caller
/// shape is covered.
const RESULT_EXC_LOWERING_SCOPE: &[&str] = &[
    "pop_value",
    "store_local_value",
    "opcode_store_fast",
    "store_fast",
    "opcode_store_fast_store_fast",
    "store_fast_store_fast",
];

/// Dispatch-wrapper family rule: every `pyopcode::execute_*` wrapper —
/// the per-opcode `execute_<name>` free fns plus `execute_opcode_step`
/// itself — is scoped as a family.  The wrappers share one shape
/// (`executor.method(..)?; Ok(StepResult::..)`) and the dispatch arms
/// tail-forward their `Result`s, so scoping them one-by-one would leave
/// the dispatch graph returning raw `StepResult` on transformed arms
/// and `Result` shells on the rest — type-inconsistent for the
/// custom-match consumers (`eval_loop`, the portals).  The family lands
/// as a unit; the callee-side type gate
/// ([`tyref_is_result_of_pyerror`]) still filters non-`Result` returns.
fn in_execute_wrapper_family(name_path: &str, leaf: &str) -> bool {
    name_path.contains("pyopcode") && leaf.starts_with("execute_")
}

/// True when `name_path`'s leaf is a scoped callee.
pub(crate) fn in_result_exc_scope(name_path: &str) -> bool {
    let leaf = name_path.rsplit("::").next().unwrap_or(name_path);
    RESULT_EXC_LOWERING_SCOPE
        .iter()
        .any(|scoped| name_path == *scoped || leaf == *scoped)
        || in_execute_wrapper_family(name_path, leaf)
}

/// True when a call target's leaf names a scoped callee.
pub(crate) fn call_target_in_scope(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => RESULT_EXC_LOWERING_SCOPE.contains(&name.as_str()),
        CallTarget::FunctionPath { segments } => {
            let leaf = segments.last().map(String::as_str).unwrap_or("");
            RESULT_EXC_LOWERING_SCOPE.contains(&leaf)
                || (segments.iter().any(|s| s == "pyopcode") && leaf.starts_with("execute_"))
        }
        _ => false,
    }
}

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
fn result_ctor_kind(target: &CallTarget) -> Option<bool> {
    let CallTarget::SyntheticTransparentCtor { name, owner_path } = target else {
        return None;
    };
    if owner_path
        != &[
            "core".to_string(),
            "result".to_string(),
            "Result".to_string(),
        ]
    {
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
        // The shell must flow out through this block's single
        // unconditional exit, and through nothing else.
        if graph.blocks[bi].exits.len() != 1 || graph.blocks[bi].exitswitch.is_some() {
            return Err(format!(
                "{}: block {bi} Result shell block has a conditional exit — \
                 unsupported shape",
                graph.name
            ));
        }
        let consumers = count_var_uses(graph, &ctor_var);
        // The shell's only op use is the `__pos_0` payload FieldWrite
        // base.  Its link uses are forwarding exit args, all in the single
        // exit asserted above: the monotonic lowering forwards the shell
        // once, but the framestate-threaded lowering can carry the same
        // value in several `mergeable` slots (a value occupying both a
        // locals and a stack cell appears once per slot in
        // `getoutputargs`), so `link_uses` may exceed 1.  Every slot
        // reaches the returnblock (verified below); the `Ok` rewrite
        // replaces every occurrence with the payload and the `Err` rewrite
        // discards the exit wholesale (`set_raise_values` → `set_goto`),
        // so multiple forwarding slots lower soundly.
        if consumers.op_uses != 1 || consumers.link_uses < 1 {
            return Err(format!(
                "{}: block {bi} Result shell has unexpected consumers \
                 (op_uses={}, link_uses={}) — expected the single __pos_0 \
                 payload FieldWrite and at least one forwarding exit arg",
                graph.name, consumers.op_uses, consumers.link_uses
            ));
        }
        // Verify the value reaches returnblock through pure forwarding.
        verify_forwards_to_returnblock(graph, bi, &ctor_var)?;

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
            // the `W_ExceptionObject` ref (`BH_LAST_EXC_VALUE`'s
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
        // A scoped callee whose body is `return f(...)?` where `f` is
        // outside [`RESULT_EXC_LOWERING_SCOPE`] (e.g. a trait method
        // such as `OpcodeStepExecutor::return_value`): the optimised MIR
        // folds the `?`-diamond into a direct forward of the callee's
        // `Result` to `returnblock`, leaving no `Ok`/`Err` shell to
        // rewrite, and the caller-rule capture never recorded the site
        // (the callee is unscoped), so `tail_forwarded_returns` is 0.
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
fn op_operand_vars(kind: &OpKind) -> Vec<Variable> {
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
        | OpKind::LoadStatic { .. } => Vec::new(),

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

/// Verify `var` flows from `from_block`'s single exit through pure
/// positional forwarding (single-exit blocks re-exporting the value)
/// until it lands in `returnblock`.
fn verify_forwards_to_returnblock(
    graph: &FunctionGraph,
    from_block: usize,
    var: &Variable,
) -> Result<(), String> {
    let mut current = from_block;
    let mut tracked = var.clone();
    // Bounded by block count — forwarding chains cannot loop without
    // revisiting a block, which the bound rejects.
    for _ in 0..graph.blocks.len() {
        let block = &graph.blocks[current];
        // Every hop after `from_block` must be a pure forwarder — no
        // operations and no conditional exit — otherwise the value is
        // inspected or re-derived en route rather than carried untouched
        // to the returnblock, and deleting the shell / collapsing
        // `__pos_0` would be unsound.  `from_block` itself is the
        // producer (it keeps the ctor + `__pos_0` write, or the call),
        // so it is exempt.
        if current != from_block && (!block.operations.is_empty() || block.exitswitch.is_some()) {
            return Err(format!(
                "{}: block {current} on the Result-return forwarding chain \
                 is not a pure forwarder ({} ops, exitswitch present: {})",
                graph.name,
                block.operations.len(),
                block.exitswitch.is_some()
            ));
        }
        let [link] = block.exits.as_slice() else {
            return Err(format!(
                "{}: block {current} on the Result-return forwarding chain \
                 has {} exits — unsupported shape",
                graph.name,
                block.exits.len()
            ));
        };
        let Some(pos) = link
            .args
            .iter()
            .position(|a| matches!(a, LinkArg::Value(v) if *v == tracked))
        else {
            return Err(format!(
                "{}: Result shell var lost on the forwarding chain at block {current}",
                graph.name
            ));
        };
        if link.target == graph.returnblock {
            return Ok(());
        }
        let target = link.target.0;
        let next_block = &graph.blocks[target];
        let Some(next_var) = next_block.inputargs.get(pos) else {
            return Err(format!(
                "{}: forwarding target block {target} has no inputarg at \
                 position {pos}",
                graph.name
            ));
        };
        tracked = next_var.clone();
        current = target;
    }
    Err(format!(
        "{}: Result-return forwarding chain did not reach returnblock",
        graph.name
    ))
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
    results: &[Variable],
    enclosing_scoped: bool,
) -> Result<RewireOutcome, String> {
    let mut outcome = RewireOutcome {
        diamonds: 0,
        tail_forwards: 0,
        rewrapped: 0,
    };
    for r in results {
        match rewire_one_call_site(graph, r, enclosing_scoped)? {
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
                "{name}: tail-forwards a scoped callee's Result out of an \
                 unscoped graph — add it to RESULT_EXC_LOWERING_SCOPE or \
                 the callers' discriminant switches read garbage"
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
        catch_and_rewrap(graph, a, r)?;
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
/// an `Ok` shell, the exception arm binds the caught `W_ExceptionObject`
/// back into the `PyError` domain (`PyError::from_exc_object`, the
/// inverse of the callee rule's `to_exc_object`) and wraps it in an
/// `Err` shell.  This is the same erasure boundary the residual-call
/// ABI implements at host calls (`Ok` → value, `Err` →
/// `BH_LAST_EXC_VALUE`), value-encoded again one block later.  The
/// rebuilt shells sit in the CALLER's graph next to the sibling shells
/// the caller already builds for its own returns, so no new shell
/// exposure is introduced on walked paths.
fn catch_and_rewrap(graph: &mut FunctionGraph, a: usize, r: &Variable) -> Result<(), String> {
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
        Some(build_shell(graph, n_id, "Ok", payload))
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
        Some(build_shell(graph, e_id, "Err", v_err))
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
) -> Variable {
    use crate::model::FieldDescriptor;
    let owner = format!("core::result::Result::{variant}");
    let shell = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor_with_owner(
                    ["core", "result", "Result"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
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
fn back_substitute(
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
fn follow_single_exit(
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
fn assert_single_pred(graph: &FunctionGraph, block: usize, name: &str) -> Result<(), String> {
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
fn split_diamond_exits(exits: &[Link], name: &str) -> Result<(Link, Link), String> {
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
fn assert_block_pure_besides(
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
    verify_forwards_to_returnblock(graph, e_block, &residual_result)
}

/// In the continue-arm target, the `__pos_0` read off the inputarg at
/// `pos` collapses: the inherited value already *is* the payload.
/// Deletes the read and renames its result to the inputarg.
fn collapse_pos0_read(
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
