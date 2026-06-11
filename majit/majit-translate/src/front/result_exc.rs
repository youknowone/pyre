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
//! conformance scan lands, [`RESULT_EXC_LOWERING_SCOPE`] pins the
//! callee set; every call site of a scoped callee either matches the
//! `?`-diamond or fails the build loudly (custom `match` handlers
//! need the caught value bound from `last_exc_value` — a later
//! widening).

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::TyRef;

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType};

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

/// True when `name_path`'s leaf is a scoped callee.
pub(crate) fn in_result_exc_scope(name_path: &str) -> bool {
    RESULT_EXC_LOWERING_SCOPE
        .iter()
        .any(|leaf| name_path == *leaf || name_path.ends_with(&format!("::{leaf}")))
}

/// True when a call target's leaf names a scoped callee.
pub(crate) fn call_target_in_scope(target: &CallTarget) -> bool {
    let leaf = match target {
        CallTarget::Method { name, .. } => name.as_str(),
        CallTarget::FunctionPath { segments } => segments.last().map(String::as_str).unwrap_or(""),
        _ => return false,
    };
    RESULT_EXC_LOWERING_SCOPE.contains(&leaf)
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
                fieldwrite_idx = Some((i, value.clone()));
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
        let consumers = count_var_uses(graph, &ctor_var);
        // ctor result + FieldWrite base + one link arg.
        if consumers.link_uses != 1 || consumers.op_uses != 1 {
            return Err(format!(
                "{}: block {bi} Result shell has unexpected consumers \
                 (op_uses={}, link_uses={}) — only the payload FieldWrite \
                 and one exit arg are supported",
                graph.name, consumers.op_uses, consumers.link_uses
            ));
        }
        if graph.blocks[bi].exits.len() != 1 || graph.blocks[bi].exitswitch.is_some() {
            return Err(format!(
                "{}: block {bi} Result shell block has a conditional exit — \
                 unsupported shape",
                graph.name
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
        return Err(format!(
            "{}: scoped Result-of-PyError callee with no rewritable returns",
            graph.name
        ));
    }
    Ok(rewritten)
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

/// Operand variables of an op kind, restricted to the kinds the
/// Result-shell pattern can contain.  Every other kind returns its
/// operands through the generic arms below.
fn op_operand_vars(kind: &OpKind) -> Vec<Variable> {
    match kind {
        OpKind::Call { args, .. } => args.clone(),
        OpKind::FieldWrite { base, value, .. } => vec![base.clone(), value.clone()],
        OpKind::FieldRead { base, .. } => vec![base.clone()],
        OpKind::BinOp { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
        OpKind::UnaryOp { operand, .. } => vec![operand.clone()],
        _ => Vec::new(),
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
}

/// Caller rule.  `results` are the result `Variable`s of calls to
/// scoped callees (captured during lowering).  Each must sit at the
/// head of a `Try::branch` diamond — rewired into
/// `ExitSwitch::LastException` exits — or tail-forward to
/// `returnblock` inside a scoped enclosing graph.
pub(crate) fn rewire_result_exc_call_sites(
    graph: &mut FunctionGraph,
    results: &[Variable],
    enclosing_scoped: bool,
) -> Result<RewireOutcome, String> {
    let mut outcome = RewireOutcome {
        diamonds: 0,
        tail_forwards: 0,
    };
    for r in results {
        if rewire_one_call_site(graph, r, enclosing_scoped)? {
            outcome.diamonds += 1;
        } else {
            outcome.tail_forwards += 1;
        }
    }
    Ok(outcome)
}

/// Returns `true` for a rewired diamond, `false` for a (no-op)
/// tail-forward.
fn rewire_one_call_site(
    graph: &mut FunctionGraph,
    r: &Variable,
    enclosing_scoped: bool,
) -> Result<bool, String> {
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
        return Ok(false);
    }
    let (b, r_b) =
        follow_single_exit(graph, a, r).map_err(|e| format!("{name}: call block exit: {e}"))?;
    assert_single_pred(graph, b, &name)?;
    // Block B: `cf = Result::branch(r)`.
    let branch_op_idx = graph.blocks[b]
        .operations
        .iter()
        .position(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, args, .. }
                    if name == "branch" && args.as_slice() == std::slice::from_ref(&r_b)
            )
        })
        .ok_or_else(|| {
            format!(
                "{name}: block {b} after a scoped call does not start a \
                 Try::branch diamond — non-`?` use of a scoped callee's \
                 Result (custom match handlers are not supported yet)"
            )
        })?;
    let cf = graph.blocks[b].operations[branch_op_idx]
        .result
        .clone()
        .ok_or_else(|| format!("{name}: branch() without result var"))?;
    let (c, cf_c) =
        follow_single_exit(graph, b, &cf).map_err(|e| format!("{name}: branch block exit: {e}"))?;
    assert_single_pred(graph, c, &name)?;
    // Block C: `d = cf.__discriminant`; switch d {0 → continue, 1 → break}.
    let disc_var = graph.blocks[c]
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::FieldRead { base, field, .. }
                if *base == cf_c && field.name == "__discriminant" =>
            {
                op.result.clone()
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
    Ok(true)
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
    let payload_var = ops.iter().find_map(|op| match &op.kind {
        OpKind::FieldRead { base, field, .. } if *base == cf_e && field.name == "__pos_0" => {
            op.result.clone()
        }
        _ => None,
    });
    let Some(payload_var) = payload_var else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the __pos_0 residual read — \
             custom `?` handler shapes are not supported yet"
        ));
    };
    let residual_result = ops.iter().find_map(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            args,
            ..
        } if m == "from_residual" && args.as_slice() == std::slice::from_ref(&payload_var) => {
            op.result.clone()
        }
        _ => None,
    });
    let Some(residual_result) = residual_result else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the from_residual call — \
             custom `?` handler shapes are not supported yet"
        ));
    };
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
