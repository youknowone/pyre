//! Per-opcode specialization strategies: the `try_walker_*` entry points
//! that recognize a specializable shape (int/long/float arithmetic and
//! comparisons, attribute and method loads, container builds and subscript,
//! list-append, exception construction/raise, for-iter, slice, and the
//! module/name cell folds) and either fold or record a specialized trace,
//! returning `None` to fall through to the generic path.
//!
//! **Parity:** pyre-local trace-time folding. PyPy defers most
//! specialization to `optimizeopt/` (a separate later pass); pyre folds
//! during the walk instead. The fast-path shapes still mirror the
//! `opimpl_*` fast paths and `blackhole.py`'s `bhimpl_*` folds.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. The shared walker
//! primitives these build on (unbox/box, guard emission, operand reads)
//! stay in `mod.rs`; the specialization opname arms stay in `handle` and
//! call into these entry points.

use super::*;

/// `residual_call` shape `iRd>X` dispatcher. Reads `funcptr (i)`,
/// R-list args, and `descr`, runs `_build_allboxes` to produce the
/// callee's ABI-ordered arglist, classifies the call by `EffectInfo`
/// via [`select_residual_call_opcode`], records the matching
/// kind-coded `CallMayForce*` / `CallLoopinvariant*` / `CallPure*` /
/// `Call*` op, emits `GUARD_NOT_FORCED` on the forces branch, emits
/// `GUARD_NO_EXCEPTION` if the classification says `can_raise`, and
/// writes the recorded result OpRef into the dst register chosen by
/// `dst_bank`.
///
/// RPython parity: `pyjitpl.py:1334-1336 _opimpl_residual_call1` →
/// `do_residual_or_indirect_call` → `do_residual_call`
/// (pyjitpl.py:1995-2127). `pyjitpl.py:1346 opimpl_residual_call_r_i =
/// _opimpl_residual_call1` and `:1347 opimpl_residual_call_r_r =
/// _opimpl_residual_call1` confirm both kind variants share the
/// `_call1` body. The `_X` suffix is the *call's return kind* — mapping
/// comes from `do_residual_call`'s `descr.get_normalized_result_type()`
/// dispatch (pyjitpl.py:2022-2044) and `select_residual_call_opcode`'s
/// kind-keyed opcode tables.
///
/// `dst_bank` selects where the call's result lands:
/// * `'r'`: caller's `registers_r[dst]` — Ref-typed `Call*` family
///   (`_r_r/iRd>r`, `pyjitpl.py:1347 opimpl_residual_call_r_r`).
/// * `'i'`: caller's `registers_i[dst]` — Int-typed `Call*` family
///   (`_r_i/iRd>i`, `pyjitpl.py:1346 opimpl_residual_call_r_i`).
/// * `'v'`: void return — operand layout drops the trailing `>X` byte and
///   the writeback no-ops (`_r_v/iRd`, `pyjitpl.py:1348
///   opimpl_residual_call_r_v`, `blackhole.py:1245 bhimpl_residual_call_r_v`).
/// (`'f'` is intentionally absent: RPython does not exec-generate
/// `opimpl_residual_call_r_f`. The only float-result residual_call
/// shape is `_irf_f/iIRFd>f`, dispatched by
/// [`dispatch_residual_call_iIRFd_kind`].)
///
/// TODO: walker selects the IR opcode by EffectInfo
/// branch (`CallMayForce*` for forces, `CallLoopinvariant*` for
/// loop-invariant, `CallPure*` for elidable, otherwise `Call*`) via
/// [`select_residual_call_opcode`]. Two sub-cases route through
/// dedicated helpers before the selector:
///   - **release-gil** ([`direct_call_release_gil`], `pyjitpl.py:3671-
///     3681`) — early-return when `ei.is_call_release_gil()`,
///     reshapes the arglist to `[savebox, funcbox] + argboxes[1:]`
///     and records `CALL_RELEASE_GIL_*` instead of `CALL_MAY_FORCE_*`.
///   - **loop-invariant heapcache** ([`loopinvariant_lookup`] /
///     [`loopinvariant_now_known`], `pyjitpl.py:2088 + 2109`) —
///     short-circuits the record on a heapcache hit and populates
///     the cache after a fresh record.
///
/// Emits `GUARD_NOT_FORCED` on the forces path plus
/// `GUARD_NO_EXCEPTION` whenever `check_can_raise(False)` is true,
/// matching `pyjitpl.py:2078-2082`. After every recorded call op,
/// invalidates the heapcache via
/// `heap_cache.invalidate_caches_varargs(call_opcode, ei, allboxes)`
/// matching `pyjitpl.py:2659 _record_helper_varargs` parity (forces
/// branch's `pyjitpl.py:2072` redundantly invalidates with
/// `CALL_MAY_FORCE_*`, equivalent because `select_residual_call_opcode`
/// returns `CallMayForce*` for the forces classification).  Release-gil
/// helper invalidates with `CALL_MAY_FORCE_*` matching
/// `pyjitpl.py:2072`'s `opnum1`. The pre-call vable IR bookkeeping
/// (`pyjitpl.py:2017 vable_and_vrefs_before_residual_call`, IR-only
/// portion: FORCE_TOKEN + SETFIELD_GC) is wired via
/// [`maybe_walker_vable_and_vrefs_before_residual_call`].  The
/// after-call helpers (`pyjitpl.py:3337-3366
/// vrefs_after_residual_call` / `vable_after_residual_call`) and the
/// runtime heap mutations on `tracing_before_residual_call` run in the
/// residual-call execution path — see
/// [`walker_vable_and_vrefs_before_residual_call`] for the IR-vs-heap
/// split rationale.  The `OS_NOT_IN_TRACE` check fires up front via
/// [`do_not_in_trace_call_result`] — fail-loud guard against future
/// silent TODOs once the `majit-translate` analyzer trio
/// populates `oopspecindex`.
///
/// Still missing relative to upstream `do_residual_call`, all blocked
/// on infrastructure absent from pyre-jit-trace today:
///   - `OS_JIT_FORCE_VIRTUAL` PTR_EQ + GUARD_VALUE prelude
///     (`pyjitpl.py:2011-2014 → 2153-2172 _do_jit_force_virtual`) —
///     walker is fail-loud here via [`do_jit_force_virtual_guard`]
///     (called from each `dispatch_residual_call_*` arm); a producer
///     that emits an `OopSpecIndex::JitForceVirtual` calldescr surfaces
///     `DispatchError::JitForceVirtualRequiresConcreteResolver` instead
///     of silently recording `CALL_MAY_FORCE_*` (this was the prior
///     behaviour and is documented as STRICTER-THAN-PYPY in
///     [`do_jit_force_virtual_guard`]'s docstring). Optimizer pass
///     `OptVirtualize::optimize_jit_force_virtual` (`virtualize.rs:1226`)
///     already handles the constant-token / non-null-forced short-circuit
///     post-trace. Adding the PTR_EQ + GUARD_VALUE prelude (the only
///     way to retire the fail-loud guard) is not yet implemented and
///     would land with the walker; metainterp has a tests-only
///     orthodox port at
///     `majit-metainterp/src/pyjitpl.rs:11828 _do_jit_force_virtual`
///     that the converged walker would route through. Production reach
///     today is zero — `jtransform.rs:1903 jit.force_virtual` is the only
///     producer and pyre's interpreter does not emit it.
///   - `vrefs_after_residual_call` is unported; no `jit.virtual_ref`
///     producers exist today, so the upstream loops are empty. Vable forces
///     are detected by the residual-call execution path's heap-token bracket.
///   - `direct_libffi_call` (`pyjitpl.py:3622-3667`) — pyre's live
///     tracer also returns `None` from this helper unless a
///     `CIF_DESCRIPTION_P` parser + dynamic `calldescr` builder lands
///     (`majit-metainterp/src/pyjitpl.rs:11487-11491` defers to
///     direct_call_release_gil/may_force, which is the same fall-through
///     the walker already takes).
///   - `direct_assembler_call` (`pyjitpl.py:3589-3609`) + KEEPALIVE
///     (`pyjitpl.py:2080-2081`) — only fire when `assembler_call=True`
///     in `do_residual_call`. Walker's residual_call dispatchers are
///     never called with `assembler_call=True`; the parallel
///     `inline_call_*/dR>X` family routes through
///     [`dispatch_inline_call_dr_kind`] instead. Adding the path would
///     require the codewriter to emit a new `assembler_call` shape, not
///     a walker-side change.
///   - Per-PC liveness narrowing for the snapshot that
///     `walker_capture_snapshot_for_last_guard` attaches
///     (`pyjitpl.py:218-225 _get_list_of_active_boxes`). Walker's
///     helper today snapshots every non-`OpRef::NONE` register across
///     all three banks; RPython narrows the box list via
///     `jitcode.get_live_vars_info(pc, op_live)` so dead registers are
///     pruned before the snapshot.  The walker has no `op_live` byte
///     reader plumbed through `SubJitCodeBody` yet — follow-up
///     once the codewriter exposes the per-PC liveness table on the
///     callee body slice.  Over-capture is correctness-preserving:
///     `store_final_boxes_in_guard` filters dead boxes from the
///     snapshot via the optimizer's liveness pass.
/// STORE_SUBSCR strategy-aware walker specialization gate.  Returns
/// `Some(DispatchOutcome::Continue)` if
/// the residual_call was specialized into the trait-equivalent
/// `guard_class + guard_list_strategy + setarrayitem-family` shape;
/// `None` to fall through to the existing blackbox CallN path.
///
/// Gates (all must hold):
/// 1. `dst_bank == 'v'` (STORE_SUBSCR returns void; trait emit is `Void`).
/// 2. `r_args.len() == 3` (codewriter emits `[obj_reg, key_reg, value_reg]`).
/// 3. Runtime funcptr matches `WalkContext.store_subscr_fn_addr`, or the
///    `PYRE_WALKER_STORE_SUBSCR_FNADDR` fallback when no entry address was
///    threaded through the production dispatch path.
/// 4. All 3 concrete shadow slots (`concrete_registers_r[r_args[0..3]]`)
///    are `ConcreteValue::Ref(_)`.
/// 5. `generated_store_subscr_value` returns `true` (object is a list
///    with int key, strategy-detectable value, in-bounds index — see
///    `codegen.rs:3146-3168 generated_store_subscr_value` for the
///    detail criteria mirroring `jtransform do_resizable_list_setitem`).
///
/// Decline (any gate `false`) → `None` → dispatcher falls through to
/// `try_execute_residual_call_via_executor` which concrete-executes the
/// helper and records the blackbox `CallMayForce*` IR.  No-op for
/// non-STORE_SUBSCR residual calls.
pub(crate) fn try_walker_store_subscr_specialization<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    funcptr: OpRef,
    r_args: &[OpRef],
    dst_bank: char,
) -> Option<DispatchOutcome> {
    if dst_bank != 'v' || r_args.len() != 3 {
        return None;
    }
    // Prefer `WalkContext.store_subscr_fn_addr`, populated from
    // `cpu.store_subscr_fn` at production entry; fall back to
    // `PYRE_WALKER_STORE_SUBSCR_FNADDR` for test fixtures and runtime
    // overrides.
    let expected_fn_addr = if let Some(addr) = ctx.store_subscr_fn_addr {
        addr
    } else {
        let s = std::env::var_os("PYRE_WALKER_STORE_SUBSCR_FNADDR")?;
        let s = s.to_str()?;
        parse_hex_or_decimal_usize(s)?
    };
    let funcptr_addr = ctx.trace_ctx.box_value(funcptr).and_then(|v| match v {
        majit_ir::Value::Int(n) => Some(n as usize),
        _ => None,
    })?;
    if funcptr_addr != expected_fn_addr {
        return None;
    }
    let r_args_concrete = read_ref_var_list_concrete(code, op, 1, ctx);
    let concrete_obj = match r_args_concrete.first()? {
        crate::state::ConcreteValue::Ref(p) => *p,
        _ => return None,
    };
    let concrete_key = match r_args_concrete.get(1)? {
        crate::state::ConcreteValue::Ref(p) => *p,
        _ => return None,
    };
    let concrete_value = match r_args_concrete.get(2)? {
        crate::state::ConcreteValue::Ref(p) => *p,
        _ => return None,
    };
    let handled = crate::generated_store_subscr_value(
        ctx,
        r_args[0],
        r_args[1],
        r_args[2],
        concrete_obj,
        concrete_key,
        concrete_value,
    );
    if !handled {
        return None;
    }
    // The helper call below mutates the list; log the displaced element
    // first so a non-committing walk's legacy replay re-executes against
    // the pre-walk heap (see `FBW_STORE_JOURNAL`).  `handled` means
    // `generated_store_subscr_value` admitted an exact in-bounds
    // list[int] store, so the displaced read resolves.  The boxing
    // allocation inside `w_list_getitem` can move the operands, so
    // re-read the forwarded refs from the shadow afterwards.
    let (concrete_obj, concrete_key, concrete_value) = {
        let index = unsafe { pyre_object::w_int_get_value(concrete_key) };
        let Some(displaced) = (unsafe { pyre_object::w_list_getitem(concrete_obj, index) }) else {
            unreachable!(
                "store_subscr specialization: in-bounds index {index} has no element \
                 (generated_store_subscr_value admitted it)"
            );
        };
        let r_args_concrete = read_ref_var_list_concrete(code, op, 1, ctx);
        let (
            Some(crate::state::ConcreteValue::Ref(obj)),
            Some(crate::state::ConcreteValue::Ref(key)),
            Some(crate::state::ConcreteValue::Ref(value)),
        ) = (
            r_args_concrete.first(),
            r_args_concrete.get(1),
            r_args_concrete.get(2),
        )
        else {
            unreachable!(
                "store_subscr specialization: operand concrete vanished from the \
                 shadow across the displaced-element boxing"
            );
        };
        fbw_store_journal_push(*obj, *key, displaced);
        (*obj, *key, *value)
    };
    // Specialized IR recorded.  Heap mutation: invoke the helper
    // concretely so the next read of the container sees the updated
    // value.  `bh_store_subscr_fn(obj, key, value) -> i64` returns 1 on
    // success, 0 on raise (with the exception object stashed in
    // `BH_LAST_EXC_VALUE`).
    let success = unsafe {
        let store_subscr_fn: extern "C" fn(i64, i64, i64) -> i64 =
            std::mem::transmute(expected_fn_addr as *const ());
        store_subscr_fn(
            concrete_obj as usize as i64,
            concrete_key as usize as i64,
            concrete_value as usize as i64,
        )
    };
    if success == 0 {
        // `pyjitpl.py:2156-2168 handle_possible_exception` parity: drain
        // the helper's stashed exception into `ctx.last_exc_value*`,
        // record `GuardException` against the specialized IR, and
        // surface `SubRaise` so the caller doesn't fall through to the
        // generic residual-call path (which would re-record a second IR
        // call against the same opcode position).
        let bh_exc = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| {
            let v = c.get();
            c.set(0);
            v
        });
        if bh_exc != 0 {
            let exc = ctx.trace_ctx.const_ref(bh_exc);
            let exc_concrete = ConcreteValue::Ref(bh_exc as usize as pyre_object::PyObjectRef);
            ctx.last_exc_value = Some(exc);
            ctx.last_exc_value_concrete = exc_concrete;
            ctx.fbw_mode.class_of_last_exc_is_const = false;
            walker_record_guard_exception(ctx, op.pc);
            let exc = ctx
                .last_exc_value
                .expect("GuardException must bind the raised exception box");
            return Some(DispatchOutcome::SubRaise { exc, exc_concrete });
        }
        // Defensive: helper returned 0 but did not stash an exception.
        // Decline specialization so the generic path's
        // `execute_residual_call` decides the dispatch.
        return None;
    }
    // pyjitpl.py:2659 `_record_helper_varargs`: STORE_SUBSCR mutates the
    // heap; the specialized IR shape's setarrayitem_gc ops already
    // invalidate per-descr via the recorder, so no further explicit
    // heap-cache invalidation is needed here.
    Some(DispatchOutcome::Continue)
}

/// #124: walker-native truth specialization for the `truth_fn` residual
/// (oopspec [`majit_ir::PyreHelperKind::Truth`]).  When the sole Ref operand
/// is a concrete boxed `W_IntObject` (excluding `W_BoolObject`, which shares
/// the `intval: i64` layout but carries a distinct `BOOL_TYPE` `ob_type`, so
/// the emitted `GUARD_CLASS INT` would not match it), unbox it
/// (`GUARD_CLASS INT` + `getfield intval`) and record `int_is_true`, stamping
/// the folded concrete truth.  Returns the raw truth `OpRef` on success;
/// `None` when the operand is not a concrete non-bool int — the caller then
/// falls through to the generic may-force residual, preserving `__bool__` /
/// `__len__` semantics.
///
/// Eliding the `CALL_MAY_FORCE` here also removes its `GUARD_NOT_FORCED` /
/// `GUARD_NO_EXCEPTION`, whose kept-stack blackhole resume reads NULL peeled
/// outer-Label slots in the short-circuit value-context shape
/// (`(i % 7) and ...`).
pub(crate) fn try_walker_specialize_truth_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
) -> Result<Option<OpRef>, DispatchError> {
    let Some(obj) = walker_concrete_ref_object(ctx, operand) else {
        return Ok(None);
    };
    let val = unsafe {
        if !pyre_object::is_int(obj) || pyre_object::is_bool(obj) {
            return Ok(None);
        }
        pyre_object::w_int_get_value(obj)
    };
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let raw = walker_unbox_int(ctx, op_pc, operand, int_type_addr)?;
    let truth = ctx.trace_ctx.record_op(OpCode::IntIsTrue, &[raw]);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int((val != 0) as i64));
    Ok(Some(truth))
}

/// #57: walker-native speculative int specialization for the `BINARY_OP`
/// helper residual_call (oopspec `BinaryOp`).  Re-derives
/// the former int fast path's structure (`guard_class` + `getfield_gc_i` per
/// operand, `int_OP_ovf` + `guard_no_overflow`, `wrapint`) walker-native rather
/// than calling back into the retired trait path (which would alias the
/// reborrowed sym slices and emit `MIFrame`-style snapshots inconsistent with
/// the walker model).
///
/// The concrete boxed result is obtained from the same
/// `execute_residual_call` path the generic leg uses, so
/// `concrete_registers_r[dst]` holds the authentic runtime `W_IntObject`.
///
/// Returns `Ok(Some(()))` when the specialization was emitted (caller
/// returns `Continue`); `Ok(None)` when the operator is deferred
/// (FloorDiv / Mod / Shift / TrueDiv / Power / Subscr), the operands are
/// not both concrete `W_IntObject`, or the helper raises — the caller
/// then falls through to the generic `CallMayForce` record so the
/// Python-level `__op__` semantics are preserved.
pub(crate) fn try_walker_specialize_binary_op_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(bin_op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::BinaryOperator;
    // INT_BINOP_TABLE → (OpCode, has_overflow, needs_concrete_check).
    // Defer TrueDivide (int/int → float, separate helper) / Power /
    // Subscr to the generic leg (`_ => None`).
    let (op_code, has_overflow, needs_check) = match bin_op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => (OpCode::IntAddOvf, true, false),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => {
            (OpCode::IntSubOvf, true, false)
        }
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => {
            (OpCode::IntMulOvf, true, false)
        }
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => {
            (OpCode::IntFloorDiv, false, true)
        }
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => {
            (OpCode::IntMod, false, true)
        }
        BinaryOperator::And | BinaryOperator::InplaceAnd => (OpCode::IntAnd, false, false),
        BinaryOperator::Or | BinaryOperator::InplaceOr => (OpCode::IntOr, false, false),
        BinaryOperator::Xor | BinaryOperator::InplaceXor => (OpCode::IntXor, false, false),
        BinaryOperator::Lshift | BinaryOperator::InplaceLshift => (OpCode::IntLshift, false, true),
        BinaryOperator::Rshift | BinaryOperator::InplaceRshift => (OpCode::IntRshift, false, true),
        _ => return Ok(None),
    };

    // boolobject.py:74-76 descr_and/or/xor: when both operands are bool the
    // And/Or/Xor result is a bool (`space.newbool`), not an int.  The op runs
    // on the shared `intval` as for ints; only the boxing differs (picked
    // below).  `walker_concrete_ref_object` reads the same source as
    // `walker_int_specialization_operands`, so the flag stays consistent.
    let result_is_bool = matches!(op_code, OpCode::IntAnd | OpCode::IntOr | OpCode::IntXor)
        && match (
            walker_concrete_ref_object(ctx, r_args[0]),
            walker_concrete_ref_object(ctx, r_args[1]),
        ) {
            (Some(l), Some(r)) => unsafe { pyre_object::is_bool(l) && pyre_object::is_bool(r) },
            _ => false,
        };

    // Speculation gate + authentic boxed result (shared with COMPARE_OP).
    let Some((lhs, rhs, lhs_obj, rhs_obj, la, rb, boxed_result_i64)) =
        walker_int_specialization_operands(ctx, r_args, allboxes, call_descr)
    else {
        return Ok(None);
    };

    // intobject.py range validation (mirror the former int fast path's
    // needs_concrete_check): bail to the generic leg when the bare-IR-op
    // emission would be unsound (zero / INT_MIN-overflow divisor, oversized
    // / overflowing shift); large right-shift folds to a const.
    if needs_check {
        match op_code {
            OpCode::IntFloorDiv | OpCode::IntMod => {
                if rb == 0 || (la == i64::MIN && rb == -1) {
                    return Ok(None);
                }
            }
            OpCode::IntLshift => {
                // Don't specialize int `<<`: route to the generic (residual
                // BINARY_OP) leg, which carries the full intobject.py
                // descr_lshift semantics (promote to bignum on overflow, raise
                // ValueError on a negative count). A bare walker-native IntLshift
                // would be wrong — the trace is reused for any operands and x86
                // SHL masks the count mod 64 — and a *guarded* specialization
                // (range + round-trip guards, bail to bignum) crashes the
                // cranelift backend: when the lshift result is the loop variable
                // its box alternates small-int / bignum across the guard's
                // bridge boundary, and that trips a cranelift bridge bug (works
                // on dynasm). The generic leg handles the alternation correctly
                // on both backends.
                return Ok(None);
            }
            OpCode::IntRshift => {
                // A count >= LONG_BIT (or negative) folds to 0/-1 in
                // intobject.py:229-231, but that fold would be baked into the
                // reused trace and be wrong for an in-range count; route it to
                // the generic leg instead. An in-range recorded count is
                // specialized below behind a runtime range guard.
                let Ok(shift) = u32::try_from(rb) else {
                    return Ok(None);
                };
                if shift >= i64::BITS {
                    return Ok(None);
                }
            }
            _ => {}
        }
    }

    // --- emit the specialized IR (walker-native) ---
    // bool and int share `intval`; guard each operand against its own vtable
    // (BOOL_TYPE / INT_TYPE) so a bool unboxes through its own class.
    let (lhs_type, lhs_descr) = crate::state::int_or_bool_unbox_type_descr(lhs_obj);
    let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
    let lhs_raw = walker_unbox_int_typed(ctx, op_pc, lhs, lhs_type, lhs_descr)?;
    let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
    let (raw_result, concrete_value) = match op_code {
        OpCode::IntFloorDiv | OpCode::IntMod => {
            // rint.py:429/520 _ovf_zer guards: int_eq(rhs,0)→guard_false +
            // (lhs==INT_MIN)&(rhs==-1)→guard_false ahead of the elidable
            // helper call (so a re-used trace bails before the helper's
            // wrapping_div / wrapping_rem returns a wrap value).
            let rhs_zero = walker_int_eq_const(ctx, rhs_raw, 0, (rb == 0) as i64);
            walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[rhs_zero])?;
            let lhs_is_min = walker_int_eq_const(ctx, lhs_raw, i64::MIN, (la == i64::MIN) as i64);
            let rhs_is_neg_one = walker_int_eq_const(ctx, rhs_raw, -1, (rb == -1) as i64);
            let ovf_both = ctx
                .trace_ctx
                .record_op(OpCode::IntAnd, &[lhs_is_min, rhs_is_neg_one]);
            ctx.trace_ctx.set_opref_concrete(
                ovf_both,
                majit_ir::Value::Int(((la == i64::MIN) as i64) & ((rb == -1) as i64)),
            );
            walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[ovf_both])?;
            // jtransform.py:576-577 OS_INT_PY_DIV / OS_INT_PY_MOD elidable
            // residual call (call_typed_with_effect_pure → CallI patched via
            // record_result_of_call_pure).
            let (func_ptr, effect_info, concrete_result) = if op_code == OpCode::IntFloorDiv {
                (
                    majit_metainterp::blackhole::ll_int_py_div as *const (),
                    majit_metainterp::INT_PY_DIV_EFFECT_INFO,
                    majit_metainterp::blackhole::ll_int_py_div(la, rb),
                )
            } else {
                (
                    majit_metainterp::blackhole::ll_int_py_mod as *const (),
                    majit_metainterp::INT_PY_MOD_EFFECT_INFO,
                    majit_metainterp::blackhole::ll_int_py_mod(la, rb),
                )
            };
            let r = ctx.trace_ctx.call_typed_with_effect_pure(
                OpCode::CallI,
                func_ptr,
                &[lhs_raw, rhs_raw],
                &[majit_ir::Type::Int, majit_ir::Type::Int],
                majit_ir::Type::Int,
                effect_info,
                &[
                    majit_ir::Value::Int(func_ptr as usize as i64),
                    majit_ir::Value::Int(la),
                    majit_ir::Value::Int(rb),
                ],
                majit_ir::Value::Int(concrete_result),
            );
            (r, concrete_result)
        }
        OpCode::IntRshift => {
            // The machine SAR masks the count mod 64, so guard the count into
            // [0, LONG_BIT) — a reused trace bails rather than shifting by
            // `count & 63`. (The recorded count is < LONG_BIT here: a count
            // >= LONG_BIT const-folds to 0/-1 in the needs_check block above.)
            let in_range = walker_uint_lt_const(ctx, rhs_raw, i64::BITS as i64, 1);
            walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_range])?;
            let r = ctx
                .trace_ctx
                .record_op(OpCode::IntRshift, &[lhs_raw, rhs_raw]);
            (r, majit_metainterp::eval_binop_i(OpCode::IntRshift, la, rb))
        }
        _ => {
            let r = ctx.trace_ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
            (r, majit_metainterp::eval_binop_i(op_code, la, rb))
        }
    };
    ctx.trace_ctx
        .set_opref_concrete(raw_result, majit_ir::Value::Int(concrete_value));
    if has_overflow {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;
    }
    // A both-bool bitwise result boxes via `space.newbool` (boolobject.py:
    // 74-76) so it keeps the bool type; `boxed_result_i64` is already the
    // authentic W_Bool the forced residual produced.
    let (boxed, boxed_concrete) = if result_is_bool {
        (
            crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, raw_result, false),
            majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
        )
    } else {
        (
            walker_box_int(ctx, op_pc, raw_result, concrete_value)?,
            box_int_concrete(concrete_value, boxed_result_i64),
        )
    };
    ctx.trace_ctx.set_opref_concrete(boxed, boxed_concrete);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// Walker-native W_LongObject (bigint) arithmetic specialization for the
/// `BINARY_OP` helper residual_call (oopspec `BinaryOp`).  When both
/// operands are concrete `W_LongObject`, emit `GUARD_CLASS(LONG_TYPE)` per
/// operand + `GETFIELD_GC_PURE_R(value)` + a `CALL_PURE_R` to the elidable
/// `rbigint` payload helper (`long_binop_raw_helper`, `rbigint.py
/// @jit.elidable`) producing a bare Ref-typed bigint, then inline
/// `W_LongObject(...)` boxing via `new_with_vtable` + `setfield_gc('value')`.
/// Neither is the opaque
/// `CALL_MAY_FORCE` the generic leg records, so this sheds the per-iteration
/// force-token store + `GUARD_NOT_FORCED` + `GUARD_NO_EXCEPTION` from
/// bigint-heavy loops (e.g. `fib_loop`).
///
/// Specialized for add/sub/mul/and/or/xor (allocate → `EF_ELIDABLE_OR_MEMORYERROR`)
/// and floordiv/mod/lshift/rshift (`EF_ELIDABLE_CAN_RAISE`); both classes have
/// `check_can_raise()` true, so every op carries a trailing `GUARD_NO_EXCEPTION`.
/// True-divide has its own float fast path
/// ([`try_walker_specialize_truediv_op_long`]); pow and any non-`W_LongObject`
/// operand return `Ok(None)` so the caller falls through to the generic record,
/// preserving the `__op__` semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(spec) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag)
        .and_then(crate::trace_opcode::long_binop_raw_helper)
    else {
        return Ok(None);
    };
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(lhs_obj) && pyre_object::is_long(rhs_obj) } {
        return Ok(None);
    }
    // Authentic boxed result via the same execute path the int leg uses; a
    // NULL / raised result defers to the generic record.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    // Pyre representation demote: when the bigint result fits i64 it becomes a
    // W_IntObject in pyre's two-class int model, which the inline-NEW long box
    // cannot represent — so decline the spec here (before emitting any op) and
    // let the generic record handle the demote. Reuse the authentic boxed
    // result's payload instead of running `spec.raw_fn` a second time; the raw
    // helpers allocate/publish exception state and must not be used as a
    // trace-time probe.
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL || unsafe { pyre_object::is_int(boxed_result_obj) }
    {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_long(boxed_result_obj) } {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };
    let fits_concrete = 0_i64;
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    // Read each operand's immutable `value` payload (`GetfieldGcPure`), then call
    // the elidable `rbigint` op on the bare `*const BigInt` payloads. Passing the
    // payloads (not the wrappers) keeps the call pure on the immutable bigints, so
    // the optimizer forwards the field read and never reorders this elidable call
    // ahead of the boxing `setfield_gc` below — which would otherwise read the
    // freshly-allocated result wrapper's uninitialized `value` (the function-loop
    // unroll exposed exactly that reorder). The result is a GC-managed
    // `*mut BigInt`, Ref-typed so the JIT gcmap roots it across the collecting
    // boxing NEW. Every op allocates (`EF_ELIDABLE_OR_MEMORYERROR`) or divides
    // (`EF_ELIDABLE_CAN_RAISE`), so a trailing `GuardNoException` follows
    // (`pyjitpl.py:2110-2112`).
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let lhs_payload = unsafe { *((lhs_obj as *const u8).add(off) as *const i64) };
    let rhs_payload = unsafe { *((rhs_obj as *const u8).add(off) as *const i64) };
    let lhs_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcPureR,
        &[lhs],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        lhs_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
    );
    let rhs_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcPureR,
        &[rhs],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        rhs_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_payload as usize)),
    );
    let add_fn = spec.payload_fn as *const ();
    let concrete_args = [
        majit_ir::Value::Int(add_fn as usize as i64),
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_payload as usize)),
    ];
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        add_fn,
        &[lhs_pl, rhs_pl],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::Type::Ref,
        spec.effect,
        &concrete_args,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }
    // `newlong` demote guard: `GuardFalse(fits_int(raw))`. Passes at record time
    // (checked above), deopts to the interpreter if a future replay yields an
    // i64-fitting result. Resumes at op_pc (the BINARY_OP), like the GuardClass
    // guards.
    let fits_fn = pyre_object::longobject::jit_bigint_fits_int as *const ();
    let fits = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallI,
        fits_fn,
        &[raw],
        &[majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::cannot_raise_effect_info(),
    );
    ctx.trace_ctx
        .set_opref_concrete(fits, majit_ir::Value::Int(fits_concrete));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[fits])?;
    // Inline `W_LongObject(raw)` NEW (`new_with_vtable` + `setfield_gc('value')`).
    // NewWithVtable lowers to the collecting `CallMallocNursery` — the GC
    // safepoint that lets bigint-heavy loops reclaim dead bigints.
    let result = crate::helpers::emit_box_long_inline(
        ctx.trace_ctx,
        raw,
        crate::descr::w_long_size_descr(),
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// W_LongObject true-divide specialization — the float analogue of
/// [`try_walker_specialize_binary_op_long`].  Both operands are `int`-typed but
/// bigint-stored: guard each against `LONG_TYPE`, then `CallPureF` the elidable
/// `jit_w_long_truediv_raw` (correctly-rounded f64 quotient; raises
/// ZeroDivision/Overflow → `EF_ELIDABLE_CAN_RAISE` ⇒ trailing `GuardNoException`)
/// and box the f64 with `wrapfloat` (transparent `new_with_vtable` +
/// `setfield_gc_f`, the trace analogue of `_truediv`'s `space.newfloat(f)`), so a
/// downstream float op keeps the quotient unboxed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_truediv_op_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
        Some(BinaryOperator::TrueDivide) | Some(BinaryOperator::InplaceTrueDivide) => {}
        _ => return Ok(None),
    }
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(lhs_obj) && pyre_object::is_long(rhs_obj) } {
        return Ok(None);
    }
    // Authentic boxed float via the generic execute path; a NULL / raised result
    // (zero divisor, float overflow) defers to the generic record.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    // Pure `rbigint.truediv` → correctly-rounded f64 (CallPureF). The op already
    // ran authentically above, so the divisor is nonzero / non-overflowing here;
    // the trailing GuardNoException covers a divide-by-zero / overflow on replay.
    let truediv_fn =
        pyre_interpreter::objspace::descroperation::jit_w_long_truediv_raw as *const ();
    let f_concrete = pyre_interpreter::objspace::descroperation::jit_w_long_truediv_raw(
        lhs_obj as i64,
        rhs_obj as i64,
    );
    let concrete_args = [
        majit_ir::Value::Int(truediv_fn as usize as i64),
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_obj as usize)),
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_obj as usize)),
    ];
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallF,
        truediv_fn,
        &[lhs, rhs],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::Type::Float,
        majit_metainterp::ELIDABLE_EFFECT_INFO,
        &concrete_args,
        majit_ir::Value::Float(f_concrete),
    );
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(f_concrete));
    // pyjitpl.py:1946: no GuardNoException when the pure call folded to a Const.
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }
    // Box the f64 with the transparent float NEW (`new_with_vtable` +
    // `setfield_gc_f`), mirroring `space.newfloat(f)`.
    let result = crate::state::wrapfloat(ctx.trace_ctx, raw);
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// FBW fold of the UNPACK_SEQUENCE two-residual lowering (`unpack_sequence_fn`
/// validator + per-index `unpack_item_fn` reader emitted by the codewriter
/// UNPACK_SEQUENCE arm) for an arity-2 specialised int tuple: guard the
/// `spec_ii` class once, then read `value0` / `value1` directly
/// (`getfield_gc_pure_i` + `wrapint`) so the unpacked items stay unboxed ints
/// through the downstream BINARY_OP int fold — the walker analogue of the
/// retired MIFrame `W_SpecialisedTupleObject_ii` value0/value1 reads. Returns
/// `Ok(Some(()))` when folded (the caller returns `Continue`); `Ok(None)` to
/// fall through to the opaque residual record, which stays correct for any
/// other shape — so a non-foldable sequence is not declined.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_unpack<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::PyreHelperKind,
    i_args: &[OpRef],
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let (Some(&int_arg), Some(&seq)) = (i_args.first(), r_args.first()) else {
        return Ok(None);
    };
    let Some(majit_ir::Value::Int(int_val)) = ctx.trace_ctx.box_value(int_arg) else {
        return Ok(None);
    };
    let Some(concrete_seq) = walker_concrete_ref_object(ctx, seq) else {
        return Ok(None);
    };
    // Only the arity-2 int specialised tuple is folded today; any other shape
    // falls through to the opaque residual (correct, slower).
    let spec_ii = &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE
        as *const pyre_object::pyobject::PyType;
    if !std::ptr::eq(unsafe { (*concrete_seq).ob_type }, spec_ii) {
        return Ok(None);
    }
    match helper {
        majit_ir::PyreHelperKind::UnpackSequence => {
            // `spec_ii` is always arity 2, so the class guard subsumes the
            // exact-length check `unpack_sequence_fn` performs.
            if int_val != 2 {
                return Ok(None);
            }
            if !ctx.trace_ctx.heap_cache().is_class_known(seq) {
                let type_const = ctx.trace_ctx.const_int(spec_ii as i64);
                ctx.trace_ctx
                    .record_guard(OpCode::GuardClass, &[seq, type_const], 0);
                walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
                ctx.trace_ctx
                    .heap_cache_mut()
                    .class_now_known(seq, spec_ii as i64);
            }
            // Pass `seq` through as the validated tuple; the per-index
            // `unpack_item_fn` reads below fold off it.
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, seq)?;
            Ok(Some(()))
        }
        majit_ir::PyreHelperKind::UnpackItem => {
            if !(0..2).contains(&int_val) {
                return Ok(None);
            }
            // Authentic boxed element (small-int caching / identity); fall
            // through to the opaque residual if it cannot be executed.
            let Some(elem_ptr) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
                return Ok(None);
            };
            // The class was already guarded by the partner `unpack_sequence_fn`
            // fold (its validated-tuple passthrough reg == `seq`).
            let descr = if int_val == 0 {
                crate::descr::specialised_tuple_ii_value0_descr()
            } else {
                crate::descr::specialised_tuple_ii_value1_descr()
            };
            let raw = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, seq, descr);
            let elem =
                unsafe { pyre_object::w_int_get_value(elem_ptr as pyre_object::PyObjectRef) };
            let boxed = walker_box_int(ctx, op_pc, raw, elem)?;
            ctx.trace_ctx
                .set_opref_concrete(boxed, box_int_concrete(elem, elem_ptr as i64));
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
            Ok(Some(()))
        }
        _ => Ok(None),
    }
}

/// `mapdict.py:1479-1537 LOAD_ATTR_caching` full-body-walker fast path for a
/// plain (non-method) instance attribute.  When the concrete receiver is a
/// monomorphic instance whose attribute resolves to a boxed plain storage slot
/// or an unboxed integer/float slot, emit the guarded read PyPy compiles
/// LOAD_ATTR to under the JIT —
///   * `guard_class(obj, &INSTANCE_TYPE)` — the receiver is a `W_ObjectObject`
///     (so the `map`/`storage` field reads below are valid; `mapdict.py:1495`
///     `if map is not None:` also filters non-instances at trace time).
///   * `guard_value(getfield_gc_i(w_type, version_tag), C_version_tag)` — pins
///     the class lookup result so a later descriptor or `__getattribute__`
///     mutation deopts on trace re-entry.
///   * `guard_value(getfield_gc_i(obj, map), C_map)` — `jit.promote(self.map)`
///     (`mapdict.py:905`); pins the exact instance shape so `find_map_attr`
///     const-folds `storageindex` to a green constant.
///   * boxed: `getfield_gc_r(obj, storage)` +
///     `getarrayitem_gc_r(block, C_index)` for
///     `mapdict.py:914-916 _mapdict_read_storage`;
///   * unboxed int/float: a non-forcing typed read plus `wrapint`/`wrapfloat`,
///     matching `_prim_direct_read` (mapdict.py:577-584, 600-601).
/// — instead of the opaque `getattr_fn` `CALL_MAY_FORCE` MRO-walk residual.
///
/// Returns `Some(())` after writing the dst; `None` (fall through to the
/// residual) for every shape [`load_attr_fast_path`] declines: non-instance
/// receiver, missing map, custom `__getattribute__`, uncacheable `version_tag`,
/// a data-descriptor / `INVALID` classification, or an attribute not on this
/// instance's map.  The map `guard_value` proves the attribute is present on
/// this shape, so a successful fold provably cannot raise `AttributeError` —
/// dropping the residual's exception guard is sound even in a handler-bearing
/// body (same reasoning as the LoadGlobal fold).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    // The receiver must be a concrete instance for the map/storageindex
    // resolution below; a non-concrete or non-instance receiver declines.
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    // Resolve the attribute name from the jitcode's own PyCode `co_names`
    // (mirrors `bh_load_attr_fn`; the codewriter passes the raw co_names index).
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    // `mapdict.py:1495-1533` resolution, returning the fold ingredients (the
    // read is left to the caller so it can be folded to a guarded inline read).
    if let Some((w_type, version_tag, map, storageindex)) = unsafe {
        pyre_interpreter::objspace::std::mapdict::load_attr_fast_path(concrete_obj, &name)
    } {
        walker_guard_mapdict_instance_shape(ctx, op_pc, obj, w_type, version_tag, map)?;

        // getfield_gc_r(obj, storage) + getarrayitem_gc_r(block, C_storageindex):
        // the inline value read (`mapdict.py:914-916`).  `storageindex` is a green
        // constant (the map guard pinned it); `trace_items_block_getitem_value`
        // stamps the dst's concrete shadow from the live block slot.
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            obj,
            crate::descr::object_storage_descr(),
        );
        let idx_const = ctx.trace_ctx.const_int(storageindex as i64);
        let value = crate::state::trace_items_block_getitem_value(ctx.trace_ctx, block, idx_const);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    if let Some((slot, kind, w_type, version_tag, stored)) = unsafe {
        pyre_interpreter::baseobjspace::exception_attr_slot_fold(concrete_obj, &name, false)
    } {
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args
            && unsafe { (*(stored as *const pyre_object::listobject::W_ListObject)).strategy }
                != pyre_object::listobject::ListStrategy::Object
        {
            return Ok(None);
        }
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let raw_value = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            obj,
            crate::descr::w_exception_slot_descr(kind, slot),
        );
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[raw_value])?;
        ctx.trace_ctx.set_opref_concrete(
            raw_value,
            majit_ir::Value::Ref(majit_ir::GcRef(stored as usize)),
        );
        let value = if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
            let list = unsafe { &*(stored as *const pyre_object::listobject::W_ListObject) };
            if list.strategy != pyre_object::listobject::ListStrategy::Object {
                return Ok(None);
            }
            let list_type = &pyre_object::LIST_TYPE as *const pyre_object::PyType as i64;
            if !ctx.trace_ctx.heap_cache().is_class_known(raw_value) {
                let type_const = ctx.trace_ctx.const_int(list_type);
                walker_emit_fold_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardClass,
                    &[raw_value, type_const],
                )?;
                ctx.trace_ctx
                    .heap_cache_mut()
                    .class_now_known(raw_value, list_type);
            }
            walker_guard_exact_w_class(
                ctx,
                op_pc,
                raw_value,
                pyre_object::get_instantiate(&pyre_object::LIST_TYPE),
            )?;
            let strategy = crate::state::opimpl_getfield_gc_i(
                ctx.trace_ctx,
                raw_value,
                crate::descr::list_strategy_descr(),
            );
            let object_strategy = ctx
                .trace_ctx
                .const_int(pyre_object::listobject::ListStrategy::Object as i64);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardValue,
                &[strategy, object_strategy],
            )?;
            let len = unsafe { pyre_object::w_list_len(stored) };
            let length = crate::state::opimpl_getfield_gc_i(
                ctx.trace_ctx,
                raw_value,
                crate::descr::list_length_descr(),
            );
            let len_const = ctx.trace_ctx.const_int(len as i64);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardValue,
                &[length, len_const],
            )?;
            let block = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                raw_value,
                crate::descr::list_items_descr(),
            );
            let mut items = Vec::with_capacity(len);
            let mut concrete_items = Vec::with_capacity(len);
            for index in 0..len {
                let index_op = ctx.trace_ctx.const_int(index as i64);
                items.push(crate::state::trace_items_block_getitem_value(
                    ctx.trace_ctx,
                    block,
                    index_op,
                ));
                concrete_items.push(
                    unsafe { pyre_object::w_list_getitem(stored, index as i64) }
                        .unwrap_or(pyre_object::PY_NULL),
                );
            }
            let tuple = crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &items);
            let concrete_tuple = pyre_object::w_tuple_new(concrete_items);
            ctx.trace_ctx.set_opref_concrete(
                tuple,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_tuple as usize)),
            );
            tuple
        } else {
            raw_value
        };
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    let Some((w_type, version_tag, map, storageindex, listindex, unbox_type)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::load_attr_unboxed_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    walker_guard_mapdict_instance_shape(ctx, op_pc, obj, w_type, version_tag, map)?;

    // `_prim_direct_read` (mapdict.py:600-601): read the raw longlong from the
    // shared list through a non-forcing, non-elidable residual.  Both indices
    // are green constants pinned by the map guard; keeping boxing in the trace
    // lets an immediate integer consumer virtualize it away.
    let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
    let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
    let live = unsafe {
        pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
            concrete_obj,
            storageindex,
            listindex,
        )
    };
    let boxed = match unbox_type {
        pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
            let raw = crate::helpers::emit_trace_call_int_typed(
                ctx.trace_ctx,
                crate::helpers::jit_mapdict_unboxed_read_raw as *const (),
                &[obj, storageindex_const, listindex_const],
                &[
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ],
            );
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(live));
            let boxed = walker_box_int(ctx, op_pc, raw, live)?;
            // The `wrapint` op is a heap box, so its concrete must be a heap ptr too:
            // box the raw longlong through the same `w_int_new` the unboxed read uses
            // (mapdict.py:579-584 `_box`); `box_int_concrete` re-homes a tagged small
            // int to a fresh heap `W_IntObject` so op(NewWithVtable) == concrete(heap).
            // Without this stamp the boxed result carries no concrete, so a downstream
            // eager void residual (e.g. the STORE_ATTR that writes `self.value`) cannot
            // resolve its value arg and the walk aborts `ResidualCallArgUnbound`.
            let live_ptr = pyre_object::w_int_new(live) as i64;
            ctx.trace_ctx
                .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
            boxed
        }
        pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
            let raw = crate::helpers::emit_trace_call_float_typed(
                ctx.trace_ctx,
                crate::helpers::jit_mapdict_unboxed_read_f as *const (),
                &[obj, storageindex_const, listindex_const],
                &[
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ],
            );
            let live_f = f64::from_bits(live as u64);
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(live_f));
            let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::w_float_new(live_f) as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// `callmethod.py:25-85 LOAD_METHOD` method-cache fold for the
/// codewriter's method-form `LOAD_ATTR` residual.  The safety oracle is the
/// interpreter's `load_method_fast_path`: it declines custom
/// `__getattribute__`, uncacheable types, non-function descriptors,
/// shadowing instance attributes, and non-instance receivers.  On success the
/// walker emits the guards that keep that decision stable, then writes
/// `w_descr` as a green constant so the following `CALL` can use the existing
/// constant-callee inline path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_method_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    if name.contains("__") {
        return Ok(None);
    }
    let Some((w_type, version_tag, w_descr)) =
        (unsafe { pyre_interpreter::load_method_fast_path(concrete_obj, &name) })
    else {
        return Ok(None);
    };
    if unsafe { resolve_inlinable_callee(w_descr) }.is_none() {
        return Ok(None);
    }
    let map = unsafe {
        let inst = &*(concrete_obj as *const pyre_object::W_ObjectObject);
        inst.map
    };
    if map.is_null() {
        return Ok(None);
    }

    // guard_class(obj, &INSTANCE_TYPE): receiver is a user instance, so the
    // `w_class` and map fields read below are valid.
    let instance_type_addr = &pyre_object::pyobject::INSTANCE_TYPE as *const _ as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(instance_type_addr);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, instance_type_addr);
    }

    // Pin the Python-level receiver class (`w_class`) exactly.  This is the
    // per-frame method namespace anchor: a subclass with the same instance
    // payload vtable side-exits instead of reusing the caller's method.
    let w_class_op = walker_record_getfield_gc_r_uncached(ctx, obj, crate::descr::w_class_descr());
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[w_class_op, w_type_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(w_class_op, w_type_const);

    // typeobject.py:506 `promote(self.version_tag())`: class mutation or method
    // reassignment bumps `_version_tag`, so the old `w_descr` side-exits.
    let vt_op = walker_record_getfield_gc_i_uncached(
        ctx,
        w_type_const,
        crate::descr::type_version_tag_descr(),
    );
    let vt_const = ctx.trace_ctx.const_int(version_tag as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[vt_op, vt_const])?;
    ctx.trace_ctx.heap_cache_mut().replace_box(vt_op, vt_const);

    // mapdict.py LOAD_ATTR caching: guard the instance map so adding a
    // shadowing `obj.method` attribute changes shape and side-exits before the
    // constant descriptor is reused.
    let map_op = walker_record_getfield_gc_i_uncached(ctx, obj, crate::descr::object_map_descr());
    let map_const = ctx.trace_ctx.const_int(map as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[map_op, map_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(map_op, map_const);

    let method_const = ctx.trace_ctx.const_ref(w_descr as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, method_const)?;
    Ok(Some(()))
}

/// Fold `bh_load_method_self_fn(obj, attr, code, name_idx)` once both the
/// receiver and the attribute are concrete.  The method-attribute fold above
/// already guards class, type version, and instance map; this second residual
/// is only the pure `compute_load_method_bound` binding decision.  A plain
/// instance-method bind writes the original red receiver box, not a baked
/// `ConstRef`, matching `callmethod.py:68 f.pushvalue(w_obj)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_fold_load_method_self<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    attr: OpRef,
    _attr_reg: usize,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(concrete_attr) = walker_concrete_ref_object(ctx, attr) else {
        return Ok(None);
    };
    if unsafe { pyre_object::is_method(concrete_attr) } {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let bound =
        pyre_interpreter::eval::compute_load_method_bound(concrete_obj, concrete_attr, &name);
    let bound_op = if std::ptr::eq(bound, concrete_obj) {
        obj
    } else if bound == pyre_object::PY_NULL {
        ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64)
    } else {
        return Ok(None);
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, bound_op)?;
    Ok(Some(()))
}

pub(crate) fn try_walker_specialize_store_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    value: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    original_effect: &majit_ir::EffectInfo,
) -> Result<Option<WalkerStoreAttrSpecialization>, DispatchError> {
    if !ctx.is_authoritative_executor {
        return Ok(None);
    }
    let (Some(concrete_obj), Some(concrete_value)) = (
        walker_concrete_ref_object(ctx, obj),
        walker_concrete_ref_object(ctx, value),
    ) else {
        return Ok(None);
    };
    let name = unsafe {
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return Ok(None);
        }
        let code = &*(code_ptr as *const pyre_interpreter::CodeObject);
        match pyre_interpreter::pyframe::load_name_from_code(code, name_idx) {
            Some(n) => n.to_string(),
            None => return Ok(None),
        }
    };
    if let Some((w_type, version_tag, map, storageindex, listindex, unbox_type)) = unsafe {
        pyre_interpreter::objspace::std::mapdict::store_attr_unboxed_fast_path(concrete_obj, &name)
    } {
        match unbox_type {
            pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                // `type(w_value) is space.IntObjectCls` (mapdict.py:615): reject bool
                // and every type-changing value before emitting any guards.
                if unsafe {
                    pyre_object::pyobject::is_bool(concrete_value)
                        || !pyre_object::pyobject::is_int(concrete_value)
                } {
                    return Ok(None);
                }
            }
            pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
                // A non-float changes the slot to boxed storage and freezes further
                // unboxing (mapdict.py:615-619), so retain setattr.
                if !unsafe { pyre_object::pyobject::is_float(concrete_value) } {
                    return Ok(None);
                }
            }
        }

        walker_guard_mapdict_instance_shape(ctx, op_pc, obj, w_type, version_tag, map)?;
        let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
        let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
        let (helper_fn, raw, value_type) = match unbox_type {
            pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                let raw = walker_unbox_int(ctx, op_pc, value, int_type_addr)?;
                (
                    crate::helpers::jit_mapdict_unboxed_write_raw as *const (),
                    raw,
                    majit_ir::Type::Int,
                )
            }
            pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
                let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
                let raw = walker_unbox_float(ctx, op_pc, value, float_type_addr)?;
                let live_f = unsafe { pyre_object::w_float_get_value(concrete_value) };
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Float(live_f));
                (
                    crate::helpers::jit_mapdict_unboxed_write_f as *const (),
                    raw,
                    majit_ir::Type::Float,
                )
            }
        };
        let helper = ctx.trace_ctx.const_int(helper_fn as usize as i64);

        let mut effect = original_effect.clone();
        effect.extraeffect = majit_ir::ExtraEffect::CannotRaise;
        effect.pyre_helper = majit_ir::PyreHelperKind::StoreAttr;
        let descr = majit_metainterp::make_call_descr_with_effect(
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
                value_type,
            ],
            majit_ir::Type::Void,
            effect,
        );
        // ABI order follows the write helpers: receiver and the two guarded green
        // coordinates, then the raw symbolic value in its own bank.  No box is
        // materialized for this write.
        return Ok(Some(WalkerStoreAttrSpecialization::Residual(
            descr,
            vec![helper, obj, storageindex_const, listindex_const, raw],
        )));
    }

    if let Some((slot, kind, w_type, version_tag, _stored)) = unsafe {
        pyre_interpreter::baseobjspace::exception_attr_slot_fold(concrete_obj, &name, true)
    } {
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
            let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType;
            let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
            if !unsafe {
                std::ptr::eq((*concrete_value).ob_type, tuple_type)
                    && std::ptr::eq((*concrete_value).w_class, canonical_tuple_class)
            } {
                return Ok(None);
            }
        }
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let (stored_value, concrete_stored) =
            if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
                let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType;
                let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
                if !unsafe {
                    std::ptr::eq((*concrete_value).ob_type, tuple_type)
                        && std::ptr::eq((*concrete_value).w_class, canonical_tuple_class)
                } {
                    return Ok(None);
                }
                let tuple_type_addr = tuple_type as i64;
                if !ctx.trace_ctx.heap_cache().is_class_known(value) {
                    let type_const = ctx.trace_ctx.const_int(tuple_type_addr);
                    walker_emit_fold_guard_with_snapshot(
                        ctx,
                        op_pc,
                        OpCode::GuardClass,
                        &[value, type_const],
                    )?;
                    ctx.trace_ctx
                        .heap_cache_mut()
                        .class_now_known(value, tuple_type_addr);
                }
                walker_guard_exact_w_class(ctx, op_pc, value, canonical_tuple_class)?;
                let block = crate::state::opimpl_getfield_gc_r(
                    ctx.trace_ctx,
                    value,
                    crate::descr::tuple_wrappeditems_descr(),
                );
                let len = unsafe { pyre_object::w_tuple_len(concrete_value) };
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    block,
                    crate::state::pyobject_gcarray_descr(),
                );
                let len_const = ctx.trace_ctx.const_int(len as i64);
                walker_emit_fold_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardValue,
                    &[length, len_const],
                )?;
                let mut items = Vec::with_capacity(len);
                let mut concrete_items = Vec::with_capacity(len);
                for index in 0..len {
                    let index_op = ctx.trace_ctx.const_int(index as i64);
                    items.push(crate::state::trace_items_block_getitem_value(
                        ctx.trace_ctx,
                        block,
                        index_op,
                    ));
                    concrete_items.push(
                        unsafe { pyre_object::w_tuple_getitem(concrete_value, index as i64) }
                            .unwrap_or(pyre_object::PY_NULL),
                    );
                }
                let list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &items);
                let concrete_list = pyre_object::w_list_new_object(concrete_items);
                ctx.trace_ctx.set_opref_concrete(
                    list,
                    majit_ir::Value::Ref(majit_ir::GcRef(concrete_list as usize)),
                );
                (list, concrete_list)
            } else {
                (value, concrete_value)
            };
        let field_descr = crate::descr::w_exception_slot_descr(kind, slot);
        let field_index = field_descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[obj, stored_value], field_descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(obj, field_index, stored_value);
        // The walk is the authoritative execution path.  Apply the same raw
        // slot writer now so interpreter execution after a side exit observes
        // the store; the writer supplies the host-side remembered-set barrier.
        // Compiled SetfieldGc reference stores receive CondCallGcWb from
        // majit-gc's rewrite pass, consumed by both dynasm and cranelift.
        unsafe {
            match slot {
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args => {
                    pyre_object::interp_exceptions::w_exception_set_args(
                        concrete_obj,
                        concrete_stored,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Context
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Cause => {
                    // `exception_attr_slot_fold` declines these for stores, so
                    // the store fold never reaches here.
                    unreachable!("__context__/__cause__ slots fold on load only")
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Code => {
                    pyre_object::interp_exceptions::w_exception_set_code(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Errno => {
                    pyre_object::interp_exceptions::w_exception_set_errno(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Strerror => {
                    pyre_object::interp_exceptions::w_exception_set_strerror(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Filename => {
                    pyre_object::interp_exceptions::w_exception_set_filename(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Filename2 => {
                    pyre_object::interp_exceptions::w_exception_set_filename2(
                        concrete_obj,
                        concrete_value,
                    )
                }
            }
        }
        return Ok(Some(WalkerStoreAttrSpecialization::Direct));
    }

    let Some((w_type, version_tag, map, storageindex)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::store_attr_boxed_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    walker_guard_mapdict_instance_shape(ctx, op_pc, obj, w_type, version_tag, map)?;
    let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
    let helper = ctx
        .trace_ctx
        .const_int(crate::helpers::jit_mapdict_boxed_write as *const () as usize as i64);

    // Unlike the unboxed arm, this write stores a GC reference, so the
    // residual's original may-force effect is kept: only the opaque `setattr_fn`
    // MRO walk is replaced by the direct slot write, while the force token, the
    // virtualizable spill, and the trailing force/exception guards stay exactly
    // as the generic setattr emitted them.
    let mut effect = original_effect.clone();
    effect.pyre_helper = majit_ir::PyreHelperKind::StoreAttr;
    let descr = majit_metainterp::make_call_descr_with_effect(
        &[
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Ref,
        ],
        majit_ir::Type::Void,
        effect,
    );
    // ABI order follows `jit_mapdict_boxed_write`: receiver, guarded green
    // storage index, and the original symbolic object reference.  The value is
    // neither unboxed nor guarded by type.
    Ok(Some(WalkerStoreAttrSpecialization::Residual(
        descr,
        vec![helper, obj, storageindex_const, value],
    )))
}

/// #171: FBW virtualization of a non-escaping BUILD_LIST.
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_LIST to `new_array_clear`
/// + per-index `setarrayitem_gc` + a `newlist_from_array` residual
/// (oopspec [`majit_ir::PyreHelperKind::NewlistFromArray`]) whose single
/// r-arg is the already-built backing array.  Decompose that residual into
/// the virtualizable `opimpl_newlist` shape (`pyjitpl.py:779`) —
/// `new_with_vtable` + `new_array` + `setarrayitem_gc` + `setfield_gc` —
/// so the optimizer folds the whole list (wrapper + block) when it never
/// escapes and the array build + residual DCE.
///
/// The element boxes are recovered from the backing array (its const length
/// from `heapcache.arraylen`, then per-index element shadows via
/// `heapcache_getarrayitem`), NOT from residual args.  The storage strategy
/// is chosen from the concrete element shadows exactly as
/// `list_strategy_for` / `w_list_new` does at runtime, so the traced object
/// matches the strategy the blackhole rebuilds on deopt:
///   * `list_strategy_for` → Integer AND every element an exact
///     `W_IntObject` → Integer (`int_items` typed block, elements unboxed
///     via `walker_unbox_int`);
///   * → Float → Float (`float_items` typed block, strict `W_FloatObject`
///     elements only, so exact-type by construction);
///   * → Object → Object (boxed refs into an `ItemsBlock`).
///
/// Returns `Ok(None)` to fall through to the opaque residual (always
/// byte-correct) for any shape it cannot reproduce faithfully: empty list
/// (Empty strategy), a non-const / unrecoverable array length, an element
/// without a concrete Ref shadow, or an Integer-strategy list that carries a
/// fits-in-word `W_LongObject` / tagged immediate (which `walker_unbox_int`'s
/// `&INT_TYPE` guard does not cover).
pub(crate) fn try_walker_specialize_newlist<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }
    let arr = r_args[0];

    // Const backing-array length (`new_array_clear(Const(len))` seeded
    // `heapcache.arraylen` — a cleared array has every slot set, so read the
    // length directly rather than probing getarrayitem until a miss).  Empty
    // list → Empty strategy: decline (the residual reproduces it).
    let len = {
        let Some(len_op) = ctx.trace_ctx.heap_cache().arraylen(arr) else {
            return Ok(None);
        };
        match len_op.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) if n >= 1 => n as usize,
            _ => return Ok(None),
        }
    };

    // Recover the element boxes from the array heap-cache (the values the
    // BUILD_LIST `setarrayitem_gc` ops stored); a cache miss (clobbered array)
    // bails to the opaque residual.
    let descr_idx = crate::state::pyobject_gcarray_descr().index();
    let mut items: Vec<OpRef> = Vec::with_capacity(len);
    for i in 0..len {
        let Some(elem) =
            ctx.trace_ctx
                .heapcache_getarrayitem(arr, OpRef::ConstInt(i as i64), descr_idx)
        else {
            return Ok(None);
        };
        items.push(elem);
    }

    // Concrete element objects (needed to classify the strategy and extract
    // the payloads before any allocation).  An element without a concrete Ref
    // shadow declines to the residual.
    let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for &it in &items {
        let Some(obj) = walker_concrete_ref_object(ctx, it) else {
            return Ok(None);
        };
        concretes.push(obj);
    }

    // Strategy the runtime `w_list_new` would pick — the source of truth for
    // the concrete shadow, so the traced storage matches on deopt.
    let strategy = pyre_object::listobject::list_strategy_for(&concretes);
    use pyre_object::listobject::ListStrategy;

    // Pre-extract the machine payloads BEFORE `build_list_from_refs` allocates
    // (a minor collection there could move the boxed elements, so the raw
    // pointers must not be dereferenced afterwards).
    enum Emit {
        Int(Vec<i64>),
        Float(Vec<f64>),
        Object,
    }
    let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
    let emit = match strategy {
        ListStrategy::Integer => {
            // `list_strategy_for` accepts fits-in-word `W_LongObject` and
            // tagged immediates under Integer, but `walker_unbox_int` only
            // covers the exact `W_IntObject` (`&INT_TYPE` + `intval`) shape —
            // decline otherwise so the residual (correct for any element)
            // rebuilds the same Integer list.
            let mut vals = Vec::with_capacity(len);
            for &p in &concretes {
                if pyre_object::tagged_int::CAN_BE_TAGGED
                    && pyre_object::tagged_int::is_tagged_int(p)
                {
                    return Ok(None);
                }
                let exact_int =
                    unsafe { pyre_object::is_plain_int1(p) && std::ptr::eq((*p).ob_type, int_ty) };
                if !exact_int {
                    return Ok(None);
                }
                vals.push(unsafe { pyre_object::w_int_get_value(p) });
            }
            Emit::Int(vals)
        }
        ListStrategy::Float => {
            // `all_floats` is strict `type(w) is W_FloatObject`, so every
            // element is an exact `W_FloatObject` (`walker_unbox_float`'s
            // `&FLOAT_TYPE` guard holds).
            let mut vals = Vec::with_capacity(len);
            for &p in &concretes {
                vals.push(unsafe { pyre_object::w_float_get_value(p) });
            }
            Emit::Float(vals)
        }
        ListStrategy::Object => Emit::Object,
        // Empty is impossible here (len >= 1); decline defensively.
        ListStrategy::Empty => return Ok(None),
    };

    // Concrete shadow: a fresh list built from the element shadows
    // (`w_list_new` parity — picks the same strategy). A new allocation with
    // no heap mutation, safe during the walk like `wrapint`.
    let result_concrete = pyre_interpreter::build_list_from_refs(&concretes);
    if result_concrete.is_null() {
        return Ok(None);
    }

    // --- emit the virtualizable decomposed newlist (walker-native) ---
    let list_op = match emit {
        Emit::Int(vals) => {
            let int_type_addr = int_ty as i64;
            let mut raws: Vec<OpRef> = Vec::with_capacity(len);
            for (&it, &v) in items.iter().zip(vals.iter()) {
                let raw = walker_unbox_int(ctx, op_pc, it, int_type_addr)?;
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Int(v));
                raws.push(raw);
            }
            crate::helpers::emit_typed_list_inline(
                &mut *ctx.trace_ctx,
                &raws,
                crate::state::int_gcarray_descr(),
                crate::descr::list_int_items_len_descr(),
                crate::descr::list_int_items_block_descr(),
                ListStrategy::Integer,
            )
        }
        Emit::Float(vals) => {
            let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
            let mut raws: Vec<OpRef> = Vec::with_capacity(len);
            for (&it, &v) in items.iter().zip(vals.iter()) {
                let raw = walker_unbox_float(ctx, op_pc, it, float_type_addr)?;
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Float(v));
                raws.push(raw);
            }
            crate::helpers::emit_typed_list_inline(
                &mut *ctx.trace_ctx,
                &raws,
                crate::state::float_gcarray_descr(),
                crate::descr::list_float_items_len_descr(),
                crate::descr::list_float_items_block_descr(),
                ListStrategy::Float,
            )
        }
        Emit::Object => crate::helpers::emit_object_list_inline(&mut *ctx.trace_ctx, &items),
    };

    ctx.trace_ctx.set_opref_concrete(
        list_op,
        majit_ir::Value::Ref(majit_ir::GcRef(result_concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, list_op)?;
    Ok(Some(()))
}

/// #195 / #73: FBW virtualization of an arity-2 plain-int BUILD_TUPLE.
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_TUPLE to `new_array_clear`
/// + per-index `setarrayitem_gc` + a `newtuple_from_array` residual
/// (oopspec [`majit_ir::PyreHelperKind::NewtupleFromArray`]).  When both
/// backing-array elements are concrete plain `W_IntObject`, re-emit the
/// former trait-side spec_ii shape walker-native
/// (`new_with_vtable` + `w_class` / `value0` / `value1` `setfield_gc`),
/// reading the elements straight out of the array heap-cache so the array
/// build keeps no consumer and DCEs.  The partner
/// [`try_walker_specialize_unpack`] then folds the `value0` / `value1`
/// reads off the virtual tuple, collapsing build→unpack to a pure-int loop.
///
/// Returns `Ok(Some(()))` when folded (the caller returns `Continue`);
/// `Ok(None)` to fall through to the opaque residual, which stays correct
/// for any other shape (object tuple, arity ≠ 2, long element, cache miss)
/// — so a non-foldable build is not declined.
pub(crate) fn try_walker_specialize_newtuple<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }
    let arr = r_args[0];
    // Read the two backing-array element boxes out of the heap-cache (the
    // values the BUILD_TUPLE `setarrayitem_gc` ops stored); a cache miss
    // (non-const index / clobbered array) bails to the opaque residual.
    let descr_idx = crate::state::pyobject_gcarray_descr().index();
    let (Some(e0), Some(e1)) = (
        ctx.trace_ctx
            .heapcache_getarrayitem(arr, OpRef::ConstInt(0), descr_idx),
        ctx.trace_ctx
            .heapcache_getarrayitem(arr, OpRef::ConstInt(1), descr_idx),
    ) else {
        return Ok(None);
    };
    // Arity must be exactly 2 (the only specialised int tuple).  A BUILD_TUPLE
    // array sets every index before `newtuple_from_array`, so a cached element
    // at index 2 means arity ≥ 3 → fall through to the residual (a wrongly
    // built arity-2 spec_ii would length-mismatch the arity-N unpack).
    if ctx
        .trace_ctx
        .heapcache_getarrayitem(arr, OpRef::ConstInt(2), descr_idx)
        .is_some()
    {
        return Ok(None);
    }
    let (Some(c0), Some(c1)) = (
        walker_concrete_ref_object(ctx, e0),
        walker_concrete_ref_object(ctx, e1),
    ) else {
        return Ok(None);
    };
    // Only the arity-2 plain-int specialised tuple is folded today.  Gate
    // on `is_plain_int1` (rejects int subclasses + non-fitting longs) AND
    // an exact `&INT_TYPE` `ob_type` — that excludes the fits-in-word
    // `W_LongObject` arm `is_plain_int1` also accepts, which would need the
    // long unbox the retired trait-side payload helper did (out of scope
    // here).  Any other shape falls through to the residual (correct).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(c0)
            || pyre_object::tagged_int::is_tagged_int(c1))
    {
        // A tagged-immediate element has no real header to read for the exact
        // `&INT_TYPE` ob_type check below, and the spec_ii emit (w_class guard +
        // typed unbox) is not tag-aware. Fall through to the opaque residual,
        // which is correct for any element shape.
        return Ok(None);
    }
    let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
    let both_plain_int = unsafe {
        pyre_object::is_plain_int1(c0)
            && pyre_object::is_plain_int1(c1)
            && std::ptr::eq((*c0).ob_type, int_ty)
            && std::ptr::eq((*c1).ob_type, int_ty)
    };
    if !both_plain_int {
        return Ok(None);
    }
    // Concrete element int payloads (already proven plain `W_IntObject`).
    let (v0, v1) = unsafe {
        (
            pyre_object::w_int_get_value(c0),
            pyre_object::w_int_get_value(c1),
        )
    };

    // --- emit the virtual spec_ii walker-native ---
    // Paired `w_class` guard per element so a runtime int subclass sharing
    // `&INT_TYPE`'s payload side-exits, then the plain-int payload unbox.
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    walker_guard_exact_w_class(ctx, op_pc, e0, int_typeobj)?;
    walker_guard_exact_w_class(ctx, op_pc, e1, int_typeobj)?;
    let int_type_addr = int_ty as i64;
    let raw0 = walker_unbox_int_typed(
        ctx,
        op_pc,
        e0,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;
    let raw1 = walker_unbox_int_typed(
        ctx,
        op_pc,
        e1,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;

    let tuple = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::specialised_tuple_ii_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(tuple);
    // `ob_type` is the JIT vtable; Python-level `type()` reads `w_class`,
    // which all specialised tuple variants share at the public `tuple`
    // typedef.
    let tuple_w_class = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE);
    if !tuple_w_class.is_null() {
        let wc = ctx.trace_ctx.const_ref(tuple_w_class as i64);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[tuple, wc],
            crate::descr::specialised_tuple_ii_w_class_descr(),
        );
        ctx.trace_ctx.heapcache_setfield_cached(
            tuple,
            crate::descr::specialised_tuple_ii_w_class_descr().index(),
            wc,
        );
    }
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[tuple, raw0],
        crate::descr::specialised_tuple_ii_value0_descr(),
    );
    ctx.trace_ctx.heapcache_setfield_cached(
        tuple,
        crate::descr::specialised_tuple_ii_value0_descr().index(),
        raw0,
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[tuple, raw1],
        crate::descr::specialised_tuple_ii_value1_descr(),
    );
    ctx.trace_ctx.heapcache_setfield_cached(
        tuple,
        crate::descr::specialised_tuple_ii_value1_descr().index(),
        raw1,
    );
    // Concrete spec_ii shadow for the dst (read by the partner unpack fold's
    // `walker_concrete_ref_object` + `unpack_item_fn` execution).  Built
    // directly from the element payloads — `newtuple_from_array(arr)` cannot
    // be executed concretely (the virtual array `arr` has no concrete
    // shadow).  Constructed last so the construct→root window holds no
    // intervening runtime allocation: stamping `tuple`'s concrete roots the
    // fresh spec_ii via the trace's concrete-shadow set.
    let tuple_ptr = pyre_object::specialisedtupleobject::w_specialised_tuple_ii_new(v0, v1);
    if tuple_ptr.is_null() {
        return Ok(None);
    }
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(tuple_ptr as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, tuple)?;
    Ok(Some(()))
}

/// #57 SLICE 3b: walker-native speculative int specialization for the
/// COMPARE_OP helper residual_call (oopspec `CompareOp`).  Emits
/// `guard_class` + `getfield_gc_i` per operand + `int_<cmp>` for the raw
/// truth, then boxes it to a `W_Bool`.  NON-fused: the walker sees
/// COMPARE_OP and the following `goto_if_not` as separate JitCode ops, so
/// it always materializes the boxed bool the generic `compare_fn` would
/// have produced (the retired MIFrame compare/jump fusion does not apply).
///
/// Same gate + return contract as
/// [`try_walker_specialize_binary_op_int`].
pub(crate) fn try_walker_specialize_compare_op_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::ComparisonOperator;
    let cmp = match cmp_op {
        ComparisonOperator::Less => OpCode::IntLt,
        ComparisonOperator::LessOrEqual => OpCode::IntLe,
        ComparisonOperator::Greater => OpCode::IntGt,
        ComparisonOperator::GreaterOrEqual => OpCode::IntGe,
        ComparisonOperator::Equal => OpCode::IntEq,
        ComparisonOperator::NotEqual => OpCode::IntNe,
    };
    let Some((lhs, rhs, lhs_obj, rhs_obj, la, rb, boxed_result_i64)) =
        walker_int_specialization_operands(ctx, r_args, allboxes, call_descr)
    else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // bool and int share `intval`; guard each operand against its own vtable
    // so a bool comparand unboxes through &BOOL_TYPE.  The comparison result
    // is a bool either way.
    let (lhs_type, lhs_descr) = crate::state::int_or_bool_unbox_type_descr(lhs_obj);
    let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
    let lhs_raw = walker_unbox_int_typed(ctx, op_pc, lhs, lhs_type, lhs_descr)?;
    let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
    let truth = ctx.trace_ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
    let folded = majit_metainterp::eval_binop_i(cmp, la, rb);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(folded));
    // #62: elide the dead `box_bool` when a forward JitCode lookahead
    // PROVES the compare's boxed Ref dst is consumed solely by the
    // immediately-following `is_true` (POP_JUMP_IF_*), which folds to the
    // raw truth.  In that shape the W_Bool is never read as a Ref, so the
    // box is dead the moment it is recorded — yet it is a non-pure `CallR`
    // the optimizer cannot DCE (pure.py:222 demotes CALL_PURE→CALL and
    // emits it; the retired MIFrame path never created the box because it
    // fused COMPARE_OP+POP_JUMP at the bytecode level). Mirroring that
    // fusion walker-side: write the raw truth into the Ref dst as a marker
    // and record `bool_box_truth(truth, truth)` so the `is_true` fold
    // (dispatch_residual_call_iRd_kind:5137) resolves it to `truth`; emit
    // no box.  Gated on the lookahead proof so the marker provably never
    // escapes (no Ref consumer, not live at the branch resume) — any other
    // shape (escape to a local, arithmetic, multi-use, branch keeping the
    // value) falls back to emitting the real box.
    if dst_bank == 'r' && compare_box_provably_dead(ctx, op_pc, dst as u8) {
        bool_box_truth_record(truth, truth);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, truth)?;
        return Ok(Some(()));
    }
    // NON-fused: box the raw truth into a W_Bool (the generic compare_fn
    // residual_call lands a boxed bool in the dst Ref register; the
    // separate goto_if_not op reads it).
    let boxed = crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    // #62: remember boxed→truth so an immediately-following `is_true` residual
    // (POP_JUMP_IF_*) folds back to the raw Int instead of may-force-unboxing.
    bool_box_truth_record(boxed, truth);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// B3 (`PYRE_FBW_RAISE`): walker-native fold of the CHECK_EXC_MATCH
/// residual (`bh_compare_fn(exc, match_type, op_tag=10)`,
/// `call_jit.rs:4299`). Computes the match concretely from
/// `type(exc)` and `match_type` and emit a `const_ref` of the immortal
/// TRUE/FALSE bool singleton, eliding the opaque may-force compare (and,
/// via [`bool_box_truth_record`], the immediately-following `is_true`
/// truth-extract residual).  With the exception's constructor + raise
/// already virtualized (B3 pieces 1+2), folding the match to a constant
/// lets the whole exception de-escape and DCE.
///
/// Soundness — the fold result depends only on `(type(exc), match_type)`:
///   * `exc` (`r_args[0]`) is the in-trace inline-built virtual exception
///     whose kind/vtable are baked into the `NewWithVtable`, so its class
///     cannot differ at runtime — no guard needed.  (A `GuardClass` is
///     emitted defensively when the heapcache does not already know its
///     class, e.g. a non-construct-fold exc reaching here.)
///   * `match_type` (`r_args[1]`) is a runtime value (typically a
///     `LOAD_GLOBAL` of the handler class), so a `GuardValue` pins its
///     identity — a reassigned handler global side-exits and re-traces
///     instead of running the wrong handler.  (Stricter than the trait,
///     which elides this guard.)
///
/// Declines (`None` → generic residual) when either operand lacks a
/// concrete shadow, or `match_type` is not a valid exception class /
/// tuple (the residual then raises the correct `TypeError`).
pub(crate) fn try_walker_fold_check_exc_match<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let exc_op = r_args[0];
    let match_op = r_args[1];
    let (Some(exc), Some(match_type)) = (
        walker_concrete_ref_object(ctx, exc_op),
        walker_concrete_ref_object(ctx, match_op),
    ) else {
        return Ok(None);
    };
    // `validate_check_exc_match_class` gates `except <non-exception>:`
    // (raising `TypeError`); on a validity error decline so the residual
    // reproduces the raise instead of baking a wrong bool into the trace.
    if pyre_interpreter::eval::validate_check_exc_match_class(match_type).is_err() {
        return Ok(None);
    }
    // `eval::check_exc_match_against` = `exception_match(type(exc), match)`
    // (eval.rs), walking the exception class MRO and accepting a tuple of
    // classes. Inlined here.
    let matched = pyre_interpreter::eval::check_exc_match_against(exc, match_type);

    // --- commit to the fold: emit IR (no further declines) ---
    // Pin `match_type` identity so a runtime divergence (a reassigned
    // handler global) side-exits rather than running the wrong handler.
    if !match_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(match_op) {
        let expected = ctx.trace_ctx.const_ref(match_type as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[match_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(match_op, expected);
    }
    // Defensive `GuardClass` on the exception when its class is not yet
    // known (the construct fold marks it known, so this is a no-op for a
    // virtual inline-built exc; it pins the class for any other exc that
    // reaches this fold).
    if !exc_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(exc_op) {
        let exc_class_ptr = unsafe { (*(exc as *const pyre_object::pyobject::PyObject)).ob_type };
        let cls_const = ctx.trace_ctx.const_int(exc_class_ptr as usize as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[exc_op, cls_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(exc_op, exc_class_ptr as usize as i64);
    }

    // The match is a constant at trace time: emit the immortal bool
    // singleton as a `const_ref`, and a raw truth `const_int` so the
    // following `is_true` (the `except` clause's `POP_JUMP_IF_FALSE`)
    // folds to the constant via `bool_box_truth_record`.
    let result_obj = pyre_object::w_bool_from(matched);
    let const_bool = ctx.trace_ctx.const_ref(result_obj as i64);
    ctx.trace_ctx.set_opref_concrete(
        const_bool,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    let truth = ctx.trace_ctx.const_int(matched as i64);
    bool_box_truth_record(const_bool, truth);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, const_bool)?;
    Ok(Some(()))
}

/// W_LongObject (bigint) COMPARE_OP specialization — the long analogue of
/// [`try_walker_specialize_compare_op_int`].  Both operands are `int`-typed but
/// bigint-stored: guard each against `LONG_TYPE`, then `CallPure_I` the pure
/// `jit_w_long_cmp` (sign of `a <=> b` in {-1,0,1}; a comparison neither
/// allocates nor raises, so `EF_ELIDABLE_CANNOT_RAISE` and NO trailing guard)
/// and turn the sign into the requested truth with `int_<cmp>(sign, 0)` before
/// boxing to a `W_Bool` (same #62 dead-box elision as the int path).  Same gate
/// + return contract as [`try_walker_specialize_binary_op_long`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_compare_op_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::ComparisonOperator;
    // `a <cmp> b` ⟺ `sign(a <=> b) <cmp> 0`.
    let cmp = match cmp_op {
        ComparisonOperator::Less => OpCode::IntLt,
        ComparisonOperator::LessOrEqual => OpCode::IntLe,
        ComparisonOperator::Greater => OpCode::IntGt,
        ComparisonOperator::GreaterOrEqual => OpCode::IntGe,
        ComparisonOperator::Equal => OpCode::IntEq,
        ComparisonOperator::NotEqual => OpCode::IntNe,
    };
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(lhs_obj) && pyre_object::is_long(rhs_obj) } {
        return Ok(None);
    }
    // Authentic boxed W_Bool via the same execute path the int leg uses; also
    // advances the concrete VM state the downstream ops read.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    // Pure `rbigint` comparison → sign in {-1,0,1}. Dead after the `int_<cmp>`
    // below and never spans a guard, so it needs no blackhole reconstruction.
    let cmp_fn = pyre_object::longobject::jit_w_long_cmp as *const ();
    let sign_concrete = pyre_object::longobject::jit_w_long_cmp(lhs_obj as i64, rhs_obj as i64);
    let concrete_args = [
        majit_ir::Value::Int(cmp_fn as usize as i64),
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_obj as usize)),
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_obj as usize)),
    ];
    let sign = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        cmp_fn,
        &[lhs, rhs],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        &concrete_args,
        majit_ir::Value::Int(sign_concrete),
    );
    ctx.trace_ctx
        .set_opref_concrete(sign, majit_ir::Value::Int(sign_concrete));
    let zero = ctx.trace_ctx.const_int(0);
    let truth = ctx.trace_ctx.record_op(cmp, &[sign, zero]);
    let folded = majit_metainterp::eval_binop_i(cmp, sign_concrete, 0);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(folded));
    // #62: elide the dead box when the boxed Ref is consumed solely by the
    // following `is_true` (POP_JUMP_IF_*); else box the raw truth into a W_Bool.
    if compare_box_provably_dead(ctx, op_pc, dst as u8) {
        bool_box_truth_record(truth, truth);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, truth)?;
        return Ok(Some(()));
    }
    let boxed = crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    bool_box_truth_record(boxed, truth);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #57 SLICE 3c: walker-native speculative float specialization for the
/// `BINARY_OP` helper residual_call (oopspec `BinaryOp`), the float
/// analogue of [`try_walker_specialize_binary_op_int`].  Re-derives
/// the former float fast path's structure walker-native: per operand
/// either `guard_class FLOAT` + `getfield_gc_pure_f`, or (int operand)
/// `guard_class INT` + `getfield_gc_i` + `cast_int_to_float`; then
/// `float_OP` and `wrapfloat`.
///
/// Only the bare-primitive operators (`FloatAdd` / `FloatSub` /
/// `FloatMul` / `FloatTrueDiv`) are specialized — Power / FloorDivide /
/// Remainder have no FLOAT_* opcode and defer to the generic
/// `CALL_MAY_FORCE` leg (Power lowers to a `call_may_force` +
/// `guard_no_exception` there).  Tried as a fallback only after the int
/// specialization declines, so two-int operands keep int `__op__`
/// arithmetic.
pub(crate) fn try_walker_specialize_binary_op_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(bin_op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::BinaryOperator;
    // Power has no FLOAT_* opcode — it lowers to the raw-float
    // `float_pow_jit` call (floatobject.py:561 descr_pow → _pow), same
    // as the trait's `is_power` arm.
    let op_code = match bin_op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => Some(OpCode::FloatAdd),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => Some(OpCode::FloatSub),
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => Some(OpCode::FloatMul),
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => {
            Some(OpCode::FloatTrueDiv)
        }
        BinaryOperator::Power | BinaryOperator::InplacePower => None,
        _ => return Ok(None),
    };

    let Some((
        lhs,
        rhs,
        lhs_obj,
        rhs_obj,
        lhs_is_int,
        rhs_is_int,
        lhs_f64,
        rhs_f64,
        boxed_result_i64,
    )) = walker_float_specialization_operands(ctx, r_args, allboxes, call_descr)
    else {
        return Ok(None);
    };
    if op_code.is_none() {
        // The generic helper already executed concretely (it produced
        // `boxed_result_i64`), so a non-float result here would mean
        // `float ** x` returned a non-W_FloatObject — decline rather
        // than mis-unbox the concrete stamp.
        let boxed_obj = boxed_result_i64 as pyre_object::PyObjectRef;
        if unsafe { !pyre_object::pyobject::is_float(boxed_obj) } {
            return Ok(None);
        }
    }

    // floatobject.py:519 `_floatdiv` raises "float division by zero" — a
    // concrete zero divisor at trace time means the generic helper already
    // raised, so a non-raising specialized `FloatTrueDiv` (raw IEEE → inf)
    // would be a miscompile.  Decline so the generic CALL_MAY_FORCE leg
    // (descr_truediv) records the raising call.
    if matches!(op_code, Some(OpCode::FloatTrueDiv)) && rhs_f64 == 0.0 {
        return Ok(None);
    }

    // --- emit the specialized IR (walker-native) ---
    let lhs_raw = walker_coerce_operand_to_float(ctx, op_pc, lhs, lhs_obj, lhs_is_int, lhs_f64)?;
    let rhs_raw = walker_coerce_operand_to_float(ctx, op_pc, rhs, rhs_obj, rhs_is_int, rhs_f64)?;
    // rint.py:429 `_ovf_zer` analogue for float true-division: emit a
    // `float_eq(rhs, 0.0) → guard_false` precondition ahead of the bare
    // `FloatTrueDiv` llop so a future zero divisor deopts to the checked
    // descr_truediv path (which raises ZeroDivisionError) rather than
    // computing a raw IEEE inf.  The bare llop is sound only behind this
    // non-zero guarantee.
    if matches!(op_code, Some(OpCode::FloatTrueDiv)) {
        let rhs_zero = walker_float_eq_const(ctx, rhs_raw, 0.0, (rhs_f64 == 0.0) as i64);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[rhs_zero])?;
    }
    let raw_result = match op_code {
        Some(op_code) => {
            let r = ctx.trace_ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
            let bits = majit_metainterp::eval_binop_f(
                op_code,
                lhs_f64.to_bits() as i64,
                rhs_f64.to_bits() as i64,
            );
            ctx.trace_ctx
                .set_opref_concrete(r, majit_ir::Value::Float(f64::from_bits(bits as u64)));
            r
        }
        None => {
            let result_val = unsafe { pyre_object::w_float_get_value(boxed_result_i64 as _) };
            // _pow (floatobject.py:865) traced inline for its fast paths:
            // every special-case `if` becomes a comparison guard and only
            // the raw libm pow stays residual.
            if let Some(r) = walker_emit_float_pow_inline(
                ctx, op_pc, lhs_raw, rhs_raw, lhs_f64, rhs_f64, result_val,
            )? {
                r
            } else {
                // Cold-path fallback (nan/inf operands, negative base):
                // the opaque `_pow` helper.  It is EF_CAN_RAISE, NOT
                // force_virtual: pyjitpl.py:2084-2121 execute_varargs(
                // rop.CALL_F, ..., exc=True, pure=False) records CALL_F
                // and handle_possible_exception → GUARD_NO_EXCEPTION
                // (pyjitpl.py:3395).  The raising case never reaches
                // here: `walker_float_specialization_operands` already
                // executed the helper concretely and returns `None` on a
                // raise, falling back to the generic residual leg.
                let r = ctx.trace_ctx.call_float_typed_with_effect(
                    crate::trace_opcode::float_pow_jit as *const (),
                    &[lhs_raw, rhs_raw],
                    &[majit_ir::Type::Float, majit_ir::Type::Float],
                    majit_metainterp::default_effect_info(),
                );
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
                ctx.trace_ctx
                    .set_opref_concrete(r, majit_ir::Value::Float(result_val));
                r
            }
        }
    };
    let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw_result);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #62: walker-native speculative specialization for the `BINARY_SUBSCR`
/// helper residual_call (oopspec `BinaryOp`, op_tag `Subscr`).  Ports
/// the former subscription/list-strategy path for the object-, int-, and
/// float-storage list strategies with a
/// non-negative concrete index: `guard_class LIST` + `guard_value(strategy)`
/// + unbox index + `IntLt` bounds guard, then the strategy-specific element
/// load — `getarrayitem_gc_r` against the `Ptr(GcArray(OBJECTPTR))` items
/// block for object storage (the element is a boxed Ref read directly), or a
/// raw-array getitem + `wrapint` / `wrapfloat` rebox for int/float storage.
/// The authentic boxed result is taken from the same `execute_may_force_call`
/// path the generic leg uses.
///
/// Tuples, empty-strategy lists, negative indices, and non-`list[int]`
/// operands fall through to the generic `CallMayForce` record (`Ok(None)`),
/// preserving Python `__getitem__` semantics.
pub(crate) fn try_walker_specialize_subscr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let (Some(list_obj), Some(key_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
    ) else {
        return Ok(None);
    };

    // #171/#11 Approach C: canonical array-backed `W_TupleObject[i]`.  Two
    // gates, both required:
    //   * `ob_type == &TUPLE_TYPE` (tupleobject.py / tupleobject.rs:222) —
    //     NOT `is_tuple()` (which also accepts the three
    //     SPECIALISED_TUPLE_{II,FF,OO} variants).  Specialised tuples store
    //     `value0`/`value1` inline with no `wrappeditems` block, so a
    //     `getfield(wrappeditems)` on one yields garbage.
    //   * `w_class == canonical tuple` — a tuple SUBCLASS instance shares the
    //     payload `ob_type == &TUPLE_TYPE` but retags `w_class` and may
    //     override `__getitem__`; `baseobjspace::getitem` honours that
    //     override (subclass_special_override) so the pure `wrappeditems[i]`
    //     load must NOT be taken for it.
    // A failing gate falls to the generic residual.  The paired runtime
    // `guard_class(&TUPLE_TYPE)` + exact `w_class` guard (in
    // `try_walker_specialize_subscr_tuple`) deopt any later non-canonical
    // tuple or subclass instance flowing in.
    let tuple_canonical = unsafe {
        std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::TUPLE_TYPE)
            && std::ptr::eq(
                (*list_obj).w_class,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
            )
    };
    if tuple_canonical {
        return try_walker_specialize_subscr_tuple(
            ctx, op_pc, list_op, key_op, list_obj, key_obj, allboxes, call_descr, dst, dst_bank,
        );
    }

    // Gate: EXACT list[int], non-negative index in bounds, int- or
    // float-storage.  A bool index (`is_int` accepts `W_BoolObject`) is fine:
    // bool shares int's `intval`, so it unboxes through its own &BOOL_TYPE
    // guard below.  A list SUBCLASS instance shares `ob_type == &LIST_TYPE`
    // but retags `w_class` and may override `__getitem__`; `is_exact_list`
    // excludes it so it falls to the generic residual (which honours the
    // override) instead of this direct-storage load.
    let (sid, index, concrete_len) = unsafe {
        if !pyre_object::is_exact_list(list_obj) || !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        let index = pyre_object::w_int_get_value(key_obj);
        if index < 0 {
            return Ok(None);
        }
        let concrete_len = pyre_object::w_list_len(list_obj);
        if index as usize >= concrete_len {
            return Ok(None);
        }
        let sid = if pyre_object::w_list_uses_int_storage(list_obj) {
            1i64
        } else if pyre_object::w_list_uses_float_storage(list_obj) {
            2i64
        } else if pyre_object::w_list_uses_object_storage(list_obj) {
            0i64
        } else {
            // Empty-strategy list: no concrete element to read.
            return Ok(None);
        };
        (sid, index, concrete_len)
    };

    // Authentic boxed result from the same may-force path the generic leg uses.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // guard_class LIST (skip when class already known / operand is constant).
    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    if !list_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(list_op) {
        let type_const = ctx.trace_ctx.const_int(list_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[list_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(list_op, list_type_addr);

    // A list SUBCLASS instance shares `ob_type == &LIST_TYPE` (so it passes
    // the GuardClass above) but retags `w_class` and may override
    // `__getitem__`; guard the exact canonical `w_class` so such an instance
    // side-exits to the generic residual (which honours the override) rather
    // than taking this direct-storage load.
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        list_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
    )?;

    // guard_value(strategy == sid): getfield strategy + GuardValue + replace_box.
    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_strategy_descr(),
    );
    let sid_const = ctx.trace_ctx.const_int(sid);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[strategy, sid_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(strategy, sid_const);

    // Unbox the index operand (guard_class + getfield intval).  bool shares
    // int's `intval`, so a bool index guards its own &BOOL_TYPE.
    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(index));

    // Two-sided bounds guard `0 <= raw_index < len`.  Object storage keeps the
    // inline `length` field (rlist.py:116); int/float storage read the typed
    // items-array length field.  The trace is recorded from a non-negative
    // observed index, but a later NEGATIVE index would still satisfy
    // `raw_index < len` and reach the element load out of range; `space.getitem`
    // treats a negative index as `index + len` (listobject.py), so the
    // lower-bound guard deopts to re-execute that remap generically.
    let len_descr = match sid {
        0 => crate::descr::list_length_descr(),
        1 => crate::descr::list_int_items_len_descr(),
        _ => crate::descr::list_float_items_len_descr(),
    };
    let lenbox = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, list_op, len_descr);
    let zero = ctx.trace_ctx.const_int(0);
    let nonneg = ctx.trace_ctx.record_op(OpCode::IntGe, &[raw_index, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonneg, majit_ir::Value::Int((index >= 0) as i64));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonneg])?;
    let in_bounds = ctx.trace_ctx.record_op(OpCode::IntLt, &[raw_index, lenbox]);
    ctx.trace_ctx.set_opref_concrete(
        in_bounds,
        majit_ir::Value::Int(((index as usize) < concrete_len) as i64),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_bounds])?;

    // Element load.  Object storage reads the boxed Ref directly from the
    // `Ptr(GcArray(OBJECTPTR))` items block (no unbox/rebox).  Int/float
    // storage read the raw typed array and rebox; the raw element is stamped
    // with the true value from the authentic may-force result (the in-array
    // sanity load is skipped when `items_ptr` is not trace-time concrete) so
    // the `wrapint` / `wrapfloat` box's cached field matches a later unbox.
    let result_obj = boxed_result_i64 as pyre_object::PyObjectRef;
    let default_concrete = majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize));
    let (boxed, boxed_concrete) = match sid {
        0 => {
            let items_block = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_items_descr(),
            );
            (
                crate::state::trace_items_block_getitem_value(
                    ctx.trace_ctx,
                    items_block,
                    raw_index,
                ),
                default_concrete,
            )
        }
        1 => {
            let block = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_int_items_block_descr(),
            );
            let raw = crate::state::trace_int_block_getitem_value(ctx.trace_ctx, block, raw_index);
            let elem = unsafe { pyre_object::w_int_get_value(result_obj) };
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(elem));
            (
                walker_box_int(ctx, op_pc, raw, elem)?,
                box_int_concrete(elem, boxed_result_i64),
            )
        }
        _ => {
            let block = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_float_items_block_descr(),
            );
            let raw =
                crate::state::trace_float_block_getitem_value(ctx.trace_ctx, block, raw_index);
            let elem = unsafe { pyre_object::w_float_get_value(result_obj) };
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(elem));
            (
                crate::state::wrapfloat(ctx.trace_ctx, raw),
                default_concrete,
            )
        }
    };
    ctx.trace_ctx.set_opref_concrete(boxed, boxed_concrete);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #171/#11 Approach C, SUBSCRIPT slice: walker-native PURE element load
/// for a canonical array-backed `W_TupleObject[i]` (the tuple analogue of
/// the object-storage list arm of [`try_walker_specialize_subscr`]).
///
/// Recognition (caller already verified `ob_type == &TUPLE_TYPE`): a
/// non-negative int (or bool, which shares `intval`) index in bounds.
/// Specialised tuples never reach here — the caller gates them out — so
/// reading `wrappeditems` is always sound.
///
/// IR shape: `guard_class(&TUPLE_TYPE)` → `getfield(wrappeditems)` →
/// `arraylen_gc(wrappeditems)` for the bounds length → `IntLt` +
/// `GuardTrue` (NON-pure, so an out-of-range deopt still fires) →
/// `getarrayitem_gc_pure_r(wrappeditems, idx)` (the ONLY pure op; the
/// body is immutable per `_immutable_fields_ = ['wrappeditems[*]']`).
/// Object storage → the element is a boxed Ref read directly (no
/// unbox/rebox).  The authentic boxed result is taken from the same
/// `execute_may_force` path the generic leg uses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_subscr_tuple<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    list_op: OpRef,
    key_op: OpRef,
    tuple_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    // Gate: non-negative int index in bounds.  `w_tuple_len` reads the
    // GcArray header of `wrappeditems` (no inline length field).
    let (index, concrete_len) = unsafe {
        if !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        let index = pyre_object::w_int_get_value(key_obj);
        if index < 0 {
            return Ok(None);
        }
        let concrete_len = pyre_object::w_tuple_len(tuple_obj);
        if index as usize >= concrete_len {
            return Ok(None);
        }
        (index, concrete_len)
    };

    // Authentic boxed result from the same may-force path the generic leg uses.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // guard_class TUPLE (skip when class already known / operand is constant).
    let tuple_type_addr = &pyre_object::pyobject::TUPLE_TYPE as *const _ as i64;
    if !list_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(list_op) {
        let type_const = ctx.trace_ctx.const_int(tuple_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[list_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(list_op, tuple_type_addr);

    // A tuple SUBCLASS instance shares `ob_type == &TUPLE_TYPE` (so it passes
    // the GuardClass above) but retags `w_class` and may override
    // `__getitem__`; guard the exact canonical `w_class` so such an instance
    // side-exits to the generic residual (which honours the override) rather
    // than taking this pure `wrappeditems[i]` load.
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        list_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
    )?;

    // Unbox the index operand (guard_class + getfield intval).  bool shares
    // int's `intval`, so a bool index guards its own &BOOL_TYPE.
    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(index));

    // getfield(wrappeditems): Ptr(GcArray(OBJECTPTR)) body.
    let items_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        list_op,
        crate::descr::tuple_wrappeditems_descr(),
    );

    // Bounds length: arraylen_gc against the wrappeditems GcArray header
    // (no inline length cache).  NON-pure (G2): an out-of-range index must
    // still deopt.
    let lenbox = crate::state::opimpl_arraylen_gc(
        ctx.trace_ctx,
        items_block,
        crate::state::pyobject_gcarray_descr(),
    );
    // Two-sided bounds guard `0 <= raw_index < len`.  The trace is recorded
    // from a non-negative observed index, but a later NEGATIVE index would
    // still satisfy `raw_index < len` and reach the PURE element load out of
    // range.  `space.getitem` treats a negative index as `index + len`
    // (tupleobject.py:468); the lower-bound guard deopts so that remap
    // re-executes generically instead of reading before the array.
    let zero = ctx.trace_ctx.const_int(0);
    let nonneg = ctx.trace_ctx.record_op(OpCode::IntGe, &[raw_index, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonneg, majit_ir::Value::Int((index >= 0) as i64));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonneg])?;
    let in_bounds = ctx.trace_ctx.record_op(OpCode::IntLt, &[raw_index, lenbox]);
    ctx.trace_ctx.set_opref_concrete(
        in_bounds,
        majit_ir::Value::Int(((index as usize) < concrete_len) as i64),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_bounds])?;

    // PURE element load.  Object storage reads the boxed Ref directly from
    // the immutable `Ptr(GcArray(OBJECTPTR))` body (no unbox/rebox).
    let boxed =
        crate::state::trace_items_block_getitem_value_pure(ctx.trace_ctx, items_block, raw_index);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// `len(x)` on an exact canonical `W_ListObject` / `W_UnicodeObject` /
/// `W_TupleObject`:
/// lower the opaque `bh_call_fn(len_builtin, PY_NULL, x)` residual to the
/// inline length read the meta-tracer produces upstream
/// (descroperation.py:294 `_len`): `guard_value(callable)` +
/// `guard_class` + exact `w_class` guard + length `getfield_gc_i` +
/// `wrapint`.  For a list this reads `W_ListObject.length()` →
/// `strategy.length`, so it additionally emits `guard_value(strategy)`
/// (rlist.py); for a str it reads the codepoint field directly
/// (`W_UnicodeObject.len` → `bh_unicodelen`, no storage strategy); for a
/// tuple it reads `wrappeditems` and applies `arraylen_gc`, matching
/// `tupleobject.py` where the tuple carries no separate length field. The
/// exact `w_class` guard is required because a SUBCLASS shares
/// `ob_type == &LIST_TYPE`/`&STR_TYPE`/`&TUPLE_TYPE` but may override `__len__`
/// (`baseobjspace::len` dispatches `subclass_special_override`); it
/// side-exits to the generic residual.
///
/// Returns `None` (fall through to the generic residual, SAFE) for any
/// other shape: non-list/str/tuple arg, a subclass, empty-strategy list, a
/// bound receiver, or wrong arity.
pub(crate) fn try_walker_specialize_builtin_len<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, arg)` shape only.
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(list_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `len(x)` call.
    if concrete_callable.is_null() || !null_or_self.is_null() || list_obj.is_null() {
        return Ok(None);
    }
    if !pyre_interpreter::builtins::is_builtin_len_function(concrete_callable) {
        return Ok(None);
    }
    // Exact canonical list / str / tuple only (see the doc comment on the subclass
    // `__len__` hazard).  `arg_type_addr` / `exact_w_class` pin the guard
    // target; the booleans select the length path.
    let (arg_type_addr, exact_w_class, is_str, is_tuple) = unsafe {
        if std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::LIST_TYPE)
            && std::ptr::eq(
                (*list_obj).w_class,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
            )
        {
            (
                &pyre_object::pyobject::LIST_TYPE as *const _ as i64,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
                false,
                false,
            )
        } else if std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::STR_TYPE)
            && std::ptr::eq(
                (*list_obj).w_class,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE),
            )
        {
            (
                &pyre_object::pyobject::STR_TYPE as *const _ as i64,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE),
                true,
                false,
            )
        } else if std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::TUPLE_TYPE)
            && std::ptr::eq(
                (*list_obj).w_class,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
            )
        {
            (
                &pyre_object::pyobject::TUPLE_TYPE as *const _ as i64,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
                false,
                true,
            )
        } else {
            return Ok(None);
        }
    };
    // Length source: str reads the codepoint field directly (no storage
    // strategy); tuple reads the wrappeditems GcArray header; list resolves
    // its storage strategy (guarded below). `sid` is `None` for str/tuple.
    let (sid, concrete_len) = unsafe {
        if is_str {
            (None, pyre_object::w_str_len(list_obj))
        } else if is_tuple {
            (None, pyre_object::w_tuple_len(list_obj))
        } else {
            let concrete_len = pyre_object::w_list_len(list_obj);
            let sid = if pyre_object::w_list_uses_int_storage(list_obj) {
                1i64
            } else if pyre_object::w_list_uses_float_storage(list_obj) {
                2i64
            } else if pyre_object::w_list_uses_object_storage(list_obj) {
                0i64
            } else {
                // Empty-strategy list: no length field to read.
                return Ok(None);
            };
            (Some(sid), concrete_len)
        }
    };

    // Authentic boxed result, produced on the plain eval loop exactly as
    // the skipped residual would (len on an exact list is side-effect-free).
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[list_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // Pin the callable identity (LOAD_GLOBAL `len` is usually already a
    // constant via the namespace cell fold).
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    let list_op = r_args[2];
    // guard_class (skip when class already known / operand is constant).
    if !list_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(list_op) {
        let type_const = ctx.trace_ctx.const_int(arg_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[list_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(list_op, arg_type_addr);
    walker_guard_exact_w_class(ctx, op.pc, list_op, exact_w_class)?;
    // Length read.  list: guard the storage strategy, then read that
    // strategy's length field (rlist.py:116 inline field for object storage;
    // typed items-block length for int/float storage).  str: a plain
    // codepoint-length getfield (no strategy, `bh_unicodelen`).
    let raw_len = if let Some(sid) = sid {
        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::list_strategy_descr(),
        );
        let sid_const = ctx.trace_ctx.const_int(sid);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[strategy, sid_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy, sid_const);
        let len_descr = match sid {
            0 => crate::descr::list_length_descr(),
            1 => crate::descr::list_int_items_len_descr(),
            _ => crate::descr::list_float_items_len_descr(),
        };
        crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, list_op, len_descr)
    } else if is_tuple {
        let wrappeditems = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            list_op,
            crate::descr::tuple_wrappeditems_descr(),
        );
        crate::state::opimpl_arraylen_gc(
            ctx.trace_ctx,
            wrappeditems,
            crate::state::pyobject_gcarray_descr(),
        )
    } else {
        crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, list_op, crate::descr::str_len_descr())
    };
    ctx.trace_ctx
        .set_opref_concrete(raw_len, majit_ir::Value::Int(concrete_len as i64));
    let boxed = walker_box_int(ctx, op.pc, raw_len, concrete_len as i64)?;
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        box_int_concrete(concrete_len as i64, boxed_result as i64),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// #171 ORTHODOX descent of the real `w_list_append` charon body (WIP).
///
/// Instead of hand-rolling the int-storage append IR (the fold below), walk
/// the compiled `w_list_append` jitcode (`list_append_jitcode()`): its
/// strategy `switch` folds to `guard_value(strategy==Integer)` over the
/// concrete receiver, the `is_plain_int1` / `plain_int_w` leaves recurse via
/// `inline_call`, the `ll_list_int_*` leaves are oopspec-lowered to
/// getfield/setfield/setarrayitem, and the capacity `goto_if_not` guards the
/// spare-capacity fast path.
///
/// The sub-walk's guards must resume at the `lst.append` CALL site (re-execute
/// the append generically on deopt — any of strategy / plain-int / capacity
/// failing).  The inline-subwalk capture (`walker_capture_snapshot_for_last_
/// guard_impl` single-frame fallthrough) reads `ctx.{outer_active_boxes,
/// outer_jitcode_index,entry_py_pc}` + the vable shadow directly, so this
/// pre-publishes that ONE call-site coordinate (mapped from `op.pc`) before
/// the sub-walk with inline-subwalk mode enabled.
///
/// Like the fold, the walker only RECORDS the array-op IR; this applies the
/// append to the concrete list + journals the rewind.  Declines (`Ok(None)`)
/// BEFORE emitting any IR for a non-matching shape (SAFE fallback to the fold
/// / residual).
///
/// STATUS: the descr-pool
/// wiring, the host-static const relocation, and the list header field
/// descr-group bridge (`make_descr_from_bh` strategy/length/items →
/// `W_LIST_DESCR_GROUP`) are all in place — the strategy `switch` and the
/// inlined `is_int`/`is_bool` type predicates fold over the concrete receiver,
/// the `W_ListObject.strategy` read resolves a parent_descr, and the walk
/// descends the full append into the Integer fast-path.  The unit-`()` return
/// aggregate (`SyntheticTransparentCtor "Tuple"`) is elided to `ConstRefNull`
/// at build time (`jtransform.rs`), so the descent completes and commits a
/// working trace.  Safety net: if a stale build-time jitcode kept that ctor as
/// a symbolic (`>>47`) fnaddr, `try_execute_residual_call_via_executor`
/// declines it (`OrthodoxSubWalkTraceUnsupported`) and the descent aborts
/// gracefully (interpreter fallback) instead of baking the hash as a code
/// address and branching to garbage.
pub(crate) fn try_walker_orthodox_list_append<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(callable), ConcreteValue::Ref(null_or_self), ConcreteValue::Ref(value)) =
        (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if callable.is_null() || !null_or_self.is_null() || value.is_null() {
        return Ok(None);
    }

    // Recognition: the callable must be the bound builtin `list.append`; the
    // receiver + value then pass the shared storage/spare-capacity gate.
    let (inner_func, inner_self, len_before) = unsafe {
        if !pyre_object::function::is_method(callable) {
            return Ok(None);
        }
        let inner_func = pyre_object::function::w_method_get_func(callable);
        let inner_self = pyre_object::function::w_method_get_self(callable);
        if inner_func.is_null() || inner_self.is_null() {
            return Ok(None);
        }
        let list_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE);
        if pyre_interpreter::lookup_in_type(list_type, "append") != Some(inner_func) {
            return Ok(None);
        }
        let Some(len_before) = orthodox_list_append_recognize(inner_self, value) else {
            return Ok(None);
        };
        (inner_func, inner_self, len_before)
    };

    // Resolve the compiled `w_list_append` body + the full-body sym (the
    // resume-coordinate source) BEFORE emitting any guard — a decline must
    // leave the trace untouched.
    let Some((sub_body, sym_ptr)) = orthodox_list_append_body_and_sym(ctx) else {
        return Ok(None);
    };
    // SAFETY: `sym_ptr` is non-null with a set `jitcode` (checked in the
    // resolver) and stays live for the enclosing full-body walk.
    let sym = unsafe { &*sym_ptr };

    // ── commit (record IR; no further declines) ──
    let callable_op = r_args[0];
    let value_op = r_args[2];

    // Pin the callable to `list.append`: guard_class METHOD + guard_value on
    // the stable function slot (these guards resume via the full-body path at
    // `op.pc`, ignoring the call-site fields set below).
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    if !callable_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(callable_op) {
        let type_const = ctx.trace_ctx.const_int(method_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[callable_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(callable_op, method_type_addr);
    let func_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_function_descr(),
    );
    let func_const = ctx.trace_ctx.const_ref(inner_func as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[func_ref, func_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(func_ref, func_const);

    // Recover the receiver list OpRef (the sub-walk reads it as ref-arg 0);
    // `orthodox_list_append_commit` stamps it concrete.
    let self_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_self_descr(),
    );

    orthodox_list_append_commit(
        ctx, op, sym, &sub_body, self_ref, value_op, inner_self, value, len_before,
    )?;

    // The `list.append(x)` call's `None` return (the residual's Ref dst).
    let none_ref = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', none_ref)?;
    Ok(Some(()))
}

/// Shared recognition for the #171 orthodox list-append fold: the receiver
/// must be a list with spare capacity whose storage strategy matches the
/// value's strict type predicate (Integer / Object / Float).  Returns the
/// list length before the append (the journal rewind point) on a match, or
/// `None` (decline) otherwise.  No IR is emitted.
///
/// # Safety
/// `inner_self` / `value` must be live `PyObjectRef`s.
unsafe fn orthodox_list_append_recognize(
    inner_self: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
) -> Option<usize> {
    // `is_plain_int1` accepts a fits-int `W_LongObject` (it implies
    // `_fits_int()`), but a long is declined here: the commit path pins
    // `guard_class(value, INT_TYPE)` and a long has `ob_type == LONG_TYPE`,
    // so supporting it needs equivalent `unbox_long` machinery
    // (guard_class LONG_TYPE + `_fits_int` residual guard + long
    // extraction) threaded through the sub-walk (PR248 §2). Empirically a
    // fits-int `W_LongObject` does not reach this append: pyre normalizes
    // fits-int results to `W_IntObject` across arithmetic / `int(str)` /
    // literals, so the long arm is an unreachable optimization and the
    // decline is correctness-safe (the generic residual handles it).
    if !pyre_object::pyobject::is_list(inner_self) {
        return None;
    }
    // Empty-strategy first-append promotion (gated). `w_list_can_append_without_realloc`
    // is false for Empty (no backing block yet), so classify by the value's
    // type using switch_to_correct_strategy's int -> float -> object order
    // (listobject.py:1154) and let the commit path install the typed storage.
    if pyre_object::w_list_uses_empty_storage(inner_self) {
        if !empty_append_virt_enabled() {
            // Gate off: preserve the prior behavior (Empty always declined).
            return None;
        }
        let int_ok = pyre_object::is_plain_int1(value)
            && !pyre_object::pyobject::is_long(value)
            && !(pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(value));
        let float_ok = !value.is_null() && pyre_object::is_plain_float_strict(value);
        // switch_to_correct_strategy routes `is_plain_int1` -> Integer with no
        // tagged exclusion. Exclude any plain-int / float from the object
        // fallback so a tagged-int / fits-int `W_LongObject` DECLINES (generic
        // residual) instead of mis-routing to Object and diverging the traced
        // strategy from the concrete one the commit installs.
        let obj_ok = !value.is_null()
            && !pyre_object::is_plain_int1(value)
            && !pyre_object::is_plain_float_strict(value);
        if !int_ok && !float_ok && !obj_ok {
            return None;
        }
        // Empty length is 0 (the journal rewind point).
        return Some(0);
    }
    if !pyre_object::w_list_can_append_without_realloc(inner_self) {
        return None;
    }
    // Int-storage specialization: plain-int value stored unboxed (a
    // fits-int `W_LongObject` is declined, see note above).
    // A tagged-immediate value would need a tag-aware unboxed store and no
    // `w_class` pin; decline to the generic residual append instead.
    let int_ok = pyre_object::w_list_uses_int_storage(inner_self)
        && pyre_object::is_plain_int1(value)
        && !pyre_object::pyobject::is_long(value)
        && !(pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(value));
    // Object-storage extension: any non-null `Ref` value stored into the
    // object items block — no unboxing, so the value carries no type
    // precondition.
    let obj_ok = pyre_object::w_list_uses_object_storage(inner_self) && !value.is_null();
    // Float-storage specialization: a strict `W_FloatObject` stored
    // unboxed. `FloatListStrategy.is_correct_type` (listobject.py:2061) is
    // `type(w_obj) is W_FloatObject`, the strict predicate the body's Float
    // arm also uses. No fits-* long analogue (a float is never re-boxed
    // across arithmetic, unlike a fits-int W_LongObject).
    let float_ok = pyre_object::w_list_uses_float_storage(inner_self)
        && !value.is_null()
        && pyre_object::is_plain_float_strict(value);
    if !int_ok && !obj_ok && !float_ok {
        return None;
    }
    Some(pyre_object::w_list_len(inner_self))
}

/// Resolve the compiled `w_list_append` body + the full-body snapshot sym
/// (the resume-coordinate source) shared by both list-append fold forms.
/// Returns `None` (decline — no IR emitted yet) when the body jitcode is not
/// compiled or the snapshot sym is absent.  The returned `sym_ptr` is
/// non-null with a set `jitcode` field.
pub(crate) fn orthodox_list_append_body_and_sym<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(SubJitCodeBody, *const Sym)> {
    let jc_arc = crate::jitcode_runtime::list_append_jitcode()?;
    let sub_body = sub_jitcode_body_by_index(jc_arc.index())?;
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    // SAFETY: set for the lifetime of the enclosing full-body walk.
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return None;
    }
    Some((sub_body, sym_ptr))
}

/// Commit core of the #171 orthodox list-append fold, shared by the
/// method-call (`try_walker_orthodox_list_append`) and LIST_APPEND-opcode
/// (`try_walker_orthodox_list_append_opcode`) forms.  Stamps the receiver
/// concrete, pins the value's class (Integer/Float storage), publishes the
/// single append-site resume coordinate, descends the real `w_list_append`
/// body as a sub-jitcode walk recording its native array store, then journals
/// + applies the concrete append.  `self_ref` is the receiver list OpRef the
/// caller supplies (the bound method's `w_self` field, or the opcode's list
/// operand); `sym` / `sub_body` are the pre-resolved resume source + callee
/// body.  The caller writes any residual result (the method form's `None`; the
/// opcode form is void).  Records IR unconditionally — a body sub-walk abort
/// propagates as `DispatchError` (graceful interpreter fallback), never a wrong
/// trace.
#[allow(clippy::too_many_arguments)]
pub(crate) fn orthodox_list_append_commit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    sym: &Sym,
    sub_body: &SubJitCodeBody,
    self_ref: OpRef,
    value_op: OpRef,
    inner_self: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
    len_before: usize,
) -> Result<(), DispatchError> {
    // `w_list_append` unboxes its `value` inside an inline sub-walk.  A
    // virtual range item must be materialized at that call boundary: otherwise
    // the sub-walk's snapshot exports its raw payload as a loop-carried scalar,
    // which makes a module-cell reload retain the trace-entry value.  The
    // identity ptr→int→ptr pair is the normal forcing shape: it preserves the
    // live SSA Ref while making the virtual allocation observable to the
    // optimizer, so the descended `plain_int_w` reads the current iteration's
    // payload (as the real `w_list_append` call does).
    let value_as_int = ctx.trace_ctx.record_op(OpCode::CastPtrToInt, &[value_op]);
    ctx.trace_ctx
        .set_opref_concrete(value_as_int, Value::Int(value as usize as i64));
    let value_op = ctx
        .trace_ctx
        .record_op(OpCode::CastIntToPtr, &[value_as_int]);
    ctx.trace_ctx
        .set_opref_concrete(value_op, Value::Ref(majit_ir::GcRef(value as usize)));
    // Stamp the receiver concrete (the sub-walk reads it as ref-arg 0; its
    // strategy switch needs the concrete receiver).
    ctx.trace_ctx.set_opref_concrete(
        self_ref,
        majit_ir::Value::Ref(majit_ir::GcRef(inner_self as usize)),
    );

    // Empty-strategy first-append promotion (gated): install typed storage on
    // the receiver BEFORE the value-class pin / storage read below, so those
    // observe the post-promotion strategy. Classify the target strategy from
    // the value with recognize's int -> float -> object guards
    // (switch_to_correct_strategy, listobject.py:1154), then emit the
    // transition IR mutating the existing wrapper, promote the concrete list,
    // and journal the rewind to Empty.
    use pyre_object::listobject::ListStrategy;
    let promote_empty = empty_append_virt_enabled()
        && unsafe { pyre_object::w_list_uses_empty_storage(inner_self) };
    if promote_empty {
        let target = unsafe {
            let int_ok = pyre_object::is_plain_int1(value)
                && !pyre_object::pyobject::is_long(value)
                && !(pyre_object::tagged_int::CAN_BE_TAGGED
                    && pyre_object::tagged_int::is_tagged_int(value));
            if int_ok {
                ListStrategy::Integer
            } else if !value.is_null() && pyre_object::is_plain_float_strict(value) {
                ListStrategy::Float
            } else {
                ListStrategy::Object
            }
        };
        // Guard the current (Empty) strategy so a deopt re-enters the empty
        // path (mirror of `MIFrame::guard_list_strategy`: getfield strategy +
        // GuardValue + replace_box).
        let strategy_ref = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            self_ref,
            crate::descr::list_strategy_descr(),
        );
        let expected = ctx.trace_ctx.const_int(ListStrategy::Empty as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[strategy_ref, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy_ref, expected);
        // Emit the transition IR mutating the existing wrapper (helpers.rs).
        // The emitter seeds the new block's capacity getfield cache so the
        // append body sub-walk's spare-capacity `0 < capacity` check folds.
        crate::helpers::emit_promote_empty_list_inline(ctx.trace_ctx, self_ref, target);
        // Concrete promotion of the real list, then journal so a non-commit
        // walk rolls back to Empty.
        unsafe { pyre_object::w_list_switch_to_strategy_for(inner_self, value) };
        fbw_append_promote_journal_push(inner_self);
    }

    // Pin the appended value's class so the inlined `is_plain_int1` type
    // predicate folds during the sub-walk: guard_class(value, INT_TYPE) +
    // class_now_known, so its `is_int`/`is_bool` typeptr reads fold to the
    // INT_TYPE const (the typeptr fold in `getfield_gc_via_heapcache`).  The
    // recognition gate already proved `is_plain_int1(value)`; this guard
    // enforces ob_type==INT_TYPE at runtime.  The value's integer payload
    // stays symbolic — only its class is pinned.
    //
    // Object-storage append stores the value as a
    // plain GC ref with no unboxing, so it carries no type precondition —
    // skip the INT_TYPE pin (the sub-walk's object-storage store path does
    // not read the value's class).
    let is_obj_storage = unsafe { pyre_object::w_list_uses_object_storage(inner_self) };
    if !is_obj_storage {
        // Integer and Float storage both pin the value's class so the body's
        // strict type test folds during the sub-walk; only the ob_type const
        // differs (INT_TYPE vs FLOAT_TYPE).
        let is_float_storage = unsafe { pyre_object::w_list_uses_float_storage(inner_self) };
        let value_type_addr = if is_float_storage {
            &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64
        } else {
            &pyre_object::pyobject::INT_TYPE as *const _ as i64
        };
        if !value_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(value_op) {
            let type_const = ctx.trace_ctx.const_int(value_type_addr);
            ctx.trace_ctx
                .record_guard(OpCode::GuardClass, &[value_op, type_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(value_op, value_type_addr);
        // The strict predicate (`is_plain_int1` / `is_plain_float_strict`)
        // rejects subclasses by reading `value.w_class` and requiring it null
        // or == `get_instantiate(<type>)`. The ob_type pin above only folds the
        // `is_int`/`is_float` typeptr reads; the w_class compare stays symbolic,
        // so the inlined predicate is non-concrete and the strategy arm's
        // `if <pred>(value)` branch cannot fold — the sub-walk then descends the
        // dead else-leg `switch_to_object_strategy`, whose `ListStrategy::Object`
        // unit-variant ctor is a symbolic fnaddr the descent declines
        // (`OrthodoxSubWalkTraceUnsupported`). Pin w_class to the concrete
        // value's field so the subclass test folds too (the recognition gate
        // already proved the strict predicate).
        let concrete_w_class = unsafe { (*value).w_class } as i64;
        let w_class_ref = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            value_op,
            crate::descr::w_class_descr(),
        );
        let w_class_const = ctx.trace_ctx.const_ref(concrete_w_class);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[w_class_ref, w_class_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(w_class_ref, w_class_const);
    }

    // Pre-publish the ONE append-site resume coordinate the sub-walk's guards
    // collapse to (mirror the full-body path's last_instr / valuestackdepth
    // publication, keyed to the append op's py_pc — the CALL for the method
    // form, the LIST_APPEND for the opcode form).
    let (call_site_py_pc, vsd_value, outer_jitcode_index, call_site_marker) = unsafe {
        let jc = &*sym.jitcode();
        let jc_index = jc.index as u32;
        let marker = jc.payload.resume_marker_for_jitcode_pc(op.pc);
        let mut py = python_pc_for_jitcode_pc(&jc.payload.metadata, op.pc);
        if jc.payload.code_ptr.is_null() {
            (py, sym.valuestackdepth() as i64, jc_index, marker)
        } else {
            let codeobj = &*jc.payload.code_ptr;
            py = skip_python_trivia_forward(codeobj, py as usize) as u32;
            // Read the depth off the jitcode-pc-keyed trivia twin, which equals
            // `depth_at_py_pc()[skip_python_trivia_forward(python_pc_for_jitcode_pc(op.pc))]`
            // by construction; fall back to the py_pc-keyed static-liveness read
            // where the twin is unpopulated (skeleton / fixture install).
            let depth = if jc.payload.depth_trivia_populated() {
                jc.payload.depth_trivia_for_jitcode_pc(op.pc)
            } else {
                crate::liveness::liveness_for(jc.payload.code_ptr)
                    .depth_at_py_pc()
                    .get(py as usize)
                    .copied()
            };
            let vsd = match depth {
                Some(d) => (sym.nlocals() + d as usize) as i64,
                None => sym.valuestackdepth() as i64,
            };
            (py, vsd, jc_index, marker)
        }
    };
    if sym.owns_virtualizable_shadow() {
        let li = call_site_py_pc as i64 - 1;
        let li_op = ctx.trace_ctx.const_int(li);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "last_instr",
            li_op,
            Value::Int(li),
        );
        let vsd_op = ctx.trace_ctx.const_int(vsd_value);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "valuestackdepth",
            vsd_op,
            Value::Int(vsd_value),
        );
    }
    let call_site_word = match call_site_marker {
        Some(m) => m as i32,
        None => majit_ir::resumedata::NO_JITCODE_PC,
    };
    let active = collect_outer_active_boxes(
        sym,
        ctx.trace_ctx,
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        outer_jitcode_index,
        call_site_py_pc,
        None,
        call_site_word,
        // As above, entry metadata is keyed by the append op itself; its
        // liveness-bank query remains keyed by the resume marker.
        op.pc as i32,
        OuterActiveBoxesEntryTwin::Plain,
        "w_list_append_call_site",
        None,
        &[],
    );

    // Swap in the call-site resume context + the callee's GLOBAL descr pool
    // for the sub-walk, restore after.  `w_list_append` is a build-time
    // canonical body with no per-fn descr pool, so its `d`/`j` operands
    // resolve through `all_descr_refs()` / `RawDescrPool::Global` — NOT the
    // parent loop's per-fn pool (which mis-resolves the first residual_call
    // descr → `ResidualCallDescrNotCallDescr`).
    let saved_entry = ctx.entry_py_pc;
    let saved_marker = ctx.outer_resume_marker_jit_pc;
    let saved_oji = ctx.outer_jitcode_index;
    let saved_active = std::mem::take(&mut ctx.outer_active_boxes);
    let saved_descr_refs = ctx.descr_refs;
    let saved_raw_descrs = ctx.raw_descrs;
    let saved_lookup = ctx.sub_jitcode_lookup;
    ctx.entry_py_pc = EntryPyPc::Jit(op.pc);
    ctx.outer_resume_marker_jit_pc = call_site_marker;
    ctx.outer_jitcode_index = outer_jitcode_index;
    ctx.outer_active_boxes = active;
    ctx.descr_refs = crate::jitcode_runtime::all_descr_refs();
    ctx.raw_descrs = RawDescrPool::Global;
    ctx.sub_jitcode_lookup = &GLOBAL_SUB_JITCODE_LOOKUP_FN;

    let self_concrete = ConcreteValue::Ref(inner_self);
    let value_concrete = ConcreteValue::Ref(value);
    let saved_fbw_mode = ctx.fbw_mode;
    ctx.fbw_mode.inline_subwalk = true;
    let walk_result = run_sub_jitcode_walk(
        ctx,
        op.pc,
        sub_body,
        &[],
        &[],
        &[self_ref, value_op],
        &[self_concrete, value_concrete],
        &[],
    );
    ctx.fbw_mode = saved_fbw_mode;

    ctx.entry_py_pc = saved_entry;
    ctx.outer_resume_marker_jit_pc = saved_marker;
    ctx.outer_jitcode_index = saved_oji;
    ctx.outer_active_boxes = saved_active;
    ctx.descr_refs = saved_descr_refs;
    ctx.raw_descrs = saved_raw_descrs;
    ctx.sub_jitcode_lookup = saved_lookup;

    match walk_result? {
        DispatchOutcome::SubReturn { result: None } => {}
        _ => return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc }),
    }

    // Reaching here means the body sub-walk completed without hitting an
    // un-lowered helper: the strategy switch folded over the concrete
    // receiver, the strict type-predicate leaves recursed (`is_plain_int1`
    // for Integer / `is_plain_float_strict` for Float; Object stores with no
    // type test), the `ll_list_{int,float,obj}_*` leaves lowered to
    // getfield/setfield/setarrayitem, and the unit-`()` return aggregate
    // (`SyntheticTransparentCtor "Tuple"`) was elided to `ConstRefNull` at
    // build time.  Any residual that does NOT lower —
    // e.g. a stale build-time jitcode whose tuple ctor kept a symbolic
    // `>>47` funcbox — is declined by `try_execute_residual_call_via_executor`
    // (`OrthodoxSubWalkTraceUnsupported`) and `walk_result?` propagates that
    // abort before this point (graceful interpreter fallback, never a wrong
    // trace).  The descr-pool wiring above (strategy/header field descrs) is
    // exercised on the way in.

    // Tracing is execution: apply the append + journal the rewind (the walker
    // recorded the IR but did not mutate the concrete list).
    fbw_append_journal_push(inner_self, len_before);
    unsafe { pyre_object::w_list_append(inner_self, value) };
    Ok(())
}

/// LIST_APPEND-opcode form of the #171 orthodox list-append fold (comprehension
/// append, e.g. `[f(x) for x in xs]` inlines LIST_APPEND into the enclosing
/// function).  The codewriter lowers LIST_APPEND to a void
/// `jit_list_append(list, value)` residual tagged `ListAppendValue`; here
/// `r_args = [list, value]` (the peeked receiver + the popped value — no
/// bound-method callable).  Recognises the receiver/value against the shared
/// gate and descends the same `w_list_append` body as the method-call form
/// ([`try_walker_orthodox_list_append`]).  Returns `None` (fall through to the
/// generic residual, SAFE — identical to the retired MIFrame tracer's `jit_list_append`)
/// for any non-matching shape; the residual is void so no result is written.
pub(crate) fn try_walker_orthodox_list_append_opcode<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let _ = dst; // LIST_APPEND residual is void — no result to write.
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(list), ConcreteValue::Ref(value)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    if list.is_null() || value.is_null() {
        return Ok(None);
    }

    // Recognition: no bound-method callable to pin — the list and value are the
    // residual's two Ref operands directly.
    let Some(len_before) = (unsafe { orthodox_list_append_recognize(list, value) }) else {
        return Ok(None);
    };

    // Resolve the compiled body BEFORE emitting any IR — the opcode form emits
    // no guard before the commit, so this is the only decline point.
    let Some((sub_body, sym_ptr)) = orthodox_list_append_body_and_sym(ctx) else {
        return Ok(None);
    };
    // SAFETY: `sym_ptr` is non-null with a set `jitcode` (checked in the
    // resolver) and stays live for the enclosing full-body walk.
    let sym = unsafe { &*sym_ptr };

    // ── commit (record IR; no further declines) ──
    // The receiver list OpRef + value OpRef are the residual's Ref operands.
    orthodox_list_append_commit(
        ctx, op, sym, &sub_body, r_args[0], r_args[1], list, value, len_before,
    )?;
    Ok(Some(()))
}

/// B3 (`PYRE_FBW_RAISE`): walker-native exception-construction fold.  A
/// `Type(args)` `CallFn` residual for a canonical builtin exception class or
/// a heap subclass with the same `__new__` / `__init__` descriptors becomes a
/// traced `NewWithVtable` + `SetfieldGc` (kind / w_class / args_w) the
/// optimizer can virtualize when the exception never escapes, instead of
/// the opaque `bh_call_fn` constructor residual + its
/// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION`.
///
/// The `CallFn` arglist is `r_args = [callable, PY_NULL, args...]`
/// (the `bh_call_fn_N` shape — see `try_walker_specialize_list_append`);
/// the positional args are `r_args[2..]` (the `PY_NULL` self slot is
/// skipped).  Records the fresh `NewWithVtable` OpRef in
/// [`FBW_BUILT_EXC`] so a following `RaiseVarargs` takes the instance
/// fast path; writes the trace-time concrete exception into the dst
/// shadow so the `raise/r` GUARD_CLASS reads it.
///
/// PyPy's `W_TypeObject.descr_call` promotes the class, then resolves
/// `__new__` and `__init__` through its versioned MRO
/// (`typeobject.py:703-735`).  When both resolve to
/// `W_BaseException.descr_new` / `descr_init`
/// (`interp_exceptions.py:76-126`), a trivial subclass has the same traced
/// allocation and `args_w` store as its builtin base; only `w_class` differs.
///
/// Returns `None` (fall through to the generic residual) for any non-matching
/// shape: an overriding or uncacheable subclass, a non-trivial-args kind
/// (OSError / Unicode errors store extra fields), or a null concrete arg.
pub(crate) fn try_walker_trace_exception_new<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, args...)` shape only.
    if r_args.len() < 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `Type(args)` call.
    if concrete_callable.is_null() || !null_or_self.is_null() {
        return Ok(None);
    }

    // Concrete positional args (skip callable + PY_NULL self).  The
    // residual `args_w` list must match the runtime `descr_init` list
    // exactly, so reject any null.
    let args = &r_args[2..];
    let concrete_args: Vec<pyre_object::PyObjectRef> = arg_concretes[2..]
        .iter()
        .map(|c| match c {
            ConcreteValue::Ref(p) => *p,
            _ => std::ptr::null_mut(),
        })
        .collect();

    let is_exc_class = unsafe {
        pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(concrete_callable)
    };
    if !is_exc_class || concrete_args.iter().any(|a| a.is_null()) {
        return Ok(None);
    }

    // Decide the final runtime `args_w` list strategy and extract typed
    // payloads.  OSError can rebind the list after parsing a filename, so its
    // final slice is selected below after the concrete constructor exposes the
    // value-dependent branch result.  The shared typed-list emitter reproduces
    // `w_list_new`'s Integer layout, allowing the common `SystemExit(i)` shape
    // to virtualize alongside message-bearing Object lists.  Other strategies
    // retain the safe residual fallback.
    enum ArgsEmit {
        Object,
        Int(Vec<i64>),
    }

    let is_canonical = pyre_object::interp_exceptions::is_canonical_exc_class(concrete_callable);
    let mut subclass_lookups = None;
    let subclass_version_tag = if is_canonical {
        None
    } else {
        // A heap subclass is safe to construct concretely only after both MRO
        // lookups have been proved identical to a canonical exception class.
        // Consequently force_plain_eval below can execute only the builtin
        // Rust `descr_new` / `descr_init`, never user Python code.  This is the
        // promoted-class lookup contract of typeobject.py:703-735.
        if !unsafe { pyre_object::typeobject::w_type_is_heaptype(concrete_callable) } {
            return Ok(None);
        }
        let version_tag =
            unsafe { pyre_object::typeobject::w_type_get_version_tag(concrete_callable) };
        if version_tag == 0 {
            return Ok(None);
        }
        let Some(class_new) = (unsafe {
            pyre_interpreter::baseobjspace::lookup_in_type(concrete_callable, "__new__")
        }) else {
            return Ok(None);
        };
        let Some(class_init) = (unsafe {
            pyre_interpreter::baseobjspace::lookup_in_type(concrete_callable, "__init__")
        }) else {
            return Ok(None);
        };
        let matches_canonical = (0..pyre_object::interp_exceptions::EXC_KIND_COUNT).any(|disc| {
            // ExcKind is repr(u8) with contiguous discriminants through
            // EXC_KIND_COUNT, as required by the kind-indexed registry.
            let candidate_kind: pyre_object::interp_exceptions::ExcKind =
                unsafe { std::mem::transmute(disc as u8) };
            let candidate =
                pyre_object::interp_exceptions::lookup_exc_class_for_kind(candidate_kind);
            if candidate.is_null() {
                return false;
            }
            unsafe {
                pyre_interpreter::baseobjspace::lookup_in_type(candidate, "__new__")
                    == Some(class_new)
                    && pyre_interpreter::baseobjspace::lookup_in_type(candidate, "__init__")
                        == Some(class_init)
            }
        });
        if !matches_canonical {
            return Ok(None);
        }
        subclass_lookups = Some((class_new, class_init));
        Some(version_tag)
    };
    // Build the exception concretely on the plain eval loop (no tracer
    // re-entry) to read its kind and confirm a flat builtin instance.
    // Trace-time only; discarded after the read.
    let exc = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &concrete_args)
    };
    let Ok(exc) = exc else { return Ok(None) };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    let canonical_class = pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind);
    if is_canonical {
        // Preserve the canonical arm's registry identity check.
        if canonical_class != concrete_callable {
            return Ok(None);
        }
    } else {
        // The pre-construction descriptor check excludes Python execution;
        // repeat it for the concrete result's eventual kind so aliases whose
        // builtin wrapper produces a different physical kind still decline.
        let Some((class_new, class_init)) = subclass_lookups else {
            return Ok(None);
        };
        if canonical_class.is_null()
            || unsafe {
                pyre_interpreter::baseobjspace::lookup_in_type(canonical_class, "__new__")
                    != Some(class_new)
                    || pyre_interpreter::baseobjspace::lookup_in_type(canonical_class, "__init__")
                        != Some(class_init)
            }
        {
            return Ok(None);
        }
    }
    // `exc_new_wrapper` retags only `w_class`; the physical layout remains
    // the eventual kind's builtin pytype for canonical classes and subclasses.
    let exc_type_ptr = unsafe {
        (*(exc as *const pyre_object::interp_exceptions::W_BaseException))
            .ob_header
            .ob_type
    };
    if !std::ptr::eq(
        exc_type_ptr,
        pyre_object::interp_exceptions::exc_kind_to_pytype(kind),
    ) {
        return Ok(None);
    }
    let is_os_error_family = matches!(
        kind,
        pyre_object::interp_exceptions::ExcKind::OSError
            | pyre_object::interp_exceptions::ExcKind::FileNotFoundError
    );
    // `W_OSError._parse_init_args` / `_init_error`
    // (`interp_exceptions.py`) fill the flattened slots only for 2..=5
    // arguments.  Outside that range the ordinary args-only emit is exact.
    // Unicode constructors still require their dedicated parsing and remain
    // residual.
    let fills_os_error_slots = is_os_error_family && (2..=5).contains(&args.len());
    if !kind.has_trivial_args_constructor() && !is_os_error_family {
        return Ok(None);
    }

    let exact_os_error = pyre_interpreter::builtins::lookup_exc_class("OSError")
        .is_some_and(|w_os_error| std::ptr::eq(concrete_callable, w_os_error));
    if fills_os_error_slots && exact_os_error {
        // PyPy traces the errno-to-subclass lookup with a loop-variant errno.
        // The flat NewWithVtable emit needs a constant w_class, so pinning the
        // unboxed errno deliberately creates per-errno traces/bridges.
        let errno = concrete_args[0];
        let exact_int = pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(errno)
            || unsafe {
                pyre_object::is_plain_int1(errno)
                    && std::ptr::eq(
                        (*errno).ob_type,
                        &pyre_object::pyobject::INT_TYPE as *const _,
                    )
            };
        if !exact_int {
            return Ok(None);
        }
    }

    let concrete_w_class = unsafe { (*exc).w_class };
    let is_blocking_io_error = pyre_interpreter::builtins::lookup_exc_class("BlockingIOError")
        .is_some_and(|blocking| std::ptr::eq(concrete_w_class, blocking));
    // `W_OSError._init_error` gives an exact BlockingIOError's numeric third
    // argument the characters_written meaning.  Keep every three-or-more-arg
    // instance of that concrete class on the complete runtime path.
    if fills_os_error_slots && args.len() >= 3 && is_blocking_io_error {
        return Ok(None);
    }

    let has_filename = fills_os_error_slots
        && args.len() >= 3
        && !unsafe { pyre_object::is_none(concrete_args[2]) };
    let final_args_len = if has_filename { 2 } else { args.len() };
    let final_args = &args[..final_args_len];
    let final_concrete_args = &concrete_args[..final_args_len];
    let args_emit = match pyre_object::listobject::list_strategy_for(final_concrete_args) {
        pyre_object::listobject::ListStrategy::Object => ArgsEmit::Object,
        pyre_object::listobject::ListStrategy::Integer => {
            let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
            let mut values = Vec::with_capacity(final_concrete_args.len());
            for &arg in final_concrete_args {
                if pyre_object::tagged_int::CAN_BE_TAGGED
                    && pyre_object::tagged_int::is_tagged_int(arg)
                {
                    return Ok(None);
                }
                let exact_int = unsafe {
                    pyre_object::is_plain_int1(arg) && std::ptr::eq((*arg).ob_type, int_ty)
                };
                if !exact_int {
                    return Ok(None);
                }
                values.push(unsafe { pyre_object::w_int_get_value(arg) });
            }
            ArgsEmit::Int(values)
        }
        pyre_object::listobject::ListStrategy::Empty
        | pyre_object::listobject::ListStrategy::Float => return Ok(None),
    };

    // GuardClass pins each None-sensitive `_init_error` branch.  A tagged
    // immediate cannot be consumed by GuardClass; retain the residual path for
    // that uncommon filename shape.
    if fills_os_error_slots {
        for index in [2usize, 4] {
            if index >= args.len() || (index == 4 && args.len() != 5) {
                continue;
            }
            if pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(concrete_args[index])
            {
                return Ok(None);
            }
        }
    }
    // --- commit to the specialization: emit IR (no further declines) ---
    // Pin the callable identity so the trace-time kind / vtable stay
    // valid across iterations (`implement_guard_value`).
    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    if let Some(version_tag) = subclass_version_tag {
        // Guard the promoted class version that made both MRO descriptor
        // identities constant.  `W_TypeObject.mutated` recursively changes
        // subclass tags (`typeobject.py:266-291`), so mutating this class or a
        // base side-exits before reusing the folded constructor.
        let class_const = ctx.trace_ctx.const_ref(concrete_callable as i64);
        let live_version = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            class_const,
            crate::descr::type_version_tag_descr(),
        );
        let version_const = ctx.trace_ctx.const_int(version_tag as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[live_version, version_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(live_version, version_const);
    }

    if fills_os_error_slots && exact_os_error {
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        let raw_errno = walker_unbox_int(ctx, op.pc, args[0], int_type_addr)?;
        let errno_value = unsafe { pyre_object::w_int_get_value(concrete_args[0]) };
        let errno_const = ctx.trace_ctx.const_int(errno_value);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[raw_errno, errno_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(raw_errno, errno_const);
    }
    if fills_os_error_slots {
        for index in [2usize, 4] {
            if index >= args.len() || (index == 4 && args.len() != 5) {
                continue;
            }
            let arg = args[index];
            if !ctx.trace_ctx.heap_cache().is_class_known(arg) {
                let physical_type = unsafe { (*concrete_args[index]).ob_type } as i64;
                let type_const = ctx.trace_ctx.const_int(physical_type);
                walker_emit_fold_guard_with_snapshot(
                    ctx,
                    op.pc,
                    OpCode::GuardClass,
                    &[arg, type_const],
                )?;
                ctx.trace_ctx
                    .heap_cache_mut()
                    .class_now_known(arg, physical_type);
            }
        }
    }

    // Build `args_w` inline so its wrapper and backing block virtualize
    // alongside the exception.
    let args_list = match args_emit {
        ArgsEmit::Object => crate::helpers::emit_object_list_inline(ctx.trace_ctx, final_args),
        ArgsEmit::Int(values) => {
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            let mut raws = Vec::with_capacity(final_args.len());
            for (&arg, value) in final_args.iter().zip(values) {
                let raw = walker_unbox_int(ctx, op.pc, arg, int_type_addr)?;
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Int(value));
                raws.push(raw);
            }
            crate::helpers::emit_typed_list_inline(
                ctx.trace_ctx,
                &raws,
                crate::state::int_gcarray_descr(),
                crate::descr::list_int_items_len_descr(),
                crate::descr::list_int_items_block_descr(),
                pyre_object::listobject::ListStrategy::Integer,
            )
        }
    };
    // A raised exception can keep args_w live through the execution-context
    // slot, forcing the otherwise-virtual list.  Stamp the canonical list
    // class just as w_list_new does so that materialization preserves the
    // `space.type(args_w) is list` branch used by descr_getargs.
    let list_w_class = pyre_object::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let list_w_class = ctx.trace_ctx.const_ref(list_w_class as i64);
    let list_w_class_descr = crate::descr::list_w_class_descr();
    let list_w_class_index = list_w_class_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[args_list, list_w_class],
        list_w_class_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(args_list, list_w_class_index, list_w_class);

    // `W_OSError.descr_new` can retag exact OSError by errno while retaining
    // the OSError physical kind.  The guarded errno makes the concrete final
    // class a valid constant; dedicated classes and subclasses keep the called
    // class operand as in the ordinary constructor emit.
    let emitted_w_class = if fills_os_error_slots && exact_os_error {
        ctx.trace_ctx.const_ref(concrete_w_class as i64)
    } else {
        callable_op
    };
    let new_op =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, emitted_w_class, args_list);

    if fills_os_error_slots {
        use pyre_interpreter::baseobjspace::ExceptionAttrSlot;
        let mut stores = vec![
            (ExceptionAttrSlot::Errno, args[0]),
            (ExceptionAttrSlot::Strerror, args[1]),
        ];
        if has_filename {
            stores.push((ExceptionAttrSlot::Filename, args[2]));
            // The fourth positional argument is winerror and is ignored on
            // non-Windows builds, matching W_OSError._parse_init_args.
            if args.len() == 5 && !unsafe { pyre_object::is_none(concrete_args[4]) } {
                stores.push((ExceptionAttrSlot::Filename2, args[4]));
            }
        }
        for (slot, value) in stores {
            let descr = crate::descr::w_exception_slot_descr(kind, slot);
            let descr_index = descr.index();
            ctx.trace_ctx
                .record_op_with_descr(OpCode::SetfieldGc, &[new_op, value], descr);
            ctx.trace_ctx
                .heapcache_setfield_cached(new_op, descr_index, value);
        }
    }

    // Mark the class known so the following `raise/r` skips its
    // redundant GUARD_CLASS (mirrors the retired raise path's
    // `heapcache.class_now_known`).  The vtable on the NewWithVtable
    // already pins the class for the optimizer; this keeps the heapcache
    // model in agreement.
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new_op, exc_type_ptr as usize as i64);

    // Record the fresh instance so a following `RaiseVarargs` recovers
    // the concrete and takes the instance fast path; stamp the dst shadow
    // so the `raise/r` GUARD_CLASS reads it.
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    fbw_built_exc_insert(new_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new_op)?;
    Ok(Some(()))
}

/// B3 (`PYRE_FBW_RAISE`): walker-native RAISE_VARARGS E1 fast path. The `RaiseVarargs`
/// residual is `normalize_raise_varargs_jit(frame, exc, cause)` —
/// `r_args = [frame, exc, cause]`.  When `exc` was built inline by
/// [`try_walker_trace_exception_new`] (∈ [`FBW_BUILT_EXC`]) and there is
/// no explicit `from` cause (concrete `cause` is `PY_NULL`), skip the
/// residual publish + its `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION` and
/// emit `__context__` as a `SetfieldGc` on the (still virtual) exception:
///
///   active = GETFIELD_GC_R(ec, sys_exc_value)
///   SETFIELD_GC(exc, active, w_exception.w_context)
///
/// For a fresh exception `w_context` is null and the self-cycle is
/// impossible, so `attach_raise_cause`'s conditional `w_context = active`
/// reduces to the unconditional store (a null store when no exception is
/// active is a no-op that DCEs).  The normalized result is the same
/// instance for a flat builtin, so the inline-built `exc` OpRef is
/// written straight to the dst that fed the following `raise/r`.
///
/// Returns `None` (fall through to the residual) when `exc` was not
/// inline-built or a `from` cause is present.
pub(crate) fn try_walker_trace_raise_builtin<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let exc_op = r_args[1];
    // Take (remove) the inline-built marker: a second raise of the same
    // object (whose `w_context` is now stamped) must take the residual
    // path so its runtime `attach_raise_cause` keeps the existing
    // `__context__` and avoids the self-cycle.
    if !fbw_built_exc_take(exc_op) {
        return Ok(None);
    }
    // Explicit `raise X from Y` (concrete non-null cause) keeps the
    // residual: `attach_raise_cause` sets both `__cause__` and
    // `__suppress_context__`, which the inline `__context__` store alone
    // does not reproduce.  Re-insert the marker so the raise still routes
    // through the residual (the marker was consumed above).
    //
    // `raise X` without a cause lowers the cause operand to a const
    // `PY_NULL` (`ConstPtr(GcRef(0))`), whose concrete shadow is
    // `ConcreteValue::Null` (constant pool slots carry no `Ref` shadow);
    // `raise X from Y` passes a live non-null Ref.  Treat the const-null
    // operand AND a `ConcreteValue::Null`/`Ref(null)` shadow all as "no
    // cause"; any concrete non-null Ref is an explicit cause.
    let cause_op = r_args[2];
    let cause_concrete = read_ref_var_list_concrete(code, op, 1, ctx);
    let cause_is_null = match cause_concrete.get(2) {
        Some(ConcreteValue::Ref(p)) => p.is_null(),
        Some(ConcreteValue::Null) | None => {
            // No live concrete: the operand is "no cause" only if it is a
            // const PY_NULL.  A non-const opref with an unknown concrete
            // is conservatively treated as a possible cause (decline).
            matches!(
                ctx.trace_ctx.box_value(cause_op),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
            )
        }
        _ => false,
    };
    if !cause_is_null {
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    }

    // Recover the concrete exception + kind for the per-kind w_context
    // descr.  Always present (the construct fold stamped the dst shadow).
    let Some(exc) = walker_concrete_ref_object(ctx, exc_op) else {
        // No concrete recovered — re-insert and decline so the residual
        // runs (defensive; should not happen for a construct-fold exc).
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            fbw_built_exc_insert(exc_op);
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };

    // --- commit: emit the `__context__` chaining, skip the publish ---
    // active = GETFIELD_GC_R(ec, sys_exc_value).
    //
    // Route the EC through `walker_ensure_execution_context` so the
    // `__context__` read shares the ONE seeded EC OpRef the PUSH_EXC_INFO /
    // POP_EXCEPT exc-info lowering already consumes (`try_walker_lower_exc_
    // info_residual`).  A fresh `GETFIELD_GC_R(frame, execution_context)` here
    // would mint a DISTINCT OpRef from the seeded `input_arg` EC, so the POP
    // `sys_exc_value` store would `possible_aliasing`-mismatch the buffered
    // PUSH store and force it to materialize — keeping the virtual exception
    // escaped and defeating the balanced save/restore dead-store elimination
    // that lets the locally-caught exception DCE.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    };
    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[exc_op, active],
        crate::descr::w_exception_context_descr(kind),
    );
    // The full-body walk is also the authoritative execution of the
    // tracing iteration.  Apply the same context write to its concrete,
    // freshly-built exception that the recorded SETFIELD performs on later
    // compiled iterations; otherwise Python code reached later in this walk
    // observes a missing __context__ exactly once, while the trace itself is
    // correct.  This object is private to the inline construction, so no
    // rollback journal is needed.
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }

    // The normalized publish result is the same flat builtin instance;
    // forward the inline-built exc OpRef (carrying its concrete shadow)
    // to the dst that feeds the following `raise/r`.
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', exc_op)?;
    Ok(Some(()))
}

/// B3 piece 3 (`PYRE_FBW_RAISE`): lower the PUSH_EXC_INFO / POP_EXCEPT
/// exc-info-stack residuals to GETFIELD_GC_R / SETFIELD_GC on the EC's
/// `sys_exc_value` slot (`ec_sys_exc_value_descr`).
/// Recognised by the codewriter-stamped `pyre_helper` tag, NOT a funcptr
/// address (the residual calls the cross-crate `cpu.{get,set}_current_
/// exception_fn` wrappers in `pyre-jit`, which `pyre-jit-trace` cannot name).
///
///   * `GetCurrentException` — `get_current_exception()` (`[]→Ref`,
///     dst_bank `'r'`): the PUSH_EXC_INFO `prev` save.  Emit
///     `GETFIELD_GC_R(ec, sys_exc_value)`, stamp the live `prev` concrete
///     (the residual executor would have returned it) so a downstream read
///     of the dst sees the right value.
///   * `SetCurrentException` — `set_current_exception(exc)` (`[Ref]→void`,
///     dst_bank `'v'`): the PUSH_EXC_INFO store and the POP_EXCEPT restore.
///     Emit `SETFIELD_GC(ec, exc, sys_exc_value)` and apply the concrete
///     write the authoritative walk's residual executor would have done.
///
/// A balanced save (`GETFIELD`) + store + restore (`SETFIELD`) on the same
/// descr-identity field with no intervening read is dead-store-eliminated,
/// so a non-escaping exception virtualizes and DCEs (no per-raise
/// `CallMallocNursery`).  Declines (`None` → generic residual) when the EC
/// cannot be recovered or the operand shape does not match (SAFE).
pub(crate) fn try_walker_lower_exc_info_residual<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    pyre_helper: majit_ir::PyreHelperKind,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if pyre_helper == majit_ir::PyreHelperKind::GetCurrentException {
        // PUSH_EXC_INFO `prev = ec.sys_exc_value` — `[]→Ref`.
        if !r_args.is_empty() || dst_bank != 'r' {
            return Ok(None);
        }
        let (prev, prev_obj) = if let Some(seed) = ctx.fbw_mode.current_exception_seed {
            // resume.py:993-1007 applies pending fields before resumed
            // execution.  Bridge tracing does not mutate the live EC, so use
            // the decoded fieldbox directly; a runtime GETFIELD here would
            // read the pre-guard TLS value before the bridge applies anything.
            (seed, ctx.fbw_mode.current_exception_seed_concrete)
        } else {
            let Some(ec) = walker_ensure_execution_context(ctx) else {
                return Ok(None);
            };
            let prev = ctx.trace_ctx.record_op_with_descr(
                OpCode::GetfieldGcR,
                &[ec],
                crate::descr::ec_sys_exc_value_descr(),
            );
            (prev, pyre_interpreter::eval::get_current_exception())
        };
        // Stamp the concrete `prev` so a downstream read sees the value the
        // residual executor would have returned at this resume point.
        ctx.trace_ctx.set_opref_concrete(
            prev,
            majit_ir::Value::Ref(majit_ir::GcRef(prev_obj as usize)),
        );
        // Save (OpRef, concrete) for the matching POP_EXCEPT restore, and mark
        // the immediately-following `set_current_exception` as this PUSH's slot
        // store (not a restore).  The codewriter pushes `prev` then `exc` onto
        // the operand stack and POP_EXCEPT pops them, but the walker resolves
        // the popped `prev` operand to the caught exception, not the saved
        // prev; the LIFO stack carries the authoritative value instead.
        FBW_EXC_PREV.with(|s| s.borrow_mut().push((prev, prev_obj)));
        FBW_EXC_PENDING_PUSH_SET.with(|c| c.set(true));
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', prev)?;
        return Ok(Some(()));
    }

    // `SetCurrentException`: PUSH_EXC_INFO store (stores the caught EXC) or
    // POP_EXCEPT restore (restores the saved prev).  The two are identical at
    // the residual level; `FBW_EXC_PENDING_PUSH_SET` (set by the immediately-
    // preceding PUSH_EXC_INFO prev save) tells them apart.
    if r_args.len() != 1 || dst_bank != 'v' {
        return Ok(None);
    }
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };
    let is_push_set = FBW_EXC_PENDING_PUSH_SET.with(|c| c.replace(false));
    // POP_EXCEPT restore consumes the prev its matching PUSH_EXC_INFO saved.
    // If unbalanced (no saved prev — e.g. a POP whose PUSH was not lowered),
    // or this is the PUSH's own store, fall back to the operand value.
    let restore = if is_push_set {
        None
    } else {
        FBW_EXC_PREV.with(|s| s.borrow_mut().pop())
    };
    let (mut store_op, mut store_concrete) = match restore {
        // POP_EXCEPT: restore the saved prev, NOT the operand-stack value
        // (which the walker resolves to the just-caught exception).  Restoring
        // the saved prev makes the PUSH store + this restore a balanced no-op,
        // so a locally-caught exception de-escapes and DCEs, and keeps the slot
        // (`sys.exc_info()`) correct after the handler unwinds.
        Some((prev_op, prev_concrete)) => (prev_op, prev_concrete),
        None => {
            let exc_concrete = match read_ref_var_list_concrete(code, op, 1, ctx).first() {
                Some(ConcreteValue::Ref(p)) => *p,
                Some(ConcreteValue::Null) | None => std::ptr::null_mut(),
                _ => return Ok(None),
            };
            (r_args[0], exc_concrete)
        }
    };
    // A PUSH_EXC_INFO store publishes the exception being handled, which IS the
    // tracked active exception (`ctx.last_exc_value`, the walker's mirror of
    // RPython `metainterp.last_exc_box`).  The graph-side codewriter binds the
    // popped `exc_value`'s producer to a `last_exc_value` re-read for exactly
    // this reason (`codewriter.rs` PushExcInfo arm), but that producer is
    // graph-only — the walker reads the operand-stack slot directly on the
    // assumption that runtime register threading already holds the caught
    // exception there.  At a bridge resume into a handler the slot's per-PC
    // resume reconstruction can alias a non-exception constant (e.g. the vable
    // `f_code` scalar when the catch-landing exception slot shares its color),
    // so the published current exception would become a code object.  The
    // reconstruction can also leave the slot NULL (a bare handler entry whose
    // caught-exception slot was filled with a null sentinel), which would
    // publish `set_current_exception(NULL)` and lose the active exception for a
    // following bare `raise` / `sys.exc_info()`.  When the PUSH store's operand
    // resolves to NULL or a non-exception, recover the authoritative exception
    // from the tracked channel, matching the graph-side producer.
    if is_push_set
        && (store_concrete.is_null() || !unsafe { pyre_object::is_exception(store_concrete) })
    {
        if let (Some(tracked_op), ConcreteValue::Ref(tracked_obj)) =
            (ctx.last_exc_value, ctx.last_exc_value_concrete)
        {
            if !tracked_obj.is_null() && unsafe { pyre_object::is_exception(tracked_obj) } {
                store_op = tracked_op;
                store_concrete = tracked_obj;
            }
        }
    }
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[ec, store_op],
        crate::descr::ec_sys_exc_value_descr(),
    );
    // The walk is authoritative: apply the concrete store the residual
    // executor would have performed, so the live EC tracks the symbolic
    // SETFIELD in lock-step (a following `get_current_exception` /
    // POP_EXCEPT restore reads the right value).  Journal the displaced
    // prior value first: this store mutates the LIVE per-thread EC, so a
    // non-commit walk exit must restore it (the store journal's discipline).
    // Without the undo an exception propagating OUT of an except-handler
    // aborts the walk before its POP_EXCEPT restore, leaking the caught
    // exception into the next frame's `sys_exc_value`.
    fbw_sys_exc_journal_push(pyre_interpreter::eval::get_current_exception());
    pyre_interpreter::eval::set_current_exception(store_concrete);
    ctx.fbw_mode.current_exception_seed = Some(store_op);
    ctx.fbw_mode.current_exception_seed_concrete = store_concrete;
    Ok(Some(()))
}

/// #62: walker-native speculative specialization for the `STORE_SUBSCR`
/// helper residual_call (oopspec `StoreSubscr`, void result).  Ports
/// `generated_store_subscr_value` → `generated_list_setitem_by_strategy`
/// for the int- and float-storage list strategies with a non-negative
/// concrete index and a type-matching value: `guard_class LIST` +
/// `guard_value(strategy)` + unbox index + `IntLt` bounds guard + unbox
/// value + `setarrayitem_raw`.
///
/// No concrete execution: the recorded `setarrayitem_raw` performs the
/// mutation at runtime (the void residual was likewise not walk-executed —
/// `try_execute_residual_call_via_executor` skips Void results), so the walk's
/// concrete state is unchanged relative to the generic leg.  Object-storage
/// lists, long values, strategy mismatches, negative indices, and
/// non-`list[int]` operands fall through to the generic `CALL_MAY_FORCE`
/// record (`Ok(None)`), preserving Python `__setitem__` semantics.
pub(crate) fn try_walker_specialize_store_subscr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 3 {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let value_op = r_args[2];
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        return Ok(None);
    };

    // Gate: list[int] = value, non-negative index in bounds, storage matching
    // the value type (int storage ← W_IntObject, float storage ← W_FloatObject).
    let (sid, index, concrete_len) = unsafe {
        // A bool index is fine: bool shares int's `intval`, unboxed below via
        // its own &BOOL_TYPE guard.  A bool *value* into int storage must still
        // route through the generic path — PyPy's IntegerListStrategy rejects a
        // W_BoolObject (`is_correct_type` is exact-type), switching the list to
        // object storage, so the int-storage fast path would drop the bool type.
        // EXACT list only: a list SUBCLASS instance shares `ob_type ==
        // &LIST_TYPE` but retags `w_class` and may override `__setitem__`;
        // `is_exact_list` excludes it so it falls to the generic residual
        // (which honours the override) instead of this direct-storage store.
        if !pyre_object::is_exact_list(list_obj) || !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        let index = pyre_object::w_int_get_value(key_obj);
        if index < 0 {
            return Ok(None);
        }
        let concrete_len = pyre_object::w_list_len(list_obj);
        if index as usize >= concrete_len {
            return Ok(None);
        }
        let sid = if pyre_object::w_list_uses_int_storage(list_obj)
            && pyre_object::is_int(value_obj)
            && !pyre_object::is_bool(value_obj)
        {
            1i64
        } else if pyre_object::w_list_uses_float_storage(list_obj)
            && pyre_object::is_float(value_obj)
        {
            2i64
        } else {
            return Ok(None);
        };
        (sid, index, concrete_len)
    };

    // --- emit the specialized IR (walker-native) ---
    // guard_class LIST (skip when class already known / operand is constant).
    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    if !list_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(list_op) {
        let type_const = ctx.trace_ctx.const_int(list_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[list_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(list_op, list_type_addr);

    // A list SUBCLASS instance shares `ob_type == &LIST_TYPE` (so it passes
    // the GuardClass above) but retags `w_class` and may override
    // `__setitem__`; guard the exact canonical `w_class` so such an instance
    // side-exits to the generic residual (which honours the override) rather
    // than taking this direct-storage store.
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        list_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
    )?;

    // guard_value(strategy == sid): getfield strategy + GuardValue + replace_box.
    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_strategy_descr(),
    );
    let sid_const = ctx.trace_ctx.const_int(sid);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[strategy, sid_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(strategy, sid_const);

    // Unbox the index operand.  bool shares int's `intval`, so a bool index
    // guards its own &BOOL_TYPE.
    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(index));

    // Bounds guard (non-negative index path): IntLt(raw_index, len).
    let len_descr = if sid == 1 {
        crate::descr::list_int_items_len_descr()
    } else {
        crate::descr::list_float_items_len_descr()
    };
    let lenbox = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, list_op, len_descr);
    let in_bounds = ctx.trace_ctx.record_op(OpCode::IntLt, &[raw_index, lenbox]);
    ctx.trace_ctx.set_opref_concrete(
        in_bounds,
        majit_ir::Value::Int(((index as usize) < concrete_len) as i64),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_bounds])?;

    // Unbox the value + setarrayitem.
    if sid == 1 {
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            list_op,
            crate::descr::list_int_items_block_descr(),
        );
        // The value is a true W_IntObject (the gate excludes bool from int
        // storage), so it unboxes through the plain INT_TYPE guard.
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        let raw = walker_unbox_int(ctx, op_pc, value_op, int_type_addr)?;
        let elem = unsafe { pyre_object::w_int_get_value(value_obj) };
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(elem));
        crate::state::trace_int_block_setitem_value(ctx.trace_ctx, block, raw_index, raw);
    } else {
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            list_op,
            crate::descr::list_float_items_block_descr(),
        );
        let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
        let raw = walker_unbox_float(ctx, op_pc, value_op, float_type_addr)?;
        let elem = unsafe { pyre_object::w_float_get_value(value_obj) };
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Float(elem));
        crate::state::trace_float_block_setitem_value(ctx.trace_ctx, block, raw_index, raw);
    }

    // Tracing is execution (pyjitpl.py:2095 execute_and_record): apply the
    // store to the concrete list now, so the walk's own region — and a
    // walk-end commit that hands the END state to the interpreter with no
    // replay — sees the mutation exactly once.  The displaced element goes
    // into the undo log first: a walk that does NOT commit returns to the
    // legacy replay, which re-executes the region and must find the
    // pre-walk heap (see `FBW_STORE_JOURNAL`).
    let Some(displaced) = (unsafe { pyre_object::w_list_getitem(list_obj, index) }) else {
        unreachable!(
            "store_subscr specialization: in-bounds index {index} has no element \
             (strategy/bounds gates above admitted it)"
        );
    };
    // `w_list_getitem` boxes the displaced int/float; that allocation can
    // run a minor collection and move the operands, so re-read the
    // forwarded refs from the shadow before touching the heap.  (The
    // freshly boxed `displaced` itself cannot move before the journal
    // push roots it — nothing below allocates.)
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        unreachable!(
            "store_subscr specialization: operand concrete vanished from the shadow \
             across the displaced-element boxing"
        );
    };
    fbw_store_journal_push(list_obj, key_obj, displaced);
    let stored = unsafe { pyre_object::w_list_setitem(list_obj, index, value_obj) };
    debug_assert!(
        stored,
        "store_subscr specialization: in-bounds store failed"
    );
    Ok(Some(()))
}

/// Walker-native `GetIter` for an exact machine-word `range`.
///
/// Emits the virtual `W_IntRangeIterator` allocation shape directly — the
/// iterator PyPy's inlined `descr_iter` would trace — so a locally consumed
/// iterator stays a removable virtual `New`.
pub(crate) fn try_walker_specialize_get_iter<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    _dst: usize,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if !ctx.is_authoritative_executor
        || dst_bank != 'r'
        || r_args.len() != 1
        || ctx.fbw_mode.inline_subwalk
    {
        return Ok(None);
    }

    let range_op = r_args[0];
    let Some(range_obj) = walker_concrete_ref_object(ctx, range_op) else {
        return Ok(None);
    };

    let (concrete_start, concrete_step, concrete_length, concrete_mul, concrete_one_past) = unsafe {
        if !pyre_object::functional::is_w_range(range_obj)
            || !pyre_object::functional::is_exact_w_range(range_obj)
        {
            return Ok(None);
        }
        let (start_obj, _stop_obj, step_obj) = pyre_object::functional::w_range_fields(range_obj);
        let length_obj = pyre_object::functional::w_range_length(range_obj);
        if !pyre_object::is_int(start_obj)
            || pyre_object::is_bool(start_obj)
            || !pyre_object::is_int(step_obj)
            || pyre_object::is_bool(step_obj)
            || !pyre_object::is_int(length_obj)
            || pyre_object::is_bool(length_obj)
        {
            return Ok(None);
        }
        let Some((start, _stop, step)) = pyre_object::functional::w_range_fields_i64(range_obj)
        else {
            return Ok(None);
        };
        let Some(length) = pyre_object::functional::w_range_length_i64(range_obj) else {
            return Ok(None);
        };
        let one_past_i128 = start as i128 + length as i128 * step as i128;
        let Ok(one_past) = i64::try_from(one_past_i128) else {
            return Ok(None);
        };
        let Some(mul) = length.checked_mul(step) else {
            return Ok(None);
        };
        let Some(one_past_checked) = start.checked_add(mul) else {
            return Ok(None);
        };
        debug_assert_eq!(one_past_checked, one_past);
        (start, step, length, mul, one_past)
    };

    let range_type_addr = &pyre_object::functional::RANGE_TYPE as *const _ as i64;
    if !range_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(range_op) {
        let range_type_const = ctx.trace_ctx.const_int(range_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[range_op, range_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(range_op, range_type_addr);

    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let int_type_const = ctx.trace_ctx.const_int(int_type_addr);

    let start_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_start_descr(),
    );
    if !ctx.trace_ctx.heap_cache().is_class_known(start_r) {
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[start_r, int_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(start_r, int_type_addr);
    }
    let start_i = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        start_r,
        crate::descr::int_intval_descr(),
    );
    ctx.trace_ctx
        .set_opref_concrete(start_i, majit_ir::Value::Int(concrete_start));

    let step_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_step_descr(),
    );
    if !ctx.trace_ctx.heap_cache().is_class_known(step_r) {
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[step_r, int_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(step_r, int_type_addr);
    }
    let step_i =
        crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, step_r, crate::descr::int_intval_descr());
    ctx.trace_ctx
        .set_opref_concrete(step_i, majit_ir::Value::Int(concrete_step));

    let length_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_length_descr(),
    );
    if !ctx.trace_ctx.heap_cache().is_class_known(length_r) {
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[length_r, int_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(length_r, int_type_addr);
    }
    let length_i = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        length_r,
        crate::descr::int_intval_descr(),
    );
    ctx.trace_ctx
        .set_opref_concrete(length_i, majit_ir::Value::Int(concrete_length));

    let mul = ctx
        .trace_ctx
        .record_op(OpCode::IntMulOvf, &[length_i, step_i]);
    ctx.trace_ctx
        .set_opref_concrete(mul, majit_ir::Value::Int(concrete_mul));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;

    let one_past = ctx.trace_ctx.record_op(OpCode::IntAddOvf, &[start_i, mul]);
    ctx.trace_ctx
        .set_opref_concrete(one_past, majit_ir::Value::Int(concrete_one_past));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;

    let new = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::w_range_iter_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(new);

    let current_descr = crate::descr::range_iter_current_descr();
    let current_index = current_descr.index();
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[new, start_i], current_descr);
    ctx.trace_ctx
        .heapcache_setfield_cached(new, current_index, start_i);

    let remaining_descr = crate::descr::range_iter_remaining_descr();
    let remaining_index = remaining_descr.index();
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[new, length_i], remaining_descr);
    ctx.trace_ctx
        .heapcache_setfield_cached(new, remaining_index, length_i);

    let step_descr = crate::descr::range_iter_step_descr();
    let step_index = step_descr.index();
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[new, step_i], step_descr);
    ctx.trace_ctx
        .heapcache_setfield_cached(new, step_index, step_i);

    let range_iter_type_addr = &pyre_object::functional::RANGE_ITER_TYPE as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new, range_iter_type_addr);

    let real_iter = unsafe { pyre_object::functional::w_range_iter(range_obj) };
    ctx.trace_ctx.set_opref_concrete(
        new,
        majit_ir::Value::Ref(majit_ir::GcRef(real_iter as usize)),
    );
    ctx.vstack_last_ref = new;

    Ok(Some(new))
}

/// Walker-native `ForIterNext` for `W_IntRangeIterator`.
///
/// The generic residual advances the shared iterator before an abort can
/// occur, and forward-delivery preserves that consumed item.  This inline
/// path keeps that deliberately irreversible advance: it never journals or
/// rolls the cursor back.  It instead emits the `W_IntRangeIterator.next`
/// field-update shape with a continuation guard.  Its false side resumes at
/// the same FOR_ITER coordinate as the codewriter's ordinary exhaustion edge.
///
/// The continuation item is a normal virtualizable `W_IntObject`; allocation
/// removal elides it until an escaping consumer or a deopt needs a real box.
pub(crate) fn try_walker_specialize_for_iter_next<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    _dst: usize,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 1 {
        return Ok(None);
    }

    // The snapshot root represents the caller during an inline sub-walk, so
    // it cannot supply the callee's FOR_ITER green key for demotion.  Leave
    // that shape on the generic residual until every inlined frame threads
    // its own snapshot root.
    if ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }

    // A range class-guard failure at this FOR_ITER green key is a definitive
    // polymorphism witness.  Once the failure path has demoted it, retain the
    // generic residual rather than recreating the range guard on retrace.
    let range_green_key = walker_foriter_green_key(ctx, op_pc);
    if range_green_key.is_some_and(crate::trace::range_foriter_demoted) {
        return Ok(None);
    }

    let iter_op = r_args[0];
    let Some(iter_obj) = walker_concrete_ref_object(ctx, iter_op) else {
        return Ok(None);
    };
    let (concrete_current, concrete_remaining, concrete_step) = unsafe {
        if !pyre_object::functional::is_range_iter(iter_obj) {
            return Ok(None);
        }
        pyre_object::functional::w_range_iter_fields(iter_obj)
    };
    let concrete_continues = concrete_remaining != 0;

    // A new consume attempt completes the prior in-flight iteration before
    // this irreversible concrete advance, matching the residual executor.
    let body = fbw_foriter_body_from_op_pc(ctx.fbw_mode.snapshot_sym, op_pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body);

    // guard_class W_IntRangeIterator, unless the operand is already known.
    let range_iter_type_addr = &pyre_object::functional::RANGE_ITER_TYPE as *const _ as i64;
    if !iter_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(iter_op) {
        let type_const = ctx.trace_ctx.const_int(range_iter_type_addr);
        // Pre-mint the guard's FailDescr tagged with this FOR_ITER green key
        // so its runtime failure — a definitive polymorphism witness —
        // demotes the specialization by descr identity, independent of the
        // guard's per-trace fail index.  `store_final_boxes_in_guard`
        // preserves an existing ResumeGuardDescr (only refreshing
        // fail_arg_types), so the tag survives optimizer guard-folding and
        // unroll; a copied guard chases `prev` to this donor.  With no green
        // key available (e.g. inline sub-walk) the guard is untagged and the
        // site is simply never demoted, matching the prior behavior.
        match range_green_key {
            Some(green_key) => {
                let descr = majit_metainterp::make_resume_guard_descr_range_foriter(green_key);
                ctx.trace_ctx.record_guard_with_descr(
                    OpCode::GuardClass,
                    &[iter_op, type_const],
                    descr,
                );
            }
            None => {
                ctx.trace_ctx
                    .record_guard(OpCode::GuardClass, &[iter_op, type_const], 0);
            }
        }
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(iter_op, range_iter_type_addr);

    if !concrete_continues {
        // Exhausted arrival: the walker concretely reached remaining==0 (a nested
        // inner loop run to completion inside the outer body).  Record the
        // routing guard for the false continue predicate, then present the
        // exhaustion edge exactly as the residual does: a NULL Ref that the
        // codewriter's trailing GuardNonnull consumes as the loop exit.  The
        // iterator is already exhausted, so no cursor advance and no in-flight
        // capture.
        let zero = ctx.trace_ctx.const_int(0);
        let remaining = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            iter_op,
            crate::descr::range_iter_remaining_descr(),
        );
        let continues = ctx.trace_ctx.record_op(OpCode::IntGt, &[remaining, zero]);
        ctx.trace_ctx.set_opref_concrete(continues, Value::Int(0));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[continues])?;
        let null_item = ctx.trace_ctx.record_op(OpCode::CastIntToPtr, &[zero]);
        ctx.trace_ctx
            .set_opref_concrete(null_item, Value::Ref(majit_ir::GcRef(0)));
        return Ok(Some(null_item));
    }

    // Guard the continue arm before constructing the item.  The false arm
    // resumes at this FOR_ITER, where the interpreter takes the existing
    // exhaustion edge (iterator retained, no item pushed).  This avoids the
    // pointer-mask representation which forced the item to be materialized.
    let current = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_current_descr(),
    );
    let remaining = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_remaining_descr(),
    );
    let step = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_step_descr(),
    );
    let zero = ctx.trace_ctx.const_int(0);
    let continues = ctx.trace_ctx.record_op(OpCode::IntGt, &[remaining, zero]);
    ctx.trace_ctx
        .set_opref_concrete(continues, Value::Int(concrete_continues as i64));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[continues])?;

    // The continue guard establishes `continues == 1` on the trace path. Keep
    // Slice-1's wrapping IntAdd and live-iterator SetfieldGc updates intact.
    let delta = ctx.trace_ctx.record_op(OpCode::IntMul, &[step, continues]);
    ctx.trace_ctx.set_opref_concrete(
        delta,
        Value::Int(concrete_step.wrapping_mul(concrete_continues as i64)),
    );
    let next_current = ctx.trace_ctx.record_op(OpCode::IntAdd, &[current, delta]);
    let next_current_concrete =
        concrete_current.wrapping_add(concrete_step.wrapping_mul(concrete_continues as i64));
    ctx.trace_ctx
        .set_opref_concrete(next_current, Value::Int(next_current_concrete));
    let current_descr = crate::descr::range_iter_current_descr();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[iter_op, next_current],
        current_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(iter_op, current_descr.index(), next_current);

    let next_remaining = ctx
        .trace_ctx
        .record_op(OpCode::IntSub, &[remaining, continues]);
    let next_remaining_concrete = concrete_remaining.wrapping_sub(concrete_continues as i64);
    ctx.trace_ctx
        .set_opref_concrete(next_remaining, Value::Int(next_remaining_concrete));
    let remaining_descr = crate::descr::range_iter_remaining_descr();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[iter_op, next_remaining],
        remaining_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(iter_op, remaining_descr.index(), next_remaining);

    // `wrapint` is the transparent `NewWithVtable(W_IntObject)` +
    // `SetfieldGc(intval=current)` shape allocation removal virtualizes.  Do
    // not feed it through pointer arithmetic: locally consumed items stay
    // virtual, while normal forcing materializes escaping items.
    let item = crate::state::wrapint(ctx.trace_ctx, current);

    // Tracing executes the real range cursor advance.  The direct helper is
    // the same `W_IntRangeIterator.next` implementation used by the residual;
    // do not journal it, because abort recovery forwards this exact item.
    let concrete_item = unsafe { pyre_object::functional::w_range_iter_next(iter_obj) };
    debug_assert_eq!(concrete_item.is_some(), concrete_continues);
    let concrete_item_ptr = concrete_item.expect("GuardTrue(continues) implies a range item");
    ctx.trace_ctx.set_opref_concrete(
        item,
        Value::Ref(majit_ir::GcRef(concrete_item_ptr as usize)),
    );

    // Keep the virtual payload's concrete shadow paired with the concrete New.
    // A later body guard can then encode the virtual `i` in its snapshot and
    // blackhole will rematerialize the right item on deopt.
    ctx.trace_ctx
        .set_opref_concrete(current, Value::Int(concrete_current));

    fbw_foriter_inflight_capture(concrete_item_ptr, body);
    // Range iteration stays at the C level, so the operand-stack mirror
    // remains valid and must receive the item produced by FOR_ITER.  Its
    // virtual state is captured by subsequent body-guard snapshots.
    ctx.vstack_last_ref = item;

    Ok(Some(item))
}

/// Specialize `STORE_SUBSCR target[const_slice] = source` for a same-length,
/// step-1 slice between two Integer-strategy exact lists, eliding the
/// `CALL_MAY_FORCE` `store_subscr` residual that would force the virtualizable
/// source list (the freshly built BUILD_LIST temp from
/// [`try_walker_specialize_newlist`]) every iteration.  The same-length gate
/// makes the assignment `slice_len` independent in-bounds setitems —
/// `target[start + j] = source[j]` — with no resize and no strategy change, so
/// it rides the existing `FBW_STORE_JOURNAL` per-element undo log.
///
/// Reads the source elements through `getfield_gc(int_items)` +
/// `getarrayitem_gc` ops keyed on the source `OpRef`, so when the source is the
/// freshly built virtual list the optimizer folds the reads against its
/// recorded `SetarrayitemGc` stores and removes the whole temporary.
///
/// The slice key must be a trace constant (a `slice(...)` from `co_consts`);
/// `start` / `stop` are read off the slice object and baked into the emitted
/// index constants.  Falls through to the generic residual (returns `Ok(None)`)
/// for anything outside the gate: a non-constant / `None` / negative bound, a
/// non-unit step, a resizing (length-changing) slice, an empty slice, a
/// non-Integer-storage target or source, or a list subclass (which may override
/// `__setitem__` / `__iter__`).
pub(crate) fn try_walker_specialize_setslice<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 3 {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let value_op = r_args[2];
    // The slice key must be a trace constant (a `slice(...)` from `co_consts`):
    // its `start` / `stop` are baked into the emitted index constants, so a
    // non-constant slice (whose bounds could differ at runtime) cannot be
    // specialized this way.
    if !key_op.is_constant() {
        return Ok(None);
    }
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        return Ok(None);
    };

    // Gate, all read from the concrete shadows: `target[start:stop:1] =
    // source`, both exact-list Integer storage, `stop - start == len(source)`
    // (no resize), `1 <= slice_len`, `0 <= start <= stop <= len(target)`.
    let (start, slice_len) = unsafe {
        // EXACT list for BOTH target and source: a list subclass shares
        // `ob_type == &LIST_TYPE` but retags `w_class` and may override
        // `__setitem__` (target) or `__iter__` (source); both must route
        // through the generic residual.
        if !pyre_object::pyobject::is_exact_list(list_obj)
            || !pyre_object::is_slice(key_obj)
            || !pyre_object::pyobject::is_exact_list(value_obj)
        {
            return Ok(None);
        }
        // step == 1 (None defaults to 1; an explicit non-1 step needs the
        // strided path).
        let step_o = pyre_object::w_slice_get_step(key_obj);
        let step_is_one = pyre_object::is_none(step_o)
            || (pyre_object::is_int(step_o)
                && !pyre_object::is_bool(step_o)
                && pyre_object::w_int_get_value(step_o) == 1);
        if !step_is_one {
            return Ok(None);
        }
        // start / stop must be explicit non-negative plain ints (None bounds and
        // negative indices route through the generic residual, which normalises
        // them).
        let start_o = pyre_object::w_slice_get_start(key_obj);
        let stop_o = pyre_object::w_slice_get_stop(key_obj);
        if !(pyre_object::is_int(start_o)
            && !pyre_object::is_bool(start_o)
            && pyre_object::is_int(stop_o)
            && !pyre_object::is_bool(stop_o))
        {
            return Ok(None);
        }
        let start = pyre_object::w_int_get_value(start_o);
        let stop = pyre_object::w_int_get_value(stop_o);
        let target_len = pyre_object::w_list_len(list_obj) as i64;
        if start < 0 || stop < start || stop > target_len {
            return Ok(None);
        }
        let slice_len = stop - start;
        let src_len = pyre_object::w_list_len(value_obj) as i64;
        // Same-length only — a resizing slice changes the target length and can
        // switch strategy.
        if slice_len != src_len || slice_len < 1 {
            return Ok(None);
        }
        if !(pyre_object::w_list_uses_int_storage(list_obj)
            && pyre_object::w_list_uses_int_storage(value_obj))
        {
            return Ok(None);
        }
        (start, slice_len)
    };

    // --- emit the specialized IR (walker-native) ---
    // For BOTH target (`list_op`) and source (`value_op`): guard_class LIST +
    // exact `w_class` (a list subclass sharing `ob_type == &LIST_TYPE` but with
    // an overridden `__setitem__` / `__iter__` side-exits to the generic
    // residual) + guard strategy == Integer.  Folds away when the operand is the
    // just-built virtual list.
    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    let list_instantiate =
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let sid_const_val = pyre_object::listobject::ListStrategy::Integer as i64;
    for &lst_op in &[list_op, value_op] {
        if !lst_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(lst_op) {
            let type_const = ctx.trace_ctx.const_int(list_type_addr);
            ctx.trace_ctx
                .record_guard(OpCode::GuardClass, &[lst_op, type_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        }
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(lst_op, list_type_addr);
        walker_guard_exact_w_class(ctx, op_pc, lst_op, list_instantiate)?;

        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            lst_op,
            crate::descr::list_strategy_descr(),
        );
        let sid_const = ctx.trace_ctx.const_int(sid_const_val);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[strategy, sid_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy, sid_const);
    }

    // Bounds guard on the target: the highest written index `start + slice_len -
    // 1` must be in range.  For an Integer-strategy list the `W_ListObject`
    // `length` field is 0 — the authoritative length is `int_items.len`, so read
    // it via `list_int_items_len_descr` (exactly as store_subscr's bounds
    // guard).  IntLt(start+slice_len-1, target.int_items.len).
    let tgt_len_box = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_int_items_len_descr(),
    );
    let last_idx_const = ctx.trace_ctx.const_int(start + slice_len - 1);
    let in_bounds = ctx
        .trace_ctx
        .record_op(OpCode::IntLt, &[last_idx_const, tgt_len_box]);
    let concrete_target_len = unsafe { pyre_object::w_list_len(list_obj) as i64 };
    ctx.trace_ctx.set_opref_concrete(
        in_bounds,
        majit_ir::Value::Int(((start + slice_len - 1) < concrete_target_len) as i64),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_bounds])?;

    // Length guard on the source: source.int_items.len == slice_len (folds for
    // the virtual temp; protects a non-virtual source).
    let src_len_box = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        value_op,
        crate::descr::list_int_items_len_descr(),
    );
    let src_len_const = ctx.trace_ctx.const_int(slice_len);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[src_len_box, src_len_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(src_len_box, src_len_const);

    // items[start + j] = source.items[j] for j in 0..slice_len, through the
    // int_items blocks (`list_int_items_block_descr`, matching slice-1's
    // `emit_typed_list_inline` `SetfieldGc`), so a virtual source temp's
    // `SetarrayitemGc` stores fold against these reads.
    let src_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        value_op,
        crate::descr::list_int_items_block_descr(),
    );
    let tgt_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_int_items_block_descr(),
    );
    for j in 0..slice_len {
        let src_idx = ctx.trace_ctx.const_int(j);
        let src_raw =
            crate::state::trace_int_block_getitem_value(ctx.trace_ctx, src_block, src_idx);
        let tgt_idx = ctx.trace_ctx.const_int(start + j);
        crate::state::trace_int_block_setitem_value(ctx.trace_ctx, tgt_block, tgt_idx, src_raw);
    }

    // Tracing is execution (pyjitpl.py:2095 execute_and_record): apply the
    // assignment to the concrete lists now as `slice_len` in-bounds setitems,
    // journaling each displaced element first so a non-committing walk's legacy
    // replay re-executes against the pre-walk heap (FBW_STORE_JOURNAL).  Each
    // `w_list_getitem` / `w_int_new` boxes, and a minor collection there can
    // move any live GC object.  Following the push_roots/pop_roots reload
    // discipline, every live ref is reloaded after each boxing allocation,
    // before its next use: walker operands (`list_obj`/`value_obj`) from the
    // forwarded shadow via `walker_concrete_ref_object`, and the pinned fresh
    // boxes (`src_item`/`displaced`) from their shadow-stack slot via
    // `shadow_stack_get` (the slot index captured just before the pin).
    {
        let _roots = pyre_object::gc_roots::push_roots();
        for j in 0..slice_len {
            let tgt_index = start + j;
            let Some(value_obj) = walker_concrete_ref_object(ctx, value_op) else {
                unreachable!("setslice specialization: operand concrete vanished from the shadow");
            };
            let Some(src_item) = (unsafe { pyre_object::w_list_getitem(value_obj, j) }) else {
                unreachable!("setslice specialization: source index {j} has no element");
            };
            let src_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(src_item);
            let Some(list_obj) = walker_concrete_ref_object(ctx, list_op) else {
                unreachable!("setslice specialization: operand concrete vanished from the shadow");
            };
            let Some(displaced) = (unsafe { pyre_object::w_list_getitem(list_obj, tgt_index) })
            else {
                unreachable!(
                    "setslice specialization: target index {tgt_index} has no element \
                     (bounds gate admitted it)"
                );
            };
            let disp_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(displaced);
            let key_box = pyre_object::w_int_new(tgt_index);
            pyre_object::gc_roots::pin_root(key_box);
            let Some(list_obj) = walker_concrete_ref_object(ctx, list_op) else {
                unreachable!("setslice specialization: list concrete vanished mid-apply");
            };
            let src_item = pyre_object::gc_roots::shadow_stack_get(src_slot);
            let displaced = pyre_object::gc_roots::shadow_stack_get(disp_slot);
            fbw_store_journal_push(list_obj, key_box, displaced);
            let stored = unsafe { pyre_object::w_list_setitem(list_obj, tgt_index, src_item) };
            debug_assert!(stored, "setslice specialization: in-bounds store failed");
        }
    }
    Ok(Some(()))
}

/// #57 SLICE 3c (compare): walker-native speculative float specialization
/// for the `COMPARE_OP` helper residual_call (oopspec `CompareOp`), the
/// float analogue of [`try_walker_specialize_compare_op_int`] and the
/// former float-compare arm.  Per operand
/// either `guard_class FLOAT` + `getfield_gc_pure_f`, or (int operand)
/// `guard_class INT` + `getfield_gc_i` + `cast_int_to_float`; then
/// `float_<cmp>` for the raw truth, then NON-fused box to a `W_Bool`.
///
/// Tried as a fallback only after the int compare specialization declines,
/// so two-int operands keep int comparison.  All six `ComparisonOperator`
/// variants are handled (float compare has no deferred operators).
pub(crate) fn try_walker_specialize_compare_op_float<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::ComparisonOperator;
    let cmp = match cmp_op {
        ComparisonOperator::Less => OpCode::FloatLt,
        ComparisonOperator::LessOrEqual => OpCode::FloatLe,
        ComparisonOperator::Greater => OpCode::FloatGt,
        ComparisonOperator::GreaterOrEqual => OpCode::FloatGe,
        ComparisonOperator::Equal => OpCode::FloatEq,
        ComparisonOperator::NotEqual => OpCode::FloatNe,
    };
    let Some((
        lhs,
        rhs,
        lhs_obj,
        rhs_obj,
        lhs_is_int,
        rhs_is_int,
        lhs_f64,
        rhs_f64,
        boxed_result_i64,
    )) = walker_float_specialization_operands(ctx, r_args, allboxes, call_descr)
    else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    let lhs_raw = walker_coerce_operand_to_float(ctx, op_pc, lhs, lhs_obj, lhs_is_int, lhs_f64)?;
    let rhs_raw = walker_coerce_operand_to_float(ctx, op_pc, rhs, rhs_obj, rhs_is_int, rhs_f64)?;
    let truth = ctx.trace_ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
    let folded =
        majit_metainterp::eval_float_cmp(cmp, lhs_f64.to_bits() as i64, rhs_f64.to_bits() as i64);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(folded));
    // #62: elide the dead box when the compare's boxed dst is consumed
    // solely by the immediately-following `is_true` (see
    // [`compare_box_provably_dead`] / the int-compare twin for rationale).
    if dst_bank == 'r' && compare_box_provably_dead(ctx, op_pc, dst as u8) {
        bool_box_truth_record(truth, truth);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, truth)?;
        return Ok(Some(()));
    }
    // NON-fused: box the raw truth into a W_Bool (the generic compare_fn
    // residual_call lands a boxed bool; the separate goto_if_not reads it).
    let boxed = crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    // #62: remember boxed→truth so an immediately-following `is_true` residual
    // (POP_JUMP_IF_*) folds back to the raw Int instead of may-force-unboxing.
    bool_box_truth_record(boxed, truth);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #62 LoadGlobal cell-cache fold — walker mirror of the retired trait
/// LOAD_GLOBAL fast path.
///
/// When `ns` is a `W_ModuleDictObject` still in `ModuleDictStrategy` mode
/// whose slot for `name` holds a raw value or an `ObjectMutableCell`, emit
/// `QUASIIMMUT_FIELD(ns, slot)` + `RECORD_KNOWN_RESULT` + an elidable cell
/// lookup that the optimizer folds to the constant cell pointer.  The
/// strategy's `version?` watcher invalidates the loop (GUARD_NOT_INVALIDATED)
/// on any rebind, so the fold is sound while `load_global_fn` itself stays
/// `CallFlavor::Plain`.  Returns `Ok(true)` when the fold was emitted;
/// `Ok(false)` when the receiver is not a foldable cell (the caller then
/// falls through to the generic residual, which stays correct).
///
/// DEV-GATED + INCOMPLETE: callers gate this on `PYRE_FBW_LOADGLOBAL_FOLD`
/// (default off).  When the loaded global is a function that is then CALLed,
/// folding it to a loop-invariant constant callee routes the call through the
/// in-progress FBW call-inlining path (#68), which mis-resolves the callee and
/// produces wrong output.  Keep default-off until #68 lands.
///
/// Builtins fallback (`PYRE_FBW_BUILTIN_FOLD`): when `name` is ABSENT from the
/// module dict but resolves through `frame.get_builtin()` (e.g.
/// `raise ValueError` / `except ValueError`), the same cell fold is emitted
/// against the BUILTINS dict, guarded additionally by a `QUASIIMMUT_FIELD` on
/// the module dict so adding `name` to globals (shadowing the builtin) bumps
/// the module-dict `version` and fails the loop's GUARD_NOT_INVALIDATED.  This
/// mirrors `bh_load_global_fn`'s `finditem_str(globals)` →
/// `get_builtin().getdictvalue` fallback chain.
pub(crate) fn try_walker_load_global_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    ns_ptr: usize,
    w_code_ptr: usize,
    frame_ptr: usize,
    namei: i64,
) -> Result<bool, DispatchError> {
    let w_globals = ns_ptr as pyre_object::PyObjectRef;
    // `namei` is the raw `LOAD_GLOBAL` oparg; bit 0 is the push-NULL flag,
    // so the `co_names` index is `namei >> 1` (mirror `bh_load_global_fn`).
    let name_idx = (namei as usize) >> 1;
    let name = unsafe {
        let code = &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject);
        match pyre_interpreter::pyframe::load_name_from_code(code, name_idx) {
            Some(n) => n.to_string(),
            None => return Ok(false),
        }
    };
    if emit_module_dict_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, &name)? {
        return Ok(true);
    }
    // `emit_module_dict_cell_fold` returns `false` for BOTH an absent name and
    // a present-but-unfoldable one (`IntMutableCell` / movable / strategy
    // switched).  Only an ABSENT name may fall through to the builtins fold — a
    // present global shadows the builtin, so keep the residual (which reads the
    // live globals slot) when the slot still exists.
    if crate::state::module_dict_cell_slot_direct(w_globals, &name).is_some() {
        return Ok(false);
    }

    // Builtins fallback (`PYRE_FBW_BUILTIN_FOLD`): the name is absent from the
    // `ns_ptr` module dict.  Mirror `bh_load_global_fn`'s second leg —
    // `frame.get_builtin().getdictvalue(name)` — and fold the builtins cell
    // when the name resolves there.  Requires the live frame operand.
    if !fbw_builtin_fold_enabled() || frame_ptr == 0 {
        return Ok(false);
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::PyFrame) };
    // Faithfulness: `bh_load_global_fn` re-resolves the globals it actually
    // consults from the LIVE frame (`frame.get_w_globals()` when the frame
    // owns `w_code`, else the code's bound globals) and IGNORES the
    // `namespace_ptr` operand.  The `ns_ptr` hint usually equals that live
    // dict, but verify the name is ALSO absent from the authoritative
    // globals before falling through to builtins — a present name there must
    // resolve from globals (residual), not the builtin, or the fold would be
    // wrong.
    let live_globals = if frame.pycode as usize == w_code_ptr {
        frame.get_w_globals()
    } else {
        unsafe { pyre_interpreter::w_code_get_w_globals(w_code_ptr as pyre_object::PyObjectRef) }
    };
    if !live_globals.is_null()
        && live_globals as usize != w_globals as usize
        && crate::state::module_dict_cell_slot_direct(live_globals, &name).is_some()
    {
        return Ok(false);
    }
    let w_builtin = frame.get_builtin();
    if w_builtin.is_null() || !unsafe { pyre_object::is_module(w_builtin) } {
        return Ok(false);
    }
    let w_builtin_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
    if w_builtin_dict.is_null() {
        return Ok(false);
    }
    let Some(b_slot) = crate::state::module_dict_cell_slot_direct(w_builtin_dict, &name) else {
        return Ok(false);
    };
    let Some(b_stored) = crate::state::module_dict_cell_value_direct(w_builtin_dict, b_slot) else {
        return Ok(false);
    };
    if b_stored.is_null() || unsafe { pyre_object::celldict::is_int_mutable_cell(b_stored) } {
        return Ok(false);
    }
    if majit_gc::can_move(majit_ir::GcRef(b_stored as usize)) {
        return Ok(false);
    }
    // Guard (a): the name must stay ABSENT from the module dict so the lookup
    // keeps falling through to builtins.  A `QUASIIMMUT_FIELD` on the module
    // dict registers the loop flag on the module-dict `version` watcher; a
    // later `mutated()` (the new-key insert that shadows the builtin) fails
    // GUARD_NOT_INVALIDATED.  The slot operand is unused for version keying
    // (the watcher is per-`version`, not per-slot, and the registration
    // ignores the slot); use `usize::MAX` as a past-the-end sentinel so the
    // `quasi_immut_cache` key cannot collide with a real cell fold's slot for
    // a DIFFERENT present name on the same module dict.
    if !guard_current_frame_globals_identity(ctx, op_pc, w_globals)? {
        return Ok(false);
    }
    let abs_ns_const = ctx.trace_ctx.const_ref(w_globals as i64);
    let abs_slot_const = ctx.trace_ctx.const_int(usize::MAX as i64);
    crate::state::record_namespace_quasiimmut_field(
        ctx.trace_ctx,
        abs_ns_const,
        abs_slot_const,
        u32::MAX,
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    // Guard (b): the builtins value for `name` must be unchanged.  The
    // `emit_namespace_cell_fold` below records a `QUASIIMMUT_FIELD` on the
    // builtins dict + the elidable cell lookup, so a rebind/del of the
    // builtin bumps the builtins-dict `version` and fails the loop.
    if majit_gc::can_move(majit_ir::GcRef(w_builtin_dict as usize)) {
        return Ok(false);
    }
    if !emit_namespace_cell_fold(
        ctx,
        op_pc,
        dst,
        dst_bank,
        w_builtin_dict,
        b_slot,
        b_stored,
        false,
    )? {
        return Ok(false);
    }
    Ok(true)
}

/// LoadName cell fold — module-scope LOAD_NAME mirror of
/// [`try_walker_load_global_cell_fold`].  At module scope the frame's
/// `w_locals` is null and `w_locals` aliases `w_globals`
/// (`createframe` sets `debugdata.w_locals = w_globals_storage`,
/// pyframe.rs:1323), so `load_name_value`'s probe + LOAD_GLOBAL fallthrough
/// both resolve in `w_globals` — the same dict the global cell fold reads.
/// A non-module frame (class body / `exec(code, g, l)` with separate locals)
/// has a non-null `w_locals`, so the gate routes it to the live
/// residual `bh_load_name_fn`.
pub(crate) fn try_walker_load_name_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    frame_ptr: usize,
    w_name_ptr: usize,
) -> Result<bool, DispatchError> {
    if frame_ptr == 0 {
        return Ok(false);
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return Ok(false);
    }
    // Only module scope (w_locals IS w_globals) is foldable. Module frames bind
    // `w_locals = w_globals` (pyframe.py:216-218); a `w_locals`
    // that is a DIFFERENT object means the LOAD_NAME probe targets a separate
    // locals namespace the module-dict cell fold (keyed on `w_globals`) would
    // skip. (Class bodies / `exec(code, g, l)` set a separate one; they also do
    // not portal-trace, so the only LOAD_NAME the walker reaches in practice is
    // module-scope.)
    let w_locals = frame.get_w_locals();
    if !w_locals.is_null() && !std::ptr::eq(w_locals, w_globals) {
        return Ok(false);
    }
    let name = unsafe {
        pyre_object::unicodeobject::w_str_get_value(w_name_ptr as pyre_object::PyObjectRef)
    };
    emit_module_dict_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, name)
}

/// StoreName/StoreGlobal cell fold — module-scope store dual of
/// [`try_walker_load_name_cell_fold`].  Folds `i = <int>` on a hot module
/// global whose slot has stabilised to an `IntMutableCell` (the in-place
/// shape `write_cell` reaches after the 2nd int store) to a single
/// `setfield_gc_i(cell, intvalue)`, eliding the value boxing + residual dict
/// setitem.  Declines (→ residual `bh_store_name_fn`, which runs the full
/// `write_cell`) when the frame is non-module, the slot is not an immovable
/// `IntMutableCell`, or the value is not a provably-plain-int box (bool /
/// int-subclass / long / object all fall through — `write_cell` REPLACES the
/// cell + bumps the version for those, which the setfield fast path must not).
pub(crate) fn try_walker_store_name_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    frame_ptr: usize,
    w_name_ptr: usize,
    value_opref: OpRef,
) -> Result<bool, DispatchError> {
    if frame_ptr == 0 {
        return Ok(false);
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return Ok(false);
    }
    // Module scope gate: `w_locals` aliases `w_globals` (same as the LOAD
    // fold); a distinct `w_locals` routes to the live residual.
    let w_locals = frame.get_w_locals();
    if !w_locals.is_null() && !std::ptr::eq(w_locals, w_globals) {
        return Ok(false);
    }
    let name = unsafe {
        pyre_object::unicodeobject::w_str_get_value(w_name_ptr as pyre_object::PyObjectRef)
    };
    // Slot must hold an immovable `IntMutableCell`.  `can_move` gates the same
    // baked-address relocation hazard as the LOAD fold; mutable cells are
    // `malloc_typed` (never nursery) so a stabilised int global folds.
    let Some(slot) = crate::state::module_dict_cell_slot_direct(w_globals, name) else {
        return Ok(false);
    };
    let Some(stored) = crate::state::module_dict_cell_value_direct(w_globals, slot) else {
        return Ok(false);
    };
    if stored.is_null() || !unsafe { pyre_object::celldict::is_int_mutable_cell(stored) } {
        return Ok(false);
    }
    if majit_gc::can_move(majit_ir::GcRef(stored as usize)) {
        return Ok(false);
    }
    // The stored value must be a provably-plain-int box.  `is_plain_int1` on
    // the trace-time concrete rejects `bool` / int-subclass / `long` (whose
    // `write_cell` replaces the cell rather than mutating `intvalue`); the
    // heapcache lookup recovers the box's raw `intvalue` (populated only by
    // JIT int boxes, `emit_box_int_inline`), so the setfield needs no runtime
    // class guard — exactly as pypy's optimized trace folds the
    // `is_plain_int1` check away for an `int_add` result.
    let is_plain_int = matches!(
        ctx.trace_ctx.box_value(value_opref),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(p)))
            if p != 0 && unsafe { pyre_object::listobject::is_plain_int1(p as pyre_object::PyObjectRef) }
    );
    if !is_plain_int {
        return Ok(false);
    }
    let Some(raw_int) = ctx
        .trace_ctx
        .heapcache_getfield_cached(value_opref, crate::descr::int_intval_descr().index())
    else {
        return Ok(false);
    };
    // The eager concrete write needs the raw int the store applies; a
    // raw-int box with no concrete shadow declines to the residual.
    let Some(majit_ir::Value::Int(new_int)) = ctx.trace_ctx.box_value(raw_int) else {
        return Ok(false);
    };
    emit_namespace_cell_store_fold(ctx, op_pc, w_globals, slot, stored, raw_int, new_int)?;
    Ok(true)
}
