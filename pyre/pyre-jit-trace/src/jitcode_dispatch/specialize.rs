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
use rustpython_wtf8::Wtf8;

/// Replace an authentically executed builtin raise with ordinary trace
/// allocations.  The exception's stored args are the authority for both
/// wording and arity; keeping their objects as rooted trace constants avoids
/// re-deriving messages while still letting escape analysis virtualize the
/// exception and its args list when a handler discards them.
fn walker_recorded_builtin_raise_is_supported(
    exc: pyre_object::PyObjectRef,
    expected_kind: pyre_object::interp_exceptions::ExcKind,
) -> bool {
    if exc.is_null()
        || unsafe { !pyre_object::is_exception(exc) }
        || unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc) } != expected_kind
    {
        return false;
    }
    let args_storage = unsafe { pyre_object::interp_exceptions::w_exception_get_args_storage(exc) };
    if args_storage.is_null() || unsafe { !pyre_object::is_list(args_storage) } {
        return false;
    }
    let args_len = unsafe { pyre_object::w_list_len(args_storage) };
    (0..args_len).all(|index| {
        let Some(arg) = (unsafe { pyre_object::w_list_getitem(args_storage, index as i64) }) else {
            return false;
        };
        unsafe { pyre_object::is_str(arg) && pyre_object::is_exact_builtin_instance(arg) }
    })
}

fn walker_emit_recorded_builtin_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ec: OpRef,
    exc: pyre_object::PyObjectRef,
    expected_kind: pyre_object::interp_exceptions::ExcKind,
) -> DispatchOutcome {
    debug_assert!(walker_recorded_builtin_raise_is_supported(
        exc,
        expected_kind
    ));
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(exc);
    let args_storage = unsafe { pyre_object::interp_exceptions::w_exception_get_args_storage(exc) };
    let args_len = unsafe { pyre_object::w_list_len(args_storage) };
    let mut concrete_args = Vec::with_capacity(args_len);
    for index in 0..args_len {
        let arg = unsafe { pyre_object::w_list_getitem(args_storage, index as i64) }
            .expect("recorded exception args were validated before trace emission");
        pyre_object::gc_roots::pin_root(arg);
        concrete_args.push(arg);
    }
    let args = concrete_args
        .iter()
        .map(|&arg| ctx.trace_ctx.const_ref(arg as i64))
        .collect::<Vec<_>>();
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &args);
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

    let class = pyre_object::interp_exceptions::lookup_exc_class_for_kind(expected_kind);
    let class = ctx.trace_ctx.const_ref(class as i64);
    let raised =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, expected_kind, class, args_list);
    let exc_type =
        pyre_object::interp_exceptions::exc_kind_to_pytype(expected_kind) as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(raised, exc_type);
    ctx.trace_ctx
        .set_opref_concrete(raised, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    fbw_built_exc_insert(raised);

    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[raised, active],
        crate::descr::w_exception_context_descr(expected_kind),
    );
    // The authentic helper has published the raised exception but has not
    // chained it yet during the authoritative walk.  Mirror the recorded
    // field write so that iteration observes the same context as replay.
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }

    fbw_count_executed_residual(false, true);
    let exc_concrete = ConcreteValue::Ref(exc);
    ctx.last_exc_value = Some(raised);
    ctx.last_exc_value_concrete = exc_concrete;
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));
    DispatchOutcome::SubRaise {
        exc: raised,
        exc_concrete,
    }
}

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
/// RPython parity: `pyjitpl.py _opimpl_residual_call1` →
/// `do_residual_or_indirect_call` → `do_residual_call`
/// (pyjitpl.py). `pyjitpl.py opimpl_residual_call_r_i =
/// _opimpl_residual_call1` and `:1347 opimpl_residual_call_r_r =
/// _opimpl_residual_call1` confirm both kind variants share the
/// `_call1` body. The `_X` suffix is the *call's return kind* — mapping
/// comes from `do_residual_call`'s `descr.get_normalized_result_type()`
/// dispatch (pyjitpl.py) and `select_residual_call_opcode`'s
/// kind-keyed opcode tables.
///
/// `dst_bank` selects where the call's result lands:
/// * `'r'`: caller's `registers_r[dst]` — Ref-typed `Call*` family
///   (`_r_r/iRd>r`, `pyjitpl.py opimpl_residual_call_r_r`).
/// * `'i'`: caller's `registers_i[dst]` — Int-typed `Call*` family
///   (`_r_i/iRd>i`, `pyjitpl.py opimpl_residual_call_r_i`).
/// * `'v'`: void return — operand layout drops the trailing `>X` byte and
///   the writeback no-ops (`_r_v/iRd`, `pyjitpl.py
///   opimpl_residual_call_r_v`, `blackhole.py bhimpl_residual_call_r_v`).
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
///   - **release-gil** ([`direct_call_release_gil`], `pyjitpl.py-
///     3681`) — early-return when `ei.is_call_release_gil()`,
///     reshapes the arglist to `[savebox, funcbox] + argboxes[1:]`
///     and records `CALL_RELEASE_GIL_*` instead of `CALL_MAY_FORCE_*`.
///   - **loop-invariant heapcache** ([`loopinvariant_lookup`] /
///     [`loopinvariant_now_known`], `pyjitpl.py`) —
///     short-circuits the record on a heapcache hit and populates
///     the cache after a fresh record.
///
/// Emits `GUARD_NOT_FORCED` on the forces path plus
/// `GUARD_NO_EXCEPTION` whenever `check_can_raise(False)` is true,
/// matching `pyjitpl.py`. After every recorded call op,
/// invalidates the heapcache via
/// `heap_cache.invalidate_caches_varargs(call_opcode, ei, allboxes)`
/// matching `pyjitpl.py _record_helper_varargs` parity (forces
/// branch's `pyjitpl.py` redundantly invalidates with
/// `CALL_MAY_FORCE_*`, equivalent because `select_residual_call_opcode`
/// returns `CallMayForce*` for the forces classification).  Release-gil
/// helper invalidates with `CALL_MAY_FORCE_*` matching
/// `pyjitpl.py`'s `opnum1`. The pre-call vable IR bookkeeping
/// (`pyjitpl.py vable_and_vrefs_before_residual_call`, IR-only
/// portion: FORCE_TOKEN + SETFIELD_GC) is wired via
/// [`maybe_walker_vable_and_vrefs_before_residual_call`].  The
/// after-call helpers (`pyjitpl.py
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
///     (`pyjitpl.py _do_jit_force_virtual`) —
///     walker is fail-loud here via [`do_jit_force_virtual_guard`]
///     (called from each `dispatch_residual_call_*` arm); a producer
///     that emits an `OopSpecIndex::JitForceVirtual` calldescr surfaces
///     `DispatchError::JitForceVirtualRequiresConcreteResolver` instead
///     of silently recording `CALL_MAY_FORCE_*` (this was the prior
///     behaviour and is documented as STRICTER-THAN-PYPY in
///     [`do_jit_force_virtual_guard`]'s docstring). Optimizer pass
///     `OptVirtualize::optimize_jit_force_virtual` (`virtualize.rs`)
///     already handles the constant-token / non-null-forced short-circuit
///     post-trace. Adding the PTR_EQ + GUARD_VALUE prelude (the only
///     way to retire the fail-loud guard) is not yet implemented and
///     would land with the walker; metainterp has a tests-only
///     orthodox port at
///     `majit-metainterp/src/pyjitpl.rs _do_jit_force_virtual`
///     that the converged walker would route through. Production reach
///     today is zero — `jtransform.rs jit.force_virtual` is the only
///     producer and pyre's interpreter does not emit it.
///   - `vrefs_after_residual_call` is ported on `TraceCtx` but the walker
///     never calls it; no `jit.virtual_ref` producers exist today, so the
///     upstream loops are empty either way. Vable forces are detected by the
///     residual-call execution path's heap-token bracket.
///   - `direct_libffi_call` (`pyjitpl.py`) — pyre's live
///     tracer also returns `None` from this helper unless a
///     `CIF_DESCRIPTION_P` parser + dynamic `calldescr` builder lands
///     (`majit-metainterp/src/pyjitpl.rs` defers to
///     direct_call_release_gil/may_force, which is the same fall-through
///     the walker already takes).
///   - `direct_assembler_call` (`pyjitpl.py`) + KEEPALIVE
///     (`pyjitpl.py`) — only fire when `assembler_call=True`
///     in `do_residual_call`. Walker's residual_call dispatchers are
///     never called with `assembler_call=True`; the parallel
///     `inline_call_*/dR>X` family routes through
///     [`dispatch_inline_call_dr_kind`] instead. Adding the path would
///     require the codewriter to emit a new `assembler_call` shape, not
///     a walker-side change.
///   - Per-PC liveness narrowing for the snapshot that
///     `walker_capture_snapshot_for_last_guard` attaches
///     (`pyjitpl.py _get_list_of_active_boxes`). Walker's
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
///    `codegen.rs generated_store_subscr_value` for the
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
        // `pyjitpl.py handle_possible_exception` parity: drain
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
    // pyjitpl.py `_record_helper_varargs`: STORE_SUBSCR mutates the
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

/// Truth specialization for a concrete `W_BoolObject` operand — the sibling
/// [`try_walker_specialize_truth_int`] declines it, because it emits
/// `GUARD_CLASS INT` and a bool carries `BOOL_TYPE`.  Same `intval: i64`
/// layout, so only the guarded class constant differs; `is_true` on the
/// unboxed field is `W_BoolObject.is_true`'s `self.intval != 0`.
///
/// This is the shape every `if a == b:` reaches: `COMPARE_OP` leaves a boxed
/// bool the following `TO_BOOL` / `POP_JUMP_IF_*` tests, so without this arm a
/// comparison costs two `CALL_MAY_FORCE`s and two force/exception guard pairs
/// instead of one call and a field read.  When the comparison itself already
/// specialized, [`bool_box_truth_lookup`] folds the test first and this never
/// runs; it covers the case where the comparison stayed a residual.
pub(crate) fn try_walker_specialize_truth_bool<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
) -> Result<Option<OpRef>, DispatchError> {
    let Some(obj) = walker_concrete_ref_object(ctx, operand) else {
        return Ok(None);
    };
    let val = unsafe {
        if !pyre_object::is_bool(obj) {
            return Ok(None);
        }
        pyre_object::w_int_get_value(obj)
    };
    let bool_type_addr = &pyre_object::pyobject::BOOL_TYPE as *const _ as i64;
    let raw = walker_unbox_int(ctx, op_pc, operand, bool_type_addr)?;
    let truth = ctx.trace_ctx.record_op(OpCode::IntIsTrue, &[raw]);
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int((val != 0) as i64));
    Ok(Some(truth))
}

/// #61: walker-native identity fold for the `UNARY_POSITIVE` residual
/// (oopspec [`majit_ir::PyreHelperKind::UnaryPositive`]).  The object-space
/// `pos` on an exact int returns the operand unchanged, so a concrete non-bool
/// `W_IntObject` operand folds to the operand box itself behind the same guard
/// prefix the truth / binary int folds emit (a low-bit tag test for a tagged
/// immediate, `GUARD_CLASS INT` for a heap box).  The unboxed raw is discarded
/// (DCE): the result is the box, not its `intval`.
///
/// Returns `Ok(Some(()))` when the fold was emitted (caller returns
/// `Continue`); `Ok(None)` for a bool (`+True` is int `1`, not identity), a
/// numeric subclass, or a non-int operand — the caller then falls through to
/// the generic `CallMayForce` residual so a subclass / user `__pos__` still
/// runs.
pub(crate) fn try_walker_specialize_unary_positive_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    // `+x` is identity only for an EXACT builtin int.  A bool shares the
    // `intval` but `+True` is int `1`, not identity, and a numeric subclass
    // must reach its own `__pos__` rather than have the operand forwarded.
    let Some((_, x_class)) = walker_unary_int_operand(ctx, operand) else {
        return Ok(None);
    };
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    // Emit the guard prefix (`GUARD_CLASS INT` / tag test) so a later non-int
    // arrival deopts; the returned raw is unused because the result is the
    // operand box itself.
    let _ = walker_unbox_int(ctx, op_pc, operand, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, x_class)?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, operand)?;
    Ok(Some(()))
}

/// Shared gate for the `UNARY_POSITIVE` / `UNARY_NEGATIVE` / `UNARY_INVERT` int
/// folds: the operand must be a concrete EXACT builtin non-bool `W_IntObject`.
/// A bool unboxes through its own `&BOOL_TYPE` guard (declined here for
/// simplicity — `+True` / `-True` / `~True` stay on the residual).
///
/// Returns the concrete `intval` and the canonical `int` type object the caller
/// must pin with [`walker_guard_exact_w_class`].  `is_exact_builtin_instance`
/// only settles the operand the trace RECORDED; a numeric subclass keeps the
/// builtin `ob_type`, so the `GUARD_CLASS INT` the fold emits does not stop one
/// from entering the trace later and being answered by the fold instead of its
/// own `__neg__` / `__invert__` / `__pos__`.  Pinning `w_class` is what makes
/// that arrival side-exit, and the operand that carries the null spelling of
/// "exact builtin" has no value to pin, so it declines — the same shape the
/// long folds use (`walker_exact_builtin_class` + guard).
fn walker_unary_int_operand<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    operand: OpRef,
) -> Option<(i64, pyre_object::PyObjectRef)> {
    let obj = walker_concrete_ref_object(ctx, operand)?;
    // SAFETY: `obj` is a live concrete `PyObjectRef` from the walker shadow.
    unsafe {
        if !pyre_object::is_int(obj)
            || pyre_object::is_bool(obj)
            || !pyre_object::is_exact_builtin_instance(obj)
        {
            return None;
        }
        let class = walker_exact_builtin_class(obj)?;
        Some((pyre_object::w_int_get_value(obj), class))
    }
}

/// The `W_LongObject.value` payload of a concrete long, read the way the folds
/// that pass a payload to an `rbigint` helper need it.
///
/// # Safety
/// `obj` must be a live concrete `W_LongObject` from the walker shadow.
unsafe fn long_payload_of(obj: pyre_object::PyObjectRef) -> i64 {
    unsafe { *((obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET) as *const i64) }
}

/// Record the `getfield_gc_r` that reads a long operand's `value` payload.
/// A box the same trace built with [`crate::helpers::emit_box_long_inline`]
/// answers this out of the heap cache, so the read costs nothing and the box
/// keeps no reason to escape.
fn walker_read_long_payload<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    boxed: OpRef,
    concrete_payload: i64,
) -> OpRef {
    let payload = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[boxed],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        payload,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_payload as usize)),
    );
    payload
}

/// `intobject.py:494 _make_ovf2long`: the tail every int arithmetic fold shares
/// once its own guard has pinned the promoting branch — `GUARD_OVERFLOW` for
/// the `BINARY_OP` arm, `GUARD_VALUE` on the operand for unary negate. The tail
/// is the elidable raw-int bigint helper (`rbigint.py:717/788/873`) under
/// `EF_ELIDABLE_OR_MEMORYERROR`, then the inline `W_LongObject` box around the
/// payload it returns. `payload_fn` takes the two machine ints in
/// `(raw, concrete)` pairs, which is also the shape the concrete-args vector
/// wants.
///
/// The box needs no preceding fits_int guard. `newlong_from_rbigint`
/// (objspace.py:316-320) demotes through `rbigint.toint()`, whose
/// `numdigits() > MAX_DIGITS_THAT_CAN_FIT_IN_INT` test (rbigint.py:470) that
/// guard already answers: the helper is the *exact* int-pair sum / difference /
/// product, so a value that just overflowed a machine int cannot fit one back.
/// The same fold is what lets `try_walker_specialize_binary_op_long_int_pow`
/// skip its result-fits guard.
fn walker_emit_ovf2long_box<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    payload_fn: *const (),
    lhs: (OpRef, i64),
    rhs: (OpRef, i64),
    boxed_result_i64: i64,
) -> Result<OpRef, DispatchError> {
    let (lhs_raw, la) = lhs;
    let (rhs_raw, rb) = rhs;
    let payload_concrete =
        unsafe { long_payload_of(boxed_result_i64 as usize as pyre_object::PyObjectRef) };
    let concrete_args = [
        majit_ir::Value::Int(payload_fn as usize as i64),
        majit_ir::Value::Int(la),
        majit_ir::Value::Int(rb),
    ];
    let payload = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        payload_fn,
        &[lhs_raw, rhs_raw],
        &[majit_ir::Type::Int, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &concrete_args,
        majit_ir::Value::Ref(majit_ir::GcRef(payload_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        payload,
        majit_ir::Value::Ref(majit_ir::GcRef(payload_concrete as usize)),
    );
    if payload.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }
    let result = crate::helpers::emit_box_long_inline(
        ctx.trace_ctx,
        payload,
        crate::descr::w_long_size_descr(),
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
    );
    Ok(result)
}

/// #61: walker-native int specialization for the `UNARY_NEGATIVE` residual
/// (oopspec [`majit_ir::PyreHelperKind::UnaryNegative`]).  `-x` on an exact
/// int is `0 - x`; the object-space `neg` promotes only `-INT_MIN` to a
/// `W_LongObject` (`intobject.py:628` `descr_neg` → `_make_ovf2long`).  Since
/// majit has no overflow-checked unary negate, the fold expresses `-x` as
/// `IntSubOvf(0, x)` behind a `GUARD_CLASS INT`, reusing the binary-sub
/// overflow discipline in both directions: a record value other than `INT_MIN`
/// emits `GUARD_NO_OVERFLOW` so an `INT_MIN` arrival on the reused trace deopts
/// rather than wrapping back to `INT_MIN`, and a record value of `INT_MIN`
/// pins the operand with `GUARD_VALUE` and takes the same `_make_ovf2long` tail
/// the `BINARY_OP` overflow arm takes, so the `2**63` long is built from the
/// elidable bigint helper instead of the `CallMayForce` residual.
///
/// Returns `Ok(Some(()))` when the fold was emitted (caller returns
/// `Continue`); `Ok(None)` for a bool / subclass / non-int operand, when the
/// residual result box is unavailable, or when an `INT_MIN` operand did not
/// produce the promoted `W_LongObject` the payload read expects.
pub(crate) fn try_walker_specialize_unary_negative_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some((x, x_class)) = walker_unary_int_operand(ctx, operand) else {
        return Ok(None);
    };
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    // `0 - INT_MIN` is the one operand `descr_neg` promotes.
    let overflows = x == i64::MIN;
    if overflows {
        let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
        if boxed_result_obj == pyre_object::PY_NULL
            || !unsafe { pyre_object::is_long(boxed_result_obj) }
        {
            return Ok(None);
        }
    }
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let x_raw = walker_unbox_int(ctx, op_pc, operand, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, x_class)?;
    let zero_raw = ctx.trace_ctx.const_int(0);
    let boxed = if overflows {
        // `0 - x` overflows an i64 for exactly one operand, so "the negate
        // promoted" and "the operand is INT_MIN" name the same set: guarding
        // the value admits what `GUARD_OVERFLOW` would and nothing more. The
        // value form is the one the tail can use — with `x_raw` constant the
        // elidable bigint call folds to the `2**63` payload it returned while
        // recording instead of running once per iteration. This is the
        // `guard_value` spelling the version-tag promotes already use.
        let int_min = ctx.trace_ctx.const_int(i64::MIN);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[x_raw, int_min])?;
        walker_emit_ovf2long_box(
            ctx,
            op_pc,
            pyre_object::longobject::jit_bigint_sub_int_int as *const (),
            (zero_raw, 0),
            (x_raw, x),
            boxed_result_i64,
        )?
    } else {
        let result_value = 0i64.wrapping_sub(x);
        let raw_result = ctx
            .trace_ctx
            .record_op(OpCode::IntSubOvf, &[zero_raw, x_raw]);
        ctx.trace_ctx
            .set_opref_concrete(raw_result, majit_ir::Value::Int(result_value));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;
        let boxed = walker_box_int(ctx, op_pc, raw_result, result_value)?;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(result_value, boxed_result_i64));
        boxed
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// #61: walker-native int specialization for the `UNARY_INVERT` residual
/// (oopspec [`majit_ir::PyreHelperKind::UnaryInvert`]).  `~x` on an exact int
/// is `!x`, which always fits an i64 (`~INT_MIN == INT_MAX`, `~INT_MAX ==
/// INT_MIN`), so `descr_invert` never promotes to a long: the fold emits a
/// plain `IntInvert` behind a `GUARD_CLASS INT`, with no overflow guard.
///
/// Returns `Ok(Some(()))` when the fold was emitted (caller returns
/// `Continue`); `Ok(None)` for a bool / subclass / non-int operand, or when
/// the residual result box is unavailable.
pub(crate) fn try_walker_specialize_unary_invert_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some((x, x_class)) = walker_unary_int_operand(ctx, operand) else {
        return Ok(None);
    };
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let x_raw = walker_unbox_int(ctx, op_pc, operand, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, operand, x_class)?;
    let result_value = !x;
    let raw_result = ctx.trace_ctx.record_op(OpCode::IntInvert, &[x_raw]);
    ctx.trace_ctx
        .set_opref_concrete(raw_result, majit_ir::Value::Int(result_value));
    let boxed = walker_box_int(ctx, op_pc, raw_result, result_value)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(result_value, boxed_result_i64));
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
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
/// Returns `Ok(Some(outcome))` when the specialization was emitted; the
/// outcome is `Continue` for a value arm or `SubRaise` for a zero divisor.
/// `Ok(None)` means the operator is deferred
/// (FloorDiv / Mod / Shift / TrueDiv / Power / Subscr), the operands are
/// not both concrete `W_IntObject`, or an unsupported helper arm is reached — the caller
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
) -> Result<Option<DispatchOutcome>, DispatchError> {
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

    // boolobject.py descr_and/or/xor: when both operands are bool the
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

    // Inspect the operands before executing the authentic helper.  A raising
    // zero-divisor arm needs the helper-produced exception as its concrete
    // shadow, but records no helper call in the trace.
    let Some((lhs, rhs, lhs_obj, rhs_obj, la, rb)) =
        walker_int_specialization_input_operands(ctx, r_args)
    else {
        return Ok(None);
    };

    if matches!(op_code, OpCode::IntFloorDiv | OpCode::IntMod) && rb == 0 {
        let Some(Err(exc_i64)) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };
        // The helper publishes through both the blackhole cell (drained by
        // `execute_residual_call`) and the backend exception cells.  The
        // latter belong to compiled execution; drain the trace-time publish
        // before the walk continues into the Python handler, exactly as the
        // generic residual executor's Err arm does.
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        let exc = exc_i64 as usize as pyre_object::PyObjectRef;
        let kind = pyre_object::interp_exceptions::ExcKind::ZeroDivisionError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

        // Commit to the raising arm only after every decline.  Exact-class
        // guards preserve builtin dispatch; GuardTrue(rhs == 0) is the branch
        // guard a bridge can invert when the divisor changes mid-loop.
        let (lhs_type, lhs_descr) = crate::state::int_or_bool_unbox_type_descr(lhs_obj);
        let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
        let _lhs_raw = walker_unbox_int_typed(ctx, op_pc, lhs, lhs_type, lhs_descr)?;
        walker_guard_exact_w_class(ctx, op_pc, lhs, walker_numeric_builtin_class(lhs_obj))?;
        let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
        walker_guard_exact_w_class(ctx, op_pc, rhs, walker_numeric_builtin_class(rhs_obj))?;
        let rhs_zero = walker_int_eq_const(ctx, rhs_raw, 0, 1);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[rhs_zero])?;

        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }

    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };

    // intobject.py range validation (mirror the former int fast path's
    // needs_concrete_check): bail to the generic leg when the bare-IR-op
    // emission would be unsound (zero / INT_MIN-overflow divisor, oversized
    // / overflowing shift); large right-shift folds to a const.
    if needs_check {
        match op_code {
            OpCode::IntFloorDiv | OpCode::IntMod => {
                if la == i64::MIN && rb == -1 {
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
                // intobject.py, but that fold would be baked into the
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

    // pyjitpl.py:1881 handle_possible_overflow_error follows the concrete
    // Add/Sub/Mul outcome. The overflowing arm mirrors intobject.py:494
    // _make_ovf2long: guard_overflow, call the elidable raw-int bigint helper
    // (rbigint.py:717/788/873), and inline the W_LongObject box instead of
    // falling through to the generic CallMayForceR BINARY_OP leg.
    let overflows = has_overflow
        && match op_code {
            OpCode::IntAddOvf => la.checked_add(rb).is_none(),
            OpCode::IntSubOvf => la.checked_sub(rb).is_none(),
            OpCode::IntMulOvf => la.checked_mul(rb).is_none(),
            _ => false,
        };
    if overflows {
        let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
        if boxed_result_obj == pyre_object::PY_NULL
            || !unsafe { pyre_object::is_long(boxed_result_obj) }
        {
            return Ok(None);
        }
    }

    // --- emit the specialized IR (walker-native) ---
    // bool and int share `intval`; guard each operand against its own vtable
    // (BOOL_TYPE / INT_TYPE) so a bool unboxes through its own class.
    let (lhs_type, lhs_descr) = crate::state::int_or_bool_unbox_type_descr(lhs_obj);
    let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
    let lhs_raw = walker_unbox_int_typed(ctx, op_pc, lhs, lhs_type, lhs_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, walker_numeric_builtin_class(lhs_obj))?;
    let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, walker_numeric_builtin_class(rhs_obj))?;
    if overflows {
        let concrete_value = match op_code {
            OpCode::IntAddOvf => la.wrapping_add(rb),
            OpCode::IntSubOvf => la.wrapping_sub(rb),
            OpCode::IntMulOvf => la.wrapping_mul(rb),
            _ => unreachable!("overflow arm requires Add/Sub/Mul"),
        };
        let raw_result = ctx.trace_ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
        ctx.trace_ctx
            .set_opref_concrete(raw_result, majit_ir::Value::Int(concrete_value));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardOverflow, &[])?;

        let payload_fn = match op_code {
            OpCode::IntAddOvf => pyre_object::longobject::jit_bigint_add_int_int as *const (),
            OpCode::IntSubOvf => pyre_object::longobject::jit_bigint_sub_int_int as *const (),
            OpCode::IntMulOvf => pyre_object::longobject::jit_bigint_mul_int_int as *const (),
            _ => unreachable!("overflow arm requires Add/Sub/Mul"),
        };
        let result = walker_emit_ovf2long_box(
            ctx,
            op_pc,
            payload_fn,
            (lhs_raw, la),
            (rhs_raw, rb),
            boxed_result_i64,
        )?;
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
        return Ok(Some(DispatchOutcome::Continue));
    }
    let (raw_result, concrete_value) = match op_code {
        OpCode::IntFloorDiv | OpCode::IntMod => {
            walker_emit_int_div_domain_guards(ctx, op_pc, lhs_raw, rhs_raw, la, rb)?;
            walker_emit_int_py_div_or_mod(
                ctx,
                lhs_raw,
                rhs_raw,
                la,
                rb,
                op_code == OpCode::IntFloorDiv,
            )
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
    let boxed = if result_is_bool {
        match walker_newbool_guarded(
            ctx,
            op_pc,
            raw_result,
            concrete_value != 0,
            dst as u8,
            dst_bank,
        )? {
            Some(boxed) => boxed,
            None => {
                let boxed = crate::helpers::emit_trace_bool_value_from_truth(
                    ctx.trace_ctx,
                    raw_result,
                    false,
                );
                ctx.trace_ctx.set_opref_concrete(
                    boxed,
                    majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
                );
                boxed
            }
        }
    } else {
        let boxed = walker_box_int(ctx, op_pc, raw_result, concrete_value)?;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(concrete_value, boxed_result_i64));
        boxed
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// rint.py `_ovf_zer` guards for a machine-int division: `int_eq(rhs,0)` →
/// `guard_false` plus `(lhs==INT_MIN)&(rhs==-1)` → `guard_false`.  Both must
/// precede the elidable `ll_int_py_div` / `ll_int_py_mod` call so a re-used
/// trace bails before the helper's `wrapping_div` / `wrapping_rem` returns a
/// wrap value.  A `divmod` site shares one guard pair across both halves.
fn walker_emit_int_div_domain_guards<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    lhs_raw: OpRef,
    rhs_raw: OpRef,
    la: i64,
    rb: i64,
) -> Result<(), DispatchError> {
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
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[ovf_both])
}

/// jtransform.py `OS_INT_PY_DIV` / `OS_INT_PY_MOD` elidable residual call
/// (`call_typed_with_effect_pure` → `CallI` patched via
/// `record_result_of_call_pure`), returning the result op and its recorded
/// value.  The caller must have emitted
/// [`walker_emit_int_div_domain_guards`] over the same operand pair first.
fn walker_emit_int_py_div_or_mod<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    lhs_raw: OpRef,
    rhs_raw: OpRef,
    la: i64,
    rb: i64,
    is_div: bool,
) -> (OpRef, i64) {
    let (func_ptr, effect_info, concrete_result) = if is_div {
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
    ctx.trace_ctx
        .set_opref_concrete(r, majit_ir::Value::Int(concrete_result));
    (r, concrete_result)
}

/// Walker-native mixed `W_LongObject` / `W_IntObject` arithmetic
/// specialization for the `BINARY_OP` helper residual_call.
///
/// This is the trace shape of
/// `pypy/objspace/std/longobject.py:_make_generic_descr_binop` and
/// `descr_sub`: add/mul/and/or/xor select `rbigint.int_*` for either operand
/// order (the operations are commutative), while sub selects `int_sub` only
/// for `long - int`.  The opposite subtraction follows upstream's
/// `descr_rsub` bigint/bigint path and is deliberately left to the generic
/// record.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long_int<Sym: WalkSym>(
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
    use pyre_interpreter::objspace::descroperation as desc;
    type PayloadFn = extern "C" fn(i64, i64) -> pyre_object::longobject::JitBigIntResult;
    let (helper, commutative): (PayloadFn, bool) =
        match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
            Some(BinaryOperator::Add | BinaryOperator::InplaceAdd) => {
                (desc::jit_bigint_int_add, true)
            }
            Some(BinaryOperator::Subtract | BinaryOperator::InplaceSubtract) => {
                (desc::jit_bigint_int_sub, false)
            }
            Some(BinaryOperator::Multiply | BinaryOperator::InplaceMultiply) => {
                (desc::jit_bigint_int_mul, true)
            }
            Some(BinaryOperator::And | BinaryOperator::InplaceAnd) => {
                (desc::jit_bigint_int_and, true)
            }
            Some(BinaryOperator::Or | BinaryOperator::InplaceOr) => (desc::jit_bigint_int_or, true),
            Some(BinaryOperator::Xor | BinaryOperator::InplaceXor) => {
                (desc::jit_bigint_int_xor, true)
            }
            _ => return Ok(None),
        };
    let (lhs_obj, rhs_obj) = match (
        walker_concrete_ref_object(ctx, r_args[0]),
        walker_concrete_ref_object(ctx, r_args[1]),
    ) {
        (Some(lhs), Some(rhs)) => (lhs, rhs),
        _ => return Ok(None),
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(lhs_obj)
            || pyre_object::tagged_int::is_tagged_int(rhs_obj))
    {
        return Ok(None);
    }
    let (long, int, long_obj, int_obj) = unsafe {
        if pyre_object::is_long(lhs_obj) && pyre_object::is_int(rhs_obj) {
            (r_args[0], r_args[1], lhs_obj, rhs_obj)
        } else if commutative && pyre_object::is_int(lhs_obj) && pyre_object::is_long(rhs_obj) {
            (r_args[1], r_args[0], rhs_obj, lhs_obj)
        } else {
            return Ok(None);
        }
    };
    let (Some(long_class), Some(int_class)) = (unsafe {
        (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        )
    }) else {
        return Ok(None);
    };
    let int_value = unsafe { pyre_object::w_int_get_value(int_obj) };

    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL
        || unsafe {
            pyre_object::is_int(boxed_result_obj) || !pyre_object::is_long(boxed_result_obj)
        }
    {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, long, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let int_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let long_payload = unsafe { *((long_obj as *const u8).add(off) as *const i64) };
    let long_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        long_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
    );
    let helper_ptr = helper as *const ();
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper_ptr,
        &[long_pl, int_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper_ptr as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
            majit_ir::Value::Int(int_value),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }

    // The box needs no preceding fits_int guard. `_make_generic_descr_binop`
    // and `descr_sub` (longobject.py:304-349) wrap with
    // `W_LongObject(intop(...))`, and the interpreter arms they model
    // (`long_add`, `long_sub`, `long_mul`, `long_bitand`, `long_bitor`,
    // `long_bitxor`) wrap with `w_long_new`. Neither side demotes a
    // machine-sized result to a `W_IntObject`, so a result that fits is the
    // same object shape as one that does not, and declining on it left every
    // `x & 0xff`-shaped operation on the generic residual. The
    // `is_int(boxed_result_obj)` test above is what catches a path that does
    // demote.
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

/// Walker-native `W_LongObject // W_IntObject` / `%` specialization for the
/// `BINARY_OP` helper residual_call.
///
/// `pypy/objspace/std/longobject.py:424,441 _make_descr_binop` selects
/// `_int_floordiv` / `_int_mod` when the right operand is a `W_IntObject`.
/// The two legs differ in their *result* representation, and that difference
/// is the whole point of specialising them apart:
///   * `_int_floordiv` (`longobject.py:417-423`) → `rbigint.int_floordiv` →
///     a bigint quotient, boxed as a `W_LongObject` — the same shape
///     [`try_walker_specialize_binary_op_long_int_shift`] emits.
///   * `_int_mod` (`longobject.py:434-440`) → `rbigint.int_mod_int_result` →
///     `space.newint`: the remainder of a long by a machine int always fits a
///     machine int, so this leg allocates **no** result bigint and boxes a
///     plain `W_IntObject`.
///
/// `long_floordiv` / `long_mod` (`descroperation.rs`) raise
/// `ZeroDivisionError` before reaching the `_nonzero` rbigint seam, so the
/// divisor test is traced as a `GUARD_TRUE(int_ne(divisor, 0))` — the guard a
/// meta-tracer records for that interpreter branch. A replay with a zero
/// divisor therefore bails to the interpreter, which raises the authentic
/// error rather than re-deriving its wording here.
///
/// `int // long` and `int % long` are `descr_rfloordiv` / `descr_rmod`, which
/// coerce the left operand to a long and take the bigint/bigint path; they are
/// deliberately left to [`try_walker_specialize_binary_op_long`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long_int_div<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    let is_floordiv = match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
        Some(BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide) => true,
        Some(BinaryOperator::Remainder | BinaryOperator::InplaceRemainder) => false,
        _ => return Ok(None),
    };
    let long = r_args[0];
    let int = r_args[1];
    let (Some(long_obj), Some(int_obj)) = (
        walker_concrete_ref_object(ctx, long),
        walker_concrete_ref_object(ctx, int),
    ) else {
        return Ok(None);
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(long_obj)
            || pyre_object::tagged_int::is_tagged_int(int_obj))
    {
        return Ok(None);
    }
    let (long_class, int_class, int_value) = unsafe {
        if !pyre_object::is_long(long_obj) || !pyre_object::is_int(int_obj) {
            return Ok(None);
        }
        let (Some(long_class), Some(int_class)) = (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        ) else {
            return Ok(None);
        };
        (long_class, int_class, pyre_object::w_int_get_value(int_obj))
    };
    if int_value == 0 {
        let Some(Err(exc_i64)) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        let exc = exc_i64 as usize as pyre_object::PyObjectRef;
        let kind = pyre_object::interp_exceptions::ExcKind::ZeroDivisionError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

        let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
        walker_guard_class(ctx, op_pc, long, long_type_addr)?;
        walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
        let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
        let int_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
        walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
        let zero = ctx.trace_ctx.const_int(0);
        let is_zero = ctx.trace_ctx.record_op(OpCode::IntEq, &[int_raw, zero]);
        ctx.trace_ctx
            .set_opref_concrete(is_zero, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[is_zero])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }

    // Execute the authentic Python operation first: it supplies both the
    // observable result and the concrete payload for the pure call, without
    // running an allocating rbigint helper twice at record time.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL {
        return Ok(None);
    }
    // Everything that can decline must do so before the first guard is
    // recorded — a later bail-out would leave the operand's class pinned in
    // the heap cache with no matching guard in the trace.
    let quotient_concrete = if is_floordiv {
        if unsafe {
            pyre_object::is_int(boxed_result_obj) || !pyre_object::is_long(boxed_result_obj)
        } {
            return Ok(None);
        }
        let raw_concrete = unsafe {
            *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
                as *const i64)
        };
        Some(raw_concrete)
    } else {
        if unsafe { !pyre_object::is_int(boxed_result_obj) } {
            return Ok(None);
        }
        None
    };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, long, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let int_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
    let zero = ctx.trace_ctx.const_int(0);
    let nonzero = ctx.trace_ctx.record_op(OpCode::IntNe, &[int_raw, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonzero, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonzero])?;

    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let long_payload = unsafe { *((long_obj as *const u8).add(off) as *const i64) };
    let long_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        long_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
    );

    let result = match quotient_concrete {
        Some(raw_concrete) => {
            let helper =
                pyre_interpreter::objspace::descroperation::jit_bigint_int_div_floor as *const ();
            let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
                OpCode::CallR,
                helper,
                &[long_pl, int_raw],
                &[majit_ir::Type::Ref, majit_ir::Type::Int],
                majit_ir::Type::Ref,
                majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
                &[
                    majit_ir::Value::Int(helper as usize as i64),
                    majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
                    majit_ir::Value::Int(int_value),
                ],
                majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
            );
            ctx.trace_ctx.set_opref_concrete(
                raw,
                majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
            );
            if raw.inline_const_to_value().is_none() {
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
            }
            // The box needs no preceding fits_int guard. `_floordiv`/
            // `_int_floordiv` (longobject.py:409-424) wrap with `newlong` and
            // `long_floordiv` with `w_long_new`, so the quotient is a
            // `W_LongObject` whatever its magnitude, and the inline box carries
            // the payload by pointer — nothing about it varies with the digit
            // count the guard was testing.
            let boxed = crate::helpers::emit_box_long_inline(
                ctx.trace_ctx,
                raw,
                crate::descr::w_long_size_descr(),
                crate::descr::long_value_descr(),
            );
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            boxed
        }
        None => {
            let mod_concrete = unsafe { pyre_object::w_int_get_value(boxed_result_obj) };
            let helper = pyre_interpreter::objspace::descroperation::jit_bigint_int_mod_int_result
                as *const ();
            let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
                OpCode::CallI,
                helper,
                &[long_pl, int_raw],
                &[majit_ir::Type::Ref, majit_ir::Type::Int],
                majit_ir::Type::Int,
                majit_metainterp::ELIDABLE_EFFECT_INFO,
                &[
                    majit_ir::Value::Int(helper as usize as i64),
                    majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
                    majit_ir::Value::Int(int_value),
                ],
                majit_ir::Value::Int(mod_concrete),
            );
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(mod_concrete));
            if raw.inline_const_to_value().is_none() {
                walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
            }
            let boxed = walker_box_int(ctx, op_pc, raw, mod_concrete)?;
            ctx.trace_ctx
                .set_opref_concrete(boxed, box_int_concrete(mod_concrete, boxed_result_i64));
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// Walker-native `W_LongObject ** W_IntObject` specialization for the
/// `BINARY_OP` helper residual_call.
///
/// `longobject.py:206-231 descr_pow` keeps a `W_IntObject` exponent unwrapped
/// (`exp_bigint` stays `None`) and calls `rbigint.int_pow`; only a long
/// exponent reaches `rbigint.pow`. `long_pow` (`descroperation.rs`) reaches
/// that call past four short-circuits — a negative exponent goes to the float
/// path, a zero exponent returns 1, and a base of 0 / 1 / -1 returns a
/// constant. The exponent tests unbox to a machine word, so they record as a
/// single `GUARD_TRUE(int_gt(exp, 0))`.
///
/// The three base tests collapse into one guard: a base whose payload does not
/// fit a machine word has magnitude at least `2**63`, so it can be none of
/// 0 / 1 / -1. That is strictly stronger than the interpreter's three
/// comparisons — it can only bail more often, never take a different path —
/// and it costs one `jit_bigint_fits_int` call instead of three bigint
/// comparisons against baked constants. It also makes the usual trailing
/// result-fits guard redundant: `|base| >= 2**63` with `exp >= 1` cannot
/// produce a result that fits a machine word, so unlike the floordiv and shift
/// legs this one emits no guard on the result payload.
///
/// `descr_pow` wraps every branch as `W_LongObject` (no `newlong` demotion),
/// so the result boxes inline exactly like the other long legs.
/// The three-argument `pow(a, b, m)` is not a `BINARY_OP` and never reaches
/// here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long_int_pow<Sym: WalkSym>(
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
    if !matches!(
        pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
        Some(BinaryOperator::Power | BinaryOperator::InplacePower)
    ) {
        return Ok(None);
    }
    let long = r_args[0];
    let int = r_args[1];
    let (Some(long_obj), Some(int_obj)) = (
        walker_concrete_ref_object(ctx, long),
        walker_concrete_ref_object(ctx, int),
    ) else {
        return Ok(None);
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(long_obj)
            || pyre_object::tagged_int::is_tagged_int(int_obj))
    {
        return Ok(None);
    }
    let (long_class, int_class, exp_value) = unsafe {
        if !pyre_object::is_long(long_obj) || !pyre_object::is_int(int_obj) {
            return Ok(None);
        }
        let (Some(long_class), Some(int_class)) = (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        ) else {
            return Ok(None);
        };
        (long_class, int_class, pyre_object::w_int_get_value(int_obj))
    };
    // A non-positive exponent leaves through one of the short-circuits above
    // the `rbigint.int_pow` call; record those through the generic helper.
    if exp_value <= 0 {
        return Ok(None);
    }
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let base_payload = unsafe { *((long_obj as *const u8).add(off) as *const i64) };
    if pyre_object::longobject::jit_bigint_fits_int(base_payload) != 0 {
        return Ok(None);
    }

    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL
        || unsafe {
            pyre_object::is_int(boxed_result_obj) || !pyre_object::is_long(boxed_result_obj)
        }
    {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, long, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let exp_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
    let zero = ctx.trace_ctx.const_int(0);
    let positive = ctx.trace_ctx.record_op(OpCode::IntGt, &[exp_raw, zero]);
    ctx.trace_ctx
        .set_opref_concrete(positive, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[positive])?;

    let base_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        base_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(base_payload as usize)),
    );
    let fits_fn = pyre_object::longobject::jit_bigint_fits_int as *const ();
    let base_fits = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallI,
        fits_fn,
        &[base_pl],
        &[majit_ir::Type::Ref],
        majit_ir::Type::Int,
        majit_metainterp::cannot_raise_effect_info(),
    );
    ctx.trace_ctx
        .set_opref_concrete(base_fits, majit_ir::Value::Int(0));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[base_fits])?;

    let helper = pyre_interpreter::objspace::descroperation::jit_bigint_int_pow_nomod as *const ();
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper,
        &[base_pl, exp_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(base_payload as usize)),
            majit_ir::Value::Int(exp_value),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }

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

/// Walker-native `W_LongObject << W_IntObject` / `>>` specialization for the
/// `BINARY_OP` helper residual_call.
///
/// `pypy/objspace/std/longobject.py:_make_descr_binop` selects
/// `_int_lshift` / `_int_rshift` when the right operand is a `W_IntObject`;
/// those pass its machine-word `int_w` directly to `rbigint.lshift/rshift`.
/// Preserve that distinct source shape here: guard/unbox the right operand as
/// an Int (or Bool), guard the count non-negative, and call the matching
/// `[Ref, Int] -> Ref` rbigint residual.  Converting the count to a temporary
/// bigint and routing through the two-long helper would not be the upstream
/// program and would add an allocation to every shift.
///
/// As in [`try_walker_specialize_binary_op_long`], the helper result is the
/// bare immutable bigint payload and the walker boxes it as a
/// `W_LongObject`.  A result that demotes to `W_IntObject` declines before
/// emitting IR; replay is protected by the same fits-int guard.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_long_int_shift<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    let is_lshift = match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
        Some(BinaryOperator::Lshift | BinaryOperator::InplaceLshift) => true,
        Some(BinaryOperator::Rshift | BinaryOperator::InplaceRshift) => false,
        _ => return Ok(None),
    };
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    let (lhs_class, rhs_class, rhs_value) = unsafe {
        if !pyre_object::is_long(lhs_obj) || !pyre_object::is_int(rhs_obj) {
            return Ok(None);
        }
        let (Some(lhs_class), Some(rhs_class)) = (
            walker_exact_builtin_class(lhs_obj),
            walker_exact_builtin_class(rhs_obj),
        ) else {
            return Ok(None);
        };
        (lhs_class, rhs_class, pyre_object::w_int_get_value(rhs_obj))
    };
    if rhs_value < 0 {
        let Some(Err(exc_i64)) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        let exc = exc_i64 as usize as pyre_object::PyObjectRef;
        let kind = pyre_object::interp_exceptions::ExcKind::ValueError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

        let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
        walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
        walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
        let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
        let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
        walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
        let zero = ctx.trace_ctx.const_int(0);
        let is_negative = ctx.trace_ctx.record_op(OpCode::IntLt, &[rhs_raw, zero]);
        ctx.trace_ctx
            .set_opref_concrete(is_negative, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[is_negative])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }

    // Execute the authentic Python operation first.  It supplies both the
    // observable result and the concrete payload for the pure call without
    // running an allocating rbigint helper twice at record time.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL
        || unsafe {
            pyre_object::is_int(boxed_result_obj) || !pyre_object::is_long(boxed_result_obj)
        }
    {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
    let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
    let rhs_raw = walker_unbox_int_typed(ctx, op_pc, rhs, rhs_type, rhs_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
    let zero = ctx.trace_ctx.const_int(0);
    let nonnegative = ctx.trace_ctx.record_op(OpCode::IntGe, &[rhs_raw, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonnegative, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonnegative])?;

    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let lhs_payload = unsafe { *((lhs_obj as *const u8).add(off) as *const i64) };
    let lhs_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[lhs],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        lhs_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
    );
    let helper = if is_lshift {
        pyre_interpreter::objspace::descroperation::jit_bigint_lshift_count as *const ()
    } else {
        pyre_interpreter::objspace::descroperation::jit_bigint_shr as *const ()
    };
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper,
        &[lhs_pl, rhs_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
            majit_ir::Value::Int(rhs_value),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }

    // The box needs no preceding fits_int guard: `_int_lshift` wraps with
    // `W_LongObject(...)` and `_int_rshift` with `newlong` (longobject.py:383,
    // 402), and `long_lshift`/`long_rshift` both end in `w_long_new`. Neither
    // demotes a machine-sized result — `newlong` only reaches
    // `W_SmallLongObject`, which `withsmalllong` leaves off — so declining on
    // a fitting result put every `x >> 32`-shaped shift back on the generic
    // residual for no observable difference.
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
    Ok(Some(DispatchOutcome::Continue))
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
    use pyre_interpreter::bytecode::BinaryOperator;
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        return Ok(None);
    };
    let Some(spec) = crate::trace_opcode::long_binop_raw_helper(op) else {
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
    let (Some(lhs_class), Some(rhs_class)) = (unsafe {
        (
            walker_exact_builtin_class(lhs_obj),
            walker_exact_builtin_class(rhs_obj),
        )
    }) else {
        return Ok(None);
    };
    // Authentic boxed result via the same execute path the int leg uses; a
    // NULL / raised result defers to the generic record.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    // A NULL result means the op raised — defer to the generic record. `newlong`
    // never demotes, so an arithmetic long op always yields a W_LongObject the
    // inline-NEW box below can represent. The shift ops are the only ones that
    // can still yield a W_IntObject (`space.newint(-1)`/`(0)` on a shift count
    // that overflows a machine int); the `!is_long` decline routes that
    // huge-count case to the generic leg. Reuse the authentic boxed result's
    // payload instead of running `spec.raw_fn` a second time; the raw helpers
    // allocate/publish exception state and must not be used as a trace-time
    // probe.
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_long(boxed_result_obj) } {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_result_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET)
            as *const i64)
    };
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
    // Read each operand's immutable `value` payload, then call the
    // elidable `rbigint` op on the bare `*const BigInt` payloads. Passing
    // the payloads (not the wrappers) keeps the call pure on the immutable
    // bigints, so the optimizer forwards the field read and never reorders
    // this elidable call ahead of the boxing `setfield_gc` below — which
    // would otherwise read the freshly-allocated result wrapper's
    // uninitialized `value` (the function-loop unroll exposed exactly that
    // reorder). The forwarding is descr-keyed (`long_value_descr()` is
    // immutable), so the plain `GETFIELD_GC_R` opnum gets identical OptHeap
    // treatment — there is no pure getfield opnum. The result is a
    // GC-managed `*mut BigInt`, Ref-typed so the JIT gcmap roots it across
    // the collecting boxing NEW. Every op allocates
    // (`EF_ELIDABLE_OR_MEMORYERROR`) or divides (`EF_ELIDABLE_CAN_RAISE`),
    // so a trailing `GuardNoException` follows (`pyjitpl.py`).
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let lhs_payload = unsafe { *((lhs_obj as *const u8).add(off) as *const i64) };
    let rhs_payload = unsafe { *((rhs_obj as *const u8).add(off) as *const i64) };
    let lhs_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[lhs],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        lhs_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
    );
    let rhs_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
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
    // Shift-count demote guard: `_lshift`/`_rshift` demote to `space.newint`
    // (-1/0) when the shift count overflows a machine int (`toint()`
    // OverflowError), so guard that the count fits and let a huge-count replay
    // deopt to the generic leg. The arithmetic ops (`newlong`, no demote) emit
    // no such guard — a fitting result stays a W_LongObject in the trace.
    if matches!(
        op,
        BinaryOperator::Lshift
            | BinaryOperator::InplaceLshift
            | BinaryOperator::Rshift
            | BinaryOperator::InplaceRshift
    ) {
        let fits_fn = pyre_object::longobject::jit_bigint_fits_int as *const ();
        let count_fits = ctx.trace_ctx.call_typed_with_effect(
            OpCode::CallI,
            fits_fn,
            &[rhs_pl],
            &[majit_ir::Type::Ref],
            majit_ir::Type::Int,
            majit_metainterp::cannot_raise_effect_info(),
        );
        let count_fits_concrete =
            unsafe { pyre_object::longobject::jit_bigint_fits_int(rhs_payload) };
        ctx.trace_ctx
            .set_opref_concrete(count_fits, majit_ir::Value::Int(count_fits_concrete));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[count_fits])?;
    }
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
    let (Some(lhs_class), Some(rhs_class)) = (unsafe {
        (
            walker_exact_builtin_class(lhs_obj),
            walker_exact_builtin_class(rhs_obj),
        )
    }) else {
        return Ok(None);
    };
    // Authentic boxed float via the generic execute path; a NULL / raised result
    // (zero divisor, float overflow) defers to the generic record.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
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
    // pyjitpl.py: no GuardNoException when the pure call folded to a Const.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpecialisedPairKind {
    Int,
    Float,
    Object,
}

/// Identify the three classes produced by
/// `specialisedtupleobject.py:169-179 makespecialisedtuple2`.
pub(crate) fn specialised_pair_kind(
    seq_type: *const pyre_object::pyobject::PyType,
) -> Option<SpecialisedPairKind> {
    use pyre_object::specialisedtupleobject::{
        SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
    };
    if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_II_TYPE) {
        Some(SpecialisedPairKind::Int)
    } else if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_FF_TYPE) {
        Some(SpecialisedPairKind::Float)
    } else if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_OO_TYPE) {
        Some(SpecialisedPairKind::Object)
    } else {
        None
    }
}

/// FBW fold of the UNPACK_SEQUENCE two-residual lowering (`unpack_sequence_fn`
/// validator + per-index `unpack_item_fn` reader emitted by the codewriter
/// UNPACK_SEQUENCE arm) for an arity-2 specialised tuple: guard the
/// specialisation's class once, then read `value0` / `value1` directly instead
/// of leaving three opaque `CALL_MAY_FORCE` residuals in the loop.
///
/// `objspace.py:519-523 fixedview` reaches `tolist()` for every
/// `W_AbstractTupleObject`, and `specialisedtupleobject.py:32,58-64 tolist`
/// unrolls over `_immutable_fields_` value slots, so upstream traces the whole
/// unpack inline and the optimizer virtualizes the pair away. Both arity-2
/// layouts are covered here because `makespecialisedtuple2`
/// (`specialisedtupleobject.py:169-179`) never falls back to a plain tuple:
///   * `ii` — `value0`/`value1` are inline machine ints, so the read is
///     `getfield_gc_pure_i` + `wrapint` and the items stay unboxed through the
///     downstream BINARY_OP int fold (the walker analogue of the retired
///     MIFrame `W_SpecialisedTupleObject_ii` reads);
///   * `ff` — the same shape with `getfield_gc_pure_f` + `wrapfloat`; this is
///     the representation `zip` produces for a pair of exact floats;
///   * `oo` — `wraps[i]` for an object slot is the identity
///     (`specialisedtupleobject.py:26-27`), so the `getfield_gc_r` result is
///     already the item. This is the layout a `divmod(long, long)` result pair
///     takes, since neither half satisfies `is_plain_int1`.
///
/// Returns `Ok(Some(()))` when folded (the caller returns `Continue`);
/// `Ok(None)` to fall through to the opaque residual record, which stays
/// correct for any other shape — so a non-foldable sequence is not declined.
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
    // `objspace.py:507-541 StdObjSpace.{unpackiterable,fixedview}` takes an
    // exact tuple straight to its immutable `wrappeditems` list; and
    // `pyopcode.py:889 UNPACK_SEQUENCE` calls `fixedview_unroll`, so a
    // constant item count exposes each tuple item directly to the trace.
    // Preserve that shape for pyre's split `UnpackSequence` / `UnpackItem`
    // helpers.  In particular, `zip` produces an ordinary array-backed tuple
    // each iteration; leaving these helpers residual makes the three-item
    // comprehension trace execute one validation call plus one call per
    // projected item.
    let tuple_type = &pyre_object::pyobject::TUPLE_TYPE as *const pyre_object::pyobject::PyType;
    let canonical_tuple_class = pyre_object::pyobject::get_instantiate(unsafe { &*tuple_type });
    if unsafe {
        std::ptr::eq((*concrete_seq).ob_type, tuple_type)
            && std::ptr::eq((*concrete_seq).w_class, canonical_tuple_class)
    } {
        let concrete_len = unsafe { pyre_object::w_tuple_len(concrete_seq) };
        if int_val < 0 {
            return Ok(None);
        }
        walker_guard_class(ctx, op_pc, seq, tuple_type as i64)?;
        walker_guard_exact_w_class(ctx, op_pc, seq, canonical_tuple_class)?;
        let items = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            seq,
            crate::descr::tuple_wrappeditems_descr(),
        );
        match helper {
            majit_ir::PyreHelperKind::UnpackSequence => {
                if int_val as usize != concrete_len {
                    return Ok(None);
                }
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    items,
                    crate::state::pyobject_gcarray_descr(),
                );
                let expected = ctx.trace_ctx.const_int(int_val);
                walker_emit_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardValue,
                    &[length, expected],
                )?;
                write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, seq)?;
                return Ok(Some(()));
            }
            majit_ir::PyreHelperKind::UnpackItem => {
                let index = int_val as usize;
                if index >= concrete_len {
                    return Ok(None);
                }
                // `index < concrete_len` only holds for the tuple recorded
                // here. A trace can enter between the `UnpackSequence` helper
                // and this one — a bridge resumes mid-unpack — so this arm
                // cannot rely on that helper's length guard being in the same
                // trace. Emit it; when it is present the optimizer folds this
                // one away, so a whole unpack still guards the length once.
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    items,
                    crate::state::pyobject_gcarray_descr(),
                );
                let expected = ctx.trace_ctx.const_int(concrete_len as i64);
                walker_emit_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardValue,
                    &[length, expected],
                )?;
                let index_op = ctx.trace_ctx.const_int(int_val);
                let item = crate::state::trace_items_block_getitem_value_pure(
                    ctx.trace_ctx,
                    items,
                    index_op,
                );
                let concrete_item = unsafe {
                    pyre_object::w_tuple_getitem(concrete_seq, int_val)
                        .unwrap_or(pyre_object::PY_NULL)
                };
                ctx.trace_ctx.set_opref_concrete(
                    item,
                    majit_ir::Value::Ref(majit_ir::GcRef(concrete_item as usize)),
                );
                write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, item)?;
                return Ok(Some(()));
            }
            _ => {}
        }
    }
    // Both arity-2 specialisations fold; any other shape (a plain tuple, a
    // list, or a non-canonical tuple) falls through to the opaque residual
    // (correct, slower).
    let seq_type = unsafe { (*concrete_seq).ob_type };
    let Some(pair_kind) = specialised_pair_kind(seq_type) else {
        return Ok(None);
    };
    let spec_type = seq_type;
    match helper {
        majit_ir::PyreHelperKind::UnpackSequence => {
            // Either specialisation is always arity 2, so the class guard
            // subsumes the exact-length check `unpack_sequence_fn` performs.
            if int_val != 2 {
                return Ok(None);
            }
            walker_guard_specialised_pair_class(ctx, op_pc, seq, spec_type)?;
            // Pass `seq` through as the validated tuple; the per-index
            // `unpack_item_fn` reads below fold off it.
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, seq)?;
            Ok(Some(()))
        }
        majit_ir::PyreHelperKind::UnpackItem => {
            if !(0..2).contains(&int_val) {
                return Ok(None);
            }
            // Normally the partner `unpack_sequence_fn` fold already guarded
            // the class (its validated-tuple passthrough reg == `seq`), in
            // which case this is a no-op; guard here too so a fold that only
            // catches the item reads still proves the layout it loads from.
            walker_guard_specialised_pair_class(ctx, op_pc, seq, spec_type)?;
            let Some(item) = walker_emit_specialised_pair_item(
                ctx, op_pc, seq, pair_kind, int_val, allboxes, call_descr,
            )?
            else {
                return Ok(None);
            };
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, item)?;
            Ok(Some(()))
        }
        _ => Ok(None),
    }
}

/// The tuple `descr_getargs` would build from an exception's `args_w` list.
/// Built only to read the representation `newtuple` picks off it, so the caller
/// keeps the shape and drops the tuple.
unsafe fn args_tuple_shape_probe(stored: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    let len = unsafe { pyre_object::w_list_len(stored) };
    let items = (0..len)
        .map(|index| {
            unsafe { pyre_object::w_list_getitem(stored, index as i64) }
                .unwrap_or(pyre_object::PY_NULL)
        })
        .collect();
    pyre_object::w_tuple_new(items)
}

/// `guard_class(seq, spec)` for one of the arity-2 tuple specialisations,
/// emitted once per traced `seq` — the heap cache turns every later fold on the
/// same register into a no-op, the way upstream's optimizer keeps a single
/// class guard for a value it already proved.
fn walker_guard_specialised_pair_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq: OpRef,
    spec_type: *const pyre_object::pyobject::PyType,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.heap_cache().is_class_known(seq) {
        return Ok(());
    }
    let type_const = ctx.trace_ctx.const_int(spec_type as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardClass, &[seq, type_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(seq, spec_type as i64);
    Ok(())
}

/// Read slot `index` (0 or 1) of an arity-2 tuple specialisation whose class
/// the caller has already guarded, applying that slot's `wraps[i]`
/// (`specialisedtupleobject.py:26-27`, and `:134-142 getitem`, which unrolls
/// `iter_n` to the matching `value%s`).
///
/// `Ok(None)` declines: the `ii` / `ff` slots need the authentic box for its
/// identity, and that execution can fail.
///
/// The `ff` arm currently has no producer to serve. Upstream builds `Cls_ff`
/// from `makespecialisedtuple2` (`specialisedtupleobject.py:178`) and from
/// `specialized_zip_2_lists` (`:230`); pyre does not port the latter, and
/// `w_tuple_new` (tupleobject.rs:174-186) sends a plain-float pair to `Cls_oo`
/// instead so that `(x, x)` keeps the exact `x` object. It is kept because it
/// is the layout upstream reads, not because a trace reaches it today.
#[allow(clippy::too_many_arguments)]
fn walker_emit_specialised_pair_item<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq: OpRef,
    pair_kind: SpecialisedPairKind,
    index: i64,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Result<Option<OpRef>, DispatchError> {
    let first = index == 0;
    if pair_kind == SpecialisedPairKind::Object {
        let descr = if first {
            crate::descr::specialised_tuple_oo_value0_descr()
        } else {
            crate::descr::specialised_tuple_oo_value1_descr()
        };
        // `wraps[i]` is the identity for an object slot, so the field read is
        // the whole item — no re-boxing, and no `may_force` execution needed
        // to recover a box identity.
        return Ok(Some(crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            seq,
            descr,
        )));
    }
    // Authentic boxed element supplies the concrete shadow / identity while
    // the emitted field read and transparent wrapper replace the residual call
    // in machine code. Fall through if execution raises or cannot provide that
    // box.
    let Some(elem_ptr) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    if pair_kind == SpecialisedPairKind::Float {
        let descr = if first {
            crate::descr::specialised_tuple_ff_value0_descr()
        } else {
            crate::descr::specialised_tuple_ff_value1_descr()
        };
        let raw = majit_metainterp::box_trace::getfield_gc_f_pureornot(ctx.trace_ctx, seq, descr);
        let elem = unsafe { pyre_object::w_float_get_value(elem_ptr as pyre_object::PyObjectRef) };
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Float(elem));
        let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(elem_ptr as usize)),
        );
        return Ok(Some(boxed));
    }
    // `ii`: preserve authentic small-int caching / identity.
    let descr = if first {
        crate::descr::specialised_tuple_ii_value0_descr()
    } else {
        crate::descr::specialised_tuple_ii_value1_descr()
    };
    let raw = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, seq, descr);
    let elem = unsafe { pyre_object::w_int_get_value(elem_ptr as pyre_object::PyObjectRef) };
    let boxed = walker_box_int(ctx, op_pc, raw, elem)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(elem, elem_ptr as i64));
    Ok(Some(boxed))
}

/// One hop of a `while tb is not None: names.append(tb.tb_frame.f_code.co_name);
/// tb = tb.tb_next` traceback walk.
///
/// Each of these is a `GetSetProperty` whose getter body is a slot read on a
/// receiver [`walker_specialize_traceback_walk_field`] pins by class:
/// `pytraceback.py descr_get_next` / `descr_get_tb_frame` /
/// `descr_get_tb_lineno` / `descr_get_tb_lasti` and `pyframe.py fget_code`.
/// None of them dispatches anywhere or can raise.  Left residual, every hop of
/// the walk costs a forcing call — measured at 207 ns per `tb_lineno` read
/// against 0 for the folded `tb_next` — which is what makes each traceback
/// fixture dominated by the walk rather than by the raise.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TracebackWalkField {
    /// `tb.tb_next` — the chain link; a null slot is the terminator and
    /// surfaces as `None`.
    TbNext,
    /// `tb.tb_frame` — the node's frame.  Unlike its two siblings the getter
    /// ALSO runs `mark_as_escaped()`; see the escape emit.
    TbFrame,
    /// `frame.f_code` — `fget_f_code` is `self.pycode as PyObjectRef`.
    FCode,
    /// `tb.tb_lineno` — the line the node froze at.  `get_lineno` resolves it
    /// lazily upstream; pyre stamps it at `record_application_traceback` time,
    /// so the getter is the slot read plus the `LINENO_NOT_COMPUTED` mapping.
    ///
    /// `tb_lasti` is deliberately absent: its getter reports `lasti * 2`, so
    /// the fold would have to carry the doubling rather than hand back the
    /// slot.
    TbLineno,
    /// `code.co_name` — `code_get_field` answers it with `w_code_name_obj`,
    /// which realizes the string once and retains it on the code object, so
    /// every later read is the retained slot.
    CoName,
    /// `code.co_firstlineno` — the `co_firstlineno_raw` slot, reboxed.  The
    /// other code fields are deliberately absent: they read the host
    /// `CodeObject` behind `code_ptr` rather than a slot on the `PyCode`, so
    /// folding one means a raw load through a second indirection.
    CoFirstlineno,
}

/// Which walk hop, if any, this `(receiver, attribute)` pair is.
fn traceback_walk_field(
    concrete_obj: pyre_object::PyObjectRef,
    name: &str,
) -> Option<TracebackWalkField> {
    let ob_type = unsafe { (*concrete_obj).ob_type };
    if std::ptr::eq(ob_type, &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE) {
        return match name {
            "tb_next" => Some(TracebackWalkField::TbNext),
            "tb_frame" => Some(TracebackWalkField::TbFrame),
            "tb_lineno" => Some(TracebackWalkField::TbLineno),
            _ => None,
        };
    }
    if std::ptr::eq(ob_type, &pyre_interpreter::pyframe::FRAME_TYPE) && name == "f_code" {
        return Some(TracebackWalkField::FCode);
    }
    if std::ptr::eq(ob_type, &pyre_interpreter::pycode::CODE_TYPE) {
        return match name {
            "co_name" => Some(TracebackWalkField::CoName),
            "co_firstlineno" => Some(TracebackWalkField::CoFirstlineno),
            _ => None,
        };
    }
    None
}

/// Prove the receiving code object still owns its host `CodeObject`, the
/// `require_code` check every code-field getter runs before reading a slot.
fn walker_guard_code_ptr_present<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    let code_ptr = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        obj,
        crate::descr::pycode_code_ptr_descr(),
    );
    let live = unsafe { pyre_interpreter::w_code_get_ptr(concrete_obj) } as i64;
    ctx.trace_ctx
        .set_opref_concrete(code_ptr, majit_ir::Value::Int(live));
    let zero = ctx.trace_ctx.const_int(0);
    let absent = ctx.trace_ctx.record_op(OpCode::IntEq, &[code_ptr, zero]);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[absent])
}

/// Emit one traceback-walk hop as a guarded inline field read instead of the
/// opaque `getattr_fn` residual.
///
/// Returns `None` (fall through to the residual) BEFORE recording any guard for
/// every shape it cannot settle — an uncacheable `version_tag`, or a null slot
/// on a hop whose null is not a documented value.  A bail-out after a guard
/// would leave the caller reading the attribute as already pinned.
fn walker_specialize_traceback_walk_field<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    field: TracebackWalkField,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::pyframe::PyFrame;

    let receiver_type = match field {
        TracebackWalkField::FCode => &pyre_interpreter::pyframe::FRAME_TYPE,
        TracebackWalkField::CoName | TracebackWalkField::CoFirstlineno => {
            &pyre_interpreter::pycode::CODE_TYPE
        }
        _ => &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE,
    };
    let descr = match field {
        TracebackWalkField::TbNext => crate::descr::pytraceback_w_next_descr(),
        TracebackWalkField::TbFrame => crate::descr::pytraceback_frame_descr(),
        TracebackWalkField::FCode => crate::descr::pyframe_code_descr(),
        TracebackWalkField::TbLineno => crate::descr::pytraceback_lineno_descr(),
        TracebackWalkField::CoName => crate::descr::pycode_w_name_descr(),
        TracebackWalkField::CoFirstlineno => crate::descr::pycode_co_firstlineno_descr(),
    };
    let w_type = pyre_interpreter::typedef::gettypeobject(receiver_type);
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 {
        return Ok(None);
    }
    // The slot guard pins the receiver's `w_class` against `w_type`.  A frame
    // built before `init_typeobjects` carries a null `w_class`, which would
    // make that guard fail on its first execution, so decline instead of
    // recording a doomed trace.
    if unsafe { (*concrete_obj).w_class } != w_type {
        return Ok(None);
    }

    // Every code-field getter resolves the host `CodeObject` first
    // (`code_get_field` -> `require_code`) and raises when it is absent, so a
    // code fold owes that check.  It is a slot on the receiver, so the trace
    // proves it the same way — a read plus a non-null guard — rather than
    // trusting the record-time object.
    let code_receiver = matches!(
        field,
        TracebackWalkField::CoName | TracebackWalkField::CoFirstlineno
    );
    if code_receiver
        && unsafe { pyre_interpreter::w_code_get_ptr(concrete_obj) }
            .cast::<u8>()
            .is_null()
    {
        return Ok(None);
    }

    if field == TracebackWalkField::CoFirstlineno {
        let live =
            i64::from(unsafe { pyre_interpreter::pycode::w_code_firstlineno_raw(concrete_obj) });
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        walker_guard_code_ptr_present(ctx, op_pc, obj, concrete_obj)?;
        let raw_value = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, obj, descr);
        ctx.trace_ctx
            .set_opref_concrete(raw_value, majit_ir::Value::Int(live));
        // Reboxed for the same reason `TbLineno` is: the getter hands back a
        // Python int, and the boxed op is a heap `NewWithVtable`.
        let boxed = walker_box_int(ctx, op_pc, raw_value, live)?;
        let live_ptr = pyre_object::w_int_new(live) as i64;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
        return Ok(Some(()));
    }

    if field == TracebackWalkField::TbLineno {
        let live =
            unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_lineno_raw(concrete_obj) };
        // `get_lineno` answers the sentinel with `-1`, so the slot value is the
        // getter's value only once it is pinned against the sentinel.  A node
        // that already carries it — built from a frame with no `pycode`, or
        // handed the sentinel through `TracebackType(..., -sys.maxsize-1)` or
        // the `tb_lineno` setter — has nothing to pin, so decline before
        // recording anything.
        if live == pyre_interpreter::pytraceback::LINENO_NOT_COMPUTED {
            return Ok(None);
        }
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let raw_value = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, obj, descr);
        ctx.trace_ctx
            .set_opref_concrete(raw_value, majit_ir::Value::Int(live));
        let not_computed = ctx
            .trace_ctx
            .const_int(pyre_interpreter::pytraceback::LINENO_NOT_COMPUTED);
        let is_not_computed = ctx
            .trace_ctx
            .record_op(OpCode::IntEq, &[raw_value, not_computed]);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[is_not_computed])?;
        // The getter returns a Python int, so the raw slot is reboxed the way
        // the unboxed mapdict read does; the boxed op is a heap `NewWithVtable`
        // so its concrete has to be a heap pointer too.
        let boxed = walker_box_int(ctx, op_pc, raw_value, live)?;
        let live_ptr = pyre_object::w_int_new(live) as i64;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
        return Ok(Some(()));
    }

    let stored = match field {
        TracebackWalkField::TbNext => unsafe {
            pyre_interpreter::pytraceback::w_pytraceback_get_w_next(concrete_obj)
        },
        TracebackWalkField::TbFrame => {
            (unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(concrete_obj) })
                as pyre_object::PyObjectRef
        }
        // `w_name` is realized on first demand, so an unread code object
        // carries a null here; that declines below and the residual realizes
        // it for the next attempt.
        TracebackWalkField::CoName => unsafe {
            (*(concrete_obj as *const pyre_interpreter::pycode::PyCode)).w_name
        },
        _ => (unsafe { (*(concrete_obj as *const PyFrame)).pycode }) as pyre_object::PyObjectRef,
    };
    // Only `tb_next` has a null with a defined meaning.  A null frame is a
    // torn-down traceback and a null `pycode` a half-built frame; both are
    // answered by a `sys.namespace` stub or `None` the residual owns.
    if stored.is_null() && field != TracebackWalkField::TbNext {
        return Ok(None);
    }
    walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
    if code_receiver {
        walker_guard_code_ptr_present(ctx, op_pc, obj, concrete_obj)?;
    }
    let raw_value = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, descr);
    let value = if stored.is_null() {
        // End of the chain.  There is no is-null guard opcode, so pin the
        // slot against the null constant the way the exception `w_dict`
        // shadow guard does, then produce the None the getter returns.
        let null_const = ctx.trace_ctx.const_ref(0);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[raw_value, null_const],
        )?;
        ctx.trace_ctx.const_ref(pyre_object::w_none() as i64)
    } else {
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[raw_value])?;
        ctx.trace_ctx.set_opref_concrete(
            raw_value,
            majit_ir::Value::Ref(majit_ir::GcRef(stored as usize)),
        );
        raw_value
    };

    if field == TracebackWalkField::TbFrame {
        // `descr_get_tb_frame` also runs `frame.mark_as_escaped()`
        // (`pyframe.py:176 mark_as_escaped`): the reference it hands out has to
        // keep the frame materialised.  `set_escaped` ORs `FLAG_ESCAPED` into
        // the `flags` byte, so the trace reads that byte, sets the bit, and
        // stores it back.
        //
        // The bit has to be set BY THE TRACE, not only stamped now: the trace
        // is reused, and each replay walks a different traceback naming a
        // different frame, so a trace-time-only mark would leave every later
        // frame unmarked.  The concrete write below is the one the
        // authoritative walk's residual executor would have performed, applied
        // here for the same reason `try_walker_lower_exc_info_residual` applies
        // its own.
        let flags_descr = crate::descr::pyframe_flags_descr();
        let live_flags =
            crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, raw_value, flags_descr.clone());
        let escaped_bit = ctx.trace_ctx.const_int(i64::from(PyFrame::FLAG_ESCAPED));
        let new_flags = ctx
            .trace_ctx
            .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[raw_value, new_flags],
            flags_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(raw_value, flags_descr.index(), new_flags);
        unsafe { (*(stored as *mut PyFrame)).mark_as_escaped() };
    }

    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// `mapdict.py LOAD_ATTR_caching` full-body-walker fast path for a
/// plain (non-method) instance attribute.  When the concrete receiver is a
/// monomorphic instance whose attribute resolves to a boxed plain storage slot
/// or an unboxed integer/float slot, emit the guarded read PyPy compiles
/// LOAD_ATTR to under the JIT —
///   * `guard_class(obj, concrete_layout)` — the receiver keeps the exact
///     layout vtable whose mapdict-carrier prefix was proved at trace time (so
///     the `map`/`storage` reads below are valid; `mapdict.py` `if map is not
///     None:` also filters non-carriers at trace time).
///   * `guard_value(getfield_gc_i(w_type, version_tag), C_version_tag)` — pins
///     the class lookup result so a later descriptor or `__getattribute__`
///     mutation deopts on trace re-entry.
///   * `guard_value(getfield_gc_i(obj, map), C_map)` — `jit.promote(self.map)`
///     (`mapdict.py`); pins the exact instance shape so `find_map_attr`
///     const-folds `storageindex` to a green constant.
///   * boxed: `getfield_gc_r(obj, storage)` +
///     `getarrayitem_gc_r(block, C_index)` for
///     `mapdict.py _mapdict_read_storage`;
///   * unboxed int/float: a non-forcing typed read plus `wrapint`/`wrapfloat`,
///     matching `_prim_direct_read` (mapdict.py).
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
    use pyre_interpreter::pyframe::PyFrame;

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
    // `mapdict.py` resolution, returning the fold ingredients (the
    // read is left to the caller so it can be folded to a guarded inline read).
    if let Some((w_type, version_tag, map, storageindex)) = unsafe {
        pyre_interpreter::objspace::std::mapdict::load_attr_fast_path(concrete_obj, &name)
    } {
        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;

        // getfield_gc_r(obj, storage) + getarrayitem_gc_r(block, C_storageindex):
        // the inline value read (`mapdict.py`).  `storageindex` is a green
        // constant (the map guard pinned it); `trace_mapdict_storage_getitem`
        // stamps the dst's concrete shadow from the live block slot.
        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, unsafe {
            crate::descr::mapdict_storage_descr(concrete_obj)
        });
        let idx_const = ctx.trace_ctx.const_int(storageindex as i64);
        let value = crate::state::trace_mapdict_storage_getitem(ctx.trace_ctx, block, idx_const);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    if let Some(walk_field) = traceback_walk_field(concrete_obj, &name) {
        if let Some(()) = walker_specialize_traceback_walk_field(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            walk_field,
            dst,
            dst_bank,
        )? {
            return Ok(Some(()));
        }
    }

    // A user attribute on an exception instance. `mapdict.py:1483-1490
    // LOAD_ATTR_caching` declines this receiver upstream too — an exception is
    // not a `MapdictStorageMixin` and `_get_mapdict_map` answers None
    // (`baseobjspace.py:204-205`) — and it reaches its speed by inlining
    // `getdictvalue -> MapDictStrategy.getitem_str -> AbstractAttribute.read`
    // (`mapdict.py:55-66`, `:442-444`) instead. The attribute lives in the
    // `newdict(instance=True)` dictionary, two hops out: `w_dict` ->
    // `W_DictObject.dstorage` -> the fake carrier that holds the map.
    //
    // `w_exception_get_kind` and `w_exception_peek_dict` both cast straight to
    // `W_BaseException`, so the `is_exception` test is load-bearing: this
    // function runs for every `LOAD_ATTR` receiver that reached it.
    let exc_dict = unsafe {
        pyre_object::is_exception(concrete_obj)
            .then(|| pyre_object::interp_exceptions::w_exception_peek_dict(concrete_obj))
            .filter(|dict| !dict.is_null())
    };
    if let Some(dict) = exc_dict
        && let Some((w_type, _version_tag, carrier, map, storageindex, unboxed)) = unsafe {
            pyre_interpreter::objspace::std::mapdict::instance_dict_attr_fast_path(
                concrete_obj,
                dict,
                &name,
            )
        }
        // An unboxed float slot keeps the `f64` bit pattern in the same
        // longlong block as an int, so folding it needs a bits-to-float
        // reinterpret the trace has no operation for; leave it on the residual.
        && !matches!(
            unboxed,
            Some((pyre_interpreter::objspace::std::mapdict::UnboxType::Float, _))
        )
    {
        let kind = unsafe { pyre_object::w_exception_get_kind(concrete_obj) };
        let phys_type = unsafe { (*concrete_obj).ob_type as i64 };
        if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
            let type_const = ctx.trace_ctx.const_int(phys_type);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardClass,
                &[obj, type_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(obj, phys_type);
        }
        let w_class = walker_record_getfield_gc_r_uncached(ctx, obj, crate::descr::w_class_descr());
        let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[w_class, w_type_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(w_class, w_type_const);
        walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

        let dict_op = walker_record_getfield_gc_r_uncached(
            ctx,
            obj,
            crate::descr::w_exception_dict_descr(kind),
        );
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[dict_op])?;
        ctx.trace_ctx.set_opref_concrete(
            dict_op,
            majit_ir::Value::Ref(majit_ir::GcRef(dict as usize)),
        );

        // `instance_dict_attr_fast_path` declines a dictionary that is not
        // `MapDictStrategy`-backed, and the carrier read below is out of bounds
        // on a devolved one, so pin the strategy before dereferencing
        // `dstorage`.
        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            dict_op,
            crate::descr::dict_strategy_word_descr(),
        );
        let strategy_const = ctx.trace_ctx.const_int(
            &pyre_interpreter::objspace::std::mapdict::MAP_DICT_STRATEGY_REF as *const _ as i64,
        );
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[strategy, strategy_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy, strategy_const);

        let carrier_op =
            walker_record_getfield_gc_r_uncached(ctx, dict_op, crate::descr::dict_dstorage_descr());
        ctx.trace_ctx.set_opref_concrete(
            carrier_op,
            majit_ir::Value::Ref(majit_ir::GcRef(carrier as usize)),
        );
        let map_op = walker_record_getfield_gc_i_uncached(ctx, carrier_op, unsafe {
            crate::descr::mapdict_map_descr(carrier)
        });
        let map_const = ctx.trace_ctx.const_int(map as i64);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[map_op, map_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(map_op, map_const);

        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, carrier_op, unsafe {
            crate::descr::mapdict_storage_descr(carrier)
        });
        let index = ctx.trace_ctx.const_int(storageindex as i64);
        let slot = crate::state::trace_mapdict_storage_getitem(ctx.trace_ctx, block, index);
        let value = match unboxed {
            None => slot,
            // `_prim_direct_read` (mapdict.py:600-601): the storage slot holds
            // the shared longlong block, and the value is `items[listindex]`.
            // Keeping the boxing in the trace lets an immediate integer
            // consumer virtualize it away.
            Some((_, listindex)) => {
                let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
                let live = unsafe {
                    pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
                        carrier,
                        storageindex,
                        listindex,
                    )
                };
                let raw = crate::state::trace_int_block_getitem_value(
                    ctx.trace_ctx,
                    slot,
                    listindex_const,
                );
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Int(live));
                let boxed = walker_box_int(ctx, op_pc, raw, live)?;
                let live_ptr = pyre_object::w_int_new(live) as i64;
                ctx.trace_ctx
                    .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
                boxed
            }
        };
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
        // `descr_getargs` copies `args_w` through `newtuple`, which picks the
        // tuple representation from the arity and the element types.  Ask
        // `newtuple` itself which shape this read produces, rather than second-
        // guessing its dispatch, and settle it here — before any guard is
        // recorded, so a decline stays clean (a bail-out after the class pin
        // would leave the caller reading this attribute as already guarded).
        //
        // Only the shape crosses into the emit below; the probe tuple is
        // dropped rather than held, since nothing roots it across the guards.
        let args_specialised_oo = if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args
        {
            let ob_type = unsafe { (*args_tuple_shape_probe(stored)).ob_type };
            if std::ptr::eq(
                ob_type,
                &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
            ) {
                true
            } else if std::ptr::eq(ob_type, &pyre_object::TUPLE_TYPE) {
                false
            } else {
                // The unboxed arity-2 specialisations (`Cls_ii` / `Cls_ff`)
                // hold machine values in their inline fields, which this copy
                // has no unboxed operand for.  Only an Object-strategy `args_w`
                // holding exactly two plain ints or two plain floats gets here.
                return Ok(None);
            }
        } else {
            false
        };
        let traceback_frame = if slot
            == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback
        {
            let frame = unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(stored) };
            // `mark_traceback_escaped` leaves a torn-down traceback alone.
            // Decline before recording the receiver guards when the
            // authoritative read sees that shape; compiled replays guard
            // the frame load below and side-exit to the same residual path.
            if frame.is_null() {
                return Ok(None);
            }
            Some(frame)
        } else {
            None
        };
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
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback {
            // The fold replaces `descr_gettraceback`, whose read marks the
            // traceback's frame escaped so `ExecutionContext::leave` forces
            // its vref.  `descr_settraceback` admits only None or PyTraceback,
            // so the non-null slot is already type-safe without a class guard.
            // Read the node's frame, require the non-null case handled by the
            // traced path, and mirror `PyFrame.mark_as_escaped` directly.
            let frame_ref = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                raw_value,
                crate::descr::pytraceback_frame_descr(),
            );
            walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[frame_ref])?;
            let concrete_frame = traceback_frame.expect("traceback fold has no frame");
            ctx.trace_ctx.set_opref_concrete(
                frame_ref,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_frame as usize)),
            );
            let flags_descr = crate::descr::pyframe_flags_descr();
            let live_flags =
                crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, frame_ref, flags_descr.clone());
            let escaped_bit = ctx.trace_ctx.const_int(i64::from(PyFrame::FLAG_ESCAPED));
            let new_flags = ctx
                .trace_ctx
                .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
            ctx.trace_ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[frame_ref, new_flags],
                flags_descr.clone(),
            );
            ctx.trace_ctx
                .heapcache_setfield_cached(frame_ref, flags_descr.index(), new_flags);

            // The walk is the authoritative execution path, so mark its
            // concrete frame now as well as on every compiled re-execution.
            unsafe { pyre_interpreter::pytraceback::mark_traceback_escaped(stored) };
        }
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
            // Emit the representation `newtuple` picks, settled above.  Emitting
            // the array-backed shape for an arity the runtime specialises leaves
            // the trace disagreeing with its own record-time concrete, and the
            // `except <tuple>:` match fold — which dispatches on that concrete's
            // layout — then guards for a shape this trace never builds, so the
            // loop aborts instead of compiling.
            let tuple = if args_specialised_oo {
                crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, items[0], items[1])
            } else {
                crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &items)
            };
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

    // Module attribute (`math.sqrt`): the receiver is an exact module and the
    // name is present in its dict.  Fold the module-dict read to a
    // `QUASIIMMUT_FIELD(dict, slot)` version guard + elidable cell lookup —
    // celldict.py `_getdictvalue_no_unwrapping_pure` (`@jit.elidable_promote`) —
    // so a hot `math.sqrt(x)` loop drops its per-iteration LOAD_ATTR may-force
    // residual and the `math.sqrt` callable becomes a trace constant.  A rebind
    // of the attribute bumps the module dict `version` and fails the guard.
    // All resolution below is read-only; a missing / movable / non-canonical
    // shape falls through to the residual with no IR emitted.  An exact
    // `module` `w_class` excludes a module subclass with a custom
    // `__getattribute__`; a module-level PEP 562 `__getattr__` is irrelevant
    // because the name is present (the dict lookup wins before `__getattr__`).
    // A data descriptor on the module type (e.g. `__dict__`) outranks a
    // same-named dict entry in generic getattr, so decline when the name
    // resolves to one — the descriptor result, not the dict cell, is what a
    // read returns.
    if unsafe { pyre_object::is_module(concrete_obj) }
        && std::ptr::eq(
            unsafe { (*concrete_obj).w_class },
            pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::MODULE_TYPE),
        )
        && !unsafe {
            pyre_interpreter::baseobjspace::type_lookup_is_data_descr(
                (*concrete_obj).w_class,
                &name,
            )
        }
    {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(concrete_obj) };
        if !w_dict.is_null() && !majit_gc::can_move(majit_ir::GcRef(w_dict as usize)) {
            if let Some(slot) = crate::state::module_dict_cell_slot_direct(w_dict, &name) {
                if let Some(stored) = crate::state::module_dict_cell_value_direct(w_dict, slot) {
                    if !stored.is_null() && !majit_gc::can_move(majit_ir::GcRef(stored as usize)) {
                        // Pin the receiver to THIS module so the baked dict
                        // address is correct: a constant receiver is already
                        // pinned; a non-constant one gets a `guard_value`.
                        if !obj.is_constant() {
                            let expected = ctx.trace_ctx.const_ref(concrete_obj as i64);
                            ctx.trace_ctx
                                .record_guard(OpCode::GuardValue, &[obj, expected], 0);
                            walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
                            ctx.trace_ctx.heap_cache_mut().replace_box(obj, expected);
                        }
                        // `guard_frame_globals=false`: the receiver pin above
                        // (not a frame-globals-identity guard) proves the dict.
                        emit_namespace_cell_fold(
                            ctx, op_pc, dst, dst_bank, w_dict, slot, stored, false,
                        )?;
                        return Ok(Some(()));
                    }
                }
            }
        }
    }

    let Some((w_type, version_tag, map, storageindex, listindex, unbox_type)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::load_attr_unboxed_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    walker_guard_mapdict_instance_shape(ctx, op_pc, obj, concrete_obj, w_type, version_tag, map)?;
    let terminator = unsafe { (*map).terminator() };
    let term = unsafe { (*terminator).as_terminator() as *const _ };
    walker_pin_terminator_allow_unboxing(ctx, op_pc, term)?;

    // `_prim_direct_read` (mapdict.py): read the raw longlong from the
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
            // A trace-allocated receiver reads inline so the heap cache folds
            // the chain and the instance stays virtual (a residual Ref arg
            // would force it); an escaped receiver keeps the residual.
            let raw = if ctx.trace_ctx.heap_cache().is_unescaped(obj) {
                let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, unsafe {
                    crate::descr::mapdict_storage_descr(concrete_obj)
                });
                let slot = crate::state::trace_mapdict_storage_getitem(
                    ctx.trace_ctx,
                    block,
                    storageindex_const,
                );
                crate::state::trace_int_block_getitem_value(ctx.trace_ctx, slot, listindex_const)
            } else {
                crate::helpers::emit_trace_call_int_typed(
                    ctx.trace_ctx,
                    crate::helpers::jit_mapdict_unboxed_read_raw as *const (),
                    &[obj, storageindex_const, listindex_const],
                    &[
                        majit_ir::Type::Ref,
                        majit_ir::Type::Int,
                        majit_ir::Type::Int,
                    ],
                )
            };
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(live));
            let boxed = walker_box_int(ctx, op_pc, raw, live)?;
            // The `wrapint` op is a heap box, so its concrete must be a heap ptr too:
            // box the raw longlong through the same `w_int_new` the unboxed read uses
            // (mapdict.py `_box`); `box_int_concrete` re-homes a tagged small
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

/// The receiver-layout-specific half of the LOAD_METHOD fold: which field the
/// walker guards to keep "no instance attribute shadows this method" true for
/// the life of the trace.
///
/// `load_method_fast_path` proved the property once, at record time; this is
/// what re-proves it on every execution.  One variant per layout whose
/// non-allocating dictionary peek that predicate covers.
enum ShadowGuard {
    /// A `W_ObjectObject` receiver: pin the mapdict map, so adding
    /// `obj.<name>` grows the map chain and side-exits.
    InstanceMap(*const u8),
    /// A `W_BaseException` receiver: pin `w_dict` at null, so the lazy
    /// allocation `e.<name> = ...` performs side-exits.  Carries the kind
    /// because the field descrs are grouped per `ExcKind` vtable.
    ExceptionDictIsNull(pyre_object::interp_exceptions::ExcKind),
}

/// `callmethod.py LOAD_METHOD` method-cache fold for the
/// codewriter's method-form `LOAD_ATTR` residual.  The safety oracle is the
/// interpreter's `load_method_fast_path`: it declines custom
/// `__getattribute__`, uncacheable types, non-function descriptors, and
/// shadowing instance attributes.  On success the
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
    let Some((w_type, version_tag, w_descr)) =
        (unsafe { pyre_interpreter::load_method_fast_path(concrete_obj, &name) })
    else {
        return Ok(None);
    };
    if unsafe { resolve_inlinable_callee(w_descr) }.is_none() {
        return Ok(None);
    }
    // `space.type` reaches an exception's class through the kind registry when
    // the generic stub is still installed, and the `w_class` guard below can
    // only pin a class the slot actually holds.
    if !std::ptr::eq(unsafe { (*concrete_obj).w_class }, w_type) {
        return Ok(None);
    }
    let shadow = unsafe {
        if pyre_object::is_instance(concrete_obj) {
            let map = (*(concrete_obj as *const pyre_object::W_ObjectObject)).map;
            if map.is_null() {
                return Ok(None);
            }
            // A devolved instance holds its attributes in a dictionary and
            // keeps the same map across a later `e.<name> = ...`, so pinning
            // the map would not observe the shadow the assignment installs.
            // `W_ObjectObject.map` is stored untyped; the map layer owns the
            // node type.
            if pyre_interpreter::objspace::std::mapdict::map_is_devolved(map.cast()) {
                return Ok(None);
            }
            ShadowGuard::InstanceMap(map)
        } else if pyre_object::is_exception(concrete_obj) {
            ShadowGuard::ExceptionDictIsNull(pyre_object::w_exception_get_kind(concrete_obj))
        } else {
            // `load_method_fast_path` admits only the layouts above; keep the
            // two in step so a new layout there cannot reach an emit that has
            // no shadowing guard for it.
            return Ok(None);
        }
    };

    // guard_class(obj, ob_type): pins the payload layout, so the `w_class` and
    // shadowing-slot reads below name the fields they were recorded against.
    let physical_type = unsafe { (*concrete_obj).ob_type } as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(physical_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, physical_type);
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

    // typeobject.py `promote(self.version_tag())`: class mutation or method
    // reassignment bumps `_version_tag`, so the old `w_descr` side-exits.
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    // Re-prove the shadowing precondition: growing an instance attribute named
    // like the method must side-exit before the constant descriptor is reused.
    // mapdict.py LOAD_ATTR caching does this by pinning the map; an exception
    // has no map, and pins the still-unallocated `w_dict` slot instead.
    let (slot_op, slot_const) = match shadow {
        ShadowGuard::InstanceMap(map) => (
            walker_record_getfield_gc_i_uncached(ctx, obj, unsafe {
                crate::descr::mapdict_map_descr(concrete_obj)
            }),
            ctx.trace_ctx.const_int(map as i64),
        ),
        ShadowGuard::ExceptionDictIsNull(kind) => (
            walker_record_getfield_gc_r_uncached(
                ctx,
                obj,
                crate::descr::w_exception_dict_descr(kind),
            ),
            ctx.trace_ctx.const_ref(0),
        ),
    };
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[slot_op, slot_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(slot_op, slot_const);

    let method_const = ctx.trace_ctx.const_ref(w_descr as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, method_const)?;
    Ok(Some(()))
}

/// `LOAD_METHOD` classmethod fold for a type receiver (`Type.cmethod(...)`).
/// The safety oracle is [`pyre_interpreter::classmethod_on_type_fast_path`]: it
/// declines a custom metaclass, a metatype-defined name, an uncacheable type,
/// and any non-`classmethod` descriptor.  On success the walker pins the exact
/// type and its version tag, then writes the classmethod's `__func__` as a
/// green constant.  Because the method-load result is the plain `__func__` (not
/// a bound `Method`), the paired [`try_walker_fold_load_method_self`] runs
/// `compute_load_method_bound`, whose `is_type` + `is_classmethod` arm binds the
/// type as `cls`, and the following `CALL` inlines `__func__(cls, ...)` — the
/// instance-method shape with the class in the receiver slot.
///
/// Restricted to the top full-body frame for the reason
/// [`try_walker_specialize_load_bound_method_attr`] carries: a fold guard
/// inside an inlined callee sub-walk resumes at the caller's CALL, re-running
/// side effects.  The `getattr` residual resumes past the call, so declining
/// there re-runs nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_classmethod_attr<Sym: WalkSym>(
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
    if ctx.fbw_mode.inline_subwalk {
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
    let Some((w_type, version_tag, w_func)) =
        (unsafe { pyre_interpreter::classmethod_on_type_fast_path(concrete_obj, &name) })
    else {
        return Ok(None);
    };
    if unsafe { resolve_inlinable_callee(w_func) }.is_none() {
        return Ok(None);
    }

    // Pin the exact class.  The receiver IS the type, so a single GuardValue
    // anchors both the metaclass (exact `type`, via `is_type`) and the MRO the
    // classmethod lookup walks; the version tag below covers method reassignment.
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[obj, w_type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(obj, w_type_const);

    // typeobject.py `promote(self.version_tag())`: class mutation or classmethod
    // reassignment in the class or any base bumps `_version_tag`, so the pinned
    // `__func__` side-exits.
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    let func_const = ctx.trace_ctx.const_ref(w_func as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, func_const)?;
    Ok(Some(()))
}

/// `Cls.__name__` `LOAD_ATTR` fold, on the safety oracle
/// [`pyre_interpreter::baseobjspace::type_name_obj_fast_path`]: a class whose
/// metaclass is exactly `type` resolves `__name__` through the metatype data
/// descriptor `type.__name__`, whose getter returns the class's `w_name` slot.
/// The fold is that slot read, under the guards that prove the receiver is
/// such a class:
///
///   guard_class(obj, W_TypeObject layout)      — `is_type`
///   guard_value(getfield(obj, w_class), type)  — the metaclass is `type`
///   getfield(obj, w_name) + guard_nonnull      — what the getter returns
///
/// The class itself is NOT pinned, unlike the classmethod fold beside this
/// one: nothing here depends on which class arrived, so a loop reading
/// `cls.__name__` over several classes keeps one trace.  Nor is the version
/// tag pinned — it is not what a rename moves (`descr_set__name__` skips
/// `mutated()`), and the live slot read already reports one.
///
/// `w_name` is filled in on first read and never cleared, so the null the
/// guard covers is a class this trace has not served before; its side exit
/// runs the residual, which materialises the slot, and re-entry folds.
///
/// Attempted inside an inlined callee sub-walk as well, on the same terms as
/// [`try_walker_specialize_load_attr`]: the guards prove the read, so the fold
/// cannot raise where the residual would not, and `cls.__name__` in a method
/// body is precisely where the opaque residual costs the most — it is what
/// makes the enclosing `FOR_ITER` decline to inline the callee at all.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_type_name_attr<Sym: WalkSym>(
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
    if name != "__name__" {
        return Ok(None);
    }
    let Some((metatype, w_name)) =
        (unsafe { pyre_interpreter::baseobjspace::type_name_obj_fast_path(concrete_obj) })
    else {
        return Ok(None);
    };
    // The metaclass guard below reads the raw `w_class` slot, while the fast
    // path answers through `typedef::type`, which falls back to
    // `gettypefor(ob_type)` when that slot is null.  A receiver reached through
    // that fallback would be guarded against a value its field never holds, and
    // nothing writes the slot afterwards, so the guard would fail on every
    // execution forever — one bridge per `trace_eagerness` bucket, without ever
    // converging.  Fold only what the guard can discharge.
    if !std::ptr::eq(unsafe { (*concrete_obj).w_class }, metatype) {
        return Ok(None);
    }

    // guard_class(obj, ob_type): the `W_TypeObject` layout both field reads
    // below index into.  `is_type` is this check.
    let phys_type = unsafe { (*concrete_obj).ob_type } as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(phys_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, phys_type);
    }

    // The metaclass, pinned to `type`: only then is `type.__name__` the
    // descriptor `descr_getattribute` selects, and only then is it fixed —
    // `type` is immutable, a user metaclass is not.
    let w_class_op = walker_record_getfield_gc_r_uncached(ctx, obj, crate::descr::w_class_descr());
    let metatype_const = ctx.trace_ctx.const_ref(metatype as i64);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[w_class_op, metatype_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(w_class_op, metatype_const);

    let name_op =
        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, crate::descr::type_name_obj_descr());
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[name_op])?;
    ctx.trace_ctx.set_opref_concrete(
        name_op,
        majit_ir::Value::Ref(majit_ir::GcRef(w_name as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, name_op)?;
    Ok(Some(()))
}

/// Fold `LOAD_ATTR` on a type receiver when
/// [`pyre_interpreter::type_attr_value_fast_path`] proves that
/// `typeobject.py:811-828` returns the class-MRO value unchanged.  The exact
/// receiver and its version tag are pinned before the value is written as a
/// green constant.  [`pyre_interpreter::mutated`] recursively invalidates
/// subclasses, so the one receiver pin covers reassignment or deletion on any
/// base class as well.
///
/// A name the metatype answers with a data descriptor is refused by the oracle,
/// so `__name__` never reaches here — [`try_walker_specialize_load_type_name_attr`]
/// is its fold, and the two cover the disjoint arms of `descr_getattribute`.
///
/// The name needs no operand guard: the codewriter baked its `co_names` index
/// into the residual.  This read-only, present-attribute fold cannot raise, so
/// unlike the classmethod method-load fold it is safe inside an inlined callee
/// sub-walk; resuming past it cannot repeat a side effect.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_type_attr<Sym: WalkSym>(
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
    let Some((w_type, _version_tag, w_value)) = (unsafe {
        pyre_interpreter::type_attr_value_fast_path(concrete_obj, Wtf8::new(name.as_str()))
    }) else {
        return Ok(None);
    };

    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[obj, w_type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(obj, w_type_const);
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    let value_const = ctx.trace_ctx.const_ref(w_value as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, 'r', value_const)?;
    Ok(Some(()))
}

/// Fold the `LOAD_ATTR`-method `getattr` residual for a receiver whose name
/// resolves to a plain builtin-code function on its type — the `lst.append`
/// shape [`try_walker_specialize_load_method_attr`] declines because upstream
/// restricts its `[w_descr, w_obj]` push to `flag_method_descriptor` types.
///
/// Both engines materialise a `Method` here (`space.getattr`), but PyPy traces
/// *through* that `getattr` — it is ordinary RPython — so the type lookup folds
/// to a constant under `guard_class` + the version pin and the `Method` itself
/// virtualizes away: its optimized LOAD_METHOD emits no ops at all in a
/// steady-state loop (`pypy/objspace/std/callmethod.py:25-80`). pyre's `getattr`
/// is an opaque `CALL_MAY_FORCE` residual, which additionally drags a
/// `GUARD_NOT_FORCED` (forcing the virtualizable frame) and a `GUARD_NO_EXCEPTION`
/// through every iteration. This fold reproduces PyPy's shape directly:
///
///   guard_class(obj, ob_type)
///   guard_value(getfield(obj, w_class), the type)
///   guard_value(getfield(the type, version_tag), the tag)
///   new_with_vtable(Method) + setfield(w_function/w_self/w_class/header)
///
/// The guards make `lookup_in_type` constant exactly as the version-tag promote
/// does upstream, and the emitted `Method` is dead once the following `CALL`
/// folds — the append fold reads `w_function` / `w_self` straight back off it.
///
/// Returns `None` (fall through to the residual, SAFE) for every shape
/// [`pyre_interpreter::baseobjspace::bound_method_attr_fast_path`] declines.
///
/// Restricted to the top full-body frame for the reason
/// [`try_walker_orthodox_list_append`] documents: inside an inlined callee
/// sub-walk a fold's guards collapse their resume to the caller's CALL
/// boundary, so a guard failure re-runs the callee from its entry and doubles
/// any side effect it sequenced before this `LOAD_ATTR`. The residual resumes
/// past the call instead, so declining here re-runs nothing extra.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_bound_method_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_descr)) = (unsafe {
        pyre_interpreter::baseobjspace::bound_method_attr_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };

    // guard_class(obj, ob_type): the physical layout the `w_class` read below
    // needs, and what pins the receiver's builtin kind.
    let phys_type = unsafe { (*concrete_obj).ob_type } as i64;
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(phys_type);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(obj, phys_type);
    }

    // Pin the Python-level class exactly: a subclass reaching the same
    // physical layout can define its own `name` and must side-exit.
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

    // typeobject.py `promote(self.version_tag())`: reassigning the method on
    // the type bumps the tag, so the constant `w_descr` side-exits.
    walker_pin_type_version_tag(ctx, op_pc, w_type_const)?;

    // `w_method_new(w_descr, obj, w_type)` + the header stamp its allocation
    // performs (`ob_type` comes from the NewWithVtable's size descr).
    let func_const = ctx.trace_ctx.const_ref(w_descr as i64);
    let header_w_class = ctx
        .trace_ctx
        .const_ref(pyre_object::get_instantiate(&pyre_object::function::METHOD_TYPE) as i64);
    let method_op = crate::helpers::emit_bound_method_inline(
        ctx.trace_ctx,
        func_const,
        obj,
        w_type_const,
        header_w_class,
    );
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(method_op, method_type_addr);
    // The concrete bound method the walker's own execution must observe; a
    // fresh `Method` per evaluation is what `getattr` produces anyway, so the
    // trace allocating its own is not an identity divergence.
    let bound = pyre_object::w_method_new(w_descr, concrete_obj, w_type);
    ctx.trace_ctx.set_opref_concrete(
        method_op,
        majit_ir::Value::Ref(majit_ir::GcRef(bound as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, method_op)?;
    Ok(Some(()))
}

/// Fold `bh_load_method_self_fn(obj, attr, code, name_idx)` once both the
/// receiver and the attribute are concrete.  The method-attribute fold above
/// already guards class, type version, and instance map; this second residual
/// is only the pure `compute_load_method_bound` binding decision.  A plain
/// instance-method bind writes the original red receiver box, not a baked
/// `ConstRef`, matching `callmethod.py f.pushvalue(w_obj)`.
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
    // `compute_load_method_bound` answers PY_NULL for an already-bound method
    // without inspecting anything else, so pinning the attribute's class is
    // the whole precondition.  Left as a residual this is a second per-iteration
    // call on top of the `getattr` one (`lst.append(x)` pays both).
    if unsafe { pyre_object::is_method(concrete_attr) } {
        let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
        let class_pinned = attr.is_constant() || ctx.trace_ctx.heap_cache().is_class_known(attr);
        if !class_pinned {
            // A guard here would resume at the caller's CALL inside an inlined
            // callee sub-walk, re-running whatever that callee already did;
            // leave those to the residual (which resumes past the call).
            if ctx.fbw_mode.inline_subwalk {
                return Ok(None);
            }
            let type_const = ctx.trace_ctx.const_int(method_type_addr);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op_pc,
                OpCode::GuardClass,
                &[attr, type_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .class_now_known(attr, method_type_addr);
        }
        let null_const = ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, null_const)?;
        return Ok(Some(()));
    }
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

/// How the add-transition fold pins the stored value's type, resolved before
/// any guard is emitted so a decline falls cleanly to the residual.
enum StoreAttrAddValuePin {
    Boxed,
    /// Fresh unboxed int slot; `Some` pins a heap operand's canonical
    /// `w_class`, `None` means tagged (the unbox's tag guard is the pin).
    UnboxedInt(Option<pyre_object::PyObjectRef>),
}

/// `None` (unpinnable `w_class`, or a float pick) keeps the residual.
fn store_attr_add_value_pin(
    add: &pyre_interpreter::objspace::std::mapdict::StoreAttrAdd,
    concrete_value: pyre_object::PyObjectRef,
) -> Option<StoreAttrAddValuePin> {
    match add.unbox_type {
        None => Some(StoreAttrAddValuePin::Boxed),
        Some(pyre_interpreter::objspace::std::mapdict::UnboxType::Int) => {
            if pyre_object::tagged_int::CAN_BE_TAGGED
                && unsafe { pyre_object::tagged_int::is_tagged_int(concrete_value) }
            {
                return Some(StoreAttrAddValuePin::UnboxedInt(None));
            }
            unsafe { walker_exact_builtin_class(concrete_value) }
                .map(|canonical| StoreAttrAddValuePin::UnboxedInt(Some(canonical)))
        }
        // `store_attr_add_fast_path` never resolves a float pick; defensive.
        Some(pyre_interpreter::objspace::std::mapdict::UnboxType::Float) => None,
    }
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
    if !ctx.is_authoritative_executor || w_code_ptr == 0 {
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
    if let Some((w_type, version_tag, map, storageindex, listindex, unbox_type, attr)) = unsafe {
        pyre_interpreter::objspace::std::mapdict::store_attr_unboxed_fast_path(concrete_obj, &name)
    } {
        match unbox_type {
            pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                // `type(w_value) is space.IntObjectCls` (mapdict.py): reject bool
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
                // unboxing (mapdict.py), so retain setattr.
                if !unsafe { pyre_object::pyobject::is_float(concrete_value) } {
                    return Ok(None);
                }
            }
        }

        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;
        unsafe { pyre_interpreter::objspace::std::mapdict::mark_attr_ever_mutated(attr) };
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

        // `original_effect` is the generic setattr residual's
        // `EF_RANDOM_EFFECTS`, whose six raw sets and six bitstrings are
        // all `None` (`effectinfo.py:149-155`). Cloning it and lowering
        // only `extraeffect` would build the "non-random + raw=None"
        // shape `effectinfo.py:149-162` asserts against, and
        // `check_write_descr_field` would then unwrap a `None`
        // bitstring. Build the downgraded effect from scratch so the
        // sets match the extraeffect: the raw slot write cannot raise,
        // and it touches no field or array descr the heap cache holds
        // (the storage is reached only through this helper pair, never
        // through `getfield_gc` / `getarrayitem_gc`).
        let mut effect = majit_ir::EffectInfo::const_new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        );
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
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Cause
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Name
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::AttrObj
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeObject
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeStart
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeEnd
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeReason
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeEncoding => {
                    // `exception_attr_slot_fold` declines these for stores, so
                    // the store fold never reaches here.
                    unreachable!("load-only exception slots fold on load only")
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

    // The attribute is not in the map yet: fold the `map -> PlainAttribute`
    // transition and the grow-by-one storage rewrite into trace ops instead of
    // leaving the generic `setattr` residual, which would force the receiver.
    //
    // Only for a receiver this trace allocated and has not let escape.  The
    // emitted transition is a pair of raw field stores, so unlike the
    // interpreter's it does not hold the striped `instance_lock`, and unlike
    // the single-slot in-place write it publishes two fields: a concurrent
    // mutator of the same instance could pair one thread's `map` with
    // another's `storage`.  `is_unescaped` is what rules that out — no other
    // thread has a reference yet.  It costs almost nothing, because the fold's
    // payoff is exactly the unescaped case: an escaped receiver is one the
    // optimizer cannot remove anyway.
    if ctx.trace_ctx.heap_cache().is_unescaped(obj)
        && let Some(add) = unsafe {
            pyre_interpreter::objspace::std::mapdict::store_attr_add_fast_path(
                concrete_obj,
                &name,
                concrete_value,
            )
        }
        && let Some(value_pin) = store_attr_add_value_pin(&add, concrete_value)
    {
        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            add.w_type,
            add.version_tag,
            add.map,
        )?;
        walker_pin_holder_typ(ctx, op_pc, add.holder)?;
        // This marker is not redundant with the instance-map GuardValue. The
        // fold bakes `add.new_map`, the transition target, as a green
        // `const_int`, while the guard pins `add.map`, the instance map before
        // the transition. `holder_pick_attr` can replace `holder.attr` with a
        // fresh `PlainAttribute` without changing any instance's current map;
        // this marker protects the baked target.
        walker_pin_holder_attr(ctx, op_pc, add.holder)?;
        // General rule: plant the `allow_unboxing` marker only where the fold
        // read the flag as true. Marking a false read re-arms a permanently
        // dead field and lets later writes invalidate without bound.
        if add.picked_unbox.is_some() {
            let terminator = unsafe { (*add.map).terminator() };
            let term = unsafe { (*terminator).as_terminator() as *const _ };
            walker_pin_terminator_allow_unboxing(ctx, op_pc, term)?;
        }
        let new_map_const = ctx.trace_ctx.const_int(add.new_map as i64);
        match value_pin {
            StoreAttrAddValuePin::Boxed => {
                crate::helpers::emit_mapdict_add_attr_inline(
                    ctx.trace_ctx,
                    obj,
                    add.storageindex,
                    new_map_const,
                    value,
                );
            }
            StoreAttrAddValuePin::UnboxedInt(canonical) => {
                // Only an exactly-`int` runtime value may unbox: the tag
                // guard pins a tagged operand, the `w_class` pin a heap one
                // (a subclass shares `W_IntObject`'s `ob_type`).
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                let raw = walker_unbox_int(ctx, op_pc, value, int_type_addr)?;
                if let Some(canonical) = canonical {
                    walker_guard_exact_w_class(ctx, op_pc, value, canonical)?;
                }
                crate::helpers::emit_mapdict_add_unboxed_attr_inline(
                    ctx.trace_ctx,
                    obj,
                    add.storageindex,
                    new_map_const,
                    raw,
                );
            }
        }
        // The walk is the authoritative execution path, so apply the resolved
        // transition now; the emitted operations reproduce it in compiled code.
        unsafe {
            pyre_interpreter::objspace::std::mapdict::store_attr_add_commit(
                concrete_obj,
                &add,
                concrete_value,
            )
        };
        return Ok(Some(WalkerStoreAttrSpecialization::Direct));
    }

    let Some((w_type, version_tag, map, storageindex, attr)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::store_attr_boxed_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    walker_guard_mapdict_instance_shape(ctx, op_pc, obj, concrete_obj, w_type, version_tag, map)?;
    unsafe { pyre_interpreter::objspace::std::mapdict::mark_attr_ever_mutated(attr) };
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
/// the virtualizable `opimpl_newlist` shape (`pyjitpl.py`) —
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
/// tagged immediate (which has no `&INT_TYPE`/`&LONG_TYPE` header for the
/// unbox guard). A fits-in-word `W_LongObject` is accepted: `is_plain_int1`
/// covers it and `walker_unbox_long` supplies the `&LONG_TYPE` + `_fits_int`
/// guarded extraction.
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
        // Per element: `(unboxed i64, is_fits_long)`.  `is_fits_long` selects
        // `walker_unbox_long` (`&LONG_TYPE` + `_fits_int` guard) over the
        // plain `walker_unbox_int`.
        Int(Vec<(i64, bool)>),
        Float(Vec<f64>),
        Object,
    }
    let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
    let emit = match strategy {
        ListStrategy::Integer => {
            // `IntegerListStrategy.is_correct_type` is `is_plain_int1`, which
            // accepts an exact `W_IntObject` or a fits-in-word `W_LongObject`;
            // both store the unboxed i64 (`plain_int_w`). A tagged immediate
            // has no header for the unbox guard, so decline it to the residual
            // (correct for any element).
            let mut vals = Vec::with_capacity(len);
            for &p in &concretes {
                if pyre_object::tagged_int::CAN_BE_TAGGED
                    && pyre_object::tagged_int::is_tagged_int(p)
                {
                    return Ok(None);
                }
                if !unsafe { pyre_object::is_plain_int1(p) } {
                    return Ok(None);
                }
                let is_fits_long = unsafe { pyre_object::pyobject::is_long(p) };
                let val = if is_fits_long {
                    pyre_object::longobject::jit_w_long_toint(p as usize as i64)
                } else {
                    unsafe { pyre_object::w_int_get_value(p) }
                };
                vals.push((val, is_fits_long));
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
        // The interpreter stores this as encoded signed-longlong values.
        // The walker does not yet have an encoded numeric payload variant;
        // leave construction to the ordinary residual instead of emitting an
        // Integer array whose values would have the wrong representation.
        ListStrategy::IntOrFloat => return Ok(None),
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
            let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
            let mut raws: Vec<OpRef> = Vec::with_capacity(len);
            for (&it, &(v, is_fits_long)) in items.iter().zip(vals.iter()) {
                let raw = if is_fits_long {
                    walker_unbox_long(ctx, op_pc, it, long_type_addr)?
                } else {
                    walker_unbox_int(ctx, op_pc, it, int_type_addr)?
                };
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

/// FBW virtualization of the array-backed BUILD_TUPLE — the arities
/// `makespecialisedtuple2` does not claim.  Sibling of
/// [`try_walker_specialize_newtuple`] (arity-2 plain-int `spec_ii`) and
/// [`try_walker_specialize_newlist`], reached only after the `spec_ii` fold
/// declines, so that path stays byte-identical.
///
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_TUPLE to `new_array_clear` +
/// per-index `setarrayitem_gc` + a `newtuple_from_array` residual.  Re-emit the
/// canonical `W_TupleObject` shape walker-native (`new_with_vtable` +
/// `w_class` / `wrappeditems` `setfield_gc` over a fresh items block), reading
/// the elements straight out of the array heap-cache so the array build keeps
/// no consumer and DCEs.  A tuple that never escapes then folds away entirely,
/// and one that does escape materializes from the same fields the residual
/// would have written.
///
/// Arity 2 is `makespecialisedtuple2` territory (`Cls_ii` / `Cls_ff` /
/// `Cls_oo`, `specialisedtupleobject.py`): the runtime never builds an
/// array-backed tuple there, so emitting one would diverge from what the
/// blackhole rebuilds on deopt.  Declined here — the `spec_ii` fold owns the
/// int-int case and the residual owns the rest.  The empty tuple is declined
/// too (no element to recover a length from).
///
/// Lifting that decline is not a trace-local question: the trace stays
/// self-consistent, but a side exit hands a real pair — inline `value0` /
/// `value1`, no `wrappeditems` block — to whatever consumer the trace picked
/// for the canonical layout, and
/// [`try_walker_specialize_subscr_specialised_pair`] then reads a field that is
/// not there.
///
/// Returns `Ok(Some(()))` when folded; `Ok(None)` falls through to the opaque
/// residual, which stays correct for any shape — a non-const array length or
/// an element without a concrete Ref shadow is not declined, just not folded.
pub(crate) fn try_walker_specialize_newtuple_object<Sym: WalkSym>(
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
    // `heapcache.arraylen`; a cleared array has every slot set, so read the
    // length directly rather than probing getarrayitem until a miss).
    let len = {
        let Some(len_op) = ctx.trace_ctx.heap_cache().arraylen(arr) else {
            return Ok(None);
        };
        match len_op.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) if n >= 1 => n as usize,
            _ => return Ok(None),
        }
    };
    if len == 2 {
        return Ok(None);
    }

    // Element boxes the BUILD_TUPLE `setarrayitem_gc` ops stored; a cache miss
    // (clobbered array / non-const index) bails to the opaque residual.
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
    let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for &it in &items {
        let Some(obj) = walker_concrete_ref_object(ctx, it) else {
            return Ok(None);
        };
        concretes.push(obj);
    }

    // Concrete shadow: a fresh array-backed tuple from the element shadows
    // (`w_tuple_new` parity for every arity but 2). A new allocation with no
    // heap mutation, safe during the walk like `wrapint`.  Built before the
    // emit so a failure leaves no orphan ops in the trace.
    let result_concrete = pyre_object::w_tuple_new_array_backed(concretes);
    if result_concrete.is_null() {
        return Ok(None);
    }

    let tuple_op = crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &items);
    ctx.trace_ctx.set_opref_concrete(
        tuple_op,
        majit_ir::Value::Ref(majit_ir::GcRef(result_concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, tuple_op)?;
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
/// for any other shape (object tuple, arity ≠ 2, out-of-range long, tagged
/// immediate, cache miss) — so a non-foldable build is not declined.
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
    // The arity-2 int specialised tuple `Cls_ii` (`makespecialisedtuple2`,
    // specialisedtupleobject.py) is built when both elements pass
    // `is_plain_int1` — an exact `W_IntObject` or a fits-in-word
    // `W_LongObject`; the stored payload is `plain_int_w` of each.  A tagged
    // immediate has no real header for the unbox guard and the emit is not
    // tag-aware, so decline it to the residual (correct for any shape).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(c0)
            || pyre_object::tagged_int::is_tagged_int(c1))
    {
        return Ok(None);
    }
    let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
    let both_plain_int =
        unsafe { pyre_object::is_plain_int1(c0) && pyre_object::is_plain_int1(c1) };
    if !both_plain_int {
        return Ok(None);
    }
    let c0_long = unsafe { pyre_object::pyobject::is_long(c0) };
    let c1_long = unsafe { pyre_object::pyobject::is_long(c1) };
    // Concrete element int payloads (`plain_int_w`: `W_IntObject`'s `intval`
    // or a fits-int `W_LongObject`'s `toint()`).
    let v0 = if c0_long {
        pyre_object::longobject::jit_w_long_toint(c0 as usize as i64)
    } else {
        unsafe { pyre_object::w_int_get_value(c0) }
    };
    let v1 = if c1_long {
        pyre_object::longobject::jit_w_long_toint(c1 as usize as i64)
    } else {
        unsafe { pyre_object::w_int_get_value(c1) }
    };

    // --- emit the virtual spec_ii walker-native ---
    // Paired `w_class` guard per element so a runtime int subclass sharing
    // the public `int` `w_class` side-exits, then the plain-int payload unbox.
    // A fits-int `W_LongObject` also carries the public `int` `w_class`
    // (`is_plain_int1`), so the same guard covers it; the payload extraction
    // switches to `walker_unbox_long` (`&LONG_TYPE` + `_fits_int`).
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    walker_guard_exact_w_class(ctx, op_pc, e0, int_typeobj)?;
    walker_guard_exact_w_class(ctx, op_pc, e1, int_typeobj)?;
    let int_type_addr = int_ty as i64;
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    let raw0 = if c0_long {
        walker_unbox_long(ctx, op_pc, e0, long_type_addr)?
    } else {
        walker_unbox_int_typed(
            ctx,
            op_pc,
            e0,
            int_type_addr,
            crate::descr::int_intval_descr(),
        )?
    };
    let raw1 = if c1_long {
        walker_unbox_long(ctx, op_pc, e1, long_type_addr)?
    } else {
        walker_unbox_int_typed(
            ctx,
            op_pc,
            e1,
            int_type_addr,
            crate::descr::int_intval_descr(),
        )?
    };

    let tuple = walker_emit_specialised_tuple_ii(ctx, op_pc, raw0, raw1, v0, v1)?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, tuple)?;
    Ok(Some(()))
}

/// Emit the virtual `Cls_ii` arity-2 int tuple (`makespecialisedtuple2`,
/// specialisedtupleobject.py) walker-native: `new_with_vtable` + the `w_class`
/// / `value0` / `value1` `setfield_gc`s over the two raw int payloads.
///
/// `v0` / `v1` are the recorded payloads; they build the concrete shadow the
/// partner unpack fold reads back (`walker_concrete_ref_object` +
/// `unpack_item_fn`).  The shadow is constructed LAST so the construct→root
/// window holds no intervening runtime allocation: stamping `tuple`'s concrete
/// roots the fresh spec_ii via the trace's concrete-shadow set. A concrete
/// allocation failure aborts the whole walk because the virtual allocation
/// and stores have already been recorded.
fn walker_emit_specialised_tuple_ii<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw0: OpRef,
    raw1: OpRef,
    v0: i64,
    v1: i64,
) -> Result<OpRef, DispatchError> {
    let tuple = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::specialised_tuple_ii_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(tuple);
    crate::helpers::emit_tuple_hash_sentinel(
        ctx.trace_ctx,
        tuple,
        crate::descr::specialised_tuple_ii_hash_descr(),
    );
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
    for (raw, descr) in [
        (raw0, crate::descr::specialised_tuple_ii_value0_descr()),
        (raw1, crate::descr::specialised_tuple_ii_value1_descr()),
    ] {
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[tuple, raw], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(tuple, descr_index, raw);
    }
    let tuple_ptr = pyre_object::specialisedtupleobject::w_specialised_tuple_ii_new(v0, v1);
    if tuple_ptr.is_null() {
        return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op_pc });
    }
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(tuple_ptr as usize)),
    );
    Ok(tuple)
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
    // the optimizer cannot DCE (pure.py demotes CALL_PURE→CALL and
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
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, folded != 0, dst as u8, dst_bank)? {
        // The guarded arm already pinned the truth to a constant and filed
        // `bool_box_truth_record` against it, so the following `is_true`
        // folds without re-reading the runtime truth.
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            // #62: remember boxed→truth so an immediately-following `is_true` residual
            // (POP_JUMP_IF_*) folds back to the raw Int instead of may-force-unboxing.
            bool_box_truth_record(boxed, truth);
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// B3: walker-native fold of the CHECK_EXC_MATCH
/// residual (`bh_compare_fn(exc, match_type, op_tag=10)`,
/// `call_jit.rs`). Computes the match concretely from
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
/// Pin the elements of a tuple `except` clause's match target, returning
/// whether the target was a tuple layout this could read through.
///
/// `w_tuple_new` picks the layout by arity: two object elements become a
/// `W_SpecialisedTupleObject_oo` holding them in inline immutable `value0` /
/// `value1` slots, everything else an array-backed `W_TupleObject` behind
/// `wrappeditems`. Both are read here with the same guarded-load shape the
/// other tuple folds use, and each element is pinned to the class seen while
/// tracing. The `_oo` loads are pure (immutable fields), so a tuple built from
/// constants leaves nothing behind after optimization.
///
/// A match target that is neither layout — a bare class, the int/float
/// specialisations, or a tuple subclass whose `w_class` diverges — returns
/// `false` so the caller pins the object identity instead. That is correct for
/// a target that is loaded rather than built, which is the only case where the
/// identity can hold across iterations.
fn walker_guard_exc_match_tuple_items<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    match_op: OpRef,
    match_type: pyre_object::PyObjectRef,
) -> Result<bool, DispatchError> {
    let ob_type = unsafe { (*(match_type as *const pyre_object::pyobject::PyObject)).ob_type };
    let spec_oo = &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE
        as *const pyre_object::pyobject::PyType;
    let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::pyobject::PyType;

    // Either layout may carry a subclass `w_class`, and the element reads below
    // are only the whole target when it is a plain tuple.
    let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    if !std::ptr::eq(
        unsafe { (*(match_type as *const pyre_object::pyobject::PyObject)).w_class },
        canonical_tuple_class,
    ) {
        return Ok(false);
    }

    let mut items: Vec<(OpRef, pyre_object::PyObjectRef)> = Vec::new();
    if std::ptr::eq(ob_type, spec_oo) {
        walker_guard_exc_match_tuple_class(ctx, op_pc, match_op, spec_oo as i64)?;
        for index in 0..2usize {
            let descr = if index == 0 {
                crate::descr::specialised_tuple_oo_value0_descr()
            } else {
                crate::descr::specialised_tuple_oo_value1_descr()
            };
            let item = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, match_op, descr);
            let concrete = unsafe {
                pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(
                    match_type, index,
                )
            };
            items.push((item, concrete));
        }
    } else if std::ptr::eq(ob_type, tuple_type) {
        // Read every element before recording anything: a bail-out after a
        // guard has been emitted would leave the target's class pinned, and the
        // caller reads that as "already guarded" and drops its own pin.
        let len = unsafe { pyre_object::w_tuple_len(match_type) };
        let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
        for index in 0..len {
            let Some(concrete) =
                (unsafe { pyre_object::w_tuple_getitem(match_type, index as i64) })
            else {
                return Ok(false);
            };
            concretes.push(concrete);
        }
        walker_guard_exc_match_tuple_class(ctx, op_pc, match_op, tuple_type as i64)?;
        walker_guard_exact_w_class(ctx, op_pc, match_op, canonical_tuple_class)?;
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            match_op,
            crate::descr::tuple_wrappeditems_descr(),
        );
        let length = crate::state::opimpl_arraylen_gc(
            ctx.trace_ctx,
            block,
            crate::state::pyobject_gcarray_descr(),
        );
        let len_const = ctx.trace_ctx.const_int(len as i64);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[length, len_const])?;
        for (index, concrete) in concretes.into_iter().enumerate() {
            let index_op = ctx.trace_ctx.const_int(index as i64);
            let item =
                crate::state::trace_items_block_getitem_value(ctx.trace_ctx, block, index_op);
            items.push((item, concrete));
        }
    } else {
        return Ok(false);
    }

    for (item, concrete) in items {
        if item.is_constant() {
            continue;
        }
        let expected = ctx.trace_ctx.const_ref(concrete as i64);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[item, expected])?;
        ctx.trace_ctx.heap_cache_mut().replace_box(item, expected);
    }
    Ok(true)
}

/// `GuardClass` on a match target's layout, skipped when the class is already
/// pinned.
fn walker_guard_exc_match_tuple_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    match_op: OpRef,
    type_addr: i64,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.heap_cache().is_class_known(match_op) {
        return Ok(());
    }
    let type_const = ctx.trace_ctx.const_int(type_addr);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[match_op, type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(match_op, type_addr);
    Ok(())
}

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
    // Pin `match_type` so a runtime divergence (a reassigned handler global)
    // side-exits rather than running the wrong handler.
    //
    // A tuple clause is pinned through its ELEMENTS. `except (A, B):` lowers to
    // `BUILD_TUPLE`, which allocates a fresh tuple on every visit, so an
    // identity guard on the container is unsatisfiable — it fails once per
    // visit and the loop collapses into side exits and bridge churn. The
    // elements are what the match actually reads and what a rebinding would
    // change, so guarding them is both sound and stable across the
    // re-allocation.
    if !match_op.is_constant()
        && !walker_guard_exc_match_tuple_items(ctx, op_pc, match_op, match_type)?
        && !ctx.trace_ctx.heap_cache().is_class_known(match_op)
    {
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

/// Trace `space.newbool` as its directional truth guard and prebuilt result.
/// `baseobjspace.py:896-900` chooses `w_True` or `w_False`, the prebuilt
/// singletons from `boolobject.py:79-80`; `pyjitpl.py:511-534` records the
/// matching `GUARD_TRUE` / `GUARD_FALSE`, and `pyjitpl.py:525-526` replaces
/// the truth box with the promoted constant.
///
/// Restricted to the shape [`classify_compare_box_use`] recognizes — the box
/// decides one branch and nothing else.  There the guard is the branch's own
/// `goto_if_not` guard, so it adds no guard the trace would not already carry.
/// A bool that escapes (kept on the stack for a short-circuit, stored to a
/// local) keeps the residual box: guarding it would pin a value the trace
/// otherwise carries unconstrained, and every later re-entry with the other
/// truth bails.
///
/// The shape survives the `push_and_bump!` publish
/// ([`VablePublish::Tolerated`]).  That store mirrors the box into the
/// operand-stack slot the guard's own resume image describes; swapping a
/// prebuilt singleton in for the recorded call result leaves it storing the
/// same value, so it is not a second use that makes the bool escape.
fn walker_newbool_guarded<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    truth: OpRef,
    observed: bool,
    dst_reg: u8,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if ctx.fbw_mode.snapshot_sym.is_null() || dst_bank != 'r' {
        return Ok(None);
    }
    if !matches!(
        classify_compare_box_use(ctx, op_pc, dst_reg, VablePublish::Tolerated),
        CompareBoxUse::FeedsBranchOnly { .. }
    ) {
        return Ok(None);
    }
    let guard = if observed {
        OpCode::GuardTrue
    } else {
        OpCode::GuardFalse
    };
    ctx.trace_ctx.record_guard(guard, &[truth], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    let promoted = ctx.trace_ctx.const_int(observed as i64);
    let result_obj = pyre_object::w_bool_from(observed);
    let const_bool = ctx.trace_ctx.const_ref(result_obj as i64);
    ctx.trace_ctx.set_opref_concrete(
        const_bool,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    bool_box_truth_record(const_bool, promoted);
    Ok(Some(const_bool))
}

/// Does `tp` name a layout whose class overrides `is_w` with a value
/// comparison?  `baseobjspace::is_w` gates one branch per overriding class,
/// each demanding both operands be that exact type: `int`
/// (`intobject.py:44`), `float` (`floatobject.py:196`), `complex`
/// (`complexobject.py:287`), `tuple` (`tupleobject.py:47`), `bytes`
/// (`bytesobject.py:25`), `str` (`unicodeobject.py:101`) and `frozenset`
/// (`setobject.py:592`).  Every other class keeps the default pointer
/// identity (`baseobjspace.py:246`).
fn is_w_compares_by_value(tp: *const pyre_object::pyobject::PyType) -> bool {
    [
        &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::pyobject::FLOAT_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::pyobject::COMPLEX_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::pyobject::TUPLE_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::bytesobject::BYTES_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::pyobject::STR_TYPE as *const pyre_object::pyobject::PyType,
        &pyre_object::setobject::FROZENSET_TYPE as *const pyre_object::pyobject::PyType,
    ]
    .iter()
    .any(|special| std::ptr::eq(*special, tp))
}

/// Walker-native fold of the `IS_OP` residual — `bh_compare_fn(lhs, rhs,
/// tag)` with tag 8 (`is`) or 9 (`is_not`), the tags
/// `compare_op_tag_for_opname` assigns those two opnames.
///
/// `IS_OP` is `space.is_w(w_1, w_2)` plus a `newbool`
/// (`pyopcode.py:1078-1092`), and `is_w` (`baseobjspace.py:833`) dispatches
/// to `w_two.is_w(space, w_one)`, whose default is pointer identity.  Two
/// tiers, mirroring `FASTPATHS_SAME_BOXES`' `ptr_eq`/`ptr_ne` entries
/// (`pyjitpl.py:326-336`):
///
///   * Same box — `baseobjspace::is_w` answers at its opening `ptr::eq`
///     whatever the class, so the result is the constant `True`/`False`.
///     No op and no guard: this is the `b1 is b2` fast check itself.
///   * Distinct boxes whose layouts both keep the default `is_w` — no
///     value-comparison branch can fire, so `is_w` again reduces to that
///     `ptr::eq`.  A `GuardClass` per operand pins the layout, then
///     `ptr_eq`/`ptr_ne` replaces the may-force `compare_fn` and the
///     `GuardNotForced` behind it.
///
/// Declining on a value-comparing layout is what keeps the second tier
/// sound: `GuardClass` pins `ob_type`, and an `int` subclass instance
/// shares `INT_TYPE` with a plain `int` while answering the exact-type gate
/// differently, so the layout alone cannot separate them.  A tagged
/// immediate is declined outright — it carries no `ob_type` to guard.
pub(crate) fn try_walker_fold_is_op<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 || dst_bank != 'r' || !ctx.is_authoritative_executor {
        return Ok(None);
    }
    let invert = match op_tag {
        8 => false,
        9 => true,
        _ => return Ok(None),
    };
    let lhs = r_args[0];
    let rhs = r_args[1];

    if lhs.same_box(rhs) {
        // `x is x` / `x is not x` — statically determined.
        return walker_write_const_bool_result(ctx, op_pc, !invert, dst, dst_bank).map(Some);
    }

    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    if lhs_obj.is_null() || rhs_obj.is_null() {
        return Ok(None);
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(lhs_obj)
            || pyre_object::tagged_int::is_tagged_int(rhs_obj))
    {
        return Ok(None);
    }
    let (lhs_type, rhs_type) = unsafe {
        (
            (*(lhs_obj as *const pyre_object::pyobject::PyObject)).ob_type,
            (*(rhs_obj as *const pyre_object::pyobject::PyObject)).ob_type,
        )
    };
    if is_w_compares_by_value(lhs_type) || is_w_compares_by_value(rhs_type) {
        return Ok(None);
    }
    // The layout test above is the proof that `is_w` reduces to `ptr::eq`
    // here; cross-check it against the real thing and decline rather than
    // record a concrete the emitted `ptr_eq` disagrees with.
    let same = std::ptr::eq(lhs_obj, rhs_obj);
    if same != pyre_interpreter::baseobjspace::is_w(lhs_obj, rhs_obj) {
        return Ok(None);
    }

    // --- commit to the fold: emit IR (no further declines) ---
    for (operand, operand_type) in [(lhs, lhs_type), (rhs, rhs_type)] {
        if operand.is_constant() || ctx.trace_ctx.heap_cache().is_class_known(operand) {
            continue;
        }
        let type_addr = operand_type as usize as i64;
        let type_const = ctx.trace_ctx.const_int(type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[operand, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(operand, type_addr);
    }
    let cmp = if invert { OpCode::PtrNe } else { OpCode::PtrEq };
    let truth = ctx.trace_ctx.record_op(cmp, &[lhs, rhs]);
    let result = same != invert;
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(result as i64));
    // Same boxed-bool elision as the int compare specialization: when the
    // Ref dst is provably consumed only by the following `is_true`, write
    // the raw truth and let `bool_box_truth_record` resolve it.
    if compare_box_provably_dead(ctx, op_pc, dst as u8) {
        bool_box_truth_record(truth, truth);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, truth)?;
        return Ok(Some(()));
    }
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, result, dst as u8, dst_bank)? {
        // The guarded arm already pinned the truth to a constant and filed
        // `bool_box_truth_record` against it, so the following `is_true`
        // folds without re-reading the runtime truth.
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::w_bool_from(result) as usize)),
            );
            bool_box_truth_record(boxed, truth);
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// Write an immortal `bool` singleton into a residual call's Ref dst, along
/// with the raw truth `bool_box_truth_record` needs so an immediately
/// following `is_true` (`POP_JUMP_IF_*`) folds to the constant instead of
/// unboxing through a residual.
fn walker_write_const_bool_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    value: bool,
    dst: usize,
    dst_bank: char,
) -> Result<(), DispatchError> {
    let result_obj = pyre_object::w_bool_from(value);
    let const_bool = ctx.trace_ctx.const_ref(result_obj as i64);
    ctx.trace_ctx.set_opref_concrete(
        const_bool,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    let truth = ctx.trace_ctx.const_int(value as i64);
    bool_box_truth_record(const_bool, truth);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, const_bool)
}

/// MAKE_FUNCTION inline emission: replace the
/// `jit_make_function_from_globals(globals, code)` residual with the
/// `NewWithVtable` + `SetfieldGc` set `function.py:47-57 Function.__init__`
/// performs, so a `def` in a loop body virtualizes away instead of allocating a
/// `Function` per iteration.
///
/// Everything the constructor stores is loop-invariant here: `globals` and
/// `code` arrive as baked constants (`codewriter.rs` MakeFunction arm bakes the
/// frame's globals object, and the code object comes from a `LOAD_CONST`), and
/// the remaining slots are derived from them:
///
/// * `name` — `function.py:51 self.name = code.co_name`, a pointer into the
///   `Box::into_raw`'d `CodeObject`, which is never rewritten in place nor
///   freed.  This is the same pointer the residual stores
///   (`function_new_from_code` borrows it too), so a materialized function is
///   indistinguishable from an interpreted one.
/// * `w_name` / `w_qualname` — the code object's single realized `co_name` and
///   `co_qualname`, shared by every function built from it.
/// * `w_builtins` — CPython 3.14 `_PyEval_BuiltinsFromGlobals`, frozen at
///   construction from `globals['__builtins__']`.  Only the allocation-free
///   shape is reproduced: `__builtins__` naming a module, reduced to its dict.
///   An absent or non-module `__builtins__` routes
///   `pick_builtin_obj_checked` through `w_module_new_aliasing_dict` / the
///   default-module build, which mint a fresh object per call and so cannot be
///   baked — those decline to the residual.
///
/// Soundness rests on one guard beyond the constant operands: the module
/// dict's `version?` is pinned, so rebinding `globals['__builtins__']` runs
/// `mutated()` and revokes the loop, exactly as it does for a shadowing insert
/// under the LOAD_GLOBAL cell fold.  Nothing watches the code object's
/// `co_name` / `co_qualname` because neither is mutable in place —
/// `code.replace()` clones first and yields a different code object, which is a
/// different constant.
///
/// Declines (each falls through to the residual, which stays correct): a
/// non-constant operand, a non-`PyCode` or bodyless code object, globals that
/// are not a module dict, an unbakeable `__builtins__`, and any baked pointer
/// the collector may relocate.
pub(crate) fn try_walker_specialize_make_function<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 {
        return Ok(None);
    }
    let (globals_op, code_op) = (r_args[0], r_args[1]);
    if !globals_op.is_constant() || !code_op.is_constant() {
        return Ok(None);
    }
    let Some(w_code) = walker_concrete_ref_object(ctx, code_op) else {
        return Ok(None);
    };
    // A `BuiltinCode`-backed carrier is immortal and boxes its name through
    // `malloc_raw`; `is_code` admits only the `PyCode` shape this reproduces.
    if !unsafe { pyre_interpreter::is_code(w_code) } {
        return Ok(None);
    }
    let code_ptr =
        unsafe { pyre_interpreter::w_code_get_ptr(w_code) } as *const pyre_interpreter::CodeObject;
    if code_ptr.is_null() {
        return Ok(None);
    }

    // Realizing the name/qualname are the fold's collection points (they
    // allocate once per code object and hit the cache afterwards), so they run
    // here, while the only live pointers are the `Box::into_raw`'d code wrapper
    // and its `CodeObject` — neither of which the collector relocates.
    let w_name = unsafe { pyre_interpreter::pycode::w_code_name_obj(w_code) };
    if w_name.is_null() {
        return Ok(None);
    }
    let w_qualname = unsafe { pyre_interpreter::pycode::w_code_qualname_obj(w_code) };
    if w_qualname.is_null() {
        return Ok(None);
    }
    let Some(w_globals) = walker_concrete_ref_object(ctx, globals_op) else {
        return Ok(None);
    };
    // Restrict to a module namespace before probing it directly, so the slot
    // read below walks the same storage `pick_builtin_obj_checked`'s
    // `finditem_str` does.  This is also what `walker_pin_namespace_version`
    // needs, but that one emits IR, so it runs only once the fold commits.
    if unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(w_globals) }.is_null()
    {
        return Ok(None);
    }
    // `function_new_impl`'s `w_builtins` derivation, restricted to the branch
    // that allocates nothing and therefore answers the same object each run.
    let w_builtins_module = unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }
        .unwrap_or(pyre_object::PY_NULL);
    if w_builtins_module.is_null() || !unsafe { pyre_object::is_module(w_builtins_module) } {
        return Ok(None);
    }
    let w_builtins = unsafe { pyre_object::w_module_get_w_dict(w_builtins_module) };
    if w_builtins.is_null() {
        return Ok(None);
    }
    for baked in [w_builtins, w_name, w_qualname] {
        if majit_gc::can_move(majit_ir::GcRef(baked as usize)) {
            return Ok(None);
        }
    }

    // --- commit to the fold: emit IR (no further declines) ---
    // The only mutable input: `globals['__builtins__']` may be rebound after
    // this function is built, and a later iteration must then see the new
    // mapping.  Pinning the namespace `version?` revokes the loop instead.
    walker_pin_namespace_version(ctx, op_pc, w_globals)?;
    let header_w_class = ctx
        .trace_ctx
        .const_ref(pyre_object::get_instantiate(&pyre_interpreter::FUNCTION_TYPE) as i64);
    // `function.py:33 can_change_code = True` for a plain `def`.
    let can_change_code = ctx.trace_ctx.const_int(1);
    let name = ctx
        .trace_ctx
        .const_ref(unsafe { &(*code_ptr).obj_name } as *const String as i64);
    let w_name_const = ctx.trace_ctx.const_ref(w_name as i64);
    let w_builtins_const = ctx.trace_ctx.const_ref(w_builtins as i64);
    let w_qualname_const = ctx.trace_ctx.const_ref(w_qualname as i64);
    let func_op = crate::helpers::emit_make_function_inline(
        ctx.trace_ctx,
        header_w_class,
        code_op,
        can_change_code,
        name,
        w_name_const,
        globals_op,
        w_builtins_const,
        w_qualname_const,
    );
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(func_op, &pyre_interpreter::FUNCTION_TYPE as *const _ as i64);
    // Tracing is execution: build the concrete function the rest of the walk
    // observes.  A fresh `Function` per evaluation is what MAKE_FUNCTION
    // produces anyway, so the trace allocating its own is not an identity
    // divergence.
    let func = pyre_interpreter::runtime_ops::make_function_from_code_obj_with_globals_obj(
        w_code, w_globals,
    );
    ctx.trace_ctx.set_opref_concrete(
        func_op,
        majit_ir::Value::Ref(majit_ir::GcRef(func as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, func_op)?;
    Ok(Some(()))
}

/// Mixed W_LongObject/W_IntObject COMPARE_OP specialization.
///
/// `pypy/objspace/std/longobject.py:_make_descr_cmp` selects the corresponding
/// `rbigint.int_<cmp>` method for a machine-int other operand.  For the
/// reflected order, select the inverse comparison with the bigint kept as the
/// first residual argument (`int < long` becomes `long > int`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_compare_op_long_int<Sym: WalkSym>(
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
    use pyre_interpreter::bytecode::ComparisonOperator;
    use pyre_interpreter::objspace::descroperation as desc;
    type CompareFn = extern "C" fn(i64, i64) -> i64;
    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    let (lhs_obj, rhs_obj) = match (
        walker_concrete_ref_object(ctx, r_args[0]),
        walker_concrete_ref_object(ctx, r_args[1]),
    ) {
        (Some(lhs), Some(rhs)) => (lhs, rhs),
        _ => return Ok(None),
    };
    let lhs_is_long = unsafe { pyre_object::is_long(lhs_obj) };
    let rhs_is_long = unsafe { pyre_object::is_long(rhs_obj) };
    let lhs_is_int = unsafe { pyre_object::is_int(lhs_obj) };
    let rhs_is_int = unsafe { pyre_object::is_int(rhs_obj) };
    let (long, int, long_obj, int_obj, reflected) = if lhs_is_long && rhs_is_int {
        (r_args[0], r_args[1], lhs_obj, rhs_obj, false)
    } else if lhs_is_int && rhs_is_long {
        (r_args[1], r_args[0], rhs_obj, lhs_obj, true)
    } else {
        return Ok(None);
    };
    let (Some(long_class), Some(int_class)) = (unsafe {
        (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        )
    }) else {
        return Ok(None);
    };
    let effective_cmp = if reflected {
        match cmp_op {
            ComparisonOperator::Less => ComparisonOperator::Greater,
            ComparisonOperator::LessOrEqual => ComparisonOperator::GreaterOrEqual,
            ComparisonOperator::Greater => ComparisonOperator::Less,
            ComparisonOperator::GreaterOrEqual => ComparisonOperator::LessOrEqual,
            ComparisonOperator::Equal => ComparisonOperator::Equal,
            ComparisonOperator::NotEqual => ComparisonOperator::NotEqual,
        }
    } else {
        cmp_op
    };
    let helper: CompareFn = match effective_cmp {
        ComparisonOperator::Less => desc::jit_bigint_int_lt,
        ComparisonOperator::LessOrEqual => desc::jit_bigint_int_le,
        ComparisonOperator::Greater => desc::jit_bigint_int_gt,
        ComparisonOperator::GreaterOrEqual => desc::jit_bigint_int_ge,
        ComparisonOperator::Equal => desc::jit_bigint_int_eq,
        ComparisonOperator::NotEqual => desc::jit_bigint_int_ne,
    };
    let int_value = unsafe { pyre_object::w_int_get_value(int_obj) };

    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let boxed_result_obj = boxed_result_i64 as usize as pyre_object::PyObjectRef;
    if boxed_result_obj == pyre_object::PY_NULL
        || !unsafe { pyre_object::is_bool(boxed_result_obj) }
    {
        return Ok(None);
    }
    let concrete_truth = unsafe { pyre_object::w_bool_get_value(boxed_result_obj) as i64 };

    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, long, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, long, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let int_raw = walker_unbox_int_typed(ctx, op_pc, int, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, int, int_class)?;
    let off = pyre_object::longobject::LONG_VALUE_OFFSET;
    let long_payload = unsafe { *((long_obj as *const u8).add(off) as *const i64) };
    let long_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        long_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
    );
    let helper_ptr = helper as *const ();
    let truth = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        helper_ptr,
        &[long_pl, int_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Int,
        majit_metainterp::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper_ptr as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
            majit_ir::Value::Int(int_value),
        ],
        majit_ir::Value::Int(concrete_truth),
    );
    ctx.trace_ctx
        .set_opref_concrete(truth, majit_ir::Value::Int(concrete_truth));
    if compare_box_provably_dead(ctx, op_pc, dst as u8) {
        bool_box_truth_record(truth, truth);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, truth)?;
        return Ok(Some(()));
    }
    let boxed = match walker_newbool_guarded(
        ctx,
        op_pc,
        truth,
        concrete_truth != 0,
        dst as u8,
        dst_bank,
    )? {
        // The guarded arm already pinned the truth to a constant and filed
        // `bool_box_truth_record` against it, so the following `is_true`
        // folds without re-reading the runtime truth.
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            bool_box_truth_record(boxed, truth);
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// W_LongObject (bigint) COMPARE_OP specialization — the long analogue of
/// [`try_walker_specialize_compare_op_int`].  Both operands are `int`-typed but
/// bigint-stored: guard each against `LONG_TYPE`, read each `value` payload,
/// then `CallPure_I` the pure
/// `jit_bigint_cmp` (sign of `a <=> b` in {-1,0,1}; a comparison neither
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
    let (Some(lhs_class), Some(rhs_class)) = (unsafe {
        (
            walker_exact_builtin_class(lhs_obj),
            walker_exact_builtin_class(rhs_obj),
        )
    }) else {
        return Ok(None);
    };
    // Authentic boxed W_Bool via the same execute path the int leg uses; also
    // advances the concrete VM state the downstream ops read.
    let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, lhs, long_type_addr)?;
    walker_guard_class(ctx, op_pc, rhs, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, lhs_class)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, rhs_class)?;
    // `_make_descr_cmp` (longobject.py:383-391) compares `self.num` against
    // `w_other.num`, so the two payload reads are trace ops rather than work
    // hidden inside the callee. Spelling them out is also what keeps a
    // `W_LongObject` this same trace built from having to escape into the
    // comparison: the read hits the heap cache entry `emit_box_long_inline`
    // filed and the box stays virtual.
    let lhs_payload = unsafe { long_payload_of(lhs_obj) };
    let rhs_payload = unsafe { long_payload_of(rhs_obj) };
    let lhs_pl = walker_read_long_payload(ctx, lhs, lhs_payload);
    let rhs_pl = walker_read_long_payload(ctx, rhs, rhs_payload);
    // Pure `rbigint` comparison → sign in {-1,0,1}. Dead after the `int_<cmp>`
    // below and never spans a guard, so it needs no blackhole reconstruction.
    let cmp_fn = pyre_object::longobject::jit_bigint_cmp as *const ();
    let sign_concrete = pyre_object::longobject::jit_bigint_cmp(lhs_payload, rhs_payload);
    let concrete_args = [
        majit_ir::Value::Int(cmp_fn as usize as i64),
        majit_ir::Value::Ref(majit_ir::GcRef(lhs_payload as usize)),
        majit_ir::Value::Ref(majit_ir::GcRef(rhs_payload as usize)),
    ];
    let sign = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        cmp_fn,
        &[lhs_pl, rhs_pl],
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
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, folded != 0, dst as u8, dst_bank)? {
        // The guarded arm already pinned the truth to a constant and filed
        // `bool_box_truth_record` against it, so the following `is_true`
        // folds without re-reading the runtime truth.
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            bool_box_truth_record(boxed, truth);
            boxed
        }
    };
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
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(bin_op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        return Ok(None);
    };
    use pyre_interpreter::bytecode::BinaryOperator;
    // Power has no FLOAT_* opcode — it lowers to the raw-float
    // `float_pow_jit` call (floatobject.py descr_pow → _pow), same
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

    let Some((lhs, rhs, lhs_obj, rhs_obj, lhs_is_int, rhs_is_int, lhs_f64, rhs_f64)) =
        walker_float_specialization_input_operands(ctx, r_args)
    else {
        return Ok(None);
    };

    if matches!(op_code, Some(OpCode::FloatTrueDiv)) && rhs_f64 == 0.0 {
        let Some(Err(exc_i64)) = walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };
        if let Some(cb) = crate::callbacks::try_get() {
            (cb.drain_backend_jit_exc)();
        }
        let exc = exc_i64 as usize as pyre_object::PyObjectRef;
        let kind = pyre_object::interp_exceptions::ExcKind::ZeroDivisionError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

        let _lhs_raw =
            walker_coerce_operand_to_float(ctx, op_pc, lhs, lhs_obj, lhs_is_int, lhs_f64, false)?;
        walker_guard_exact_w_class(ctx, op_pc, lhs, walker_numeric_builtin_class(lhs_obj))?;
        let rhs_raw =
            walker_coerce_operand_to_float(ctx, op_pc, rhs, rhs_obj, rhs_is_int, rhs_f64, false)?;
        walker_guard_exact_w_class(ctx, op_pc, rhs, walker_numeric_builtin_class(rhs_obj))?;
        let rhs_zero = walker_float_eq_const(ctx, rhs_raw, 0.0, 1);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[rhs_zero])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }

    let Some(Ok(boxed_result_i64)) =
        walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr)
    else {
        return Ok(None);
    };
    if boxed_result_i64 == 0 {
        return Ok(None);
    }
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

    // --- emit the specialized IR (walker-native) ---
    let lhs_raw =
        walker_coerce_operand_to_float(ctx, op_pc, lhs, lhs_obj, lhs_is_int, lhs_f64, false)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, walker_numeric_builtin_class(lhs_obj))?;
    let rhs_raw =
        walker_coerce_operand_to_float(ctx, op_pc, rhs, rhs_obj, rhs_is_int, rhs_f64, false)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, walker_numeric_builtin_class(rhs_obj))?;
    // rint.py `_ovf_zer` analogue for float true-division: emit a
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
            // _pow (floatobject.py) traced inline for its fast paths:
            // every special-case `if` becomes a comparison guard and only
            // the raw libm pow stays residual.
            if let Some(r) = walker_emit_float_pow_inline(
                ctx, op_pc, lhs_raw, rhs_raw, lhs_f64, rhs_f64, result_val,
            )? {
                r
            } else {
                // Cold-path fallback (nan/inf operands, negative base):
                // the opaque `_pow` helper.  It is EF_CAN_RAISE, NOT
                // force_virtual: pyjitpl.py execute_varargs(
                // rop.CALL_F, ..., exc=True, pure=False) records CALL_F
                // and handle_possible_exception → GUARD_NO_EXCEPTION
                // (pyjitpl.py).  The raising case never reaches
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
    Ok(Some(DispatchOutcome::Continue))
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
/// A canonical Unicode-strategy dict with an exact-str key and a concrete hit
/// records the `rordereddict.py dict.lookup` oopspec producer: exact dict/key
/// guards, a strategy guard, elidable `rstr.ll_strhash`, `dict.lookup`, a
/// non-negative guard on the returned entry index, then a guarded value read.
///
/// Tuples, dict misses, empty-strategy lists, negative indices, and
/// non-`list[int]` operands fall through to the generic `CallMayForce` record
/// (`Ok(None)`), preserving Python `__getitem__` semantics.
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

    if let Some(hit) = walker_probe_exact_dict_hit(list_obj, key_obj)? {
        return walker_emit_exact_dict_hit(
            ctx, op_pc, list_op, key_op, list_obj, hit, dst, dst_bank,
        );
    }

    // #171/#11 Approach C: canonical array-backed `W_TupleObject[i]`.  Two
    // gates, both required:
    //   * `ob_type == &TUPLE_TYPE` (tupleobject.py / tupleobject.rs) —
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

    // The arity-2 specialisations reach the same `getitem`, but their items
    // are the inline `value0`/`value1` slots, so they need their own reader —
    // which is why the gate above is `ob_type == &TUPLE_TYPE` and not
    // `is_tuple()`. No `w_class` guard: only a specialisation carries its own
    // `ob_type`, so a tuple subclass (which keeps `&TUPLE_TYPE`) can never
    // pass the class guard below.
    if let Some(pair_kind) = specialised_pair_kind(unsafe { (*list_obj).ob_type }) {
        return try_walker_specialize_subscr_specialised_pair(
            ctx, op_pc, list_op, key_op, list_obj, key_obj, pair_kind, allboxes, call_descr, dst,
            dst_bank,
        );
    }

    // The `dict.lookup` gate.  Both `w_class` checks are load-bearing: a dict
    // SUBCLASS shares `ob_type == &DICT_TYPE` but retags `w_class` and reaches
    // `__missing__` on a miss, and a str SUBCLASS key may override `__hash__` /
    // `__eq__`, so neither may take the exact-str probe.  The strategy check is
    // what makes the probe non-raising: `UnicodeDictStrategy` hands the dict to
    // `ObjectDictStrategy` the moment a non-exact-str key is stored, so while
    // it holds, every stored key is an exact str and the comparisons are WTF-8
    // byte equality (`dictmultiobject.py:1286+` `r_dict(unicode_eq,
    // unicode_hash)`).
    let canonical_dict = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    let dict_unicode = !canonical_dict.is_null()
        && unsafe {
            std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::DICT_TYPE)
                && std::ptr::eq((*list_obj).w_class, canonical_dict)
                && pyre_object::dictmultiobject::w_dict_get_strategy(list_obj).strategy_kind()
                    == pyre_object::dictmultiobject::StrategyKind::Unicode
        };
    let canonical_str = if dict_unicode {
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE)
    } else {
        std::ptr::null_mut()
    };
    let dict_unicode_hit = !canonical_str.is_null()
        && unsafe {
            std::ptr::eq((*key_obj).ob_type, &pyre_object::pyobject::STR_TYPE)
                && std::ptr::eq((*key_obj).w_class, canonical_str)
        };
    if dict_unicode_hit {
        let hash = unsafe { pyre_object::dictmultiobject::w_dict_unicode_key_hash(key_obj) };
        let index = unsafe {
            pyre_object::dictmultiobject::w_dict_unicode_lookup_index(list_obj, key_obj, hash, 0)
        };
        if index < 0 {
            return Ok(None);
        }

        let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };

        walker_guard_class(
            ctx,
            op_pc,
            list_op,
            &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
        )?;
        walker_guard_exact_w_class(ctx, op_pc, list_op, canonical_dict)?;
        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::dict_strategy_word_descr(),
        );
        let unicode_strategy_const = ctx
            .trace_ctx
            .const_int(&pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF as *const _ as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[strategy, unicode_strategy_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(strategy, unicode_strategy_const);
        walker_guard_class(
            ctx,
            op_pc,
            key_op,
            &pyre_object::pyobject::STR_TYPE as *const _ as i64,
        )?;
        walker_guard_exact_w_class(ctx, op_pc, key_op, canonical_str)?;

        let hash_effect = majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::ElidableCannotRaise,
            majit_ir::OopSpecIndex::None,
        );
        // Both residuals bind the macro-emitted `__majit_call_target_*`
        // trampoline rather than the raw fn: the wasm backend derives a
        // residual's `call_indirect` type from the descr alone — `(i64 x n) ->
        // i64` — so a raw `*mut PyObject` argument, `i32` on wasm32, traps
        // `indirect call type mismatch`. The trampoline takes and returns the
        // uniform machine word everywhere, and is the address `jit_fnaddr`
        // registers for these paths.
        let hash_fn = {
            let f: extern "C" fn(i64) -> i64 =
                pyre_object::dictmultiobject::__majit_call_target_w_dict_unicode_key_hash;
            f as *const ()
        };
        let hash_op = ctx.trace_ctx.call_typed_with_effect_pure(
            OpCode::CallI,
            hash_fn,
            &[key_op],
            &[majit_ir::Type::Ref],
            majit_ir::Type::Int,
            hash_effect,
            &[
                majit_ir::Value::Int(hash_fn as i64),
                majit_ir::Value::Ref(majit_ir::GcRef(key_obj as usize)),
            ],
            majit_ir::Value::Int(hash),
        );

        let mut lookup_effect = majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::DictLookup,
        );
        lookup_effect.extradescrs = Some(vec![
            crate::descr::dict_lookup_namespace_descr(),
            crate::descr::dict_lookup_entries_array_descr(),
        ]);
        let lookup_flag = ctx.trace_ctx.const_int(0);
        let lookup_fn: extern "C" fn(i64, i64, i64, i64) -> i64 =
            pyre_object::dictmultiobject::__majit_call_target_w_dict_unicode_lookup_index;
        let index_op = ctx.trace_ctx.call_typed_with_effect(
            OpCode::CallI,
            lookup_fn as *const (),
            &[list_op, key_op, hash_op, lookup_flag],
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
            ],
            majit_ir::Type::Int,
            lookup_effect,
        );
        ctx.trace_ctx
            .set_opref_concrete(index_op, majit_ir::Value::Int(index));

        let zero = ctx.trace_ctx.const_int(0);
        let nonneg = ctx.trace_ctx.record_op(OpCode::IntGe, &[index_op, zero]);
        ctx.trace_ctx
            .set_opref_concrete(nonneg, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonneg])?;

        let value = ctx.trace_ctx.call_ref_typed_with_effect(
            crate::helpers::jit_dict_value_at as *const (),
            &[list_op, index_op, key_op, hash_op],
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
            ],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[value])?;
        ctx.trace_ctx.set_opref_concrete(
            value,
            majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
        );
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
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
    // inline `length` field (rlist.py); int/float storage read the typed
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
    // (tupleobject.py); the lower-bound guard deopts so that remap
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

/// SUBSCRIPT arm for the arity-2 tuple specialisations
/// (`specialisedtupleobject.py:134-142 getitem`), the analogue of
/// [`try_walker_specialize_subscr_tuple`] for a receiver whose items are the
/// inline `value0` / `value1` slots instead of a `wrappeditems` block.
///
/// Upstream `getitem` normalises a negative index against the constant
/// `typelen` and then runs the unrolled `index == i` chain to pick the slot,
/// so the recorded slot is pinned here with a single `guard_value` on the
/// unboxed index: it re-proves both the sign test and the comparison at once,
/// and a literal subscript makes it constant, which the optimizer drops.
///
/// The caller matched `ob_type` against one of the three specialisation
/// classes, which is also why no `w_class` guard follows: a tuple subclass
/// instance keeps the canonical `ob_type == &TUPLE_TYPE`, so it can neither
/// reach this arm nor pass the class guard below.
///
/// A slice key, a non-int key, or an out-of-range index falls through to the
/// generic residual (`Ok(None)`), which raises `IndexError` the way `getitem`
/// does.
#[allow(clippy::too_many_arguments)]
fn try_walker_specialize_subscr_specialised_pair<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq_op: OpRef,
    key_op: OpRef,
    seq_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    pair_kind: SpecialisedPairKind,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    const TYPELEN: i64 = 2;
    // The object-slot read is WRONG CODE from here and stays residual until it
    // is understood.  `test.test_datetime` reaches it through
    // `self.lt = (array('q', ut), array('q', ut))` read as `self.lt[dt.fold]`,
    // and the next call in that frame comes out one positional argument short —
    // `bisect.bisect_right(lt, timestamp)` answers `missing 1 required
    // positional argument: 'x'`.  Declining only this kind restores the module;
    // the `ii` / `ff` arms, which share the class guard and the pinned index,
    // are unaffected, and UNPACK reaches the same slots through
    // [`walker_emit_specialised_pair_item`] with no index operand and is sound.
    if pair_kind == SpecialisedPairKind::Object {
        return Ok(None);
    }
    // `bool` shares int's `intval`, so it indexes through its own &BOOL_TYPE
    // guard in the unbox below.
    let raw_key = unsafe {
        if !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        pyre_object::w_int_get_value(key_obj)
    };
    let index = if raw_key < 0 {
        raw_key + TYPELEN
    } else {
        raw_key
    };
    if !(0..TYPELEN).contains(&index) {
        return Ok(None);
    }

    let spec_type = unsafe { (*seq_obj).ob_type };
    walker_guard_specialised_pair_class(ctx, op_pc, seq_op, spec_type)?;

    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(raw_key));
    let expected = ctx.trace_ctx.const_int(raw_key);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[raw_index, expected])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(raw_index, expected);

    let Some(item) = walker_emit_specialised_pair_item(
        ctx, op_pc, seq_op, pair_kind, index, allboxes, call_descr,
    )?
    else {
        return Ok(None);
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, item)?;
    Ok(Some(()))
}

/// Builtin `type(x)`:
///
/// ```python
/// # pypy/objspace/std/objspace.py:441-443
/// jit.promote(w_obj.__class__)
/// return w_obj.getclass(self)
/// ```
///
/// Pyre's generic `bh_call_fn` otherwise enters `type_descr_call_impl`, which
/// performs the complete type-constructor protocol for every loop iteration.
/// Pin the callable and the argument's physical/Python class, then return the
/// promoted class object.  The generic-exception representation is declined:
/// its observable class can come from `ExcKind` even when physical type and
/// `w_class` match, so it needs a separate kind guard.
pub(crate) fn try_walker_specialize_builtin_type<Sym: WalkSym>(
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
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || obj.is_null() {
        return Ok(None);
    }
    let builtin_type = pyre_object::get_instantiate(&pyre_object::pyobject::TYPE_TYPE);
    if builtin_type.is_null() || !std::ptr::eq(concrete_callable, builtin_type) {
        return Ok(None);
    }

    let tagged =
        pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj);
    let (physical_type, stored_w_class) = if tagged {
        (
            &pyre_object::pyobject::INT_TYPE as *const _ as i64,
            pyre_object::get_instantiate(&pyre_object::pyobject::INT_TYPE),
        )
    } else {
        let physical_type = unsafe { (*obj).ob_type } as i64;
        let stored_w_class = unsafe { (*obj).w_class };
        if unsafe { pyre_object::is_exception(obj) } {
            let generic_exception =
                pyre_object::get_instantiate(&pyre_object::interp_exceptions::EXCEPTION_TYPE);
            if stored_w_class.is_null() || std::ptr::eq(stored_w_class, generic_exception) {
                return Ok(None);
            }
        }
        (physical_type, stored_w_class)
    };
    let Some(result_type) = pyre_interpreter::typedef::r#type(obj) else {
        return Ok(None);
    };
    let result_type = result_type.as_ptr();

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

    let obj_op = r_args[2];
    walker_guard_class(ctx, op.pc, obj_op, physical_type)?;
    // A tagged int has no dereferenceable `w_class` field; GuardClass's boxed
    // leg is enough because both representations return the canonical int
    // class.  Every other populated `w_class` is the live promoted field.
    if !tagged && !stored_w_class.is_null() {
        walker_guard_exact_w_class(ctx, op.pc, obj_op, stored_w_class)?;
    }
    let result = ctx.trace_ctx.const_ref(result_type as i64);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result)?;
    Ok(Some(()))
}

fn is_builtin_dict_get_function(callable: pyre_object::PyObjectRef) -> bool {
    if callable.is_null() || !unsafe { pyre_interpreter::is_function(callable) } {
        return false;
    }
    let code = unsafe { pyre_interpreter::function_get_code(callable) } as pyre_object::PyObjectRef;
    !code.is_null()
        && unsafe { pyre_interpreter::is_builtin_code(code) }
        && unsafe { pyre_interpreter::builtin_code_get(code) as usize }
            == pyre_interpreter::type_methods::dict_method_get as *const () as usize
}

#[derive(Clone, Copy)]
enum DictFoldKeyProbe {
    Int,
    Unicode,
}

#[derive(Clone, Copy)]
struct DictFoldHit {
    concrete_value: pyre_object::PyObjectRef,
    key_probe: DictFoldKeyProbe,
}

fn walker_probe_exact_dict_hit(
    dict: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
) -> Result<Option<DictFoldHit>, DispatchError> {
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    if canonical_dict.is_null()
        || !unsafe {
            std::ptr::eq((*dict).ob_type, &pyre_object::pyobject::DICT_TYPE)
                && std::ptr::eq((*dict).w_class, canonical_dict)
        }
    {
        return Ok(None);
    }

    // Only the two homogeneous strategies fold: their lookups probe a native
    // table and so cannot run Python-level `__hash__` or `__eq__`.  Every other
    // strategy, a mapdict-backed instance dict included, keeps the real lookup.
    let strategy_kind =
        unsafe { pyre_object::dictmultiobject::w_dict_get_strategy(dict).strategy_kind() };
    let int_probe = strategy_kind == pyre_object::dictmultiobject::StrategyKind::Int
        && unsafe { pyre_object::listobject::is_plain_int1(key) && pyre_object::is_int(key) };
    let unicode_probe = strategy_kind == pyre_object::dictmultiobject::StrategyKind::Unicode
        && unsafe {
            pyre_object::is_exact_type(key, &pyre_object::pyobject::STR_TYPE)
                && pyre_object::w_str_get_value_opt(key).is_some()
                && pyre_object::dict_eq_hook::try_hash_str(
                    pyre_object::w_str_get_value_opt(key).unwrap().as_bytes(),
                )
                .is_some()
        };

    let found = if int_probe {
        let index =
            unsafe { pyre_object::dictmultiobject::w_dict_index_of_int_strategy(dict, key) };
        index.and_then(|index| {
            unsafe { pyre_object::dictmultiobject::w_dict_nth_value(dict, index) }
                .map(|value| (value, DictFoldKeyProbe::Int))
        })
    } else if unicode_probe {
        let index =
            unsafe { pyre_object::dictmultiobject::w_dict_index_of_unicode_strategy(dict, key) };
        index.and_then(|index| {
            unsafe { pyre_object::dictmultiobject::w_dict_nth_value(dict, index) }
                .map(|value| (value, DictFoldKeyProbe::Unicode))
        })
    } else {
        None
    };

    let Some((concrete_value, key_probe)) = found else {
        return Ok(None);
    };
    Ok(Some(DictFoldHit {
        concrete_value,
        key_probe,
    }))
}

fn walker_emit_exact_dict_hit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dict_op: OpRef,
    key_op: OpRef,
    dict: pyre_object::PyObjectRef,
    hit: DictFoldHit,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    walker_guard_class(
        ctx,
        op_pc,
        dict_op,
        &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
    )?;
    walker_guard_exact_w_class(ctx, op_pc, dict_op, canonical_dict)?;

    let (key_type, canonical_key, strategy_ref, lookup_helper) = match hit.key_probe {
        DictFoldKeyProbe::Int => (
            &pyre_object::pyobject::INT_TYPE as *const _ as i64,
            pyre_object::get_instantiate(&pyre_object::pyobject::INT_TYPE),
            &pyre_object::dictmultiobject::INT_DICT_STRATEGY_REF as *const _ as i64,
            crate::helpers::jit_dict_exact_int_lookup_or_null as *const (),
        ),
        DictFoldKeyProbe::Unicode => (
            &pyre_object::pyobject::STR_TYPE as *const _ as i64,
            pyre_object::get_instantiate(&pyre_object::pyobject::STR_TYPE),
            &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF as *const _ as i64,
            crate::helpers::jit_dict_exact_unicode_lookup_or_null as *const (),
        ),
    };

    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        dict_op,
        crate::descr::dict_strategy_word_descr(),
    );
    let strategy_const = ctx.trace_ctx.const_int(strategy_ref);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardValue,
        &[strategy, strategy_const],
    )?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(strategy, strategy_const);

    walker_guard_class(ctx, op_pc, key_op, key_type)?;
    walker_guard_exact_w_class(ctx, op_pc, key_op, canonical_key)?;

    let value = ctx.trace_ctx.call_ref_typed_with_effect(
        lookup_helper,
        &[dict_op, key_op],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardNonnull, &[value])?;
    ctx.trace_ctx.set_opref_concrete(
        value,
        majit_ir::Value::Ref(majit_ir::GcRef(hit.concrete_value as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// `dict.get` on an exact dictionary and an Int/Unicode strategy hit.
///
/// `dictmultiobject.py:1095-1098` probes an Int strategy with the unboxed key
/// value, and `dictmultiobject.py:1315-1318` probes a Unicode strategy with
/// exact-str bytes. The trace guards the exact dict, strategy vtable, and exact
/// key type, then performs a live strategy-specific lookup and guards that it
/// hit. Misses and object/identity strategy keys remain residual so their
/// hash/equality effects stay observable (`rdict.py:576`).
pub(crate) fn try_walker_specialize_builtin_dict_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if !(r_args.len() == 3 || r_args.len() == 4) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(callable_operand),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(key),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if callable_operand.is_null() || key.is_null() {
        return Ok(None);
    }
    // LOAD_ATTR's generic method path produces `[Method, PY_NULL, key]`;
    // LOAD_METHOD's split path produces `[Function, receiver, key]`.
    // `_Method._immutable_fields_` lets both converge on the same live
    // function/receiver reads.
    let bound_method =
        null_or_self.is_null() && unsafe { pyre_object::is_method(callable_operand) };
    let (callable, dict) = if bound_method {
        (
            unsafe { pyre_object::w_method_get_func(callable_operand) },
            unsafe { pyre_object::w_method_get_self(callable_operand) },
        )
    } else {
        (callable_operand, null_or_self)
    };
    if callable.is_null() || dict.is_null() || !is_builtin_dict_get_function(callable) {
        return Ok(None);
    }
    let Some(hit) = walker_probe_exact_dict_hit(dict, key)? else {
        return Ok(None);
    };

    let mut callable_op = r_args[0];
    let dict_op;
    if bound_method {
        walker_guard_class(
            ctx,
            op.pc,
            r_args[0],
            &pyre_object::function::METHOD_TYPE as *const _ as i64,
        )?;
        callable_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_function_descr(),
        );
        dict_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_self_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            dict_op,
            majit_ir::Value::Ref(majit_ir::GcRef(dict as usize)),
        );
    } else {
        dict_op = r_args[1];
    }
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(callable as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[callable_op, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }
    walker_emit_exact_dict_hit(ctx, op.pc, dict_op, r_args[2], dict, hit, dst, 'r')
}

/// Where the guarded receiver's length comes from — the `length()` body each
/// layout has upstream.
#[derive(Clone, Copy)]
enum BuiltinLenSource {
    /// `W_ListObject.length()` → `strategy.length` (rlist.py). Carries the
    /// storage-strategy id the read is guarded on.
    ListStrategy(i64),
    /// `EmptyListStrategy.length()` returns zero (`listobject.py:1131-1132`).
    /// The strategy still needs a guard because a reused list may transition
    /// to typed or object storage after tracing.
    EmptyList,
    /// `W_UnicodeObject.len` → `bh_unicodelen`; no storage strategy.
    StrField,
    /// `tupleobject.py` carries no separate length field, so the length is
    /// `arraylen_gc(wrappeditems)`.
    TupleArrayLen,
    /// `specialisedtupleobject.py:54-55 length()` returns the constant
    /// `typelen`.
    PairArity,
}

/// `len(x)` on an exact canonical `W_ListObject` / `W_UnicodeObject` /
/// `W_TupleObject`, or on an arity-2 tuple specialisation:
/// lower the opaque `bh_call_fn(len_builtin, PY_NULL, x)` residual to the
/// inline length read the meta-tracer produces upstream
/// (descroperation.py `_len`): `guard_value(callable)` +
/// `guard_class` + exact `w_class` guard + the [`BuiltinLenSource`] read +
/// `wrapint`.  The exact `w_class` guard is required because a SUBCLASS shares
/// `ob_type == &LIST_TYPE`/`&STR_TYPE`/`&TUPLE_TYPE` but may override `__len__`
/// (`baseobjspace::len` dispatches `subclass_special_override`); it
/// side-exits to the generic residual.
///
/// Returns `None` (fall through to the generic residual, SAFE) for any
/// other shape: non-list/str/tuple arg, a subclass, a bound receiver, or wrong
/// arity.
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
    // Exact canonical list / str / tuple, or one of the arity-2 tuple
    // specialisations.  `arg_type_addr` pins the `guard_class` target;
    // `exact_w_class` is the subclass-`__len__` guard (see the doc comment),
    // absent for a specialisation because only `makespecialisedtuple2` builds
    // that `ob_type` and always with the canonical tuple `w_class`, so the
    // class guard alone already excludes every subclass instance.
    let (arg_type_addr, exact_w_class, len_source, concrete_len) = unsafe {
        let ob_type = (*list_obj).ob_type;
        let w_class = (*list_obj).w_class;
        if std::ptr::eq(ob_type, &pyre_object::pyobject::LIST_TYPE) {
            let exact = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            let len_source = if pyre_object::w_list_uses_int_storage(list_obj) {
                BuiltinLenSource::ListStrategy(
                    pyre_object::listobject::ListStrategy::Integer as i64,
                )
            } else if pyre_object::w_list_uses_float_storage(list_obj) {
                BuiltinLenSource::ListStrategy(pyre_object::listobject::ListStrategy::Float as i64)
            } else if pyre_object::w_list_uses_object_storage(list_obj) {
                BuiltinLenSource::ListStrategy(pyre_object::listobject::ListStrategy::Object as i64)
            } else if pyre_object::w_list_uses_empty_storage(list_obj) {
                BuiltinLenSource::EmptyList
            } else {
                return Ok(None);
            };
            (
                &pyre_object::pyobject::LIST_TYPE as *const _ as i64,
                Some(exact),
                len_source,
                pyre_object::w_list_len(list_obj),
            )
        } else if std::ptr::eq(ob_type, &pyre_object::pyobject::STR_TYPE) {
            let exact = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            (
                &pyre_object::pyobject::STR_TYPE as *const _ as i64,
                Some(exact),
                BuiltinLenSource::StrField,
                pyre_object::w_str_len(list_obj),
            )
        } else if std::ptr::eq(ob_type, &pyre_object::pyobject::TUPLE_TYPE) {
            let exact = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE);
            if !std::ptr::eq(w_class, exact) {
                return Ok(None);
            }
            (
                &pyre_object::pyobject::TUPLE_TYPE as *const _ as i64,
                Some(exact),
                BuiltinLenSource::TupleArrayLen,
                pyre_object::w_tuple_len(list_obj),
            )
        } else if specialised_pair_kind(ob_type).is_some() {
            (
                ob_type as i64,
                None,
                BuiltinLenSource::PairArity,
                pyre_object::w_tuple_len(list_obj),
            )
        } else {
            return Ok(None);
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
    if let Some(exact_w_class) = exact_w_class {
        walker_guard_exact_w_class(ctx, op.pc, list_op, exact_w_class)?;
    }
    // Length read.  list: guard the storage strategy, then read that
    // strategy's length field (rlist.py inline field for object storage;
    // typed items-block length for int/float storage).  str: a plain
    // codepoint-length getfield (no strategy, `bh_unicodelen`).
    let raw_len = match len_source {
        BuiltinLenSource::ListStrategy(sid) => {
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
        }
        BuiltinLenSource::EmptyList => {
            let strategy = crate::state::opimpl_getfield_gc_i(
                ctx.trace_ctx,
                list_op,
                crate::descr::list_strategy_descr(),
            );
            let empty = ctx
                .trace_ctx
                .const_int(pyre_object::listobject::ListStrategy::Empty as i64);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[strategy, empty], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            ctx.trace_ctx.heap_cache_mut().replace_box(strategy, empty);
            ctx.trace_ctx.const_int(0)
        }
        BuiltinLenSource::TupleArrayLen => {
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
        }
        BuiltinLenSource::StrField => crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::str_len_descr(),
        ),
        // `specialisedtupleobject.py:54-55 length()` returns the constant
        // `typelen`; there is no field to read, so the class guard above is
        // the whole proof and the box below folds to a constant.
        BuiltinLenSource::PairArity => ctx.trace_ctx.const_int(concrete_len as i64),
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

/// Fold plain `getattr(type, name)` when
/// [`pyre_interpreter::type_attr_value_fast_path`] proves that
/// `typeobject.py:811-828` returns the class-MRO value unchanged.  The exact
/// callable, exact receiver, exact name object, and receiver version are pinned
/// before the value is written as a green constant.  Pinning the callable makes
/// a rebound `getattr` side-exit instead of continuing to use the folded value.
/// The operand guards are tautologies when their inputs are already constants
/// and disappear during optimization.
/// [`pyre_interpreter::mutated`] recursively invalidates subclasses, so the
/// receiver's one quasi-immutable version watcher covers base-class mutation
/// and emits no per-iteration operations.
///
/// Like the `len` fold this is safe in an inlined callee sub-walk: the oracle
/// proves a read-only present attribute, so it cannot raise or introduce a
/// side effect that resume would repeat.  Every other shape declines before
/// emitting IR and falls through to the generic residual.
pub(crate) fn try_walker_specialize_builtin_type_getattr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, obj, name)` shape only.
    if r_args.len() != 4 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_name),
    ) = (
        arg_concretes[0],
        arg_concretes[1],
        arg_concretes[2],
        arg_concretes[3],
    )
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `getattr(type, name)` call.
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || concrete_obj.is_null()
        || concrete_name.is_null()
    {
        return Ok(None);
    }
    if !pyre_interpreter::builtins::is_builtin_getattr_function(concrete_callable) {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_exact_type(concrete_name, &pyre_object::pyobject::STR_TYPE) } {
        return Ok(None);
    }
    let name = unsafe { pyre_object::w_str_get_wtf8(concrete_name) };
    let Some((w_type, _version_tag, w_value)) =
        (unsafe { pyre_interpreter::type_attr_value_fast_path(concrete_obj, name) })
    else {
        return Ok(None);
    };

    let callable_op = r_args[0];
    if !callable_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[callable_op, expected],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_op, expected);
    }

    let obj_ref = r_args[2];
    let w_type_const = ctx.trace_ctx.const_ref(w_type as i64);
    if !obj_ref.is_constant() {
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[obj_ref, w_type_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(obj_ref, w_type_const);
    }

    // The baked WTF-8 bytes remain constant only while this exact string is
    // the name operand.  Constant operands make this guard a removable
    // tautology, so it costs nothing in the steady loop.
    let name_ref = r_args[3];
    let name_const = ctx.trace_ctx.const_ref(concrete_name as i64);
    if !name_ref.is_constant() {
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op.pc,
            OpCode::GuardValue,
            &[name_ref, name_const],
        )?;
    }

    // typeobject.py `promote(self.version_tag())`: this quasi-immutable watcher
    // emits no per-iteration op. `mutated` (baseobjspace.rs) recurses through
    // subclasses, so changing the attribute on any base invalidates this pin.
    walker_pin_type_version_tag(ctx, op.pc, w_type_const)?;

    let value_const = ctx.trace_ctx.const_ref(w_value as i64);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', value_const)?;
    Ok(Some(()))
}

/// `range(stop)` / `range(start, stop)` / `range(start, stop, step)` with
/// exact canonical machine-word ints: lower the opaque constructor residual
/// to a virtual `W_Range` and four virtual wrapped-int fields.  This lets the
/// existing GET_ITER specialization consume the range without forcing either
/// allocation.  All other callables and argument shapes fall through to the
/// generic residual.
pub(crate) fn try_walker_specialize_builtin_range<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !(3..=5).contains(&r_args.len()) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    let range_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::functional::RANGE_TYPE);
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !std::ptr::eq(concrete_callable, range_type)
    {
        return Ok(None);
    }

    let exact_int_class = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let mut concrete_args = Vec::with_capacity(r_args.len() - 2);
    let mut concrete_values = Vec::with_capacity(r_args.len() - 2);
    for concrete in &arg_concretes[2..] {
        let ConcreteValue::Ref(arg_obj) = *concrete else {
            return Ok(None);
        };
        if arg_obj.is_null()
            || unsafe {
                !std::ptr::eq((*arg_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
                    || !std::ptr::eq((*arg_obj).w_class, exact_int_class)
            }
        {
            return Ok(None);
        }
        concrete_args.push(arg_obj);
        concrete_values.push(unsafe { pyre_object::w_int_get_value(arg_obj) });
    }
    // Produce the authentic result before emitting IR, keeping every decline
    // point side-effect-free with respect to the trace under construction.
    let authentic_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &concrete_args)
    };
    if concrete_values.len() == 3 && concrete_values[2] == 0 {
        let Err(mut err) = authentic_result else {
            return Ok(None);
        };
        let exc = err.to_exc_object();
        let kind = pyre_object::interp_exceptions::ExcKind::ValueError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return Ok(None);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return Ok(None);
        };

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
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        let mut raw_args = Vec::with_capacity(concrete_values.len());
        for (&arg_op, &concrete_value) in r_args[2..].iter().zip(&concrete_values) {
            walker_guard_class(ctx, op.pc, arg_op, int_type_addr)?;
            walker_guard_exact_w_class(ctx, op.pc, arg_op, exact_int_class)?;
            let raw = crate::state::opimpl_getfield_gc_i(
                ctx.trace_ctx,
                arg_op,
                crate::descr::int_intval_descr(),
            );
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(concrete_value));
            raw_args.push(raw);
        }
        let step_raw = raw_args[2];
        let zero = ctx.trace_ctx.const_int(0);
        let is_zero = ctx.trace_ctx.record_op(OpCode::IntEq, &[step_raw, zero]);
        ctx.trace_ctx
            .set_opref_concrete(is_zero, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[is_zero])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }
    let Ok(authentic_range) = authentic_result else {
        return Ok(None);
    };
    let (authentic_start, authentic_stop, authentic_step) =
        unsafe { pyre_object::functional::w_range_fields(authentic_range) };
    let authentic_length = unsafe { pyre_object::functional::w_range_length(authentic_range) };
    let authentic_fields = [
        authentic_start,
        authentic_stop,
        authentic_step,
        authentic_length,
    ];
    if authentic_fields.iter().any(|&field| unsafe {
        !std::ptr::eq((*field).ob_type, &pyre_object::pyobject::INT_TYPE)
            || !std::ptr::eq((*field).w_class, exact_int_class)
    }) {
        return Ok(None);
    }
    let concrete_fields =
        authentic_fields.map(|field| unsafe { pyre_object::w_int_get_value(field) });
    let [
        concrete_start,
        concrete_stop,
        concrete_step,
        concrete_length,
    ] = concrete_fields;

    // The bound test below reads the unboxed `intval`, and that read is only
    // safe behind the class guards emitted here, so the decline cannot be
    // hoisted ahead of the emission — it rewinds instead.
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();

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

    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let int_type_const = ctx.trace_ctx.const_int(int_type_addr);
    let mut raw_args = Vec::with_capacity(concrete_values.len());
    for (&arg_op, &concrete_value) in r_args[2..].iter().zip(&concrete_values) {
        if !arg_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(arg_op) {
            ctx.trace_ctx
                .record_guard(OpCode::GuardClass, &[arg_op, int_type_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(arg_op, int_type_addr);
        walker_guard_exact_w_class(ctx, op.pc, arg_op, exact_int_class)?;
        let raw = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            arg_op,
            crate::descr::int_intval_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(concrete_value));
        raw_args.push(raw);
    }

    let zero = ctx.trace_ctx.const_int(0);
    let one = ctx.trace_ctx.const_int(1);
    let (start, stop, step) = match raw_args.as_slice() {
        [stop] => (zero, *stop, one),
        [start, stop] => (*start, *stop, one),
        [start, stop, step] => (*start, *stop, *step),
        _ => unreachable!("range arity gate admitted an invalid argument count"),
    };
    // Trace-constant bounds only.  This is what makes `length` sound as a
    // record-time constant: a bound that varies per iteration would pair a
    // stale length with fresh start/stop/step.  The alternative — emitting
    // `compute_range_length` — costs a division chain plus its overflow
    // guards, and it regressed two shapes the gate pins: a bound that
    // alternates empty and non-empty needs the emptiness folded in
    // branchlessly (a guard side-exits every other call), and the resulting op
    // run aborts the wasm trace when the `range` sits inside a self-recursive
    // callee that is inlined per level.  A variable bound keeps the residual
    // until that is worked out.
    if !start.is_constant() || !stop.is_constant() || !step.is_constant() {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    let length = ctx.trace_ctx.const_int(concrete_length);

    let new = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::w_range_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(new);

    let field_descrs = [
        crate::descr::range_start_descr(),
        crate::descr::range_stop_descr(),
        crate::descr::range_step_descr(),
        crate::descr::range_length_descr(),
    ];
    let raw_fields = [start, stop, step, length];
    for (((descr, raw), concrete_value), authentic_field) in field_descrs
        .into_iter()
        .zip(raw_fields)
        .zip(concrete_fields)
        .zip(authentic_fields)
    {
        let boxed = crate::state::wrapint(ctx.trace_ctx, raw);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(authentic_field as usize)),
        );
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new, boxed], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new, descr_index, boxed);
    }

    let range_type_addr = &pyre_object::functional::RANGE_TYPE as *const _ as i64;
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new, range_type_addr);
    ctx.trace_ctx.set_opref_concrete(
        new,
        majit_ir::Value::Ref(majit_ir::GcRef(authentic_range as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// Virtualize PyPy's exact `zip(tuple0, tuple1, strict=True)` allocation chain.
pub(crate) fn try_walker_specialize_builtin_zip<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // CALL_KW keeps the keyword-name tuple after all argument values:
    // [callable, null_or_self, p, q, strict, kwnames].
    if r_args.len() != 6 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let [
        ConcreteValue::Ref(callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(tuple0),
        ConcreteValue::Ref(tuple1),
        ConcreteValue::Ref(strict),
        ConcreteValue::Ref(kwnames),
    ] = arg_concretes.as_slice()
    else {
        return Ok(None);
    };
    let zip_callable = pyre_interpreter::typedef::gettypeobject(&pyre_object::functional::ZIP_TYPE);
    if callable.is_null()
        || !std::ptr::eq(*callable, zip_callable)
        || !null_or_self.is_null()
        || kwnames.is_null()
        || unsafe { !pyre_object::is_tuple(*kwnames) }
        || unsafe { pyre_object::w_tuple_len(*kwnames) } != 1
        || !std::ptr::eq(*strict, pyre_object::w_bool_from(true))
    {
        return Ok(None);
    }
    let Some(keyword) = (unsafe { pyre_object::w_tuple_getitem(*kwnames, 0) }) else {
        return Ok(None);
    };
    if unsafe {
        !pyre_object::is_str(keyword)
            || pyre_object::w_str_get_wtf8(keyword).as_str().ok() != Some("strict")
    } {
        return Ok(None);
    }
    let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType;
    let tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    for tuple in [*tuple0, *tuple1] {
        if tuple.is_null()
            || unsafe {
                !std::ptr::eq((*tuple).ob_type, tuple_type)
                    || !std::ptr::eq((*tuple).w_class, tuple_class)
            }
        {
            return Ok(None);
        }
    }

    // Recognition is complete; build the authentic shadow before emitting.
    let roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(*tuple0);
    pyre_object::gc_roots::pin_root(*tuple1);
    let concrete_iter0 = pyre_object::w_tuple_iter_new(unsafe {
        pyre_object::gc_roots::shadow_stack_get(root_base)
    });
    pyre_object::gc_roots::pin_root(concrete_iter0);
    let iter0_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let concrete_iter1 = pyre_object::w_tuple_iter_new(unsafe {
        pyre_object::gc_roots::shadow_stack_get(root_base + 1)
    });
    pyre_object::gc_roots::pin_root(concrete_iter1);
    let iter1_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let concrete_list = pyre_object::w_list_new(vec![
        unsafe { pyre_object::gc_roots::shadow_stack_get(iter0_slot) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(iter1_slot) },
    ]);
    pyre_object::gc_roots::pin_root(concrete_list);
    let list_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let concrete_zip = pyre_object::functional::w_zip_new(
        unsafe { pyre_object::gc_roots::shadow_stack_get(list_slot) },
        true,
    );
    if concrete_zip.is_null() {
        drop(roots);
        return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op.pc });
    }
    pyre_object::gc_roots::pin_root(concrete_zip);
    let zip_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    walker_guard_builtin_callable_identity(ctx, op.pc, r_args[0], *callable)?;
    for (arg_op, concrete) in [(r_args[5], *kwnames), (r_args[4], *strict)] {
        if !arg_op.is_constant() {
            let expected = ctx.trace_ctx.const_ref(concrete as i64);
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardValue, &[arg_op, expected])?;
            ctx.trace_ctx.heap_cache_mut().replace_box(arg_op, expected);
        }
    }

    let zero = ctx.trace_ctx.const_int(0);
    let mut iterator_ops = Vec::with_capacity(2);
    for (tuple_op, concrete_slot) in [(r_args[2], iter0_slot), (r_args[3], iter1_slot)] {
        walker_guard_class(ctx, op.pc, tuple_op, tuple_type as i64)?;
        walker_guard_exact_w_class(ctx, op.pc, tuple_op, tuple_class)?;
        let iterator = ctx.trace_ctx.record_op_with_descr(
            OpCode::NewWithVtable,
            &[],
            crate::descr::tuple_iter_size_descr(),
        );
        ctx.trace_ctx.heap_cache_mut().new_object(iterator);
        let seq_descr = crate::descr::tuple_iter_seq_descr();
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[iterator, tuple_op],
            seq_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(iterator, seq_descr.index(), tuple_op);
        let index_descr = crate::descr::tuple_iter_index_descr();
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[iterator, zero],
            index_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(iterator, index_descr.index(), zero);
        ctx.trace_ctx.heap_cache_mut().class_now_known(
            iterator,
            &pyre_object::iterobject::TUPLE_ITER_TYPE as *const _ as i64,
        );
        // Re-read after allocations that may move the shadow.
        let concrete_iter = unsafe { pyre_object::gc_roots::shadow_stack_get(concrete_slot) };
        ctx.trace_ctx.set_opref_concrete(
            iterator,
            Value::Ref(majit_ir::GcRef(concrete_iter as usize)),
        );
        iterator_ops.push(iterator);
    }

    let iterator_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &iterator_ops);
    ctx.trace_ctx.set_opref_concrete(
        iterator_list,
        Value::Ref(majit_ir::GcRef(
            unsafe { pyre_object::gc_roots::shadow_stack_get(list_slot) } as usize,
        )),
    );
    let zip = ctx.trace_ctx.record_op_with_descr(
        OpCode::NewWithVtable,
        &[],
        crate::descr::w_zip_size_descr(),
    );
    ctx.trace_ctx.heap_cache_mut().new_object(zip);
    for (value, descr) in [
        (iterator_list, crate::descr::zip_iterators_descr()),
        (ctx.trace_ctx.const_int(1), crate::descr::zip_strict_descr()),
        (zero, crate::descr::zip_iteration_progress_descr()),
    ] {
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[zip, value], descr.clone());
        ctx.trace_ctx
            .heapcache_setfield_cached(zip, descr.index(), value);
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(zip, &pyre_object::functional::ZIP_TYPE as *const _ as i64);
    ctx.trace_ctx.set_opref_concrete(
        zip,
        Value::Ref(majit_ir::GcRef(
            unsafe { pyre_object::gc_roots::shadow_stack_get(zip_slot) } as usize,
        )),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', zip)?;
    drop(roots);
    Ok(Some(()))
}

/// Unrolling bound for [`try_walker_specialize_builtin_locals`].
///
/// The modelled expansion is a straight-line unroll of `fast2locals`' slot
/// loop, one guard plus at most one store per fastlocal.  Upstream bounds the
/// same unroll by `@jit.unroll_safe` on a loop whose trip count is the code
/// object's `numlocals`; the explicit ceiling here keeps a pathologically
/// wide frame from turning one `locals()` into hundreds of trace ops.  Over
/// the bound the fold declines and the generic residual runs (SAFE).
const MAX_MODELLED_FASTLOCALS: usize = 32;

/// Which builtin [`try_walker_specialize_builtin_locals`] is standing in for.
///
/// All three resolve their frame through the same `topframe_for_locals` and
/// read the same fastlocals, so one modelled expansion serves them; they
/// differ only in what they make of the resulting mapping.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameLocalsBuiltin {
    /// `locals()` / `vars()` — the mapping itself is the result.
    Mapping,
    /// `dir()` — the mapping's sorted key set is the result.
    SortedNames,
}

/// Zero-argument `locals()` / `vars()` / `dir()` on the walk's own portal
/// frame: model
/// `pyframe.py:539-583 fast2locals` in the trace instead of residualizing
/// `interp_inspect.py:7-11 locals` → `pyframe.py:525-529 getdictscope`.
///
/// `fast2locals` is `@jit.unroll_safe`, and `policy.py:60-67` cancels
/// `contains_loop` for unroll_safe graphs, so upstream LOOKS INSIDE it: each
/// `self.locals_cells_stack_w[i]` lowers to `getarrayitem_vable_r`
/// (`jtransform.py:1877 do_fixed_list_getitem`), answered from
/// `metainterp.virtualizable_boxes`, and `jtransform.py:2164-2172
/// rewrite_op_jit_force_virtualizable` returns `[]` for a read the tracer is
/// inside.  There is no residual and no virtualizable force anywhere on the
/// upstream locals-read path.
///
/// Pyre residualizes the same read as one opaque `bh_call_fn(locals, PY_NULL)`
/// `CallMayForce`, which arms `virtualizable.py:281-291
/// force_virtualizable_if_necessary` for the whole call; the read barrier
/// `force_frame_before_locals_read` then clears `TOKEN_TRACING_RESCALL` and
/// `tracing_after_residual_call` reads that clear as an escape
/// (`VableEscapedDuringResidualCall`), losing the loop.  The deviation is the
/// residual BOUNDARY, not the barrier — `rvirtualizable.py:49-53` injects the
/// same hook on reads upstream and `pyjitpl.py:3373-3390` aborts
/// unconditionally on a detected force — so this removes the boundary and
/// leaves the barrier live for every shape it declines.
///
/// Emitted shape, mirroring `pyframe.py:555-574` line by line:
/// `guard_value(callable)`; the frame's own mapping, read as
/// `getorcreatedebug()` (the `debugdata` virtualizable field, answered from
/// `virtualizable_boxes`) followed by `getfield_gc_r(w_locals)` under a
/// non-null and exact-dict guard; one `getarrayitem_vable_r(frame,
/// ConstInt(i))` per fastlocal (the same lowering `emit_load_fast_ref!`
/// already emits for LOAD_FAST); a `guard_isnull` / `guard_nonnull` pinning
/// the slot's bound-ness; and a plain non-forcing `Call` per slot —
/// `setitem_str` when bound, `delitem` when not.  None of those ops can reach
/// `force_frame`, so nothing arms the vable protocol.
///
/// The mapping is the FRAME's whenever the frame already carries one:
/// `fast2locals` rewrites only the varname keys, so a foreign key — one an
/// `f_locals` write put there (PEP 667) — survives every call, and an
/// expansion that always started from an empty dict would drop it for as long
/// as the loop stayed compiled.  A frame that carries none keeps the empty
/// `newdict` (pyframe.py:557) the residual would have materialised, under a
/// `guard_isnull` that side-exits if one appears mid-loop; nothing else
/// references that dict, so it is already the independent copy
/// `frame_locals_snapshot` hands back and its `delitem` arm is a no-op.
///
/// One further non-forcing `Call` turns that mapping into the published
/// result: `jit_locals_dict_snapshot` (the independent PEP 667 copy
/// `frame_locals_snapshot` builds) for `locals()` / `vars()`, and
/// `jit_dir_names_from_locals` (the split-out tail of `builtin_dir`'s
/// no-argument path, which reads `getdictscope` rather than the copy) for
/// `dir()`.  Both take the mapping and not the frame, so they too cannot
/// reach `force_frame`.
///
/// Returns `None` (fall through to the generic residual, SAFE — exactly
/// today's behaviour) for every other shape: a rebound `locals` / `vars` /
/// `dir` name,
/// a bound receiver, any argument, an inline sub-walk, a frame that is not the
/// standard virtualizable the boxes describe, a hidden top frame, a
/// non-OPTIMIZED (module / class / exec) frame, cellvars / freevars /
/// `CO_FAST_HIDDEN` slots, a slot the shadow cannot answer with a Ref, a frame
/// wider than [`MAX_MODELLED_FASTLOCALS`], a shadow whose mapping is not the
/// frame's, and a frame-owned mapping that is not an exact dict.
pub(crate) fn try_walker_specialize_builtin_locals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain zero-argument `bh_call_fn(callable, PY_NULL)` shape only.
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl` prepends
    // as arg0 — not a plain `locals()` call.
    if concrete_callable.is_null() || !null_or_self.is_null() {
        return Ok(None);
    }
    // `vars()` with no argument delegates straight to `builtin_locals`
    // (`app_inspect.py:21-24`), so both names share the fold; `vars(obj)` and
    // `dir(obj)` carry an extra operand and are already excluded by the arity
    // gate.
    let fold = if pyre_interpreter::builtins::is_builtin_locals_function(concrete_callable)
        || pyre_interpreter::builtins::is_builtin_vars_function(concrete_callable)
    {
        FrameLocalsBuiltin::Mapping
    } else if pyre_interpreter::builtins::is_builtin_dir_function(concrete_callable) {
        FrameLocalsBuiltin::SortedNames
    } else {
        return Ok(None);
    };
    // The modelled reads answer from `virtualizable_boxes`, which describe the
    // PORTAL frame only.  An inline sub-walk publishes a different concrete
    // frame whose locals live in the callee shadow, not in those boxes.
    if ctx.fbw_mode.inline_subwalk || current_inline_concrete_frame() != 0 {
        return Ok(None);
    }
    let (Some(vable_op), Some(vable_ptr)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
    ) else {
        return Ok(None);
    };
    // The frame `locals()` reports on is `ec.gettopframe_nohidden()`
    // (`interp_inspect.py:7-11`).  Resolve it the same way and require it to BE
    // the standard virtualizable: a hidden portal frame, or any deeper frame
    // handed out through the backref chain, resolves elsewhere and declines.
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return Ok(None);
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() || frame as usize != vable_ptr {
        return Ok(None);
    }
    let frame_ref = unsafe { &*frame };
    let code_ptr = unsafe { pyre_interpreter::pyframe::pyframe_get_pycode(frame_ref) };
    let code_obj = unsafe { &*code_ptr };
    if !pyre_interpreter::PyFrame::code_locals_are_plain_fastlocals(code_obj) {
        return Ok(None);
    }
    let numlocals = code_obj.varnames.len();
    if numlocals > MAX_MODELLED_FASTLOCALS {
        return Ok(None);
    }
    // `locals_cells_stack_w` is PyFrame's only virtualizable array
    // (`virtualizable_gen.rs arrays`), so array index 0 names it.
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        return Ok(None);
    };
    let Some(lengths) = ctx
        .trace_ctx
        .virtualizable_array_lengths()
        .map(<[usize]>::to_vec)
    else {
        return Ok(None);
    };
    if info.num_arrays() != 1 || lengths.first().copied().unwrap_or(0) < numlocals {
        return Ok(None);
    }
    let (Some(fdescr), Some(adescr)) = (
        info.array_field_descrs().first().cloned(),
        info.array_descrs.first().cloned(),
    ) else {
        return Ok(None);
    };
    // `fast2locals` opens on `self.getorcreatedebug()` (pyframe.py:555) and
    // writes into ITS `w_locals`: the mapping is the FRAME's, carried across
    // calls, so a key written through `f_locals` outlives every `fast2locals`
    // that does not name it.  `debugdata` is a virtualizable field, so the read
    // answers from `virtualizable_boxes` and records no op — exactly like the
    // slot reads below — and the frame never becomes an operand, so nothing
    // here can reach `force_frame`.
    let Some((debugdata_op, majit_ir::Value::Ref(debugdata_ref))) = ctx
        .trace_ctx
        .virtualizable_entry_at(crate::virtualizable_spec::DEBUGDATA_VABLE_FIELD_INDEX)
    else {
        return Ok(None);
    };
    // Read the mapping through the SHADOW's payload, which is what the emitted
    // `getfield_gc_r` reads, and require it to be the one the residual would
    // have used.  The two payloads are not the same object: a root portal seed
    // bakes the vable identity against the live frame but expands the shadow
    // from the `snapshot_for_tracing` copy, whose `clone_debugdata_ptr` hands
    // out a fresh `FrameDebugData` around the same `w_locals`.  Comparing the
    // holders would decline every portal trace; comparing the mapping is the
    // invariant that actually has to hold.
    let shadow_debugdata =
        debugdata_ref.as_usize() as *const pyre_interpreter::pyframe::FrameDebugData;
    let w_locals = if shadow_debugdata.is_null() {
        pyre_object::PY_NULL
    } else {
        unsafe { (*shadow_debugdata).w_locals }
    };
    if !std::ptr::eq(w_locals, frame_ref.get_w_locals()) {
        return Ok(None);
    }
    // Two shapes, each pinned by a guard so the compiled loop side-exits when
    // the frame moves to the other one:
    //
    // * the frame already carries its mapping — rewrite THAT, so a foreign key
    //   an `f_locals` write left in it survives, as it does across the
    //   residual's `fast2locals`;
    // * the frame carries none — `fast2locals` would materialise an empty dict
    //   (pyframe.py:556-557 `d.w_locals = space.newdict()`) and fill it from
    //   the fastlocals, and the expansion builds exactly that dict instead of
    //   modelling the store.  Nothing else references it, so it is already the
    //   independent copy `frame_locals_snapshot` would hand back, and a
    //   `delitem` on a key it never held is a no-op.
    //
    // The slot helpers are dict-keyed, so a frame-owned mapping that is not an
    // exact dict declines.
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    let frame_owned = !w_locals.is_null();
    if frame_owned
        && (canonical_dict.is_null()
            || !unsafe {
                std::ptr::eq((*w_locals).ob_type, &pyre_object::pyobject::DICT_TYPE)
                    && std::ptr::eq((*w_locals).w_class, canonical_dict)
            })
    {
        return Ok(None);
    }
    // Resolve every slot's shadow entry BEFORE emitting anything, so a slot the
    // shadow cannot answer declines from a clean trace position.  The read is
    // the standard-virtualizable arm of `_opimpl_getarrayitem_vable`
    // (`virtualizable_boxes[index]`), which records no op — the emit pass below
    // re-runs it through the real entry point.
    let mut slots: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(numlocals);
    for i in 0..numlocals {
        let flat = info.get_index_in_array(0, i, &lengths);
        let Some((slot_op, entry_value)) = ctx.trace_ctx.virtualizable_entry_at(flat) else {
            return Ok(None);
        };
        // The value comes from the SHADOW, never from `locals_w!(frame)`.  An
        // unsynchronized virtualizable's heap array holds whatever the frame
        // last wrote out — measured one FOR_ITER iteration behind on the loop
        // variable — which is exactly the staleness the read barrier's
        // `force_now` repairs before the residual reads it.  The shadow already
        // holds the repaired value, so sourcing from it reproduces the forced
        // residual's answer without the force, and it is what upstream's
        // traced-in `fast2locals` reads (`getarrayitem_vable_r` answered from
        // `virtualizable_boxes`).
        //
        // Prefer the OpRef's own concrete over the `virtualizable_values` copy:
        // the op table is the GC-forwarded channel, so a Ref that moved across
        // an earlier residual is current there.
        let value = match ctx
            .trace_ctx
            .concrete_of_opref(slot_op)
            .filter(|v| matches!(v, majit_ir::Value::Ref(_)))
            .unwrap_or(entry_value)
        {
            majit_ir::Value::Ref(gcref) => gcref.as_usize() as pyre_object::PyObjectRef,
            _ => return Ok(None),
        };
        slots.push(value);
    }

    // Which helper turns the mapping into the published result.  `dir()` reads
    // `getdictscope` — the mapping itself — through `builtin_dir`'s split-out
    // sorted-key-set tail.  `locals()` / `vars()` hand back
    // `frame_locals_snapshot`'s independent PEP 667 copy, which a mapping the
    // expansion just built for itself already is.
    let tail_fn: Option<extern "C" fn(i64) -> i64> = match fold {
        FrameLocalsBuiltin::Mapping if frame_owned => {
            Some(pyre_interpreter::pyframe::jit_locals_dict_snapshot)
        }
        FrameLocalsBuiltin::Mapping => None,
        FrameLocalsBuiltin::SortedNames => {
            Some(pyre_interpreter::builtins::jit_dir_names_from_locals)
        }
    };
    // Authentic mapping, built on the plain eval loop exactly as the skipped
    // residual would — through the SAME helpers the emitted calls invoke, so
    // the recording-time value and the compiled loop's value cannot diverge.
    // On the frame-owned arm this MUTATES the frame's own mapping, which is
    // exactly what the residual `fast2locals` does; the rewrite is a pure
    // function of the fastlocals, so a decline below — or a discarded walk —
    // leaves the residual free to redo it with the same outcome.
    let (concrete_locals, concrete_result) = {
        let _roots = pyre_object::gc_roots::push_roots();
        let locals_root = pyre_object::gc_roots::shadow_stack_len();
        // Re-read rather than reuse the gate's `w_locals`: the slot resolution
        // above sits between the two, so the pin takes the address the frame
        // holds NOW.
        pyre_object::gc_roots::pin_root(if frame_owned {
            frame_ref.get_w_locals()
        } else {
            unsafe { pyre_object::w_dict_new() }
        });
        let value_roots: Vec<usize> = slots
            .iter()
            .map(|&value| {
                let slot = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(value);
                slot
            })
            .collect();
        let mut result = pyre_object::PY_NULL;
        let mut slot_failed = false;
        for (i, &value_root) in value_roots.iter().enumerate() {
            let value = pyre_object::gc_roots::shadow_stack_get(value_root);
            let locals = pyre_object::gc_roots::shadow_stack_get(locals_root) as i64;
            // pyframe.py:566-574 — a bound slot is stored, an unbound one
            // deleted.  Both allocate, so the mapping is re-read from its
            // pinned slot on every pass.  A fresh mapping never held the key,
            // so its `delitem` arm is skipped rather than emitted.
            let updated = if !value.is_null() {
                pyre_interpreter::pyframe::jit_locals_dict_setitem_local(
                    locals,
                    code_ptr as i64,
                    i as i64,
                    value as i64,
                )
            } else if frame_owned {
                pyre_interpreter::pyframe::jit_locals_dict_delitem_local(
                    locals,
                    code_ptr as i64,
                    i as i64,
                )
            } else {
                locals
            };
            if (updated as pyre_object::PyObjectRef).is_null() {
                slot_failed = true;
                break;
            }
        }
        if !slot_failed {
            let locals = pyre_object::gc_roots::shadow_stack_get(locals_root);
            // The tail runs here too, so the recorded result is produced by the
            // very helper the emitted call names.
            result = match tail_fn {
                Some(tail) => tail(locals as i64) as pyre_object::PyObjectRef,
                None => locals,
            };
        }
        (pyre_object::gc_roots::shadow_stack_get(locals_root), result)
    };
    // A slot rewrite or the tail reports a failure as PY_NULL instead of
    // publishing it; nothing has been emitted yet, so decline and let the
    // residual raise.
    if concrete_result.is_null() {
        return Ok(None);
    }
    let concrete_locals_value = majit_ir::Value::Ref(majit_ir::GcRef(concrete_locals as usize));

    // --- emit the specialized IR (walker-native) ---
    // Pin the callable identity (LOAD_GLOBAL `locals` is usually already a
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
    // The code object is the jitdriver green this trace is keyed on
    // (`interp_jit.py:23 greens = ['next_instr', 'is_being_profiled',
    // 'pycode']`), so its address is a constant for the compiled loop and
    // carries no guard of its own.
    let code_const = ctx.trace_ctx.const_int(code_ptr as i64);
    // `d = self.getorcreatedebug()` — pyframe.py:555.  An absent payload has no
    // `w_locals` to read, so the guard pins that direction and the fresh-dict
    // arm below stands in for the materialisation.
    let debugdata_present = debugdata_ref.as_usize() != 0;
    if !debugdata_op.is_constant() {
        let opcode = if debugdata_present {
            OpCode::GuardNonnull
        } else {
            OpCode::GuardIsnull
        };
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, opcode, &[debugdata_op])?;
    }
    // `d.w_locals` — pyframe.py:556.  Read whenever there is a payload to read
    // it from, and guarded in the direction recorded, so a frame that
    // materialises its mapping mid-loop side-exits instead of going on writing
    // into the expansion's own dict.
    let mut field_op = None;
    if debugdata_present {
        let op_ref = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            debugdata_op,
            crate::descr::frame_debug_data_w_locals_descr(),
        );
        if !op_ref.is_constant() {
            let opcode = if frame_owned {
                OpCode::GuardNonnull
            } else {
                OpCode::GuardIsnull
            };
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, opcode, &[op_ref])?;
        }
        field_op = Some(op_ref);
    }
    let mut dict_op = match field_op.filter(|_| frame_owned) {
        Some(op_ref) => op_ref,
        // pyframe.py:557 `self.space.newdict(instance=True)` — the mapping
        // `fast2locals` would have materialised, built here instead of
        // modelling the store back into the debug payload.
        None => ctx.trace_ctx.call_ref_typed_with_effect(
            pyre_interpreter::pyframe::jit_locals_dict_new as *const (),
            &[],
            &[],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        ),
    };
    ctx.trace_ctx
        .set_opref_concrete(dict_op, concrete_locals_value);
    if frame_owned {
        walker_guard_class(
            ctx,
            op.pc,
            dict_op,
            &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
        )?;
        walker_guard_exact_w_class(
            ctx,
            op.pc,
            dict_op,
            // Re-derived rather than reusing the gate's binding: the
            // record-time rewrite above allocates, so this takes the address
            // `dict` has NOW.
            pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE),
        )?;
    }
    for (i, &value) in slots.iter().enumerate() {
        // `self.locals_cells_stack_w[i]` — `jtransform.py:1877
        // do_fixed_list_getitem`, the identical lowering `emit_load_fast_ref!`
        // emits for LOAD_FAST.  On the standard virtualizable this resolves to
        // `virtualizable_boxes[index]` and records no op.
        let index_const = ctx.trace_ctx.const_int(i as i64);
        let (slot_op, _) = ctx.trace_ctx.vable_getarrayitem_ref_indexed(
            op.pc,
            vable_op,
            index_const,
            i as i64,
            fdescr.clone(),
            adescr.clone(),
        );
        // `pyframe.py:566-571` branches on the slot being bound; pin the
        // direction so a slot that changes bound-ness side-exits instead of
        // publishing a mapping with the wrong key set.  A slot the trace
        // already holds as a constant needs no guard.
        let bound = !value.is_null();
        if !slot_op.is_constant() {
            let opcode = if bound {
                OpCode::GuardNonnull
            } else {
                OpCode::GuardIsnull
            };
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, opcode, &[slot_op])?;
        }
        // `pyframe.py:566-574` — a bound slot is stored, an unbound one
        // deleted.  The delete is what keeps a key from a since-unbound local
        // out of a mapping the frame carries across calls; on the fresh arm
        // the mapping never held the key, so it is skipped.
        if !bound && !frame_owned {
            continue;
        }
        let (helper, args, arg_types): (_, Vec<OpRef>, Vec<majit_ir::Type>) = if bound {
            (
                pyre_interpreter::pyframe::jit_locals_dict_setitem_local as *const (),
                vec![dict_op, code_const, index_const, slot_op],
                vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                    majit_ir::Type::Ref,
                ],
            )
        } else {
            (
                pyre_interpreter::pyframe::jit_locals_dict_delitem_local as *const (),
                vec![dict_op, code_const, index_const],
                vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ],
            )
        };
        dict_op = ctx.trace_ctx.call_ref_typed_with_effect(
            helper,
            &args,
            &arg_types,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        // Every link of the chain names the SAME mapping, so the post-build
        // address is the live one for all of them.
        ctx.trace_ctx
            .set_opref_concrete(dict_op, concrete_locals_value);
        if !bound {
            // The delete reports a raising comparison as PY_NULL instead of
            // publishing it; side-exit so the residual re-runs and raises.
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[dict_op])?;
        }
    }
    // `frame_locals_snapshot`'s PEP 667 copy for `locals()` / `vars()`, or
    // `builtin_dir`'s no-argument tail for `dir()` — each split out so the
    // trace and the eval loop run one implementation.  Both report a failure
    // as PY_NULL instead of publishing it, so the guarded side exit re-runs
    // the residual and raises from the eval loop.
    let result_op = match tail_fn {
        Some(tail) => {
            let op_ref = ctx.trace_ctx.call_ref_typed_with_effect(
                tail as *const (),
                &[dict_op],
                &[majit_ir::Type::Ref],
                majit_ir::EffectInfo::new(
                    majit_ir::ExtraEffect::CannotRaise,
                    majit_ir::OopSpecIndex::None,
                ),
            );
            ctx.trace_ctx.set_opref_concrete(
                op_ref,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_result as usize)),
            );
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[op_ref])?;
            op_ref
        }
        None => dict_op,
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result_op)?;
    Ok(Some(()))
}

/// `sys._getframe()` / `sys._getframe(0)` at the top walk level: publish the
/// portal virtualizable itself instead of residualizing `vm.py:42-54 getframe`.
///
/// `getframe` is `@jit.look_inside_iff(lambda space, depth:
/// jit.isconstant(depth))` (`pypy/module/sys/vm.py:41`), so a constant depth is
/// traced THROUGH: `ec.gettopframe_nohidden()` is a vref read that
/// `pyjitpl.py:2153-2172 _do_jit_force_virtual` answers with
/// `virtualizable_boxes[-1]` under a `ptr_eq` + `implement_guard_value`, the
/// `depth == 0` test folds away, and `mark_as_escaped` is one `setfield_gc`.
/// No call and no virtualizable force anywhere — pypy3 reports `forcings: 0`
/// and `abort: vable escape: 0` on the fixtures where pyre loses the loop.
///
/// Pyre residualizes the same walk as one opaque `bh_call_fn(_getframe,
/// PY_NULL, depth)` `CallMayForce`, and [`pyre_interpreter::module::sys::vm::getframe`]'s
/// `force_frame` on the frame it returns — the stand-in for the injection
/// `rvirtualizable.py:49-53 hook_access_field` performs and pyre's rtyper
/// cannot build — clears `TOKEN_TRACING_RESCALL` inside that call whenever the
/// returned frame is the traced one, which `tracing_after_residual_call` reads
/// as an escape (`VableEscapedDuringResidualCall`).  At depth 0 the returned
/// frame is always the portal, so the residual always escapes.  Removing it
/// removes the force with it, and nothing has to replace it: `last_instr` is
/// published onto the portal frame at every may-force boundary
/// (`LiveLastInstrGuard`), and every getset that reads a virtualizable field
/// off the handed-out frame (`f_locals`, and `f_lasti` / `f_lineno` through
/// their own `jit_getattr` residual) is itself such a boundary.  Of those only
/// `f_locals` also FORCES: measured against this fold, swapping a fixture's
/// forcing read for `f_lasti` or `f_lineno` leaves `loops_aborted` at 0.
///
/// Emitted shape, following `getframe`'s body line by line:
/// `guard_value(callable)`; `guard_class` + exact-class + `getfield_gc_i` on
/// the depth box, whose resulting RAW int must be a trace constant — that
/// unboxed value is what `jit.isconstant(depth)` tests upstream, where
/// `@unwrap_spec(depth=int)` has already run OUTSIDE the looked-inside graph
/// (the wrapped `W_IntObject` the residual receives is built in-trace by
/// `NewWithVtable` + `SetfieldGc` and is never constant, so testing the box
/// declines 100% of the time); `getfield_gc_r(frame, execution_context)` +
/// `getfield_gc_r(ec, topframeref)` + `ptr_eq` + `guard_true`, the port of
/// `_do_jit_force_virtual`'s identity check; and one non-forcing void `Call`
/// for `mark_as_escaped`.  At depth 0 the result IS
/// `standard_virtualizable_box()`, exactly as `_do_jit_force_virtual` returns
/// `standard_box`.
///
/// For constant depth >= 1, the same top-frame proof seeds the walk, then each
/// hop emits the `ExecutionContext.getnextframe_nohidden(frame)` body shape:
/// raw `frame.f_backref` read, the residual `OS_JIT_FORCE_VIRTUAL`
/// `CallMayForceR(jit_force_vref, raw)` bracketed by FORCE_TOKEN/SETFIELD_GC and
/// `GuardNotForced`, `GuardNonnull` for the "call stack is not deep enough"
/// arm, then `frame.hide()` as `frame.pycode.hidden_applevel` with a
/// `GuardFalse`.  The residual force stays in the trace: `pyjitpl.py:2153-2172
/// _do_jit_force_virtual` returns `None` for a known non-standard
/// virtualizable, and that `None` result is precisely the caller's signal to
/// emit the residual `jit_force_virtual` call.  `optimize_jit_force_virtual`
/// only elides it later for a trace Virtual, matching upstream and giving the
/// cheap per-hop cost instead of the full `_getframe` residual.
///
/// The guard reads `topframeref` raw, without the vref force or the
/// hidden-frame walk `gettopframe_nohidden` performs, so the gate below
/// requires the record-time chain to need neither: `topframeref` must BE the
/// portal pointer (not a `JitVirtualRef` naming it) and the nohidden walk must
/// land on the same frame.  Any other chain declines, and at runtime a
/// `topframeref` that stops matching side-exits.
///
/// Returns `None` (fall through to the generic residual, SAFE — exactly
/// today's behaviour) for every other shape: a rebound `sys._getframe`, a
/// bound receiver, a negative / non-int / inexact / non-constant depth, an
/// inline sub-walk, a walk with no standard virtualizable, armed audit hooks,
/// a `topframeref` / `gettopframe_nohidden()` mismatch with the portal frame,
/// a hop whose forced `f_backref` is null, or a hop whose result is hidden.
/// Declines after emission rewind to the pre-specialization trace position and
/// reset the heap cache before falling through.
///
/// ⛔ The TOP walk level is the only level this may take, at any depth. Inside
/// an inline sub-walk depth 0 names the callee's virtual frame, whose
/// `last_instr` is still the `-1` its constructor wrote and which nothing
/// updates through the inlined body (`jitcode_dispatch/mod.rs` says so in the
/// tree); depth > 0 would start the hop chain from that same frame.  The
/// sub-walk gate is what makes "depth 0 == the portal" true rather than
/// assumed.
pub(crate) fn try_walker_specialize_sys_getframe<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // `sys._getframe()` (2) or `sys._getframe(depth)` (3).
    if !(2..=3).contains(&r_args.len()) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl` prepends
    // as arg0, not a plain `sys._getframe(...)` call.
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !pyre_interpreter::module::sys::vm::is_builtin_getframe_function(concrete_callable)
    {
        return Ok(None);
    }
    // The depth has to be an exact plain non-negative int before anything is
    // emitted; the guards below pin both facts for the compiled loop.
    let exact_int_class = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let (depth_arg, depth_value) = if r_args.len() == 3 {
        let ConcreteValue::Ref(depth_obj) = arg_concretes[2] else {
            return Ok(None);
        };
        if depth_obj.is_null()
            || unsafe {
                !std::ptr::eq((*depth_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
                    || !std::ptr::eq((*depth_obj).w_class, exact_int_class)
            }
        {
            return Ok(None);
        }
        let depth = unsafe { pyre_object::w_int_get_value(depth_obj) };
        if depth < 0 {
            return Ok(None);
        }
        (Some(r_args[2]), depth)
    } else {
        (None, 0)
    };
    // `virtualizable_boxes` describe the PORTAL frame only.  An inline sub-walk
    // publishes a different concrete frame, so depth 0 there names the callee —
    // the level this arm must not take.
    if ctx.fbw_mode.inline_subwalk || current_inline_concrete_frame() != 0 {
        return Ok(None);
    }
    // `vm.py:51 audit(space, "sys._getframe", [f])`.  With no hook installed
    // `audit` takes its `holder.hooks_w is None` early-out (`vm.py:481`) and the
    // event costs nothing; the emission below pins that read so a later
    // `addaudithook` revokes this loop instead of silently missing the event.
    // With a hook already installed the event reaches `trigger_audit_events`,
    // which is `@objectmodel.dont_inline` — a residual call this arm has no
    // channel for — so it declines and the generic residual `getframe` emits.
    let audit_holder = pyre_interpreter::module::sys::vm::audit_holder_ptr();
    if audit_holder.is_null() || pyre_interpreter::module::sys::vm::audit_hooks_armed() {
        return Ok(None);
    }
    let (Some(vable_op), Some(vable_ptr)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
    ) else {
        return Ok(None);
    };
    let ec =
        pyre_interpreter::call::getexecutioncontext() as *mut pyre_interpreter::PyExecutionContext;
    if ec.is_null() {
        return Ok(None);
    }
    // The emitted guard compares the RAW `topframeref` against the portal, so
    // require the record-time chain to make that comparison equivalent to
    // `getframe`'s own resolution: the slot holds the frame pointer itself
    // (an inlined callee's `JitVirtualRef` would decline here, and so would a
    // deeper portal), and the hidden-frame walk lands on that same frame.
    if unsafe { (*ec).topframeref } as usize != vable_ptr {
        return Ok(None);
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() || frame as usize != vable_ptr {
        return Ok(None);
    }

    // --- emit the specialized IR (walker-native) ---
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();

    // `sys` is an ordinary mutable module, so nothing else keeps the name bound
    // to this builtin across iterations.
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
    // `@unwrap_spec(depth=int)` and then `jit.isconstant(depth)`: unbox first,
    // and require the UNBOXED value to be the trace constant.  The unbox is
    // only sound behind the class guards, so the constness decline rewinds
    // rather than being hoisted above them.
    if let Some(depth_op) = depth_arg {
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        walker_guard_class(ctx, op.pc, depth_op, int_type_addr)?;
        walker_guard_exact_w_class(ctx, op.pc, depth_op, exact_int_class)?;
        let raw = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            depth_op,
            crate::descr::int_intval_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(depth_value));
        if !raw.is_constant() {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
    }
    // `ec = space.getexecutioncontext()` — recovered off the portal frame, the
    // same route `walker_ec_enter` takes (`inline_call.rs`), since the outer
    // frame's `execution_context` is always the true one.
    let ec_op = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[vable_op],
        crate::descr::pyframe_execution_context_descr(),
    );
    ctx.trace_ctx
        .set_opref_concrete(ec_op, majit_ir::Value::Ref(majit_ir::GcRef(ec as usize)));
    // `f = ec.gettopframe_nohidden()` followed by `pyjitpl.py:2166-2168`'s
    // `ptr_eq(vref_box, standard_box)` + `implement_guard_value`: the identity
    // this arm resolved at record time, re-checked every compiled iteration.
    let topframeref_op = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec_op],
        crate::descr::ec_topframeref_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        topframeref_op,
        majit_ir::Value::Ref(majit_ir::GcRef(vable_ptr)),
    );
    let is_standard = ctx
        .trace_ctx
        .record_op(OpCode::PtrEq, &[topframeref_op, vable_op]);
    ctx.trace_ctx
        .set_opref_concrete(is_standard, majit_ir::Value::Int(1));
    walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[is_standard])?;

    let mut cur_op = vable_op;
    let mut cur_ptr = frame;
    for _ in 0..depth_value {
        let raw_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            cur_op,
            crate::descr::pyframe_f_backref_descr(),
        );
        let raw_ptr = unsafe { (*cur_ptr).f_backref };
        ctx.trace_ctx.set_opref_concrete(
            raw_op,
            majit_ir::Value::Ref(majit_ir::GcRef(raw_ptr as usize)),
        );

        let next_ptr = pyre_interpreter::executioncontext::force_vref(raw_ptr);
        if next_ptr.is_null() || unsafe { (*next_ptr).hide() } {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }

        maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
        let force_fn = crate::helpers::jit_force_vref as *const ();
        let next_op = ctx.trace_ctx.call_typed_with_effect(
            OpCode::CallMayForceR,
            force_fn,
            &[raw_op],
            &[majit_ir::Type::Ref],
            majit_ir::Type::Ref,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::ForcesVirtualOrVirtualizable,
                majit_ir::OopSpecIndex::JitForceVirtual,
            ),
        );
        ctx.trace_ctx.set_opref_concrete(
            next_op,
            majit_ir::Value::Ref(majit_ir::GcRef(next_ptr as usize)),
        );
        // `pyjitpl.py:2163-2165` short-circuits a known non-standard
        // virtualizable to `None`; the caller turns that into this residual
        // `jit_force_virtual` call, so the ptr_eq/guard_value half is not
        // emitted for hop results.
        ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[next_op])?;

        let code_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            next_op,
            crate::descr::pyframe_code_descr(),
        );
        let code_ptr = unsafe { (*next_ptr).pycode };
        ctx.trace_ctx.set_opref_concrete(
            code_op,
            majit_ir::Value::Ref(majit_ir::GcRef(code_ptr as usize)),
        );
        // `optimizer.py:464-480` reaches for `descr.get_parent_descr()` only
        // when arg0 carries no pointer info yet; a preceding `GUARD_CLASS`
        // gives it `info.InstancePtrInfo()` and that lookup never runs.  The
        // code descrs are deliberately standalone rather than a positional
        // group (see `descr.rs` on `PYCODE_CODE_PTR_FIELD_DESCR`: a `PyCode` is
        // never allocated from a trace, and a partial layout under the live
        // `W_CODE_GC_TYPE_ID` would double-answer the collector's registry), so
        // their weak `parent_descr` is `None` by design and the guard is what
        // makes this read legal — the same order the code-field arm of
        // [`try_walker_specialize_traceback_walk`] uses.
        let code_type_addr = &pyre_interpreter::pycode::CODE_TYPE as *const _ as i64;
        if code_ptr.is_null()
            || unsafe {
                !std::ptr::eq(
                    (*code_ptr.cast::<pyre_object::PyObject>()).ob_type,
                    &pyre_interpreter::pycode::CODE_TYPE,
                )
            }
        {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        walker_guard_class(ctx, op.pc, code_op, code_type_addr)?;
        let hidden_op = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            code_op,
            crate::descr::pycode_hidden_applevel_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(hidden_op, majit_ir::Value::Int(0));
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardFalse, &[hidden_op])?;

        cur_op = next_op;
        cur_ptr = next_ptr;
    }

    // `f.mark_as_escaped()` — vm.py:54.  `escaped` is not one of the six fields
    // `interp_jit.py:25-30` declares, so the store cannot force; it is
    // load-bearing at `executioncontext.py:99-106 leave`, which forces the
    // leaving frame's own vref only for a frame that escaped.  Upstream traces
    // it as the ordinary `setfield_gc` on the flag, so it is emitted as the
    // read/or/store the `tb_frame` fold above already uses — an opaque call
    // would hide the update from the optimizer and its heap cache.
    let flags_descr = crate::descr::pyframe_flags_descr();
    let live_flags = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, cur_op, flags_descr.clone());
    let escaped_bit = ctx
        .trace_ctx
        .const_int(i64::from(pyre_interpreter::PyFrame::FLAG_ESCAPED));
    let new_flags = ctx
        .trace_ctx
        .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[cur_op, new_flags],
        flags_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(cur_op, flags_descr.index(), new_flags);
    // The walk IS the interpreter running, so the recorded store has to take
    // effect here too — the residual would have applied it before returning.
    unsafe { (*cur_ptr).mark_as_escaped() };

    // `audit(space, "sys._getframe", [f])` — vm.py:51.  The gate above resolved
    // it to the no-hook early-out, so all that is emitted is the marker for the
    // read that reached that conclusion.
    walker_pin_audit_hooks(ctx, op.pc, audit_holder)?;

    // `return f` — at depth 0 `cur_op` is still the standard virtualizable
    // `_do_jit_force_virtual` hands back as `standard_box`; each hop above
    // advanced it to the frame the walk settled on.
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', cur_op)?;
    Ok(Some(()))
}

/// `math.sqrt(x)` on an exact int/float argument: inline the domain-guarded
/// pure `CALL_F(sqrt_nonneg_jit)` (ll_math.rs `ll_math_sqrt` → `sqrt_nonneg`,
/// EF_ELIDABLE_CANNOT_RAISE) instead of the opaque
/// `bh_call_fn(sqrt_builtin, NULL, x)` residual, so the result `W_FloatObject`
/// virtualizes.  Two guards pin the branches of `ll_math_sqrt`: `x >= 0` (the
/// ValueError direction) and `isfinite(x)` (NaN/±inf take the residual).  A
/// negative argument raises in the concrete pre-exec below and declines to the
/// generic residual (which records the raise).  Any non-matching shape falls
/// through to the generic residual (SAFE).
pub(crate) fn try_walker_specialize_math_sqrt<Sym: WalkSym>(
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
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl` prepends
    // as arg0 — not a plain `sqrt(x)` call.
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    if !pyre_interpreter::module::math::interp_math::is_math_sqrt_function(concrete_callable) {
        return Ok(None);
    }
    // Exact int/bool/float argument only — a numeric subclass keeps the builtin
    // `ob_type` layout but a Python-visible `w_class`, and the `guard_class`
    // the coercion emits reads `ob_type`, so it would not catch the subclass.
    let (is_int, val) = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) {
            return Ok(None);
        }
        // `bool` shares `W_IntObject`'s `intval`; it coerces through the int arm
        // via its own `&BOOL_TYPE` guard inside `walker_coerce_operand_to_float`.
        if pyre_object::is_int(arg_obj) {
            (true, pyre_object::w_int_get_value(arg_obj) as f64)
        } else if pyre_object::is_float(arg_obj) {
            (false, pyre_object::w_float_get_value(arg_obj))
        } else {
            return Ok(None);
        }
    };
    // Cold domain: NaN/±inf and negatives take the opaque residual (the
    // negative direction also raises in the concrete pre-exec below).
    if !(val.is_finite() && val >= 0.0) {
        return Ok(None);
    }
    // Authentic boxed result, produced on the plain eval loop exactly as the
    // skipped residual would.  Declines on any raise (defensive; the cold
    // domain is already excluded above).
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
    // Pin the callable identity (the module-attr fold usually makes it a
    // constant already; guard only when it is not).
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
    // Coerce the argument to a raw float (int → guard_class + unbox +
    // CastIntToFloat; float → guard_class + unbox).
    let x = walker_coerce_operand_to_float(ctx, op.pc, r_args[2], arg_obj, is_int, val, false)?;
    // `ll_math_sqrt` domain guards: `if x < 0.0` (FloatLt pinned false) and
    // `if isfinite(x)` (FloatSub(x,x) == 0 pinned true — excludes NaN/±inf).
    let zero = ctx.trace_ctx.const_float(0.0f64.to_bits() as i64);
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatLt, &[x, zero], false)?;
    let diff = ctx.trace_ctx.record_op(OpCode::FloatSub, &[x, x]);
    ctx.trace_ctx
        .set_opref_concrete(diff, majit_ir::Value::Float(0.0));
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatEq, &[diff, zero], true)?;
    // The pure elidable libm sqrt: EF_ELIDABLE_CANNOT_RAISE → CALL_F, no
    // trailing guard, foldable/hoistable.
    let raw = ctx.trace_ctx.call_float_typed_with_effect(
        crate::trace_opcode::sqrt_nonneg_jit as *const (),
        &[x],
        &[majit_ir::Type::Float],
        majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
    );
    let result_val = unsafe { pyre_object::w_float_get_value(boxed_result) };
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(result_val));
    let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `math.log/cos/sin(x)` on an exact int/float argument.  This is the direct
/// RPython `ll_math_{log,cos,sin}` shape: pin the domain branch, unbox the
/// numeric operand, emit the raw pure `CALL_F`, and leave the result box
/// virtualizable.  Rebound callables, subclasses, exceptional domains, and
/// non-numeric inputs retain the ordinary residual call.
pub(crate) fn try_walker_specialize_math_log_trig<Sym: WalkSym>(
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
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    // Carry `is_log` out of the branch that knows it. Recovering it afterwards
    // by comparing `raw_fn` against `math_log_positive_jit` would make the
    // domain guard below depend on the three helpers keeping distinct
    // addresses, which is a linker property, not a source one.
    let (raw_fn, is_log) =
        if pyre_interpreter::module::math::interp_math::is_math_log_function(concrete_callable) {
            (
                crate::trace_opcode::math_log_positive_jit as *const (),
                true,
            )
        } else if pyre_interpreter::module::math::interp_math::is_math_cos_function(
            concrete_callable,
        ) {
            (crate::trace_opcode::math_cos_finite_jit as *const (), false)
        } else if pyre_interpreter::module::math::interp_math::is_math_sin_function(
            concrete_callable,
        ) {
            (crate::trace_opcode::math_sin_finite_jit as *const (), false)
        } else {
            return Ok(None);
        };
    let (is_int, val) = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) {
            return Ok(None);
        }
        if pyre_object::is_int(arg_obj) {
            (true, pyre_object::w_int_get_value(arg_obj) as f64)
        } else if pyre_object::is_float(arg_obj) {
            (false, pyre_object::w_float_get_value(arg_obj))
        } else {
            return Ok(None);
        }
    };
    // The hot RPython branches used here are log(finite positive) and
    // trig(finite).  Cold NaN/inf/domain cases stay on the exact builtin.
    if !val.is_finite() || (is_log && val <= 0.0) {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

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
    let x = walker_coerce_operand_to_float(ctx, op.pc, r_args[2], arg_obj, is_int, val, false)?;
    let zero = ctx.trace_ctx.const_float(0.0f64.to_bits() as i64);
    if is_log {
        walker_float_cmp_guard(ctx, op.pc, OpCode::FloatLt, &[zero, x], true)?;
    }
    let diff = ctx.trace_ctx.record_op(OpCode::FloatSub, &[x, x]);
    ctx.trace_ctx
        .set_opref_concrete(diff, majit_ir::Value::Float(0.0));
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatEq, &[diff, zero], true)?;
    let raw = ctx.trace_ctx.call_float_typed_with_effect(
        raw_fn,
        &[x],
        &[majit_ir::Type::Float],
        majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
    );
    let result_val = unsafe { pyre_object::w_float_get_value(boxed_result) };
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(result_val));
    let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `math.frexp(x)` on an exact int/float argument.  RPython lowers
/// `ll_math_frexp` to two unboxed results and `space.newtuple2`; emit the same
/// shape as two pure typed calls followed by a virtualizable object tuple.
/// This avoids the opaque builtin dispatch and the concrete tuple/element
/// allocations in numeric loops.  Rebound callables, subclasses, and other
/// coercion shapes retain the generic residual path.
pub(crate) fn try_walker_specialize_math_frexp<Sym: WalkSym>(
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
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    if !pyre_interpreter::module::math::interp_math::is_math_frexp_function(concrete_callable) {
        return Ok(None);
    }
    let (is_int, x_value) = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) {
            return Ok(None);
        }
        if pyre_object::is_int(arg_obj) {
            (true, pyre_object::w_int_get_value(arg_obj) as f64)
        } else if pyre_object::is_float(arg_obj) {
            (false, pyre_object::w_float_get_value(arg_obj))
        } else {
            return Ok(None);
        }
    };

    // Execute the skipped builtin once so all concrete tuple/boxing choices
    // match the interpreter exactly.
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    let (Some(mantissa_obj), Some(exponent_obj)) = (unsafe {
        (
            pyre_object::w_tuple_getitem(boxed_result, 0),
            pyre_object::w_tuple_getitem(boxed_result, 1),
        )
    }) else {
        return Ok(None);
    };
    let mantissa_value = unsafe { pyre_object::w_float_get_value(mantissa_obj) };
    let exponent_value = unsafe { pyre_object::w_int_get_value(exponent_obj) };

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
    let x = walker_coerce_operand_to_float(ctx, op.pc, r_args[2], arg_obj, is_int, x_value, false)?;
    let mantissa = ctx.trace_ctx.call_float_typed_with_effect(
        pyre_interpreter::module::math::interp_math::jit_math_frexp_mantissa as *const (),
        &[x],
        &[majit_ir::Type::Float],
        majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
    );
    ctx.trace_ctx
        .set_opref_concrete(mantissa, majit_ir::Value::Float(mantissa_value));
    let exponent = ctx.trace_ctx.call_int_typed_with_effect(
        pyre_interpreter::module::math::interp_math::jit_math_frexp_exponent as *const (),
        &[x],
        &[majit_ir::Type::Float],
        majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
    );
    ctx.trace_ctx
        .set_opref_concrete(exponent, majit_ir::Value::Int(exponent_value));

    let mantissa_box = crate::state::wrapfloat(ctx.trace_ctx, mantissa);
    ctx.trace_ctx.set_opref_concrete(
        mantissa_box,
        majit_ir::Value::Ref(majit_ir::GcRef(mantissa_obj as usize)),
    );
    let exponent_box = walker_box_int(ctx, op.pc, exponent, exponent_value)?;
    ctx.trace_ctx.set_opref_concrete(
        exponent_box,
        box_int_concrete(exponent_value, exponent_obj as i64),
    );
    // `space.newtuple2` selects `W_SpecialisedTupleObject_oo` for the
    // float/int pair.  The traced allocation must use that same layout:
    // UNPACK_SEQUENCE specializes from the record-time concrete object's
    // class and reads the two inline `value*` fields.
    let tuple =
        crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, mantissa_box, exponent_box);
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', tuple)?;
    Ok(Some(()))
}

/// `math.ldexp(x, exp)` on exact numeric arguments.  This is the direct
/// walker equivalent of RPython's `ll_math_ldexp`: carry `x` and `exp`
/// unboxed, call the platform operation, and guard a finite result so the
/// overflow direction resumes in the builtin and raises `OverflowError`.
/// Underflow to signed zero remains on the fast path.  Non-finite concrete
/// inputs and non-int exponents retain the generic residual path.
pub(crate) fn try_walker_specialize_math_ldexp<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 4 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(x_obj),
        ConcreteValue::Ref(exp_obj),
    ) = (
        arg_concretes[0],
        arg_concretes[1],
        arg_concretes[2],
        arg_concretes[3],
    )
    else {
        return Ok(None);
    };
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || x_obj.is_null()
        || exp_obj.is_null()
        || !pyre_interpreter::module::math::interp_math::is_math_ldexp_function(concrete_callable)
    {
        return Ok(None);
    }
    let (x_is_int, x_value, exp_value) = unsafe {
        if !pyre_object::is_exact_builtin_instance(x_obj)
            || !pyre_object::is_exact_builtin_instance(exp_obj)
        {
            return Ok(None);
        }
        let (x_is_int, x_value) = if pyre_object::is_int(x_obj) {
            (true, pyre_object::w_int_get_value(x_obj) as f64)
        } else if pyre_object::is_float(x_obj) {
            (false, pyre_object::w_float_get_value(x_obj))
        } else {
            return Ok(None);
        };
        if !pyre_object::is_int(exp_obj) {
            return Ok(None);
        }
        (x_is_int, x_value, pyre_object::w_int_get_value(exp_obj))
    };
    // The finite-result guard below represents RPython's errno/overflow
    // branch.  Trace non-finite inputs through the ordinary builtin because
    // ll_math_ldexp returns them unchanged rather than taking that guard.
    if !x_value.is_finite() {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[x_obj, exp_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    let result_value = unsafe { pyre_object::w_float_get_value(boxed_result) };

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
    let x = walker_coerce_operand_to_float(ctx, op.pc, r_args[2], x_obj, x_is_int, x_value, false)?;
    let (exp_type_addr, exp_descr) = crate::state::int_or_bool_unbox_type_descr(exp_obj);
    let exp = walker_unbox_int_typed(ctx, op.pc, r_args[3], exp_type_addr, exp_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(exp, majit_ir::Value::Int(exp_value));
    let raw = ctx.trace_ctx.call_float_typed_with_effect(
        pyre_interpreter::module::math::interp_math::jit_math_ldexp_raw as *const (),
        &[x, exp],
        &[majit_ir::Type::Float, majit_ir::Type::Int],
        majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
    );
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Float(result_value));
    // `result - result == 0` exactly for every finite value, including
    // signed zero; infinity and NaN bail to the raising/propagating builtin.
    let diff = ctx.trace_ctx.record_op(OpCode::FloatSub, &[raw, raw]);
    ctx.trace_ctx
        .set_opref_concrete(diff, majit_ir::Value::Float(0.0));
    let zero = ctx.trace_ctx.const_float(0.0f64.to_bits() as i64);
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatEq, &[diff, zero], true)?;

    let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `math.isqrt(n)` on an exact nonnegative machine integer.
///
/// PyPy exposes `isqrt` from `app_math.py`, so tracing its `W_IntObject` arm
/// carries `n` unboxed through the integer algorithm and virtualizes the
/// result.  Pyre's native module wrapper otherwise materializes an `RBigInt`
/// before the walker can see that arm.  Recreate the translated shape as an
/// exact-class guard, unbox, pure non-raising integer call, and `wrapint`.
/// Longs, bools, subclasses, negative values, rebound callables, and
/// `__index__` objects retain the generic residual path.
pub(crate) fn try_walker_specialize_math_isqrt<Sym: WalkSym>(
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
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || arg_obj.is_null()
        || !pyre_interpreter::module::math::interp_math::is_math_isqrt_function(concrete_callable)
    {
        return Ok(None);
    }
    let value = unsafe {
        // `is_int` also accepts a `bool`, whose singletons carry a `&BOOL_TYPE`
        // vtable and a NULL `w_class`.  The unbox and the exact-class guard
        // below both pin the canonical `int` class, so neither can hold for one;
        // decline it here as the int/long binop specializations do instead of
        // emitting a guard that fails on the operand it was recorded from.
        if !pyre_object::is_exact_builtin_instance(arg_obj)
            || !pyre_object::is_int(arg_obj)
            || pyre_object::is_bool(arg_obj)
        {
            return Ok(None);
        }
        pyre_object::w_int_get_value(arg_obj)
    };
    if value < 0 {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_int(boxed_result) } {
        return Ok(None);
    }
    let result_value = unsafe { pyre_object::w_int_get_value(boxed_result) };

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
    let arg_op = r_args[2];
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let raw_int = walker_unbox_int(ctx, op.pc, arg_op, int_type_addr)?;
    walker_guard_exact_w_class(
        ctx,
        op.pc,
        arg_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE),
    )?;
    ctx.trace_ctx
        .set_opref_concrete(raw_int, majit_ir::Value::Int(value));
    let zero = ctx.trace_ctx.const_int(0);
    let nonnegative = ctx.trace_ctx.record_op(OpCode::IntGe, &[raw_int, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonnegative, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[nonnegative])?;

    let raw_result = ctx.trace_ctx.call_int_typed_with_effect(
        pyre_interpreter::module::math::interp_math::jit_math_isqrt_i64 as *const (),
        &[raw_int],
        &[majit_ir::Type::Int],
        majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
    );
    ctx.trace_ctx
        .set_opref_concrete(raw_result, majit_ir::Value::Int(result_value));
    let boxed = walker_box_int(ctx, op.pc, raw_result, result_value)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(result_value, boxed_result as i64));
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `int(x)` for an exact float whose truncated value fits a machine Signed.
///
/// PyPy `floatobject.py:newint_from_float` first runs
/// `ovfcheck_float_to_int`; its success arm is exactly
/// `CAST_FLOAT_TO_INT + space.newint`.  Emit that arm with the corresponding
/// `-2**63 <= x < 2**63` guards, leaving NaN, infinity, out-of-range values,
/// subclasses, and rebound constructors on the ordinary residual path.  The
/// slow arm remains responsible for `newlong_from_float` and its exceptions.
pub(crate) fn try_walker_specialize_int_call<Sym: WalkSym>(
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
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    let int_type_obj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    if !std::ptr::eq(concrete_callable, int_type_obj) {
        return Ok(None);
    }
    let value = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) || !pyre_object::is_float(arg_obj) {
            return Ok(None);
        }
        pyre_object::w_float_get_value(arg_obj)
    };
    // `2**63` is exactly representable while `i64::MAX` is not; use a strict
    // upper bound, matching ovfcheck_float_to_int on a signed 64-bit target.
    const SIGNED_MIN_AS_FLOAT: f64 = -9223372036854775808.0;
    const SIGNED_LIMIT_AS_FLOAT: f64 = 9223372036854775808.0;
    if !(value >= SIGNED_MIN_AS_FLOAT && value < SIGNED_LIMIT_AS_FLOAT) {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_int(boxed_result) } {
        return Ok(None);
    }
    let result_value = unsafe { pyre_object::w_int_get_value(boxed_result) };

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
    let arg_op = r_args[2];
    let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
    let raw_float = walker_unbox_float(ctx, op.pc, arg_op, float_type_addr)?;
    walker_guard_exact_w_class(
        ctx,
        op.pc,
        arg_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE),
    )?;
    ctx.trace_ctx
        .set_opref_concrete(raw_float, majit_ir::Value::Float(value));
    let low = ctx
        .trace_ctx
        .const_float(SIGNED_MIN_AS_FLOAT.to_bits() as i64);
    let high = ctx
        .trace_ctx
        .const_float(SIGNED_LIMIT_AS_FLOAT.to_bits() as i64);
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatGe, &[raw_float, low], true)?;
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatLt, &[raw_float, high], true)?;

    let raw_int = ctx
        .trace_ctx
        .record_op(OpCode::CastFloatToInt, &[raw_float]);
    ctx.trace_ctx
        .set_opref_concrete(raw_int, majit_ir::Value::Int(result_value));
    let boxed = walker_box_int(ctx, op.pc, raw_int, result_value)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(result_value, boxed_result as i64));
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    Ok(Some(()))
}

/// `float(x)` on an exact int/float argument: inline the conversion
/// (`W_IntObject.descr_float` → `space.newfloat`, or the identity
/// `float(f) is f` for an exact float) instead of the opaque
/// `bh_call_fn(float_type, NULL, x)` residual, so the result virtualizes.  The
/// callable must be the exact `float` type object; a rebound name or a float
/// subclass (which reboxes rather than returning the argument) declines.  Any
/// non-matching shape falls through to the generic residual (SAFE).
pub(crate) fn try_walker_specialize_float_call<Sym: WalkSym>(
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
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(arg_obj),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if concrete_callable.is_null() || !null_or_self.is_null() || arg_obj.is_null() {
        return Ok(None);
    }
    // The callable must be the canonical `float` type object.
    let float_type_obj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE);
    if !std::ptr::eq(concrete_callable, float_type_obj) {
        return Ok(None);
    }
    let (is_int, val) = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) {
            return Ok(None);
        }
        if pyre_object::is_int(arg_obj) {
            (true, pyre_object::w_int_get_value(arg_obj) as f64)
        } else if pyre_object::is_float(arg_obj) {
            (false, pyre_object::w_float_get_value(arg_obj))
        } else {
            return Ok(None);
        }
    };
    // Authentic boxed result (float() is side-effect-free on int/float).
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

    // --- emit the specialized IR (walker-native) ---
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
    let arg_op = r_args[2];
    if is_int {
        // int/bool → CastIntToFloat + inline wrapfloat (no residual call).
        let raw = walker_coerce_operand_to_float(ctx, op.pc, arg_op, arg_obj, true, val, false)?;
        let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
        );
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    } else {
        // exact float → `float(f) is f`: forward the argument unchanged.  Only
        // sound when the constructor actually returned the same object; a
        // divergence (should not happen for an exact float) declines.
        if !std::ptr::eq(boxed_result, arg_obj) {
            return Ok(None);
        }
        let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
        if !arg_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(arg_op) {
            let type_const = ctx.trace_ctx.const_int(float_type_addr);
            ctx.trace_ctx
                .record_guard(OpCode::GuardClass, &[arg_op, type_const], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(arg_op, float_type_addr);
        walker_guard_exact_w_class(ctx, op.pc, arg_op, float_type_obj)?;
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', arg_op)?;
    }
    Ok(Some(()))
}

/// `divmod(a, b)` on two exact `W_IntObject` operands: emit the inline pair
/// shape the meta-tracer produces upstream (intobject.py `_divmod` →
/// `space.newtuple2(space.newint(z), space.newint(m))`) instead of the opaque
/// `bh_call_fn(divmod_builtin, NULL, a, b)` residual.
///
/// The divmod row rejects a zero divisor before dispatching, so the trace
/// carries the same domain guards the `//` / `%` specialization emits, then
/// runs the two `OS_INT_PY_DIV` / `OS_INT_PY_MOD` elidable calls over one
/// guarded operand pair.  The result is the virtual `Cls_ii` specialised
/// tuple, so a `q, r = divmod(...)` site pairs with
/// [`try_walker_specialize_unpack`] and the tuple never materializes.
///
/// The exact `w_class` guard is required because an `int` SUBCLASS shares
/// `ob_type == &INT_TYPE` but may override `__divmod__`; it side-exits to the
/// generic residual.
///
/// Returns `None` (fall through to the generic residual, SAFE) for any other
/// shape: wrong arity, a bound receiver, a non-`divmod` callable, an operand
/// that is not an exact `int` (long / float / bool / subclass), a tagged
/// immediate, a zero divisor, or the `INT_MIN // -1` pair that escapes to a
/// bigint result.
pub(crate) fn try_walker_specialize_builtin_divmod<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, a, b)` shape only.
    if r_args.len() != 4 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(lhs_obj),
        ConcreteValue::Ref(rhs_obj),
    ) = (
        arg_concretes[0],
        arg_concretes[1],
        arg_concretes[2],
        arg_concretes[3],
    )
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `divmod(a, b)` call.
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || lhs_obj.is_null()
        || rhs_obj.is_null()
    {
        return Ok(None);
    }
    if !pyre_interpreter::builtins::is_builtin_divmod_function(concrete_callable) {
        return Ok(None);
    }
    // A tagged immediate has no real header for the `w_class` / unbox guards
    // and this emit is not tag-aware, so decline it to the residual.
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(lhs_obj)
            || pyre_object::tagged_int::is_tagged_int(rhs_obj))
    {
        return Ok(None);
    }
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let is_exact_int = |o: pyre_object::PyObjectRef| unsafe {
        std::ptr::eq((*o).ob_type, &pyre_object::pyobject::INT_TYPE)
            && std::ptr::eq((*o).w_class, int_typeobj)
    };
    if !is_exact_int(lhs_obj) || !is_exact_int(rhs_obj) {
        // `_make_descr_binop(_divmod, _int_divmod)` (longobject.py:459) keeps a
        // dedicated long/int arm; every other operand shape stays generic.
        return try_walker_specialize_builtin_divmod_long_int(
            ctx,
            op,
            r_args,
            dst,
            concrete_callable,
            lhs_obj,
            rhs_obj,
        );
    }
    let (la, rb) = unsafe {
        (
            pyre_object::w_int_get_value(lhs_obj),
            pyre_object::w_int_get_value(rhs_obj),
        )
    };
    // A zero divisor raises ZeroDivisionError and `INT_MIN // -1` escapes to
    // the bigint pair; both are outside the guarded domain the emit below
    // covers, so decline rather than record a guard the recorded operands
    // already fail.
    if rb == 0 || (la == i64::MIN && rb == -1) {
        return Ok(None);
    }

    // --- emit the specialized IR (walker-native) ---
    walker_guard_builtin_callable_identity(ctx, op.pc, r_args[0], concrete_callable)?;
    let (lhs_op, rhs_op) = (r_args[2], r_args[3]);
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    // `GuardClass` before the `w_class` read: it is the guard that proves the
    // operand is a real heap header rather than a tagged immediate, so it has
    // to precede any `getfield` off that header.
    walker_guard_class(ctx, op.pc, lhs_op, int_type_addr)?;
    walker_guard_class(ctx, op.pc, rhs_op, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op.pc, lhs_op, int_typeobj)?;
    walker_guard_exact_w_class(ctx, op.pc, rhs_op, int_typeobj)?;
    let lhs_raw = walker_unbox_int_typed(
        ctx,
        op.pc,
        lhs_op,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;
    let rhs_raw = walker_unbox_int_typed(
        ctx,
        op.pc,
        rhs_op,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;
    walker_emit_int_div_domain_guards(ctx, op.pc, lhs_raw, rhs_raw, la, rb)?;
    let (div_raw, div_value) = walker_emit_int_py_div_or_mod(ctx, lhs_raw, rhs_raw, la, rb, true);
    let (mod_raw, mod_value) = walker_emit_int_py_div_or_mod(ctx, lhs_raw, rhs_raw, la, rb, false);
    let tuple =
        walker_emit_specialised_tuple_ii(ctx, op.pc, div_raw, mod_raw, div_value, mod_value)?;
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', tuple)?;
    Ok(Some(()))
}

/// One element of a concrete `W_SpecialisedTupleObject_oo`, or `None` when the
/// value is any other tuple layout. `newtuple` picks the representation, so a
/// fold that emits the object-pair shape has to confirm the record-time value
/// took the same one before reading it through those offsets.
fn walker_specialised_tuple_oo_item(
    tuple: pyre_object::PyObjectRef,
    index: usize,
) -> Option<pyre_object::PyObjectRef> {
    if tuple.is_null() {
        return None;
    }
    let ob_type = unsafe { (*(tuple as *const pyre_object::pyobject::PyObject)).ob_type };
    if !std::ptr::eq(
        ob_type,
        &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE
            as *const pyre_object::pyobject::PyType,
    ) {
        return None;
    }
    let item = unsafe {
        pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(tuple, index)
    };
    (!item.is_null()).then_some(item)
}

/// Pin a builtin's identity before folding its call away. `LOAD_GLOBAL divmod`
/// is usually already a constant via the namespace cell fold, in which case the
/// guard is unnecessary; a rebound global takes the side exit.
fn walker_guard_builtin_callable_identity<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    callable_op: OpRef,
    concrete_callable: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    if callable_op.is_constant() {
        return Ok(());
    }
    let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(callable_op, expected);
    Ok(())
}

/// `divmod(W_LongObject, W_IntObject)` — `longobject.py:451 _int_divmod`.
///
/// One `rbigint.int_divmod` (rbigint.py:1050 `@jit.elidable`) produces both
/// halves, so the trace is the upstream shape: a single `CallR` returning the
/// RPython `tuple2`, two `GetfieldGcR` off it, then `newlong` ×2 and the
/// arity-2 tuple — all three allocations trace-visible, so the shipped oo
/// unpack fold can virtualize the tuple away at an unpacking use.
///
/// Emitting `int_div_floor` and `int_mod_int_result` instead would run the
/// division twice; `_int_divmod` exists precisely to avoid that.
///
/// `int % long` and `divmod(int, long)` are `descr_rdivmod`, which coerces the
/// left operand and takes the bigint/bigint path — not this arm.
fn try_walker_specialize_builtin_divmod_long_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
    concrete_callable: pyre_object::PyObjectRef,
    long_obj: pyre_object::PyObjectRef,
    int_obj: pyre_object::PyObjectRef,
) -> Result<Option<()>, DispatchError> {
    let (long_class, int_class, int_value) = unsafe {
        if !pyre_object::is_long(long_obj) || !pyre_object::is_int(int_obj) {
            return Ok(None);
        }
        let (Some(long_class), Some(int_class)) = (
            walker_exact_builtin_class(long_obj),
            walker_exact_builtin_class(int_obj),
        ) else {
            return Ok(None);
        };
        (long_class, int_class, pyre_object::w_int_get_value(int_obj))
    };
    // A zero divisor raises before reaching rbigint; the interpreter owns the
    // authentic message.
    if int_value == 0 {
        return Ok(None);
    }

    // Take the concretes from the authentic builtin, not from the residual:
    // the residual's payload allocator collects, and it is only safe to do so
    // under a gcmap-carrying `CallR`, which the host-side walker is not. This
    // is the `math.isqrt` fold's route, and it is also what makes the recorded
    // tuple the one `newtuple` actually picked for these two operands.
    let tuple_concrete = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[long_obj, int_obj])
    };
    let Ok(tuple_concrete) = tuple_concrete else {
        return Ok(None);
    };
    // `_int_divmod` boxes both halves with `newlong`, which does not demote, so
    // a pair of longs is the only shape this arm may emit — and two longs are
    // what routes `newtuple` to the object-pair variant.
    let (Some(w_div), Some(w_mod)) = (
        walker_specialised_tuple_oo_item(tuple_concrete, 0),
        walker_specialised_tuple_oo_item(tuple_concrete, 1),
    ) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(w_div) && pyre_object::is_long(w_mod) } {
        return Ok(None);
    }
    // The call above allocates, so the operand pointers this function was
    // handed may have been forwarded. Re-fetch them from the walker's op cells,
    // which the collector does update, before reading anything through them.
    let (long_op, int_op) = (r_args[2], r_args[3]);
    let (Some(long_obj), Some(int_obj)) = (
        walker_concrete_ref_object(ctx, long_op),
        walker_concrete_ref_object(ctx, int_op),
    ) else {
        return Ok(None);
    };
    let read_payload = |o: pyre_object::PyObjectRef| unsafe {
        *((o as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET) as *const i64)
            as *mut majit_rlib::rbigint::RBigInt
    };
    let long_payload = read_payload(long_obj) as i64;
    let (div_payload, mod_payload) = (read_payload(w_div), read_payload(w_mod));
    // Both halves are already reachable from `tuple_concrete`, and this
    // allocation cannot collect, so neither can move under it.
    let pair = pyre_object::longobject::alloc_bigint_pair_no_collect(div_payload, mod_payload);
    if pair.is_null() {
        return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op.pc });
    }

    // --- emit ---
    walker_guard_builtin_callable_identity(ctx, op.pc, r_args[0], concrete_callable)?;
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    walker_guard_class(ctx, op.pc, long_op, long_type_addr)?;
    walker_guard_exact_w_class(ctx, op.pc, long_op, long_class)?;
    let (int_type, int_descr) = crate::state::int_or_bool_unbox_type_descr(int_obj);
    let int_raw = walker_unbox_int_typed(ctx, op.pc, int_op, int_type, int_descr)?;
    walker_guard_exact_w_class(ctx, op.pc, int_op, int_class)?;
    let zero = ctx.trace_ctx.const_int(0);
    let nonzero = ctx.trace_ctx.record_op(OpCode::IntNe, &[int_raw, zero]);
    ctx.trace_ctx
        .set_opref_concrete(nonzero, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[nonzero])?;

    let long_pl = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[long_op],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        long_pl,
        majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
    );

    let helper = pyre_interpreter::objspace::descroperation::jit_bigint_int_divmod as *const ();
    let pair_op = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper,
        &[long_pl, int_raw],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as usize as i64),
            majit_ir::Value::Ref(majit_ir::GcRef(long_payload as usize)),
            majit_ir::Value::Int(int_value),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(pair as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        pair_op,
        majit_ir::Value::Ref(majit_ir::GcRef(pair as usize)),
    );
    if pair_op.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNoException, &[])?;
    }

    let mut boxed = Vec::with_capacity(2);
    for (descr, payload, wrapper) in [
        (crate::descr::rbigint_pair_item0_descr(), div_payload, w_div),
        (crate::descr::rbigint_pair_item1_descr(), mod_payload, w_mod),
    ] {
        let half = ctx
            .trace_ctx
            .record_op_with_descr(OpCode::GetfieldGcR, &[pair_op], descr);
        ctx.trace_ctx.set_opref_concrete(
            half,
            majit_ir::Value::Ref(majit_ir::GcRef(payload as usize)),
        );
        let w = crate::helpers::emit_box_long_inline(
            ctx.trace_ctx,
            half,
            crate::descr::w_long_size_descr(),
            crate::descr::long_value_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(w, majit_ir::Value::Ref(majit_ir::GcRef(wrapper as usize)));
        boxed.push(w);
    }

    let tuple = crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, boxed[0], boxed[1]);
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(tuple_concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', tuple)?;
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
/// append to the concrete list + journals the rewind. Recognition declines
/// before emitting IR; an unsupported body sub-walk rolls its tentative IR
/// back before falling through to the residual call.
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
/// a symbolic (tagged) fnaddr, `try_execute_residual_call_via_executor`
/// declines it (`OrthodoxSubWalkTraceUnsupported`) and the method-call form
/// records the append as a residual call instead of baking the hash as a code
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

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // Mirror the commit's own promotion predicate exactly: a rollback must
    // undo a promotion only when one was performed, or it would pop another
    // list's journal entry.
    let promoted_empty = unsafe { pyre_object::w_list_uses_empty_storage(inner_self) };

    // ── tentative commit ──
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

    // Recover the receiver list OpRef; the sub-walk reads it as ref-arg 0.
    let self_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_self_descr(),
    );

    let commit_result = orthodox_list_append_commit(
        ctx, op, sym, &sub_body, self_ref, value_op, inner_self, value, len_before,
    );
    match commit_result {
        Ok(()) => {}
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-APPEND-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            bool_box_truth_reset();
            if promoted_empty {
                fbw_append_promote_journal_rollback_last(inner_self);
            }
            return Ok(None);
        }
        Err(error) => return Err(error),
    }

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
    // `is_plain_int1` accepts an exact `W_IntObject` or a fits-int
    // `W_LongObject`; both route to Integer storage. The commit path pins
    // `guard_class(value, LONG_TYPE)` for a long value (vs `INT_TYPE` for an
    // int) so the descended `w_list_append` body observes the right `ob_type`.
    // The body's `is_plain_int1(value)` / `plain_int_w(value)` then unbox the
    // long through the compiled `_fits_int` / `toint` path; when that path
    // reaches a helper the sub-walk cannot lower it declines
    // (`OrthodoxSubWalkTraceUnsupported`) and rolls back to the generic
    // residual (correctness-safe for any element).
    if !pyre_object::pyobject::is_list(inner_self) {
        return None;
    }
    // Empty-strategy first-append promotion. `w_list_can_append_without_realloc`
    // is false for Empty (no backing block yet), so classify by the value's
    // type using switch_to_correct_strategy's int -> float -> object order
    // (listobject.py) and let the commit path install the typed storage.
    if pyre_object::w_list_uses_empty_storage(inner_self) {
        let int_ok = pyre_object::is_plain_int1(value)
            && !(pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(value));
        let float_ok = !value.is_null() && pyre_object::is_plain_float_strict(value);
        // switch_to_correct_strategy routes `is_plain_int1` (exact int or
        // fits-in-word long) -> Integer with no tagged exclusion. Exclude any
        // plain-int / float from the object fallback so a tagged-int DECLINES
        // (generic residual) instead of mis-routing to Object and diverging the
        // traced strategy from the concrete one the commit installs.
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
    // Int-storage specialization: `is_plain_int1` value (exact `W_IntObject`
    // or fits-int `W_LongObject`) stored unboxed. A tagged-immediate value
    // would need a tag-aware unboxed store and no `w_class` pin; decline to
    // the generic residual append instead.
    let int_ok = pyre_object::w_list_uses_int_storage(inner_self)
        && pyre_object::is_plain_int1(value)
        && !(pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(value));
    // Object-storage extension: any non-null `Ref` value stored into the
    // object items block — no unboxing, so the value carries no type
    // precondition.
    let obj_ok = pyre_object::w_list_uses_object_storage(inner_self) && !value.is_null();
    // Float-storage specialization: a strict `W_FloatObject` stored
    // unboxed. `FloatListStrategy.is_correct_type` (listobject.py) is
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
/// non-null with a set `jitcode` field.  Word size does not enter here: the
/// `d` operands resolve through a descr pool built from the target's own
/// Charon layouts (`jitcode_runtime::build_time_field_offset`), whose array
/// descrs place the items at the first element-aligned offset past the length
/// word — the offset a 4-byte word and an 8-byte item disagree on.
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
    let allocated_before = unsafe { pyre_object::listobject::w_list_allocated(inner_self) };
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
    // (switch_to_correct_strategy, listobject.py), then emit the
    // transition IR mutating the existing wrapper, promote the concrete list,
    // and journal the rewind to Empty.
    use pyre_object::listobject::ListStrategy;
    let promote_empty = unsafe { pyre_object::w_list_uses_empty_storage(inner_self) };
    if promote_empty {
        let target = unsafe {
            let int_ok = pyre_object::is_plain_int1(value)
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
    // predicate folds during the sub-walk: guard_class(value, <TYPE>) +
    // class_now_known, so its `is_int`/`is_long`/`is_bool` typeptr reads fold
    // to the pinned const (the typeptr fold in `getfield_gc_via_heapcache`).
    // The recognition gate already proved `is_plain_int1(value)`; this guard
    // enforces the observed ob_type at runtime.  The value's integer payload
    // stays symbolic — only its class is pinned.
    //
    // Object-storage append stores the value as a
    // plain GC ref with no unboxing, so it carries no type precondition —
    // skip the class pin (the sub-walk's object-storage store path does
    // not read the value's class).
    let is_obj_storage = unsafe { pyre_object::w_list_uses_object_storage(inner_self) };
    if !is_obj_storage {
        // Integer and Float storage both pin the value's class so the body's
        // strict type test folds during the sub-walk; the ob_type const is
        // FLOAT_TYPE for float storage, and INT_TYPE / LONG_TYPE for int
        // storage depending on whether the value is an exact int or a fits-int
        // `W_LongObject` (both pass `is_plain_int1` -> Integer storage, but
        // carry distinct `ob_type`s the sub-walk's `is_plain_int1` folds on).
        let is_float_storage = unsafe { pyre_object::w_list_uses_float_storage(inner_self) };
        let value_is_long = unsafe { pyre_object::pyobject::is_long(value) };
        let value_type_addr = if is_float_storage {
            &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64
        } else if value_is_long {
            &pyre_object::pyobject::LONG_TYPE as *const _ as i64
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
        // Forward py twin first (#73 phase-3): equals the containing
        // coordinate plus trivia normalization by construction; the containing
        // lookup survives for the empty-twin class, and the trivia skip below
        // is an identity on the twin path.
        let mut py = jc
            .payload
            .forward_py_pc_for_jitcode_pc(op.pc)
            .unwrap_or_else(|| {
                crate::py_coord::note_empty_twin_fallback(
                    "list_append_commit",
                    jc.index,
                    op.pc as i32,
                );
                crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, op.pc)
            });
        if jc.payload.code_ptr.is_null() {
            (py, sym.valuestackdepth() as i64, jc_index, marker)
        } else {
            let codeobj = &*jc.payload.code_ptr;
            py = skip_python_trivia_forward(codeobj, py as usize) as u32;
            // Read the depth off the jitcode-pc-keyed trivia twin, which equals
            // `depth_at_py_pc()[skip_python_trivia_forward(containing_py_pc_for_jitcode_pc(op.pc))]`
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
        false,
        call_site_word,
        // As above, entry metadata is keyed by the append op itself; its
        // liveness-bank query remains keyed by the resume marker.
        op.pc as i32,
        OuterActiveBoxesEntryTwin::Plain,
        "w_list_append_call_site",
        None,
        &[],
        None,
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
    // symbolic-tagged funcbox — is declined by `try_execute_residual_call_via_executor`
    // (`OrthodoxSubWalkTraceUnsupported`) and `walk_result?` propagates that
    // abort before this point (graceful interpreter fallback, never a wrong
    // trace).  The descr-pool wiring above (strategy/header field descrs) is
    // exercised on the way in.

    // Tracing is execution: apply the append + journal the rewind.  The
    // journal entry is unconditional — it rewinds the receiver to
    // `len_before` on an aborted walk, whichever side actually grew it.
    //
    // The sub-walk normally records the store as IR without touching the
    // concrete list, so the append below is what applies it.  It is not
    // guaranteed to: the per-strategy store the descended arm reaches
    // (`W_ListObject::object_push`, `IntArray::push`, `FloatArray::push`) is a
    // `residual_call`, and a residual whose funcptr resolves to a real address
    // is EXECUTED by `try_execute_residual_call_via_executor` rather than only
    // recorded.  Those three carry runtime bindings, so on a target where the
    // arm keeps them as residuals the sub-walk has already appended, and
    // appending again puts the value in twice — one extra element per compiled
    // append, which is how it surfaces (`len(keep)` 20048 for 20000
    // iterations, a traceback name list with its last frame doubled).
    // Re-read the length instead of assuming which side ran: it is the
    // receiver's own state, so it answers for both.
    fbw_list_journal_push_append(inner_self, len_before, allocated_before);
    if unsafe { pyre_object::w_list_len(inner_self) } == len_before {
        unsafe { pyre_object::w_list_append(inner_self, value) };
    }
    Ok(())
}

/// Descend the guard-free Integer-strategy `w_list_pop_end_inner` body for a
/// bound `list.pop()` call, recording its length/item array operations instead
/// of an opaque residual call.
pub(crate) fn try_walker_orthodox_list_pop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    if callable.is_null() || !null_or_self.is_null() {
        return Ok(None);
    }

    let (inner_func, inner_self, len_before, raw_item) = unsafe {
        if !pyre_object::function::is_method(callable) {
            return Ok(None);
        }
        let inner_func = pyre_object::function::w_method_get_func(callable);
        let inner_self = pyre_object::function::w_method_get_self(callable);
        if inner_func.is_null() || inner_self.is_null() {
            return Ok(None);
        }
        let list_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE);
        if pyre_interpreter::lookup_in_type(list_type, "pop") != Some(inner_func) {
            return Ok(None);
        }
        let Some((len_before, raw_item)) = orthodox_list_pop_recognize(inner_self) else {
            return Ok(None);
        };
        (inner_func, inner_self, len_before, raw_item)
    };

    // Resolve every possible decline before recording a guard.
    let Some((sub_body, sym_ptr)) = orthodox_list_pop_body_and_sym(ctx) else {
        return Ok(None);
    };
    let sym = unsafe { &*sym_ptr };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let callable_op = r_args[0];

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
    let self_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_self_descr(),
    );

    match orthodox_list_pop_commit(
        ctx, op, sym, &sub_body, self_ref, inner_self, len_before, raw_item, dst,
    ) {
        Ok(()) => Ok(Some(())),
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-POP-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            bool_box_truth_reset();
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Recognize a non-empty Integer-strategy list and sample its final unboxed
/// item before the descended executor can mutate the live list.
unsafe fn orthodox_list_pop_recognize(
    inner_self: pyre_object::PyObjectRef,
) -> Option<(usize, i64)> {
    if !pyre_object::pyobject::is_list(inner_self)
        || !pyre_object::w_list_uses_int_storage(inner_self)
    {
        return None;
    }
    let len_before = pyre_object::w_list_len(inner_self);
    if len_before == 0 {
        return None;
    }
    let list = &*(inner_self as *const pyre_object::listobject::W_ListObject);
    let raw_item = pyre_object::listobject::ll_list_int_getitem_fast(list, len_before - 1);
    Some((len_before, raw_item))
}

/// Resolve the pop body and enclosing full-body snapshot before IR emission.
pub(crate) fn orthodox_list_pop_body_and_sym<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(SubJitCodeBody, *const Sym)> {
    let jc_arc = crate::jitcode_runtime::list_pop_end_jitcode()?;
    let sub_body = sub_jitcode_body_by_index(jc_arc.index())?;
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() || unsafe { (&*sym_ptr).jitcode().is_null() } {
        return None;
    }
    Some((sub_body, sym_ptr))
}

/// Publish the pop call-site resume coordinate, descend the real helper body,
/// write its Ref result, and journal/apply the concrete shrink exactly once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn orthodox_list_pop_commit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    sym: &Sym,
    sub_body: &SubJitCodeBody,
    self_ref: OpRef,
    inner_self: pyre_object::PyObjectRef,
    len_before: usize,
    raw_item: i64,
    dst: usize,
) -> Result<(), DispatchError> {
    ctx.trace_ctx
        .set_opref_concrete(self_ref, Value::Ref(majit_ir::GcRef(inner_self as usize)));

    let (call_site_py_pc, vsd_value, outer_jitcode_index, call_site_marker) = unsafe {
        let jc = &*sym.jitcode();
        let jc_index = jc.index as u32;
        let marker = jc.payload.resume_marker_for_jitcode_pc(op.pc);
        let mut py = jc
            .payload
            .forward_py_pc_for_jitcode_pc(op.pc)
            .unwrap_or_else(|| {
                crate::py_coord::note_empty_twin_fallback(
                    "list_pop_commit",
                    jc.index,
                    op.pc as i32,
                );
                crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, op.pc)
            });
        if jc.payload.code_ptr.is_null() {
            (py, sym.valuestackdepth() as i64, jc_index, marker)
        } else {
            let codeobj = &*jc.payload.code_ptr;
            py = skip_python_trivia_forward(codeobj, py as usize) as u32;
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
    let call_site_word = call_site_marker
        .map(|marker| marker as i32)
        .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
    let active = collect_outer_active_boxes(
        sym,
        ctx.trace_ctx,
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        outer_jitcode_index,
        false,
        call_site_word,
        op.pc as i32,
        OuterActiveBoxesEntryTwin::Plain,
        "w_list_pop_end_call_site",
        None,
        &[],
        // Not a branch-guard reconstruction: this is the pre-call site
        // snapshot, so there is no kept operand-stack slot to report as
        // unsourced.
        None,
    );

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

    let saved_fbw_mode = ctx.fbw_mode;
    ctx.fbw_mode.inline_subwalk = true;
    let walk_result = run_sub_jitcode_walk(
        ctx,
        op.pc,
        sub_body,
        &[],
        &[],
        &[self_ref],
        &[ConcreteValue::Ref(inner_self)],
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

    let result = match walk_result? {
        DispatchOutcome::SubReturn {
            result: Some(result),
        } => result,
        DispatchOutcome::SubReturn { result: None } => {
            return Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc });
        }
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc }),
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result)?;

    let w_item = pyre_object::w_int_new(raw_item);
    fbw_list_journal_push_pop_end(inner_self, len_before, w_item);
    if unsafe { pyre_object::w_list_len(inner_self) } == len_before {
        unsafe { pyre_object::w_list_pop_end(inner_self) };
    }
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
/// for any non-matching shape, and likewise after rolling the tentative IR back
/// when the body sub-walk hits an un-lowered helper; the residual is void so no
/// result is written.
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

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // Mirror the commit's own promotion predicate exactly: a rollback must
    // undo a promotion only when one was performed, or it would pop another
    // list's journal entry.
    let promoted_empty = unsafe { pyre_object::w_list_uses_empty_storage(list) };

    // ── tentative commit ──
    // The receiver list OpRef + value OpRef are the residual's Ref operands.
    let commit_result = orthodox_list_append_commit(
        ctx, op, sym, &sub_body, r_args[0], r_args[1], list, value, len_before,
    );
    match commit_result {
        Ok(()) => {}
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-APPEND-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            bool_box_truth_reset();
            if promoted_empty {
                fbw_append_promote_journal_rollback_last(list);
            }
            return Ok(None);
        }
        Err(error) => return Err(error),
    }
    Ok(Some(()))
}

/// B3: walker-native exception-construction fold.  A
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
/// (`typeobject.py`).  When both resolve to
/// `W_BaseException.descr_new` / `descr_init`
/// (`interp_exceptions.py`), a trivial subclass has the same traced
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

    // OSError can rebind `args_w` after parsing a filename, so its final
    // slice is selected below, once the concrete constructor has exposed the
    // value-dependent branch result.

    let is_canonical = pyre_object::interp_exceptions::is_canonical_exc_class(concrete_callable);
    let mut subclass_lookups = None;
    let subclass_version_tag = if is_canonical {
        None
    } else {
        // A heap subclass is safe to construct concretely only after both MRO
        // lookups have been proved identical to a canonical exception class.
        // Consequently force_plain_eval below can execute only the builtin
        // Rust `descr_new` / `descr_init`, never user Python code.  This is the
        // promoted-class lookup contract of typeobject.py.
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

    // Where the platform reads the fourth argument, `_parse_init_args` derives
    // the errno and the retagged class from it and stores it in its own slot.
    // Both depend on a value this emit neither guards nor writes, so those
    // instances stay on the runtime path.
    if cfg!(windows) && fills_os_error_slots && args.len() >= 4 {
        return Ok(None);
    }

    let has_filename = fills_os_error_slots
        && args.len() >= 3
        && !unsafe { pyre_object::is_none(concrete_args[2]) };
    let final_args_len = if has_filename { 2 } else { args.len() };
    let final_args = &args[..final_args_len];

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
    if subclass_version_tag.is_some() {
        // Pin the promoted class version that made both MRO descriptor
        // identities constant.  `W_TypeObject.mutated` recursively changes
        // subclass tags (`typeobject.py`), so mutating this class or a
        // base revokes the loop before the folded constructor is reused.
        let class_const = ctx.trace_ctx.const_ref(concrete_callable as i64);
        walker_pin_type_version_tag(ctx, op.pc, class_const)?;
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
    // alongside the exception.  `w_exception_args_new` pins the object
    // representation at every arity, so this reproduces that one shape
    // instead of picking a layout from the element types.
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, final_args);
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

/// B3: walker-native RAISE_VARARGS E1 fast path. The `RaiseVarargs`
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
    fbw_built_exc_insert(exc_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', exc_op)?;
    Ok(Some(()))
}

/// B3: walker-native fold for a bare-class `raise Type`
/// (no call parentheses).  Unlike `raise Type()`, a bare class has no
/// preceding `CallFn` construct residual — `normalize_raise_varargs_jit`
/// instantiates the class itself — so no virtualizable `NewWithVtable`
/// exists and `try_walker_trace_raise_builtin` declines it to the residual
/// (a per-iteration heap alloc + may-force).
///
/// `do_raise` instantiates a raised class with no arguments, so a bare
/// `raise ValueError` is `raise ValueError()`.  When the operand is a
/// canonical builtin exception class with a trivial-args constructor and no
/// explicit `from` cause, build the zero-argument instance inline (the
/// `try_walker_trace_exception_new` Empty-args shape) and chain
/// `__context__` (the `try_walker_trace_raise_builtin` tail), so the whole
/// exception virtualizes and DCEs when it never escapes.  A subclass or a
/// non-trivial-args kind (OSError / Unicode) declines to the residual.
///
/// Returns `None` (fall through to the generic residual) for any
/// non-matching shape.
pub(crate) fn try_walker_trace_raise_bare_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let class_op = r_args[1];
    // The residual arg concretes are `[frame, exc, cause]`.  Recover the live
    // exception operand (index 1) from the residual list rather than the opref
    // shadow: a bare class comes straight from `LOAD_GLOBAL`, not the inline
    // construct fold, so its shadow is not stamped.
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let Some(ConcreteValue::Ref(concrete_class)) = arg_concretes.get(1).copied() else {
        return Ok(None);
    };
    // The operand must be a canonical builtin exception CLASS.  An already
    // built instance (`raise ValueError()`) is not in the class registry and
    // is handled by `try_walker_trace_raise_builtin`; a non-exception operand
    // (`raise obj`) also declines here.
    if concrete_class.is_null()
        || !pyre_object::interp_exceptions::is_canonical_exc_class(concrete_class)
    {
        return Ok(None);
    }

    // Explicit `raise X from Y` keeps the residual: `attach_raise_cause` sets
    // `__cause__` and `__suppress_context__`, which the inline `__context__`
    // store alone does not reproduce.  The cause operand is a const `PY_NULL`
    // (or a `Null` / `Ref(null)` shadow) when there is no cause; any concrete
    // non-null Ref is an explicit cause.
    let cause_op = r_args[2];
    let cause_is_null = match arg_concretes.get(2) {
        Some(ConcreteValue::Ref(p)) => p.is_null(),
        Some(ConcreteValue::Null) | None => matches!(
            ctx.trace_ctx.box_value(cause_op),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
        ),
        _ => false,
    };
    if !cause_is_null {
        return Ok(None);
    }

    // Build the exception concretely on the plain eval loop (no tracer
    // re-entry) to read its kind and confirm a flat builtin instance.  A
    // canonical class has the builtin `descr_new` / `descr_init`, so a
    // zero-argument construction runs no user code.  Trace-time only.
    let exc = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_class, &[])
    };
    let Ok(exc) = exc else { return Ok(None) };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    if pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind) != concrete_class {
        return Ok(None);
    }
    if !kind.has_trivial_args_constructor() {
        return Ok(None);
    }
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

    // Resolve the EC while declining is still free.  `walker_ensure_execution_
    // context` returns `None` on a null snapshot sym or a frameless walk, and
    // its recovery records a `GETFIELD_GC_R` that must not land after a guard
    // referencing it — `ensure_execution_context` recovers eagerly at walk
    // entry for that reason.  A decline past the commit below would also leave
    // the construction ops orphaned and the heap-cache shadows describing an
    // object the caller's generic-residual fall-through never built.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };

    // --- commit: pin the class identity, emit the construction + raise ---
    // Guard the class operand so the trace-time kind / vtable stay valid
    // across iterations (`implement_guard_value`).
    if !class_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_class as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[class_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(class_op, expected);
    }

    // Empty `args_w` list (zero-argument construction), stamped with the
    // canonical list class exactly as `w_list_new` does.
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &[]);
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

    let new_op =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, class_op, args_list);
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new_op, exc_type_ptr as usize as i64);
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));

    // `__context__` chaining on the still-virtual exception, mirroring the
    // `try_walker_trace_raise_builtin` tail: `active = GETFIELD_GC_R(ec,
    // sys_exc_value)` then `SETFIELD_GC(exc, active, w_context)`.  `ec` came
    // from `walker_ensure_execution_context` above, so the read shares the one
    // seeded EC OpRef the PUSH_EXC_INFO / POP_EXCEPT lowering consumes.
    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_op, active],
        crate::descr::w_exception_context_descr(kind),
    );
    // Apply the same context write to the concrete, freshly-built exception so
    // Python code reached later in this authoritative walk observes the
    // `__context__` the recorded SETFIELD performs on compiled iterations.
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }

    // Mark the inline-built instance FBW-built so the following `raise/r`
    // records its frame node via the virtual `record_fresh_application_
    // traceback` (an inline PyTraceback `NewWithVtable` + SETFIELDs on the
    // exception) rather than `record_top_level_application_traceback`, whose
    // runtime hook passes the exception to a `CallN` and forces it to
    // materialize — defeating the save/restore DCE that virtualizes a
    // locally-caught raise.  Mirrors `try_walker_trace_raise_builtin`.
    fbw_built_exc_insert(new_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new_op)?;
    Ok(Some(()))
}

/// Walker-native fold for the deterministic immutable-type
/// STORE_ATTR / DELETE_ATTR raise (`int.x = v` / `del str.x` →
/// TypeError from the `object_setattr` / `object_delattr` non-heaptype
/// guard, `typeobject.py:416/437`).
///
/// The generic path records the raise as an opaque
/// `CallMayForceN(bh_store_attr_fn)` + `GuardNotForced` +
/// `GuardException`, whose result box (the exception materialised
/// *inside* the residual by `PyError::to_exc_object`) can never
/// virtualize — every compiled iteration re-allocates the TypeError,
/// its message string, and its args list through the runtime GC hooks.
/// PyPy traces `space.setattr` itself, so the same raise shows up as
/// `new_with_vtable` + `setfield_gc` ops its optimizer removes when the
/// exception never escapes.
///
/// This fold restores that shape for the one attribute-store raise
/// whose outcome is provably iteration-invariant: when
/// `type_immutable_attr_raise_is_stable` holds (constant non-heaptype
/// receiver, canonical `type` metaclass, no metaclass descriptor for
/// `name` — every consulted dict frozen), the raise and its message
/// depend only on trace-time constants.  Pin the receiver with
/// `GuardValue` (when not already constant), run the authentic
/// `setattr_str` / `delattr_str` concretely for the authoritative
/// walk's exception, and emit the [`try_walker_trace_exception_new`]
/// construction (`NewWithVtable` + args-list `SetfieldGc`s, message as
/// a rooted trace constant) in place of the residual + guards.  The
/// raise then routes through the ordinary `SubRaise` path with a
/// virtualizable exception OpRef, and a locally-caught `except` DCEs
/// the whole allocation exactly as the explicit-`raise` B3 fold does.
///
/// Returns `None` (fall through to the generic residual) for any
/// non-matching or unprovable shape.
pub(crate) fn try_walker_trace_immutable_type_attr_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    obj_op: OpRef,
    store_value: Option<OpRef>,
    w_code_ptr: usize,
    name_idx: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || w_code_ptr == 0 {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj_op) else {
        return Ok(None);
    };
    // STORE_ATTR runs the authentic concrete `setattr_str` below only for
    // its raising exception.  The value plays no role in that raise — the
    // stability predicate proves no data descriptor for `name`, so the
    // terminal raises before consulting the value — so a non-constant store
    // value (the common `int.x = i` loop case) still folds: a `None`
    // concrete value substitutes a placeholder for the authentic run, and
    // the value operand never enters the emitted trace.
    let concrete_value = match store_value {
        Some(value_op) => {
            Some(walker_concrete_ref_object(ctx, value_op).unwrap_or_else(pyre_object::w_none))
        }
        None => None,
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
    if !pyre_interpreter::baseobjspace::type_immutable_attr_raise_is_stable(
        concrete_obj,
        &name,
        store_value.is_none(),
    ) {
        return Ok(None);
    }

    // The raise decision also reads the metaclass-MRO descriptor state — the
    // branch-F `lookup_in_type_where(type, name)` walk and the forwarding
    // `type.__setattr__` / `type.__delattr__` — which the receiver
    // `GuardValue` below does not cover.  A `version_tag` guard on the
    // metaclass pins that state (the guard the sibling method/attr folds
    // carry, `typeobject.py promote(self.version_tag())`): mutating `type`'s
    // dict bumps its tag directly, and mutating `object`'s dict bumps it too
    // because `mutated()` propagates down to the `type` subclass — so one
    // guard covers the whole `(type, object)` MRO the walk reads.  Branch C
    // proved the metaclass is the canonical `type`.  A tagless metaclass
    // (`version_tag == 0`) is uncacheable, so decline before emitting guards.
    let metaclass = pyre_object::get_instantiate(&pyre_object::pyobject::TYPE_TYPE);
    let metaclass_version_tag =
        unsafe { pyre_object::typeobject::w_type_get_version_tag(metaclass) };
    if metaclass_version_tag == 0 {
        return Ok(None);
    }

    // --- commit: pin the receiver, run the authentic raise, emit inline ---
    // The stability predicate makes the raise a pure function of `(obj,
    // name)`; `GuardValue` pins the one live input (`name` is a co_names
    // constant).
    if !obj_op.is_constant() {
        let expected = ctx.trace_ctx.const_ref(concrete_obj as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[obj_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx.heap_cache_mut().replace_box(obj_op, expected);
    }
    // Pin the metaclass `version_tag` (see above): a `GETFIELD_GC_I` +
    // `GuardValue` that side-exits on any `type`/`object` dict mutation.
    let metaclass_const = ctx.trace_ctx.const_ref(metaclass as i64);
    let vt_op = walker_record_getfield_gc_i_uncached(
        ctx,
        metaclass_const,
        crate::descr::type_version_tag_descr(),
    );
    let vt_const = ctx.trace_ctx.const_int(metaclass_version_tag as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[vt_op, vt_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx.heap_cache_mut().replace_box(vt_op, vt_const);

    // The authoritative walk's concrete execution — the same call the
    // residual executor would have made, raising before any heap
    // mutation.  Plain eval: the predicate excludes every user-code path.
    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        match concrete_value {
            Some(value) => pyre_interpreter::baseobjspace::setattr_str(concrete_obj, &name, value),
            None => pyre_interpreter::baseobjspace::delattr_str(concrete_obj, &name),
        }
    };
    let Err(mut err) = result else {
        // Unreachable under the predicate (a non-heaptype dict rejects
        // every mutation).  Fail loud rather than falling through: the
        // generic residual would re-run the (somehow) committed effect.
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "immutable-type attr raise fold: stable raise unexpectedly succeeded",
        });
    };
    let exc = err.to_exc_object();
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    // The folded raise is exactly the immutable-type TypeError; any other
    // kind means the runtime path diverged from the predicate's model.
    if kind != pyre_object::interp_exceptions::ExcKind::TypeError {
        return Ok(None);
    }
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

    // Message as a trace constant: deterministic per `(obj, name)` under
    // the predicate, so one shared immutable string is exact (the same
    // sharing a `raise TypeError("...")` gets from co_consts).  Pin the
    // fresh exception across the string allocation; the recorded ConstPtr
    // slot is forwarded across minor collections by the op-graph walker
    // and rooted by the compiled loop's gcref table thereafter.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(exc);
    let msg = pyre_object::w_str_from_wtf8(err.message.clone());
    let msg_const = ctx.trace_ctx.const_ref(msg as i64);
    let args_list = crate::helpers::emit_object_list_inline(ctx.trace_ctx, &[msg_const]);
    // Stamp the canonical list class exactly as `w_list_new` does (the
    // `try_walker_trace_exception_new` args tail), so a materialised
    // `args_w` still satisfies `space.type(args_w) is list`.
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

    let class_const = ctx
        .trace_ctx
        .const_ref(pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind) as i64);
    let new_op =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, class_const, args_list);
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(new_op, exc_type_ptr as usize as i64);
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    // Inline-built marker: the downstream raise routing records the frame
    // node via the virtual `record_fresh_application_traceback` instead of
    // the forcing runtime hook (mirrors `try_walker_trace_raise_bare_class`).
    fbw_built_exc_insert(new_op);

    // The residual-executor Err-arm state, minus the call itself: seed the
    // standing exception for the `SubRaise` routing (`execute_raised`
    // analogue) and restore the blackhole cell so an aborting walk still
    // delivers the pending raise to the live frame.  The class IS proven
    // constant here — the `NewWithVtable` vtable pins it.
    fbw_count_executed_residual(true, true);
    ctx.last_exc_value = Some(new_op);
    ctx.last_exc_value_concrete = ConcreteValue::Ref(exc);
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));

    Ok(Some((
        DispatchOutcome::SubRaise {
            exc: new_op,
            exc_concrete: ConcreteValue::Ref(exc),
        },
        op.next_pc,
    )))
}

/// B3 piece 3: lower the PUSH_EXC_INFO / POP_EXCEPT
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
            // resume.py applies pending fields before resumed
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

    // Tracing is execution (pyjitpl.py execute_and_record): apply the
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

    // `W_Zip.iter_w` is identity; exact-class guards preserve overrides.
    let zip_type = &pyre_object::functional::ZIP_TYPE as *const pyre_object::PyType;
    let zip_class = pyre_object::get_instantiate(&pyre_object::functional::ZIP_TYPE);
    if unsafe {
        !range_obj.is_null()
            && std::ptr::eq((*range_obj).ob_type, zip_type)
            && std::ptr::eq((*range_obj).w_class, zip_class)
    } {
        walker_guard_class(ctx, op_pc, range_op, zip_type as i64)?;
        walker_guard_exact_w_class(ctx, op_pc, range_op, zip_class)?;
        ctx.vstack_last_ref = range_op;
        return Ok(Some(range_op));
    }

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
fn try_walker_specialize_zip_two_tuple_iters<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    zip_op: OpRef,
    zip_obj: pyre_object::PyObjectRef,
) -> Result<Option<OpRef>, DispatchError> {
    if ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }

    let zip_type = &pyre_object::functional::ZIP_TYPE as *const pyre_object::PyType;
    let zip_class = pyre_object::get_instantiate(&pyre_object::functional::ZIP_TYPE);
    let (iterators_obj, inner_objs, steps, strict) = unsafe {
        if zip_obj.is_null()
            || !std::ptr::eq((*zip_obj).ob_type, zip_type)
            || !std::ptr::eq((*zip_obj).w_class, zip_class)
        {
            return Ok(None);
        }
        let iterators_obj = pyre_object::functional::w_zip_get_iterators(zip_obj);
        if iterators_obj.is_null()
            || !pyre_object::is_list(iterators_obj)
            || !pyre_object::is_exact_builtin_instance(iterators_obj)
        {
            return Ok(None);
        }
        let list = &*(iterators_obj as *const pyre_object::listobject::W_ListObject);
        if list.strategy != pyre_object::listobject::ListStrategy::Object
            || pyre_object::w_list_len(iterators_obj) != 2
        {
            return Ok(None);
        }
        let Some(inner0) = pyre_object::w_list_getitem(iterators_obj, 0) else {
            return Ok(None);
        };
        let Some(inner1) = pyre_object::w_list_getitem(iterators_obj, 1) else {
            return Ok(None);
        };

        let mut steps = Vec::with_capacity(2);
        for inner in [inner0, inner1] {
            if !pyre_object::is_tuple_iter(inner)
                || !std::ptr::eq((*inner).ob_type, &pyre_object::iterobject::TUPLE_ITER_TYPE)
            {
                return Ok(None);
            }
            let seq = pyre_object::w_tuple_iter_seq(inner);
            let index = pyre_object::w_tuple_iter_index(inner);
            if seq.is_null()
                || !pyre_object::is_tuple(seq)
                || !pyre_object::is_exact_builtin_instance(seq)
                || index < 0
            {
                return Ok(None);
            }
            let len = pyre_object::w_tuple_len(seq) as i64;
            let item = pyre_object::w_tuple_getitem(seq, index);
            steps.push((seq, index, len, item));
        }
        (
            iterators_obj,
            [inner0, inner1],
            steps,
            pyre_object::functional::w_zip_get_strict(zip_obj),
        )
    };
    let concrete_continues = steps.iter().all(|step| step.3.is_some());
    let concrete_exhausted = steps.iter().all(|step| step.1 >= step.2);
    // Mixed bounds must fall through to the interpreter's strict error path.
    if !concrete_continues && !(strict && concrete_exhausted) {
        return Ok(None);
    }

    // Reproduce PyPy's unrolled arity-two `W_Zip.next_w` shape.
    walker_guard_class(ctx, op_pc, zip_op, zip_type as i64)?;
    walker_guard_exact_w_class(ctx, op_pc, zip_op, zip_class)?;
    let iterators_op = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        zip_op,
        crate::descr::zip_iterators_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        iterators_op,
        Value::Ref(majit_ir::GcRef(iterators_obj as usize)),
    );

    let list_type = &pyre_object::LIST_TYPE as *const pyre_object::PyType as i64;
    walker_guard_class(ctx, op_pc, iterators_op, list_type)?;
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        iterators_op,
        pyre_object::get_instantiate(&pyre_object::LIST_TYPE),
    )?;
    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iterators_op,
        crate::descr::list_strategy_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        strategy,
        Value::Int(pyre_object::listobject::ListStrategy::Object as i64),
    );
    let object_strategy = ctx
        .trace_ctx
        .const_int(pyre_object::listobject::ListStrategy::Object as i64);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[strategy, object_strategy])?;
    let list_len = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iterators_op,
        crate::descr::list_length_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(list_len, Value::Int(2));
    let two = ctx.trace_ctx.const_int(2);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[list_len, two])?;
    let iterator_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        iterators_op,
        crate::descr::list_items_descr(),
    );

    let mut inner_ops = Vec::with_capacity(2);
    for (index, inner_obj) in inner_objs.into_iter().enumerate() {
        let index_op = ctx.trace_ctx.const_int(index as i64);
        let inner_op =
            crate::state::trace_items_block_getitem_value(ctx.trace_ctx, iterator_block, index_op);
        ctx.trace_ctx
            .set_opref_concrete(inner_op, Value::Ref(majit_ir::GcRef(inner_obj as usize)));
        inner_ops.push(inner_op);
    }

    // Guard both cursors before either is advanced.
    let tuple_iter_type =
        &pyre_object::iterobject::TUPLE_ITER_TYPE as *const pyre_object::PyType as i64;
    let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType as i64;
    let tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    let mut emitted_steps = Vec::with_capacity(2);
    let mut both_match = None;
    for (inner_op, (seq_obj, index, len, item_obj)) in
        inner_ops.iter().copied().zip(steps.iter().copied())
    {
        walker_guard_class(ctx, op_pc, inner_op, tuple_iter_type)?;
        let seq_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            inner_op,
            crate::descr::tuple_iter_seq_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(seq_op, Value::Ref(majit_ir::GcRef(seq_obj as usize)));
        walker_guard_class(ctx, op_pc, seq_op, tuple_type)?;
        walker_guard_exact_w_class(ctx, op_pc, seq_op, tuple_class)?;
        let raw_index = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            inner_op,
            crate::descr::tuple_iter_index_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(raw_index, Value::Int(index));
        let items = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            seq_op,
            crate::descr::tuple_wrappeditems_descr(),
        );
        let raw_len = crate::state::opimpl_arraylen_gc(
            ctx.trace_ctx,
            items,
            crate::state::pyobject_gcarray_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(raw_len, Value::Int(len));
        let matches_arm = if concrete_continues {
            let zero = ctx.trace_ctx.const_int(0);
            let nonnegative = ctx.trace_ctx.record_op(OpCode::IntGe, &[raw_index, zero]);
            ctx.trace_ctx
                .set_opref_concrete(nonnegative, Value::Int((index >= 0) as i64));
            let in_bounds = ctx
                .trace_ctx
                .record_op(OpCode::IntLt, &[raw_index, raw_len]);
            ctx.trace_ctx
                .set_opref_concrete(in_bounds, Value::Int((index < len) as i64));
            let matches = ctx
                .trace_ctx
                .record_op(OpCode::IntAnd, &[nonnegative, in_bounds]);
            ctx.trace_ctx
                .set_opref_concrete(matches, Value::Int((index >= 0 && index < len) as i64));
            matches
        } else {
            let matches = ctx
                .trace_ctx
                .record_op(OpCode::IntGe, &[raw_index, raw_len]);
            ctx.trace_ctx
                .set_opref_concrete(matches, Value::Int((index >= len) as i64));
            matches
        };
        both_match = Some(match both_match {
            None => matches_arm,
            Some(prior) => {
                let both = ctx
                    .trace_ctx
                    .record_op(OpCode::IntAnd, &[prior, matches_arm]);
                ctx.trace_ctx.set_opref_concrete(both, Value::Int(1));
                both
            }
        });
        emitted_steps.push((inner_op, raw_index, items, item_obj));
    }
    walker_emit_guard_with_snapshot(
        ctx,
        op_pc,
        OpCode::GuardTrue,
        &[both_match.expect("zip arity-two recognition emitted two cursors")],
    )?;

    let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body);

    if concrete_exhausted {
        // PyPy clears both exhausted tuple iterators before StopIteration.
        let strict_op = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            zip_op,
            crate::descr::zip_strict_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(strict_op, Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[strict_op])?;
        let null_ref = ctx.trace_ctx.const_ref(0);
        for (inner_op, step) in inner_ops.into_iter().zip(steps.iter()) {
            let seq_descr = crate::descr::tuple_iter_seq_descr();
            ctx.trace_ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[inner_op, null_ref],
                seq_descr.clone(),
            );
            ctx.trace_ctx
                .heapcache_setfield_cached(inner_op, seq_descr.index(), null_ref);
            let inner_obj = walker_concrete_ref_object(ctx, inner_op)
                .expect("zip tuple iterator concrete survived exhaustion emission");
            if ctx.trace_ctx.is_bridge_trace {
                let pre_seq = unsafe { pyre_object::w_tuple_iter_seq(inner_obj) };
                fbw_bridge_tuple_iter_journal_push(inner_obj, pre_seq, step.1);
            }
            unsafe { pyre_object::w_tuple_iter_set_seq(inner_obj, pyre_object::PY_NULL) };
        }
        let zero = ctx.trace_ctx.const_int(0);
        let null_item = ctx.trace_ctx.record_op(OpCode::CastIntToPtr, &[zero]);
        ctx.trace_ctx
            .set_opref_concrete(null_item, Value::Ref(majit_ir::GcRef(0)));
        return Ok(Some(null_item));
    }

    let one = ctx.trace_ctx.const_int(1);
    let mut item_ops = Vec::with_capacity(2);
    for (inner_op, raw_index, items, item_obj) in emitted_steps {
        let item_op =
            crate::state::trace_items_block_getitem_value_pure(ctx.trace_ctx, items, raw_index);
        let next_index = ctx.trace_ctx.record_op(OpCode::IntAdd, &[raw_index, one]);
        let concrete_index = steps[item_ops.len()].1;
        ctx.trace_ctx
            .set_opref_concrete(next_index, Value::Int(concrete_index + 1));
        let index_descr = crate::descr::tuple_iter_index_descr();
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[inner_op, next_index],
            index_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(inner_op, index_descr.index(), next_index);
        let item_obj = item_obj.expect("continue-arm zip step has an item");
        ctx.trace_ctx
            .set_opref_concrete(item_op, Value::Ref(majit_ir::GcRef(item_obj as usize)));
        item_ops.push(item_op);
    }

    let tuple_op =
        crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, item_ops[0], item_ops[1]);

    // Advance the authentic shadows and retain the yielded pair for abort.
    let concrete_tuple = pyre_object::w_specialised_tuple_oo_new(
        steps[0].3.expect("continue-arm zip step has item 0"),
        steps[1].3.expect("continue-arm zip step has item 1"),
    );
    if concrete_tuple.is_null() {
        return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op_pc });
    }
    for (inner_op, step) in inner_ops.into_iter().zip(steps.iter()) {
        let inner_obj = walker_concrete_ref_object(ctx, inner_op)
            .expect("zip tuple iterator concrete survived tuple allocation");
        if ctx.trace_ctx.is_bridge_trace {
            let pre_seq = unsafe { pyre_object::w_tuple_iter_seq(inner_obj) };
            fbw_bridge_tuple_iter_journal_push(inner_obj, pre_seq, step.1);
        }
        unsafe { pyre_object::w_tuple_iter_set_index(inner_obj, step.1 + 1) };
    }
    let concrete_item0 = unsafe {
        pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(concrete_tuple, 0)
    };
    let concrete_item1 = unsafe {
        pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(concrete_tuple, 1)
    };
    for (item_op, concrete_item) in item_ops.into_iter().zip([concrete_item0, concrete_item1]) {
        ctx.trace_ctx
            .set_opref_concrete(item_op, Value::Ref(majit_ir::GcRef(concrete_item as usize)));
    }
    ctx.trace_ctx.set_opref_concrete(
        tuple_op,
        Value::Ref(majit_ir::GcRef(concrete_tuple as usize)),
    );
    fbw_foriter_inflight_capture(concrete_tuple, body);
    ctx.vstack_last_ref = tuple_op;
    Ok(Some(tuple_op))
}

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
    if unsafe { pyre_object::functional::is_zip(iter_obj) } {
        return try_walker_specialize_zip_two_tuple_iters(ctx, op_pc, iter_op, iter_obj);
    }
    let (concrete_current, concrete_remaining, concrete_step) = unsafe {
        if !pyre_object::functional::is_range_iter(iter_obj) {
            return Ok(None);
        }
        pyre_object::functional::w_range_iter_fields(iter_obj)
    };
    let concrete_continues = concrete_remaining != 0;

    // A new consume attempt completes the prior in-flight iteration before
    // this irreversible concrete advance, matching the residual executor.
    let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
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
    // the wrapping IntAdd and live-iterator SetfieldGc updates intact.
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

    if ctx.trace_ctx.is_bridge_trace {
        // A bridge/retrace recording walk has no in-flight forward-delivery on
        // abort, so journal the pre-advance cursor for restore if the walk does
        // not commit (keeps the aborted recording side-effect neutral).
        fbw_bridge_iter_journal_push(iter_obj, concrete_current, concrete_remaining);
    }
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
    // int_items blocks (`list_int_items_block_descr`, matching
    // `emit_typed_list_inline`'s `SetfieldGc`), so a virtual source temp's
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

    // Tracing is execution (pyjitpl.py execute_and_record): apply the
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

    // floatobject.py:139-146 — an int wider than a double represents exactly
    // is compared through its bigint, which this fold cannot express.  Decline
    // so the residual call decides it; the in-range case emits the same
    // precondition as a guard (`exact_int` below).
    let out_of_range = |is_int: bool, obj| {
        is_int && !int_is_exact_as_float(unsafe { pyre_object::w_int_get_value(obj) })
    };
    if out_of_range(lhs_is_int, lhs_obj) || out_of_range(rhs_is_int, rhs_obj) {
        return Ok(None);
    }

    // --- emit the specialized IR (walker-native) ---
    let lhs_raw =
        walker_coerce_operand_to_float(ctx, op_pc, lhs, lhs_obj, lhs_is_int, lhs_f64, true)?;
    let rhs_raw =
        walker_coerce_operand_to_float(ctx, op_pc, rhs, rhs_obj, rhs_is_int, rhs_f64, true)?;
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
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, folded != 0, dst as u8, dst_bank)? {
        // The guarded arm already pinned the truth to a constant and filed
        // `bool_box_truth_record` against it, so the following `is_true`
        // folds without re-reading the runtime truth.
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(boxed_result_i64 as usize)),
            );
            // #62: remember boxed→truth so an immediately-following `is_true` residual
            // (POP_JUMP_IF_*) folds back to the raw Int instead of may-force-unboxing.
            bool_box_truth_record(boxed, truth);
            boxed
        }
    };
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
/// Callers fall back to the residual call when this fold declines. When the
/// loaded global is a function that is then CALLed, folding it to a
/// loop-invariant constant callee routes the call through the FBW call-inlining
/// path (#68).
///
/// Builtins fallback: when `name` is ABSENT from the
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
    if w_code_ptr == 0 {
        return Ok(false);
    }
    let w_globals = ns_ptr as pyre_object::PyObjectRef;
    // The namespace operand is the fold's authority: both legs end at
    // `guard_current_frame_globals_identity`, which bakes it as the expected
    // `ConstPtr` and declines outright on a null one.  An inlined callee whose
    // namespace register is unseeded presents it as a null `Ref`, so decline
    // here instead of walking the builtins leg, which reads `__builtins__`
    // straight out of it.  The residual re-resolves the globals from the frame
    // it runs on, so declining stays correct.
    if w_globals.is_null() {
        return Ok(false);
    }
    // Raw dict access in the inlined-callee builtins leg below is valid only
    // for an exact plain dict or module dict.  Dict subclasses are legal exec
    // namespaces but use a different object layout, so leave them to the live
    // residual lookup.
    if !unsafe { pyre_object::is_dict(w_globals) } {
        return Ok(false);
    }
    // `namei` is the raw `LOAD_GLOBAL` oparg; bit 0 is the push-NULL flag,
    // so the `co_names` index is `namei >> 1` (mirror `bh_load_global_fn`).
    let name_idx = (namei as usize) >> 1;
    let name = unsafe {
        // The wrapper being non-null does not make its `code_ptr` non-null:
        // `w_code_new_with_hidden_applevel` (pycode.rs:386) leaves the field
        // null for a gateway builtin or a test fixture, and every sibling
        // name lookup screens it the same way.
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return Ok(false);
        }
        let code = &*(code_ptr as *const pyre_interpreter::CodeObject);
        match pyre_interpreter::pyframe::load_name_from_code(code, name_idx) {
            Some(n) => n.to_string(),
            None => return Ok(false),
        }
    };
    if emit_module_dict_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, &name)? {
        return Ok(true);
    }

    // Builtins fallback: the name is absent from the
    // `ns_ptr` module dict.  Mirror `bh_load_global_fn`'s second leg —
    // `frame.get_builtin().getdictvalue(name)` — and fold the builtins cell
    // when the name resolves there.  Requires the live frame operand.
    // The builtins fallback needs the module `pick_builtin(w_globals)` picks
    // (`frame.get_builtin()`).  A live frame supplies it directly and also lets
    // us double-check the operand against the frame's AUTHORITATIVE globals
    // — `bh_load_global_fn` re-resolves the globals it consults from the LIVE
    // frame (`frame.get_w_globals()` when the frame owns `w_code`, else the
    // code's bound globals) and IGNORES the `namespace_ptr` operand.  The
    // `ns_ptr` hint usually equals that live dict; when it does not, nothing
    // here can prove what the residual would read, so decline.
    // An INLINED callee has no materialised frame (`frame_ptr == 0`, its
    // `portal_frame_reg` unseeded); derive the builtin module from the concrete
    // globals' `__builtins__` cell instead — the same object `pick_builtin`
    // resolves (baseobjspace.rs:9716) and the one the interpreter fallback would
    // rebuild for the resumed callee frame.  #670 keeps `__builtins__` in every
    // module dict, `ns_ptr` is the callee's own namespace field (so it is the
    // authoritative globals), and guard (a) below watches the globals `version`,
    // so a later `__builtins__` rebind fails the loop exactly as a
    // shadowing-name insert would.
    let w_builtin = if frame_ptr != 0 {
        let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::PyFrame) };
        let live_globals = if frame.pycode as usize == w_code_ptr {
            frame.get_w_globals()
        } else {
            unsafe {
                pyre_interpreter::w_code_get_w_globals(w_code_ptr as pyre_object::PyObjectRef)
            }
        };
        // Only the SAME dict makes the absence provable.  `module_dict_cell_slot_direct`
        // answers `None` both for a name that is absent and for a dict it cannot
        // read at all — a plain dict, or a module dict that ran
        // `switch_to_object_strategy` — so on a different dict its `None` says
        // nothing.  Guard (a) below pins `w_globals`' version, which watches the
        // wrong dict in that case, and the residual it replaces resolves
        // `live_globals`; a name present there would read the builtin instead of
        // the global.
        if live_globals.is_null() || live_globals as usize != w_globals as usize {
            return Ok(false);
        }
        frame.get_builtin()
    } else {
        unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }
            .unwrap_or(pyre_object::PY_NULL)
    };
    emit_builtins_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, w_builtin, &name)
}

/// Builtins-fallback half of the LOAD_GLOBAL and module-scope LOAD_NAME cell
/// folds: the name resolves through the frame's builtin module rather than the
/// module dict.  Mirrors `_load_global`'s second leg,
/// `get_builtin().getdictvalue(varname)`, and is reached only once
/// [`emit_module_dict_cell_fold`] has declined.
///
/// Two guards carry it.  (a) The name must stay ABSENT from the module dict,
/// which pinning that dict's `version?` is what proves: the insert that would
/// shadow the builtin runs `mutated()` and fails GUARD_NOT_INVALIDATED.
/// (b) The builtins value itself folds through [`emit_namespace_cell_fold`],
/// whose `QUASIIMMUT_FIELD` on the builtins dict fails the loop on a rebind or
/// delete there.
///
/// Returns `Ok(false)` — the caller then keeps the live residual — for a name
/// still present in the module dict, a missing or non-module builtin, an
/// unfoldable builtins slot (absent / null / `IntMutableCell` / movable), or a
/// movable builtins dict.
fn emit_builtins_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    w_globals: pyre_object::PyObjectRef,
    w_builtin: pyre_object::PyObjectRef,
    name: &str,
) -> Result<bool, DispatchError> {
    // `emit_module_dict_cell_fold` returns `false` for BOTH an absent name and
    // a present-but-unfoldable one (`IntMutableCell` / strategy switched).
    // Only an ABSENT name may fall through to the builtins fold — a
    // present global shadows the builtin, so keep the residual (which reads the
    // live globals slot) when the slot still exists.
    if crate::state::module_dict_cell_slot_direct(w_globals, name).is_some() {
        return Ok(false);
    }
    if w_builtin.is_null() || !unsafe { pyre_object::is_module(w_builtin) } {
        return Ok(false);
    }
    let w_builtin_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
    if w_builtin_dict.is_null() {
        return Ok(false);
    }
    let Some(b_slot) = crate::state::module_dict_cell_slot_direct(w_builtin_dict, name) else {
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
    // keeps falling through to builtins.  Pinning the module dict's `version?`
    // is what proves that: the new-key insert that would shadow the builtin
    // runs `mutated()`, which fails GUARD_NOT_INVALIDATED.  It is the same
    // field a present-name fold on this namespace pins, so the two share one
    // marker.
    if !guard_current_frame_globals_identity(ctx, op_pc, w_globals)? {
        return Ok(false);
    }
    if !walker_pin_namespace_version(ctx, op_pc, w_globals)? {
        return Ok(false);
    }
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
/// pyframe.rs), so `load_name_value`'s probe + LOAD_GLOBAL fallthrough
/// both resolve in `w_globals` — the same dict the global cell fold reads.
/// A non-module frame (class body / `exec(code, g, l)` with separate locals)
/// has a non-null `w_locals`, so the gate routes it to the live
/// residual `bh_load_name_fn`.
///
/// Builtins fallback: when `name` is absent from the module dict, module-scope
/// `LOAD_NAME` falls through via `load_global_value` to
/// `frame.get_builtin().getdictvalue(name)`.  The builtins cell fold pins the
/// module dict `version?` so a later global insertion that shadows the builtin
/// fails GUARD_NOT_INVALIDATED, then folds the builtins dict cell like the
/// LOAD_GLOBAL fallback.
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
    // `w_locals = w_globals` (pyframe.py); a `w_locals`
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
    if emit_module_dict_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, name)? {
        return Ok(true);
    }
    emit_builtins_cell_fold(
        ctx,
        op_pc,
        dst,
        dst_bank,
        w_globals,
        frame.get_builtin(),
        name,
    )
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
    helper: majit_ir::PyreHelperKind,
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
    // STORE_NAME writes `get_or_create_w_locals`, so only a module frame —
    // where `w_locals` aliases `w_globals` — targets the dict this folds.  An
    // ABSENT `w_locals` does NOT stand in for globals on the write path (unlike
    // the LOAD fold): the store would land in a fresh locals mapping while the
    // fold set the module cell.  STORE_GLOBAL names globals outright, and its
    // frame is a function frame whose `w_locals` is legitimately null, so the
    // gate must not apply to it.
    if helper == majit_ir::PyreHelperKind::StoreName {
        let w_locals = frame.get_w_locals();
        if !std::ptr::eq(w_locals, w_globals) {
            return Ok(false);
        }
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
    // The stored value must be a provably-plain-int box. `is_plain_int1` accepts
    // a fits-int `W_LongObject`, whose `write_cell` REPLACES the cell rather than
    // mutating `intvalue`, so exclude `long` explicitly; the remaining int box's
    // raw `intvalue` (populated only by JIT int boxes, `emit_box_int_inline`) is
    // recovered by the heapcache lookup, so the setfield needs no runtime class
    // guard — exactly as pypy's optimized trace folds the `is_plain_int1` check
    // away for an `int_add` result. (bool / int-subclass are already excluded by
    // `is_plain_int1`.)
    let is_plain_int = matches!(
        ctx.trace_ctx.box_value(value_opref),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(p)))
            if p != 0
                && unsafe { pyre_object::listobject::is_plain_int1(p as pyre_object::PyObjectRef) }
                && !unsafe { pyre_object::is_long(p as pyre_object::PyObjectRef) }
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
    emit_namespace_cell_store_fold(ctx, op_pc, w_globals, slot, stored, raw_int, new_int)
}
