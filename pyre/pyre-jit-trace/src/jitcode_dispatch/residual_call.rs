//! Residual-call dispatch: the tracer path for a callee that cannot be
//! folded or inlined and must be recorded as a residual `CALL_*` operation.
//!
//! **Parity:** trace-side counterpart of `pyjitpl.py`'s
//! `opimpl_residual_call_*`; the executor fast paths call into
//! `majit-metainterp/executor.rs` (`executor.py`). PyPy keeps these opimpls
//! inside `pyjitpl.py`'s `MIFrame`; the split into this file is pyre-local
//! navigability, not a PyPy file boundary.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. Covers the per-shape
//! dispatchers (`dispatch_residual_call_{iRd,iIRd,iIRFd}_kind`), the
//! executor fast paths (`try_fold_pure_call_via_executor`,
//! `try_execute_residual_call_via_executor`), opcode selection and arg
//! binding, the pre-call vable/vref sync, result writeback, and the
//! residual-call body classification helpers. The `residual_call_*` opname
//! arms themselves stay in `handle` (mod.rs) and call into these.

use super::*;

/// Reject a residual_call whose `allboxes` (funcbox + permuted args)
/// contains an `OpRef::NONE`.  RPython's `do_residual_call` resolves
/// each argbox through `env[box]`, where an unbound box is a `KeyError`;
/// recording the op anyway lets `OpRef::NONE` reach the backend's
/// `resolve_opref`, which aborts the process.  Returning
/// `ResidualCallArgUnbound` instead lets the outer walker fall back to a
/// trace abort — the same graceful outcome a pre-seam inline arm reached
/// when its payload read surfaced `GotoIfNotValueNotConcrete`.
pub(crate) fn ensure_residual_call_args_bound(
    allboxes: &[OpRef],
    pc: usize,
) -> Result<(), DispatchError> {
    if let Some(arg_index) = allboxes.iter().position(|b| b.is_none()) {
        return Err(DispatchError::ResidualCallArgUnbound { pc, arg_index });
    }
    Ok(())
}

/// EffectInfo-driven opcode selector shared by `dispatch_residual_call_*`
/// dispatchers. Mirrors `pyjitpl.py:1995-2126 do_residual_call`'s
/// precedence:
///   1. **forces branch** (`pyjitpl.py:2007-2082`): outer check on
///      `assembler_call or check_forces_virtual_or_virtualizable()`
///      records `CALL_MAY_FORCE_*` at step 2 and unconditionally fires
///      `GUARD_NOT_FORCED` (`:2079`).  The release-gil sub-case
///      (`pyjitpl.py:2063 if effectinfo.is_call_release_gil()`) is
///      handled by [`direct_call_release_gil`] **before** this
///      selector is called — the dispatcher early-returns on
///      `ei.is_call_release_gil()` so this function only ever sees
///      EI values where the sub-case is not active.
///   2. `EF_LOOPINVARIANT` (`:2087-2110`): `CALL_LOOPINVARIANT_*`.
///   3. `check_is_elidable()` (`:2112-2126`): `CALL_PURE_*`.
///   4. default (`:2126`): plain `CALL_*`.
///
/// Returns the `Call*` opcode for the call itself, whether
/// `handle_possible_exception` should emit `GUARD_NO_EXCEPTION`
/// (`check_can_raise(False)`), and whether the unconditional
/// `GUARD_NOT_FORCED` from the forces branch (`pyjitpl.py:2079`)
/// should fire.
///
/// ★ Task #390 sub-slice 1 — non-elidable concrete-execute inventory.
///
/// PyPy `do_residual_call` (pyjitpl.py:1995-2126) concrete-executes
/// the helper at trace-record time across the **forces** /
/// **loopinvariant** (cache miss) / **elidable** / **default**
/// branches via `executor.execute_varargs(opnum, argboxes, descr,
/// exc=can_raise, pure=is_elidable)`.  Two narrower branches sit
/// outside this uniform call:
///   * `OS_NOT_IN_TRACE` short-circuits through
///     `do_not_in_trace_call` (`pyjitpl.py:2003-2006`) and never
///     reaches `executor.execute_varargs`.
///   * `is_call_release_gil()` runs the helper through
///     `do_call_release_gil` (`pyjitpl.py:3671-3681`), invoking
///     `executor.execute_varargs` directly **before** the recorded
///     `CALL_RELEASE_GIL_*` op is emitted.
/// The recorded opcode kind selected above is for the IR trace
/// only; concrete execution either fired (or was intentionally
/// skipped via the two narrow branches above) before the trace op
/// hits the recorder.
///
/// Per-branch concrete-execute status (sub-slices 3 + 4 landed — every
/// non-pure residual call now concrete-executes during the walk, matching
/// PyPy `do_residual_call` which runs `executor.execute_varargs` for the
/// whole forces branch regardless of EI):
///
/// | EI branch | Selected op | Pyre walker concrete-execute |
/// |---|---|---|
/// | `is_call_release_gil()` | (early-routed to [`direct_call_release_gil`], records `CALL_RELEASE_GIL_*`) | executed as `CallMayForce*` on the **original** `allboxes` via [`try_execute_residual_call_via_executor`] (sub-slice 4) |
/// | `check_forces_virtual_or_virtualizable()` | `CallMayForce*` + `GuardNotForced` | executed via [`try_execute_residual_call_via_executor`] (active vable bracketed by the token protocol) |
/// | `extraeffect == LoopInvariant` | `CallLoopinvariant*` | executed on cache miss; [`loopinvariant_lookup`] reuses the cached OpRef on hit (no execute, no record) |
/// | `check_is_elidable()` | `CallPure*` | executed + cached via [`try_fold_pure_call_via_executor`] (elidable_cannot_raise only — see its caveats) |
/// | default | `Call*` + (`GuardNoException` iff can_raise) | executed via [`try_execute_residual_call_via_executor`] |
///
/// All three dispatch entry points — [`dispatch_residual_call_iRd_kind`]
/// (`_opimpl_residual_call1`), [`dispatch_residual_call_iIRd_kind`]
/// (`_opimpl_residual_call2`), [`dispatch_residual_call_iIRFd_kind`]
/// (`_opimpl_residual_call3`) — call [`select_residual_call_opcode`],
/// `record_op_with_descr`, then [`try_fold_pure_call_via_executor`] (pure)
/// + [`try_execute_residual_call_via_executor`] (non-pure).  The executor
/// self-gates and degrades to recording-only when a precondition fails
/// (non-authoritative walk, non-const funcbox, unpatched symbolic fnaddr).
///
/// **Priority order for sub-slice 2 (widen):** Call* (default — the
/// store_subscr_fn / set_current_exception class, smallest blast
/// radius, root cause of M4 SIGBUS) → CallLoopinvariant* (`pure=False
/// exc=False` is safest) → CallMayForce* (riskiest — force-virtual
/// audit precondition).
pub(crate) fn select_residual_call_opcode(
    ei: &majit_ir::EffectInfo,
    dst_bank: char,
    caller: &'static str,
) -> (OpCode, bool, bool) {
    // Release-gil sub-case is handled by `direct_call_release_gil`
    // before this selector runs.  Any `is_call_release_gil()` EI
    // reaching here is a dispatcher bug.
    debug_assert!(
        !ei.is_call_release_gil(),
        "{caller}: select_residual_call_opcode received an is_call_release_gil() EI; \
         dispatcher should have routed via direct_call_release_gil first"
    );
    let (call_op, pure_op, may_force_op, loopinvariant_op): (OpCode, OpCode, OpCode, OpCode) =
        match dst_bank {
            'r' => (
                OpCode::CallR,
                OpCode::CallPureR,
                OpCode::CallMayForceR,
                OpCode::CallLoopinvariantR,
            ),
            'i' => (
                OpCode::CallI,
                OpCode::CallPureI,
                OpCode::CallMayForceI,
                OpCode::CallLoopinvariantI,
            ),
            // `_irf_f/iIRFd>f` (`pyjitpl.py:1354 opimpl_residual_call_irf_f =
            // _opimpl_residual_call3`, `blackhole.py:1250 bhimpl_residual_call_irf_f`).
            // `resoperation.py:1462 Type::Float => CallF`. The `_r_f` /
            // `_ir_f` shapes do not exist upstream — the only float-result
            // residual_call variant routes through the `iIRFd` arglist.
            'f' => (
                OpCode::CallF,
                OpCode::CallPureF,
                OpCode::CallMayForceF,
                OpCode::CallLoopinvariantF,
            ),
            // `_*_v/iRd|iIRd|iIRFd` void variants (`pyjitpl.py:1348
            // opimpl_residual_call_r_v = _opimpl_residual_call1`,
            // `:1351 opimpl_residual_call_ir_v = _opimpl_residual_call2`,
            // `:1355 opimpl_residual_call_irf_v = _opimpl_residual_call3`,
            // `blackhole.py:1245/1248/1253 bhimpl_residual_call_*_v`).
            // `resoperation.py:1463 Type::Void => CallN`. No dst writeback;
            // `write_residual_call_result_to_dst` no-ops on 'v'.
            'v' => (
                OpCode::CallN,
                OpCode::CallPureN,
                OpCode::CallMayForceN,
                OpCode::CallLoopinvariantN,
            ),
            _ => panic!("{caller}: unsupported dst_bank '{dst_bank}'"),
        };
    if ei.check_forces_virtual_or_virtualizable() {
        // pyjitpl.py:2017-2082 forces-virtual-or-virtualizable branch
        // proper: CALL_MAY_FORCE_* + GUARD_NOT_FORCED.
        // `handle_possible_exception` also fires (forces always
        // satisfies check_can_raise).
        (may_force_op, ei.check_can_raise(false), true)
    } else if ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant {
        // pyjitpl.py:2087-2110 EF_LOOPINVARIANT branch: CALL_LOOPINVARIANT_*
        // via miframe_execute_varargs(..., exc=False). LoopInvariant
        // never raises (extraeffect=1 < CannotRaise=2 → check_can_raise=False).
        //
        // The `pyjitpl.py:2088 call_loopinvariant_known_result` lookup
        // and `pyjitpl.py:2109 call_loopinvariant_now_known` cache
        // update are wired at the dispatcher level via
        // [`loopinvariant_lookup`] and [`loopinvariant_now_known`]
        // around the `record_op_with_descr` call — they require the
        // dispatcher's `descr_index` and `arg0_int` so this opcode
        // selector cannot perform them on its own.
        (loopinvariant_op, ei.check_can_raise(false), false)
    } else if ei.check_is_elidable() {
        // pyjitpl.py:2112 + 2126 elidable branch: CALL_PURE_*.
        (pure_op, ei.check_can_raise(false), false)
    } else {
        // pyjitpl.py:2126 default branch: CALL_*.
        (call_op, ei.check_can_raise(false), false)
    }
}

/// `pyjitpl.py:_record_helper_pure` (`pyjitpl.py:1346-1400`) parity for the
/// walker layer: when a residual_call routes to `CallPure*` (elidable +
/// cannot-raise EI per [`select_residual_call_opcode`]) AND every
/// argument in `allboxes` has a known concrete value
/// (`TraceCtx::box_value` returns `Some`), execute the helper at trace
/// time via [`majit_metainterp::executor::execute_pure_call`] and stamp
/// `recorded` with the result.
///
/// PyPy `_opimpl_*` methods (e.g. `_opimpl_setitem`,
/// `_opimpl_setfield_*`) concrete-execute every `do_residual_call`
/// regardless of `check_is_elidable()` — the EI flag only selects the
/// recorded opcode kind (`CALL_PURE_*` vs `CALL_*`), not whether the
/// helper runs.  This function covers the elidable arm; the
/// non-elidable arms (`CallMayForce*`, `CallLoopinvariant*`, `Call*`)
/// are concrete-executed by [`try_execute_residual_call_via_executor`]
/// with raised exceptions surfaced through `BH_LAST_EXC_VALUE` so
/// `eval_loop_jit` can route them into the bytecode exception handler.
///
/// RPython upstream `_record_helper_pure` invokes
/// `executor.execute_varargs(opnum, argboxes, descr, exc=False, pure=True)`
/// which dispatches to `cpu.bh_call_*` and stores the result on
/// `result_box.value` (`pyjitpl.py:1392`).  Pyre's walker observes the
/// same effect through the `set_opref_concrete` stamp — downstream walker
/// chain (sub-jitcode bodies that consume the call result via
/// `concrete_of_opref`) folds end-to-end instead of stalling at
/// `RefOp/IntOp(N)` unknown values.
///
/// **Caller contract**:
/// * `call_opcode` must be one of `CallPureI`/`CallPureR`/`CallPureF`/
///   `CallPureN` — the `select_residual_call_opcode` elidable arm
///   (`pyjitpl.py:2126` proper, `dispatch.rs:2688-2690`).  Other call
///   shapes (`CallMayForce*`, `CallLoopinvariant*`, `Call*`) carry
///   `can_raise=true` or escape semantics that require the full
///   `execute_varargs` MetaInterp seam — they MUST NOT route here.
/// * `allboxes[0]` is the funcbox (per `build_allboxes` layout); the
///   remaining slots are user args in `descr.arg_types()` ABI order.
///
/// Best-effort: returns silently when any operand lacks a concrete
/// `box_value` (the walker has no way to read the runtime value), or
/// when the arity exceeds `MAX_HOST_CALL_ARITY` (16) — the trace still
/// has the recorded `CallPure*` op for the optimizer to consume later,
/// just without the per-record fold.
pub(crate) fn try_fold_pure_call_via_executor<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    recorded: OpRef,
) {
    if !matches!(
        call_opcode,
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN
    ) {
        return;
    }
    // pyjitpl.py:1351-1352 — `_record_helper_pure` only fires for
    // `EF_ELIDABLE_CANNOT_RAISE`. `select_residual_call_opcode` returns
    // `CallPure*` whenever `check_is_elidable()` is true (including
    // `EF_ELIDABLE_CAN_RAISE`), so re-check the can-raise predicate here
    // before dispatching through the `execute_pure_call` no-metainterp
    // carve-out.  A `EF_ELIDABLE_CAN_RAISE` callee would silently swallow
    // the exception via `BH_LAST_EXC_VALUE` with no metainterp to
    // transcribe it.
    let ei = call_descr.get_extra_info();
    if ei.check_can_raise(false) {
        return;
    }
    if allboxes.is_empty() {
        return;
    }
    // pyjitpl.py:1960-1993 `_build_allboxes`: slot 0 is funcbox, slots
    // 1.. are user args in `descr.arg_types()` ABI order.  Walker's
    // [`build_allboxes`] preserves the same layout.
    //
    // pyjitpl.py:1352 invariant: `_record_helper_pure` requires
    // `funcbox` to be a Const so its `getint()` is the actual fn
    // pointer.  Non-constant funcboxes carry a stale-stamped Int
    // (from `cast_ptr_to_int` of a Ref-bank receiver, etc.) and
    // dereferencing as a code address yields SIGSEGV.  Skip the fold
    // when the funcbox is non-constant; the recorded `CallPure*` op
    // stays in the trace for the optimizer to consume later.
    if !allboxes[0].is_constant() {
        return;
    }
    let funcptr_val = ctx.trace_ctx.box_value(allboxes[0]);
    let func_ptr = match funcptr_val {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return,
    };
    // Cap at MAX_HOST_CALL_ARITY (`call_int_function` / `call_void_function`
    // panic on excess arity).  `allboxes.len() - 1` is the arg count
    // (funcbox doesn't pass through).
    if allboxes.len() - 1 > majit_translate::codewriter::insns::MAX_HOST_CALL_ARITY {
        return;
    }
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for &arg in &allboxes[1..] {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                // `usize::MAX` sentinel from `concrete_of_opref` means
                // "no concrete known" — never reach this path because
                // `box_value` returns `None` for un-stamped OpRefs, but
                // belt-and-suspenders against future plumbing.
                if r == majit_ir::GcRef::NO_CONCRETE {
                    return;
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => return,
        };
        args.push(v);
    }
    // Refuse to invoke the helper when any Ref argument is NULL.  Pyre's
    // getfield_gc_r walker handler propagates field reads (including
    // pointer-valued fields like `PyFrame.f_back`) as concrete values
    // when the parent struct is concrete-known; a top-level frame
    // returns NULL for `f_back`, stamping `Value::Ref(GcRef(0))` into
    // the constant pool.  Folding `helper(NULL)` would then dereference
    // NULL and SEGV.  PyPy avoids this because its optimizer inserts
    // `guard_nonnull` ahead of any pointer-deref residual call; pyre's
    // walker folds before that guard exists, so guard the executor
    // entry against NULL receivers and fall through to recording the
    // IR op as-is.  The downstream optimizer then sees the call op and
    // emits the necessary guards.
    for (i, &arg) in args.iter().enumerate() {
        if matches!(call_descr.arg_types().get(i), Some(majit_ir::Type::Ref)) && arg == 0 {
            return;
        }
    }
    let result_i64 = majit_metainterp::executor::execute_pure_call(call_descr, func_ptr, &args);
    // pyjitpl.py:1392 `result_box.value = result`: stamp the recorded
    // OpRef with the executed concrete so downstream
    // `concrete_of_opref` / `box_value` consumers see the folded value.
    let result_value = match call_descr.result_type() {
        majit_ir::Type::Int => majit_ir::Value::Int(result_i64),
        majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(result_i64 as usize)),
        majit_ir::Type::Float => majit_ir::Value::Float(f64::from_bits(result_i64 as u64)),
        // void callees discard the result upstream too (`bh_call_v` has
        // no return value); `CallPureN` is included in the matched set
        // only to mirror PyPy's `_record_helper_pure` handling of all
        // pure shapes — skip the stamp for void.
        majit_ir::Type::Void => return,
    };
    // Stamp only when the recorded result has a live slot in the active
    // recorder.  A deeper inlined / recursive frame's residual result may be
    // recorded in a context whose position is not allocated in the active
    // recorder; stamping it would violate the `*FrontendOp(pos, value)`
    // invariant.  Skipping leaves the result symbolic so the downstream branch
    // aborts the trace into the trait fallback instead of crashing.
    ctx.trace_ctx.try_set_opref_concrete(recorded, result_value);
}

/// Abort the walk when a result-bearing may-force CALL is recorded with a
/// concrete-NULL Ref argument — the specialized direct-call shape whose
/// baked `ptr(0x0)` (the `PUSH_NULL` self-slot) makes the runtime call pass
/// NULL where the callee entry expects its globals/closure, yielding a NULL
/// result (closures / locals-bound callees called in a loop). See
/// [`DispatchError::MayForceNullRefArgUnsupported`].
pub(crate) fn walker_abort_if_mayforce_null_ref_arg<Sym: WalkSym>(
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    if !matches!(
        call_opcode,
        OpCode::CallMayForceR | OpCode::CallMayForceI | OpCode::CallMayForceF
    ) {
        return Ok(());
    }
    // `allboxes[0]` is the funcbox; `allboxes[1 + i]` aligns with
    // `arg_types[i]` (see `build_allboxes`).  A Ref arg folded to the
    // NULL constant (`GcRef(0)`) is the broken self-slot; the sentinel
    // `GcRef(usize::MAX)` means "no concrete known" and is left alone.
    //
    // Exemption: `bh_call_fn_N(callable, null_or_self, args...)`'s
    // `null_or_self` (arg index 1) is a checked sentinel — `PY_NULL`
    // means "no receiver" and is never dereferenced (`bh_call_fn_impl`
    // prepends it as arg0 only when non-null), so a concrete-NULL there
    // is the normal plain-call shape, not the broken baked-NULL shape.
    let is_call_fn = call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallFn;
    // `RaiseVarargs` (`normalize_raise_varargs`) carries a trailing `cause`
    // Ref that is a checked `PY_NULL` sentinel for `raise X` without `from`
    // (never dereferenced when null); exempt it (gated `PYRE_FBW_RAISE`) so the
    // FBW path can own the raise instead of declining to the trait.
    let is_raise_varargs = fbw_raise_enabled()
        && call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs;
    // `bh_call_function_ex_fn(callable, self_or_null, starargs, kwargs_or_null)`
    // — `self_or_null` (arg 1) and `kwargs_or_null` (arg 3) are checked
    // `PY_NULL` sentinels (never dereferenced when null), so a concrete-NULL
    // there is the normal `f(*args)` / no-`**` shape, not the broken baked-NULL.
    let is_call_function_ex =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallFunctionEx;
    // `bh_call_kw_N(callable, null_or_self, kwnames, args...)` — `null_or_self`
    // (arg 1) is a checked `PY_NULL` sentinel (prepended as arg0 only when
    // non-null), so a concrete-NULL there is the normal plain-call shape.
    let is_call_kw = call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallKw;
    for (i, &ty) in call_descr.arg_types().iter().enumerate() {
        if ty != majit_ir::Type::Ref {
            continue;
        }
        if is_call_fn && i == 1 {
            continue;
        }
        if is_call_function_ex && (i == 1 || i == 3) {
            continue;
        }
        if is_call_kw && i == 1 {
            continue;
        }
        if is_raise_varargs && i + 1 == call_descr.arg_types().len() {
            continue;
        }
        if let Some(&b) = allboxes.get(1 + i) {
            if matches!(
                ctx.trace_ctx.box_value(b),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
            ) {
                // Phase-1 diagnostic (gh#343 depth-2): pinpoint which Ref arg
                // folded to concrete NULL and its provenance.  Gated on
                // `PYRE_P2_DIAG` (the depth-2 framestack-walk diag flag) and
                // computed only on the abort path, so the default trace path
                // pays nothing.
                if std::env::var_os("PYRE_P2_DIAG").is_some() {
                    eprintln!(
                        "[p2-mayforce] NULL Ref arg: pc={pc} call_opcode={call_opcode:?} \
                         helper={:?} arg_index={i} nargs={} funcbox={:?}(={:?})",
                        call_descr.get_extra_info().pyre_helper,
                        call_descr.arg_types().len(),
                        allboxes.first(),
                        allboxes.first().and_then(|&f| ctx.trace_ctx.box_value(f)),
                    );
                    for (j, &aty) in call_descr.arg_types().iter().enumerate() {
                        let ab = allboxes.get(1 + j).copied();
                        eprintln!(
                            "[p2-mayforce]   arg[{j}] ty={aty:?} opref={ab:?} val={:?}",
                            ab.and_then(|b| ctx.trace_ctx.box_value(b)),
                        );
                    }
                }
                return Err(DispatchError::MayForceNullRefArgUnsupported { pc });
            }
        }
    }
    Ok(())
}

/// Diagnostic (`PYRE_FBW_DEBUG_ABORT`): dump the Python coordinate and per-arg
/// provenance behind a ValueUnavailable residual decline — the resume py_pc,
/// the decoded Python opcode, the declined arg's OpRef and whether it is a
/// constant, and its `box_value`.  Attributes an unj_val census walk to a
/// knowable-but-unpopulated value versus a genuinely-symbolic heap object
/// without re-instrumenting.
pub(crate) fn probe_resid_decline_ctx<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    why: &str,
    op_pc: usize,
    arg_index: usize,
    arg: OpRef,
    allboxes: &[OpRef],
) {
    let sym = ctx.fbw_mode.snapshot_sym;
    let (py_pc, opcode) = if !sym.is_null() {
        let s = unsafe { &*sym };
        if !s.jitcode().is_null() {
            let jc = unsafe { &*s.jitcode() };
            let pc = python_pc_for_jitcode_pc(&jc.payload.metadata, op_pc) as usize;
            let op = if !jc.payload.code_ptr.is_null() {
                pyre_interpreter::decode_instruction_at(unsafe { &*jc.payload.code_ptr }, pc)
                    .map(|(i, _)| format!("{i:?}"))
            } else {
                None
            };
            (Some(pc), op)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };
    let box_v = ctx.trace_ctx.box_value(arg);
    let arg_id = if arg.is_constant() || arg.is_none() {
        "const".to_string()
    } else {
        format!("r{}", arg.raw())
    };
    // The declined arg's semantic register slot and its index-keyed concrete
    // shadow (`concrete_registers_r`): a `Null` shadow at a found slot with a
    // `None` box_value is the bridge-resume seed gap (neither store populated).
    let reg_slot = ctx.registers_r.iter().position(|&r| r == arg);
    let shadow = reg_slot.and_then(|s| ctx.concrete_registers_r.get(s));
    eprintln!(
        "[fbw-resid-decline] {why} op_pc={op_pc} py_pc={py_pc:?} py_op={opcode:?} \
         arg_index={arg_index} arg={arg_id} box_value={box_v:?} reg_slot={reg_slot:?} \
         reg_shadow={shadow:?} nargs={}",
        allboxes.len() - 1,
    );
}

/// Whether a residual call is a self-recursive call to the walk's own code —
/// the `CALL_ASSEMBLER` fold target running as a plain residual because the
/// fold declined (no compiled token yet, a non-concrete argument during a
/// bridge resume, etc.).  Mirrors the callee/self resolution in
/// `try_walker_call_assembler_self_recursive`.  Keeps the recursion itself out
/// of the foreign-body-residual latch so pure recursion (`fib`) still folds.
pub(crate) fn residual_callee_is_walk_self_recursive<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
    helper: majit_ir::PyreHelperKind,
) -> bool {
    if helper != majit_ir::PyreHelperKind::CallFn {
        return false;
    }
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return false;
    }
    // A `bh_call_fn` residual is `[funcptr, callable, null_or_self, arg0, ...]`;
    // the Python callable is `allboxes[1]`.
    let Some(&callable_box) = allboxes.get(1) else {
        return false;
    };
    let Some(majit_ir::Value::Ref(callable_ref)) = ctx.trace_ctx.box_value(callable_box) else {
        return false;
    };
    if callable_ref == majit_ir::GcRef::NO_CONCRETE || callable_ref.as_usize() == 0 {
        return false;
    }
    let callable = callable_ref.as_usize() as pyre_object::PyObjectRef;
    unsafe {
        let Some((w_code, _nparams, _has_closure)) = resolve_inlinable_callee(callable) else {
            return false;
        };
        let sym = &*sym_ptr;
        if sym.jitcode().is_null() {
            return false;
        }
        let caller_code =
            pyre_interpreter::live_code_wrapper((*sym.jitcode()).raw_code() as *const ())
                as *const ();
        w_code as usize == caller_code as usize
    }
}

pub(crate) fn try_execute_residual_call_via_executor<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    recorded: OpRef,
    op_pc: usize,
) -> Result<ResidualExecOutcome, DispatchError> {
    // `execute_varargs` clears the metainterp exception slot at residual-call
    // entry, before the helper can either run or leave the call recorded
    // symbolically. A handled exception from an earlier opcode must not survive
    // across the back edge and make a later linear `catch_exception/L` look like
    // it is handling a fresh raise.
    clear_walk_exception(ctx);

    // Orthodox sub-jitcode walk safety (#171 wall-5d): a residual call whose
    // funcbox is a `symbolic_fnaddr` placeholder — a 64-bit `DefaultHasher`
    // hash of an in-body helper's `CallPath`/`CallTarget`, minted when
    // `jit_trace_fnaddrs()` has no entry for it (e.g. the zero-arg
    // `SyntheticTransparentCtor "Tuple"` unit constructor inside
    // `w_list_append`) — must not be recorded while inlining a sub-jitcode
    // body.  The production fall-throughs below leave such a call symbolic when
    // folding declines, on the contract that it runs at runtime against live
    // state; but a sub-walk's recorded trace is committed and compiled, so the
    // backend bakes the hash as a code address and the trace branches straight
    // to it -> SIGSEGV.  Decline the whole descent so it aborts gracefully at
    // the first un-lowered helper.  A user-space code address fits in 47 bits
    // on 64-bit macOS/Linux; symbolic hashes set bits >= 47.
    if ctx.fbw_mode.inline_subwalk
        && allboxes.first().is_some_and(|b| b.is_constant())
        && let Some(majit_ir::Value::Int(addr)) = ctx.trace_ctx.box_value(allboxes[0])
        && (addr as u64) >> 47 != 0
    {
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc: op_pc });
    }
    // Authoritative-executor gate: fire ONLY when the walk is the sole
    // concrete-execution leg (the production full-body walk and its
    // inline sub-walks; the per-opcode arm walk is retired).  Shadow /
    // diagnostic-probe runs leave the flag `false` so the call is
    // recorded symbolically without re-running its side effects.
    if !ctx.is_authoritative_executor {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    let plain_or_loopinvariant = matches!(
        call_opcode,
        OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
    );
    // `pyjitpl.py:2017-2082 do_residual_call` forces branch: every
    // `CallMayForce*` is concrete-executed, with the active
    // virtualizable bracketed by the token protocol (set
    // TOKEN_TRACING_RESCALL before the call, probe-and-clear after —
    // see the doc bullet above).
    let is_may_force = matches!(
        call_opcode,
        OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
    );
    if !plain_or_loopinvariant && !is_may_force {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    if allboxes.is_empty() {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // Same funcbox-must-be-const invariant as `try_fold_pure_call_via_executor`:
    // a non-const funcbox carries a stale stamp and dereferencing it as a
    // code address SEGVs.  pyjitpl.py:1346-1354 forces the funcbox through
    // the executor's `cpu.bh_call_*` `ConstInt.getint()` path which
    // implicitly requires constness too (residual_call descrs always
    // carry a fixed funcptr at translation time).
    if !allboxes[0].is_constant() {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // The LOAD_CONST helper (oopspec `LoadConst`) has a dedicated fold in the
    // residual_call dispatchers: when the const index AND the code pointer
    // (`frame.pycode`) are both concrete, it materializes `co_consts[idx]`
    // directly and suppresses the residual.  When that fold declines — the
    // promoted `frame.pycode` is concrete for the portal frame but an inlined
    // callee sub-walk does not seed it — the residual is recorded so the
    // loop computes it at runtime from the live frame's real `pycode`.
    // Executing it concretely here would pass the unseeded (null/garbage)
    // code pointer to `bh_load_const_fn`, which dereferences it via
    // `w_code_get_ptr` and faults.  Leave it symbolic, mirroring the fold's
    // "falls through to the generic record" contract.
    if call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::LoadConst {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    let funcptr_val = ctx.trace_ctx.box_value(allboxes[0]);
    let func_ptr = match funcptr_val {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic)),
    };
    // Sub-slice 4 safety gate — reject `symbolic_fnaddr_for_path`
    // placeholder values that escaped runtime patching.  Pyre's
    // codewriter mints a 64-bit hash of the helper's `CallPath` when
    // the build-time `pyre_interpreter::jit_trace_fnaddrs()` snapshot
    // has no entry for it (`majit-translate/src/codewriter/call.rs:
    // 4926 symbolic_fnaddr_for_path`).  `runtime_fnaddr_patch` rewrites
    // these to real runtime addresses only when the path appears in
    // both the build-time and runtime registries; helpers absent from
    // the runtime registry retain the hash and dereferencing it as a
    // code address SIGBUSes.  Valid user-space code addresses fit in
    // 47 bits on macOS/Linux 64-bit (canonical low half); hash values
    // typically have bits ≥ 47 set.  Reject anything outside that
    // range — both the symbolic-hash leak class AND any other stray
    // non-fnptr value (e.g. an int constant mistakenly routed through
    // the funcbox slot).
    if (func_ptr as u64) >> 47 != 0 {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // A residual whose funcptr is a `PyFrame` operand-stack accessor
    // (`pop`/`push`/`peek`/`peek_at`) reads or mutates the live frame's
    // operand stack.  During a walk that stack is empty — the walk holds
    // operand values symbolically in its register banks, not on the real
    // frame (the portal lowers stack ops to vable array writes; these
    // accessors appear only inside inlined callee sub-jitcode bodies such as
    // `pop_value`).  Executing one here underflows `PyFrame::pop`'s
    // `valuestackdepth > stack_base()` assertion against the paused outer
    // frame.  Record it symbolically instead, mirroring the tracer's
    // never-mutate-the-traced-frame discipline; it runs at runtime against a
    // frame whose operand stack the compiled trace's preceding pushes have
    // populated.
    if pyre_interpreter::is_pyframe_operand_stack_accessor(func_ptr as usize) {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    if allboxes.len() - 1 > majit_translate::codewriter::insns::MAX_HOST_CALL_ARITY {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // A void residual (CALL_N family) is a side effect with no result box, so
    // `do_residual_call` executes it EAGERLY during the walk and resumes the
    // compiled loop at iteration N+1 (the commit-invariant note below) — a
    // deferred void store is lost (its symbolic op fires only for N+1+).  When a
    // void call's arg cannot be resolved to a concrete (`GcRef(usize::MAX)` =
    // "no concrete known", or `None` = unbound), eager execution is impossible
    // AND deferral drops the store, so neither path is correct: abort the trace
    // gracefully (interpreter fallback) rather than silently drop it.  This is
    // the off-by-one for a module-global loop that builds and stores a heap
    // object then reads it back (`g = [n]; ... g[0]`): the `STORE_NAME` is a
    // void `CallN` whose value is the still-virtual `BUILD_LIST` result, so the
    // iteration-N store never reaches the cell.  A value-returning call with a
    // non-concrete arg is safe to leave symbolic — the compiled loop computes
    // its result at runtime with no lost side effect.
    let is_void = matches!(
        call_opcode,
        OpCode::CallN | OpCode::CallMayForceN | OpCode::CallLoopinvariantN
    );
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for (arg_index, &arg) in allboxes[1..].iter().enumerate() {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                if r == majit_ir::GcRef::NO_CONCRETE {
                    if is_void {
                        return Err(DispatchError::ResidualCallArgUnbound {
                            pc: op_pc,
                            arg_index,
                        });
                    }
                    if fbw_debug_abort_enabled() {
                        probe_resid_decline_ctx(
                            ctx,
                            "NO_CONCRETE",
                            op_pc,
                            arg_index,
                            arg,
                            allboxes,
                        );
                    }
                    return Ok(ResidualExecOutcome::Declined(
                        ResidualDecline::ValueUnavailable,
                    ));
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => {
                if is_void {
                    return Err(DispatchError::ResidualCallArgUnbound {
                        pc: op_pc,
                        arg_index,
                    });
                }
                if fbw_debug_abort_enabled() {
                    probe_resid_decline_ctx(ctx, "box_value=None", op_pc, arg_index, arg, allboxes);
                }
                return Ok(ResidualExecOutcome::Declined(
                    ResidualDecline::ValueUnavailable,
                ));
            }
        };
        args.push(v);
    }
    // NULL-Ref-arg refusal: same SEGV-avoidance contract as the pure
    // path (see `try_fold_pure_call_via_executor`'s NULL guard).  Pyre's
    // optimizer emits `guard_nonnull` after this walker fold, so a NULL
    // receiver dereferences before that guard exists; fall through to
    // recording the call op and let the optimizer's guard emission
    // handle it at compile time.
    // Exemption: `bh_call_fn_N(callable, null_or_self, args...)`'s
    // `null_or_self` (arg index 1) is a checked sentinel — `PY_NULL`
    // means "no receiver" and is never dereferenced (`bh_call_fn_impl`
    // prepends it as arg0 only when non-null), so a concrete-NULL there
    // is the normal plain-call shape.  These exemptions MUST match
    // `walker_abort_if_mayforce_null_ref_arg`'s — otherwise a normal
    // no-receiver keyword/star call is declined as symbolic
    // (left symbolic), which drops the recording iteration's call
    // exactly once (`g(i, d=4)` in a hot loop summed to n-1, callee
    // ran n-1 times).
    let is_call_fn = call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallFn;
    // `bh_call_kw_N(callable, null_or_self, kwnames, args...)` — `null_or_self`
    // (arg index 1) is the same checked `PY_NULL` sentinel.
    let is_call_kw = call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallKw;
    // `bh_call_function_ex_fn(callable, self_or_null, starargs, kwargs_or_null)`
    // — `self_or_null` (arg 1) and `kwargs_or_null` (arg 3) are checked `PY_NULL`
    // sentinels (never dereferenced when null), so a concrete-NULL there is the
    // normal `f(*args)` / no-`**` shape.
    let is_call_function_ex =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallFunctionEx;
    // Same `RaiseVarargs` trailing-`cause` sentinel exemption as
    // `walker_abort_if_mayforce_null_ref_arg` (gated `PYRE_FBW_RAISE`).
    let is_raise_varargs = fbw_raise_enabled()
        && call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs;
    for (i, &arg) in args.iter().enumerate() {
        if is_call_fn && i == 1 {
            continue;
        }
        if is_call_kw && i == 1 {
            continue;
        }
        if is_call_function_ex && (i == 1 || i == 3) {
            continue;
        }
        if is_raise_varargs && i + 1 == args.len() {
            continue;
        }
        if matches!(call_descr.arg_types().get(i), Some(majit_ir::Type::Ref)) && arg == 0 {
            return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
        }
    }
    // #57 (Finding #1, in-place container mutation): an in-flight FOR_ITER
    // body's `acc += delta` is a bare `NB_INPLACE_*` `BinaryOp` residual (args
    // = [lhs, rhs, op_code]) that may mutate its receiver in place at the C
    // level — no Void result, no write tag, no user frame — so none of the
    // body-effect signals below see it.  A committed non-journaled in-place
    // mutation that an aborting walk delivers would be re-run (double); dropped,
    // it would lose the iteration's tail.  Two recoverable shapes are handled
    // here, decided BEFORE any vable/tracing-call state is set up so an early
    // decline strands nothing:
    //
    //  * `acc += [ints]` for two Integer-strategy lists — the extend keeps `acc`
    //    Integer-strategy, so `w_list_int_set_len` can rewind it.  Capture the
    //    pre-extend length; the success arm journals it so the abort rollback
    //    undoes the one extend and the deliver re-applies it exactly once.
    //  * an immutable receiver (`int`/`bool`/`float`/`tuple`) — `+=` yields a
    //    FRESH object and rebinds the journaled local, so a plain deliver re-run
    //    is exact with no journaling.
    //
    // Any OTHER *exact builtin* receiver — an object-/float-strategy list,
    // `bytearray`, `set`, `dict`, `array`, a mixed `int-list += non-ints` that
    // would change strategy, … — may commit a mutation the rollback cannot
    // rewind, so decline the walk here and let this loop run interpreted
    // (exact), like the gate refusing an unsupported body op.  A user instance
    // must not take that decline: its `__iadd__`/`__isub__`/… has not run yet,
    // while the in-flight `FOR_ITER` item has already been consumed.  The
    // permanent-abort path then drops that item (the conservative delivery
    // gate sees the loop's preceding `STORE_NAME`), silently skipping one
    // augmented-assignment iteration at the trace-entry boundary.  Let the
    // normal residual dispatch execute user special methods; its existing
    // user-frame effect accounting handles any later abort.
    let inplace_list_journal: Option<(pyre_object::PyObjectRef, usize)> =
        if call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::BinaryOp
            && args.len() >= 3
            && pyre_interpreter::runtime_ops::binary_op_tag_is_inplace(args[2])
            && fbw_foriter_inflight_active()
        {
            let lhs = args[0] as usize as pyre_object::PyObjectRef;
            let rhs = args[1] as usize as pyre_object::PyObjectRef;
            unsafe {
                if pyre_object::pyobject::is_exact_list(lhs)
                    && pyre_object::listobject::w_list_is_integer_strategy(lhs)
                    && pyre_object::pyobject::is_exact_list(rhs)
                    && pyre_object::listobject::w_list_is_integer_strategy(rhs)
                {
                    Some((lhs, pyre_object::w_list_len(lhs)))
                } else if pyre_object::pyobject::is_int_or_long(lhs)
                    || pyre_object::pyobject::is_bool(lhs)
                    || pyre_object::pyobject::is_float(lhs)
                    || pyre_object::pyobject::is_tuple(lhs)
                {
                    None
                } else if pyre_object::pyobject::is_exact_builtin_instance(lhs) {
                    return Err(DispatchError::InplaceContainerMutationUnsupported { pc: op_pc });
                } else {
                    None
                }
            }
        } else {
            None
        };
    // `do_residual_call` (pyjitpl.py:2040/2104/2123 for CALL_MAY_FORCE_N /
    // CALL_LOOPINVARIANT_N / CALL_N) runs `executor.execute_varargs` for a void
    // call exactly like the value-returning shapes, applying the side effect
    // once during tracing and then resuming the compiled loop at the *next*
    // iteration (`raise_continue_running_normally`, pyjitpl.py:3072-3091, hands
    // back the end-of-iteration-N state so iteration N is never re-run).  Pyre
    // mirrors the second half via the walk-end commit
    // (`flush_walk_end_state_to_frame`, run_perfn_walk): a successful commit
    // adopts the end-of-walk frame so the compiled loop enters at iteration
    // N+1, leaving the eagerly-applied side effect counted once.  Executing
    // void calls here (rather than recording-only) keeps that invariant whole —
    // a deferred void store would be lost on commit (its symbolic op only fires
    // for N+1+), which is why deferral previously forced the no-commit legacy
    // replay.  The replay path that re-runs iteration N has the symmetric
    // hazard for already-executed value calls (e.g. `list.insert` returns the
    // None ref, so it is not a void call yet still mutates) — eager-everything +
    // commit is the single consistent rule, matching `do_residual_call`.
    //
    // The standard virtualizable box pointer for a MayForce residual — a
    // force inside the callee could escape the frame.  None for non-forces
    // opcodes and when no live vable exists (the jitdriver has no standard
    // virtualizable, or unit-test init disabled the heap pointer) — nothing
    // the callee could force.  The token is armed further below, past every
    // decline gate.
    let mut vable_obj_root = if is_may_force {
        ctx.trace_ctx
            .standard_virtualizable_box()
            .and_then(|_| ctx.trace_ctx.virtualizable_heap_ptr())
            .filter(|p| !p.is_null())
            .map(|p| Box::new(p as usize as i64))
    } else {
        None
    };
    // A Python-level callee (e.g. a recursive `fib`) re-enters the
    // interpreter (`eval_loop_jit` → `jit_merge_point`) while this walk still
    // holds the driver in the tracing state.  Suspend re-entrant trace
    // continuation for the duration of the concrete call so the callee runs as
    // plain interpretation instead of starting a nested trace that would share
    // and corrupt this walk's `TraceCtx` (flaky `libsystem_malloc` freelist
    // abort during deep recursion).  Plain C-helper callees never re-enter, so
    // the guard is a no-op for them.
    //
    // In RPython the tracing metainterp and the executing (blackhole /
    // compiled) interpreter are SEPARATE objects, so `do_residual_call`
    // never perturbs the tracer's `MetaInterp.vable_ptr` /
    // `virtualizable_boxes`.  Pyre shares one `TraceCtx` across the walk
    // and any re-entrant JIT activity the concrete call triggers: a
    // self-recursive `CALL_ASSEMBLER` callee that re-enters compiled code
    // and deopts runs `set_vable_ptr` for the nested frames, leaving
    // `virtualizable_heap_ptr` pointing at a nested callee frame whose
    // `vable_token` is still the live JIT FORCE_TOKEN.  The next
    // `tracing_before_residual_call` in this same walk would then assert on
    // the non-NONE token (virtualizable.rs:565).  Snapshot the standard
    // virtualizable pointer and restore it after the call so the walk's
    // subsequent vable token protocol / field reads see the frame being
    // traced, mirroring RPython's separate-state isolation.
    let saved_vable_heap_ptr = ctx.trace_ctx.virtualizable_heap_ptr();
    // #57 Option C (Finding #1, R1 double-apply guard): whether THIS residual
    // could commit an irreversible heap mutation the journals do not cover,
    // while an in-flight FOR_ITER item is already captured (a consume ran
    // earlier this iteration).  The journaled list ops (setitem / append) run
    // OUTSIDE this executor (`try_walker_store_subscr_specialization` /
    // `try_walker_orthodox_list_append`) and roll back on abort, so they are
    // not body-effect candidates here.
    //
    // The OLD allow-list (`StoreSubscr` / `CallFn` / `SetCurrentException`
    // only) MISSED the many statement-level mutators that reach this executor
    // concretely, succeed, and carry `PyreHelperKind::None` (`store_attr_fn` /
    // `delete_subscr_fn` / `delete_attr_fn` / `list_extend_fn` / `store_name_fn`
    // / `store_global` / `store_slice` …): a missed mutator is a silent double
    // on a body re-run (correctness-FATAL).  Track residuals that WRITE live
    // heap state outside the journals.
    //
    // The write discriminator is the residual's RESULT TYPE plus the
    // value-returning-mutator helper tags — NOT `extraeffect`, which cannot
    // separate a write from a read: `getattr_fn` (a pure `.append` bound-method
    // lookup) and `store_attr_fn` are BOTH `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
    // (the analyzer is stubbed, so `write_descrs_*` are empty for both).  A
    // `Void`-result residual produces no value, so it is executed solely for
    // its heap side effect → a write.  Every statement-level mutator above
    // lowers to a `residual_call_*_v` (Void); the benign reads (`getattr`,
    // `load_global`, `load_name`, `load_deref` …) all RETURN a value, so a
    // Void result is a clean write proxy that does not over-refuse a read.
    //
    // The few VALUE-returning writes the Void test cannot see are caught by
    // helper tag: `CallFn` (an opaque Python call returning its None ref may
    // mutate arbitrary state), `StoreSubscr` (a dict/object `o[k]=v` returning
    // the stored value), `SetCurrentException` (the TLS exc-slot write), and
    // `StoreDeref` (an in-place closure-cell write returning the slot value —
    // `nonlocal n; n += 1`).  Over-refusing only these (never a benign read)
    // keeps the journaled-append body (`for_mutate`) delivering, since its
    // `getattr`/append residuals return `Ref` and carry no write tag.
    //
    // Provably read-only/elidable residuals are exempt up front: `@jit.elidable`-
    // class (`check_is_elidable`: `EF_ELIDABLE_*`, the pure executor folds
    // these) or `EF_LOOPINVARIANT` (loop-hoisted).  The `for_iter_next` consume
    // itself ([`PyreHelperKind::ForIterNext`]) is excluded — it is the SOURCE of
    // the capture (it runs while the PRIOR iteration's item is still in flight),
    // not a body effect for that prior iteration.  Sampled BEFORE the call so
    // the success arm can flag an effect that committed AFTER the in-flight
    // consume.
    let ei = call_descr.get_extra_info();
    let helper = ei.pyre_helper;
    let provably_side_effect_free = ei.check_is_elidable()
        || ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant
        || helper == majit_ir::PyreHelperKind::ForIterNext;
    let writes_live_heap = call_descr.result_type() == majit_ir::Type::Void
        || matches!(
            helper,
            majit_ir::PyreHelperKind::CallFn
                | majit_ir::PyreHelperKind::StoreSubscr
                | majit_ir::PyreHelperKind::SetCurrentException
                | majit_ir::PyreHelperKind::StoreDeref
        );
    // Inside an inline sub-walk, decline before any residual that is not
    // provably side-effect-free.  Ref-result getters/dunders/user `__next__`
    // can mutate live heap through user frames while `writes_live_heap` is
    // false, and rollback would miss that concrete mutation.  The helper no-ops
    // on an empty session framestack, so top-level depth-1 behavior is unchanged.
    if !provably_side_effect_free {
        fbw_abort_nested_unjournaled_residual(ctx, op_pc)?;
    }
    // pyjitpl.py:3329-3330 `vinfo.tracing_before_residual_call(virtualizable)`
    // heap half: every decline gate has now passed, so the helper WILL
    // execute — set TOKEN_TRACING_RESCALL on the active virtualizable so a
    // force inside the callee is observable afterwards.  Armed AFTER the
    // inline-subwalk decline so a declined residual never strands the token:
    // tracing_before pairs with the tracing_after clear below only for
    // residuals that proceed, mirroring `do_residual_call` where
    // `vable_and_vrefs_before_residual_call` (pyjitpl.py:2017) runs only past
    // the OS_NOT_IN_TRACE / force-virtual short-circuits.  Trait mirror:
    // `MIFrame::vable_and_vrefs_before_residual_call` (trace_opcode.rs:2602).
    let vable_root_depth = if let Some(obj) = vable_obj_root.as_mut() {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
        unsafe {
            majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(&mut **obj));
            info.tracing_before_residual_call(**obj as usize as *mut u8);
        }
        Some(root_depth)
    } else {
        None
    };
    let body_effect_candidate =
        !provably_side_effect_free && writes_live_heap && fbw_foriter_inflight_active();
    // #57 Option C (Finding #1, user-frame signal): the Void/helper-tag write
    // discriminator above cannot see a body effect committed through USER
    // PYTHON CODE by a value-returning (`Ref`), `PyreHelperKind::None`,
    // `MayForce` residual: `obj.prop` (a `@property` getter / `__getattr__` /
    // descriptor `__get__`), `a + b` / `a == b` (user `__add__` / `__eq__`),
    // `iter(obj)` (user `__iter__`), `str(obj)` / `f"{obj}"` (user `__str__` /
    // `__format__`), `import name`.  Each RETURNS a value (so the Void proxy
    // misses it) and carries no write tag, yet its getter/dunder/module body
    // may mutate live heap.  Those mutations all run a USER PYTHON FRAME (the
    // getter's bytecode); a pure builtin path (`seen.append`'s C-level
    // bound-method lookup, `int.__add__`) does NOT.  Snapshot the monotonic
    // frame eval-loop entry odometer before the call; if it advanced while an
    // in-flight FOR_ITER item is active, the residual ran user bytecode that
    // could have committed an irreversible body effect → flag it (the success
    // arm compares post-call).  Sampled only when an item is in flight and the
    // residual is not provably read-only (an elidable / loop-invariant fold or
    // the `for_iter_next` consume itself never counts).
    let user_frame_snapshot = (!provably_side_effect_free && fbw_foriter_inflight_active())
        .then(pyre_interpreter::call::frame_entry_count);
    // #493: a NEW consume attempt for a FOR_ITER whose prior item is still in
    // flight means that item's body ran to completion — mark the entry BEFORE
    // the call so an attempt that aborts mid-way (a kept-stack guard on the
    // exhaustion arm) still records the completion; a successful attempt
    // replaces the entry with a fresh one anyway.
    if helper == majit_ir::PyreHelperKind::ForIterNext {
        let body = fbw_foriter_body_from_op_pc(ctx.fbw_mode.snapshot_sym, op_pc)
            .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
        fbw_foriter_inflight_mark_attempt(body);
    }
    // gh#467: sample the user-frame odometer UNCONDITIONALLY (not only under an
    // in-flight FOR_ITER) so the concrete-heap-write gate can detect a callee
    // sub-walk mutation committed through a value-returning dunder body — the
    // same user-frame signal Finding #1 uses, generalized past FOR_ITER.
    let heap_write_odometer_before =
        (!provably_side_effect_free).then(pyre_interpreter::call::frame_entry_count);
    let exec_result = {
        let _suspend = majit_metainterp::TraceContinuationSuspendGuard::enter();
        majit_metainterp::executor::execute_residual_call(call_descr, func_ptr, &args)
    };
    if !provably_side_effect_free {
        fbw_mark_executed_nonpure_residual();
        // Count only a FOREIGN non-pure residual: a self-recursive call is the
        // fold target running because its fold declined, not a body side effect.
        if !residual_callee_is_walk_self_recursive(ctx, allboxes, helper) {
            fbw_mark_executed_body_residual();
        }
    }
    let restored_vable_heap_ptr = vable_obj_root
        .as_ref()
        .map(|obj| **obj as usize as *const u8)
        .or(saved_vable_heap_ptr)
        .unwrap_or(std::ptr::null());
    ctx.trace_ctx
        .set_virtualizable_heap_ptr(restored_vable_heap_ptr);
    // pyjitpl.py:3349-3353 `vinfo.tracing_after_residual_call(virtualizable)`
    // heap half: a cleared token means the callee forced the virtualizable —
    // the frame escaped, the trace must abort (pyjitpl.py:3365
    // `SwitchToBlackhole(Counters.ABORT_ESCAPE, raising_exception=True)`;
    // trait mirror `MIFrame::vable_after_residual_call`,
    // trace_opcode.rs:2646).  The interpreter resumes from the live frame,
    // which the callee's force path made heap-authoritative — no
    // `load_fields_from_virtualizable` analogue is needed because the FBW
    // abort discards the walk shadow instead of handing it to a blackhole
    // leg.  An intact token is cleared back to TOKEN_NONE.
    if let Some(obj) = vable_obj_root.as_ref() {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let forced = unsafe { info.tracing_after_residual_call(**obj as usize as *mut u8) };
        if let Some(depth) = vable_root_depth {
            majit_gc::shadow_stack::pop_resume_ref_roots_to(depth);
        }
        if forced {
            return Err(DispatchError::VableEscapedDuringResidualCall { pc: op_pc });
        }
    }
    // #57 Option C (Finding #1, R1): a residual that is not provably
    // side-effect-free has now EXECUTED AFTER the in-flight FOR_ITER consume —
    // whether it returned a value (Ok) or raised (Err).  The store/append
    // journals roll their entries back on abort (so a body re-run re-applies
    // them once), but a mutation outside those journals (a dict
    // `store_subscr_fn`, an `obj.attr = …` `store_attr_fn`, a `list.extend`,
    // a `del o[k]`, a name/global/deref store …) cannot be undone — delivering
    // the in-flight item and re-running the body would double it.  Flag it so
    // `fbw_foriter_inflight_take` refuses delivery (the legacy drop-on-abort
    // fallback) instead of doubling.
    // The user-frame signal: the residual's concrete execution entered a user
    // Python frame (the odometer advanced), so a value-returning
    // getter/dunder/module body may have mutated live heap outside the
    // journals — a body effect the Void/helper-tag discriminator misses.
    // `for_mutate`'s `seen.append` resolves its bound method at the C level (no
    // user frame), so its snapshot is unchanged and it still DELIVERS.
    //
    // The marking runs on BOTH arms.  A getter that mutates and THEN raises
    // (`for_prop_raise_abort`: `Obj.hits += 1; raise`, caught locally, walk
    // continues, later abort) takes the Err arm but still committed the
    // irreversible effect and still bumped the eval-loop entry odometer; if it
    // marked only on Ok, `fbw_foriter_inflight_take` would see no signal and
    // DELIVER, re-running the getter and DOUBLING `Obj.hits`.  The
    // `for_iter_next` consume itself is exempt (`provably_side_effect_free`
    // leaves both `body_effect_candidate` false and `user_frame_snapshot`
    // None), so a raising `__next__` never self-flags.
    //
    // The odometer bumps at frame ENTRY, so `entered_user_frame` cannot tell a
    // user frame that raised AFTER mutating from one that raised BEFORE: a
    // getter that raises before committing anything is also refused here.  That
    // is a harmless conservative DROP (the legacy bypass still runs the
    // iteration once), never a double — refusing a non-mutating raise costs
    // nothing but the never-double guarantee.
    let entered_user_frame = user_frame_snapshot
        .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before);
    if body_effect_candidate || entered_user_frame {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-foriter] body effect committed since consume (helper={helper:?} \
                 extraeffect={:?} result_type={:?} write_discriminator={body_effect_candidate} \
                 entered_user_frame={entered_user_frame})",
                ei.extraeffect,
                call_descr.result_type(),
            );
        }
        fbw_mark_foriter_body_effect_since_consume();
    }
    // gh#467: bump the concrete-heap-write odometer for any residual that is not
    // provably side-effect-free and either writes live heap (a Void / mutator-
    // tagged store the store/append journals do not cover) or entered a user
    // Python frame (a value-returning getter/dunder body that may have mutated).
    // The inline abort-forward-flush gate snapshots this at the CALL and refuses
    // the forward flush if a callee sub-walk moved it — re-executing the CALL
    // would double the effect.
    if !provably_side_effect_free
        && (writes_live_heap
            || heap_write_odometer_before
                .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before))
    {
        fbw_bump_executed_effect();
    }
    match exec_result {
        Ok(result_i64) => {
            fbw_count_executed_residual(is_void, is_may_force);
            // #57 (Finding #1): the in-place int-list extend committed; journal
            // its pre-extend length so an aborting walk's rollback rewinds it and
            // the deliver re-applies it exactly once.  `result_i64 == lhs`
            // confirms the in-place mutation (list `__iadd__`/`__imul__` return
            // self) rather than a fresh-object op that merely shared the slot.
            if let Some((lhs, len_before)) = inplace_list_journal {
                if result_i64 as usize == lhs as usize {
                    fbw_append_journal_push(lhs, len_before);
                }
            }
            // pyjitpl.py:1392 `result_box.value = result` analogue — stamp
            // the recorded OpRef with the executed concrete so downstream
            // `concrete_of_opref` / `box_value` consumers see the folded
            // value. An executed void helper has nothing to stamp.
            match call_descr.result_type() {
                majit_ir::Type::Int => {
                    ctx.trace_ctx
                        .set_opref_concrete(recorded, majit_ir::Value::Int(result_i64));
                }
                majit_ir::Type::Ref => {
                    ctx.trace_ctx.set_opref_concrete(
                        recorded,
                        majit_ir::Value::Ref(majit_ir::GcRef(result_i64 as usize)),
                    );
                }
                majit_ir::Type::Float => {
                    ctx.trace_ctx.set_opref_concrete(
                        recorded,
                        majit_ir::Value::Float(f64::from_bits(result_i64 as u64)),
                    );
                }
                majit_ir::Type::Void => {}
            }
            // #57 Option C (capture): this residual is the FOR_ITER advance
            // (`for_iter_next`) — it just advanced the real shared heap
            // iterator (an irreversible side effect with no journal undo).
            // Stash the consumed item + the FOR_ITER body pc (the continue
            // arm's `py_pc + 1` fallthrough) so an aborting walk can DELIVER
            // the in-flight iteration to the live frame instead of dropping
            // it.  A null result is the exhaustion arm (no item, no body
            // runs) — nothing to deliver, leave the stash empty.
            if call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::ForIterNext
                && result_i64 != 0
            {
                // The body pc is the FOR_ITER continue-arm fallthrough — the
                // Python bytecode pc of the FOR_ITER opcode plus one (matching
                // `opcode_for_iter`'s `next_instr() == opcode_pc + 1`).
                //
                // Finding #2: derive it from the residual op's OWN JitCode pc
                // (`op_pc`) mapped to its containing Python opcode, NOT the
                // walk-ENTRY coordinate (`entry_py_pc + 1`).  The entry
                // coordinate equals the FOR_ITER fallthrough only when FOR_ITER
                // is the loop-header / walk-entry opcode; a second/nested
                // FOR_ITER reached deeper in a traced body has its own
                // `op_pc`, so the entry coordinate would point at the WRONG
                // body and deliver to the wrong pc.  The fallback (no outer
                // full-body sym / metadata) keeps the entry coordinate, which
                // is correct for the loop-header FOR_ITER.
                let body = fbw_foriter_body_from_op_pc(ctx.fbw_mode.snapshot_sym, op_pc)
                    .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
                fbw_foriter_inflight_capture(result_i64 as usize as pyre_object::PyObjectRef, body);
                // #73/#267: the item lands on the operand-stack TOS through the
                // codewriter's `pin!` slot binding (FOR_ITER lowering), not a
                // `setarrayitem_vable_r` push, and the residual result is
                // stamped via `set_opref_concrete`, not `write_ref_reg` — so
                // neither mirror chokepoint sees the item and `vstack_last_ref`
                // still holds whatever inner box the ForIterNext produced.  Seed
                // it with the item OpRef so the FOR_ITER boundary
                // (`ResultToTos`) places the item, not a stale box, on the new
                // TOS.  This runs for every `ForIterNext` residual once it
                // returns, placing the item OpRef on the new TOS for the
                // FOR_ITER `ResultToTos` boundary.
                ctx.vstack_last_ref = recorded;
                if fbw_debug_abort_enabled() {
                    let item = result_i64 as usize as pyre_object::PyObjectRef;
                    let intval = if unsafe { pyre_object::pyobject::is_int(item) } {
                        Some(unsafe { pyre_object::w_int_get_value(item) })
                    } else {
                        None
                    };
                    eprintln!(
                        "[fbw-foriter] capture item=0x{:x} intval={intval:?} foriter_pc={} body={body:?} \
                         store_journal_len={} append_journal_len={} unjournaled={}",
                        result_i64 as usize,
                        ctx.entry_py_pc(),
                        fbw_store_journal_len(),
                        FBW_APPEND_JOURNAL.with(|j| j.borrow().len()),
                        fbw_has_unjournaled_effect(),
                    );
                }
            }
        }
        Err(bh_exc) => {
            fbw_count_executed_residual(is_void, is_may_force);
            // pyjitpl.py:1690-1696 `metainterp.execute_raised(exception,
            // constant=False)` analogue — seed the standing exception
            // state so downstream walker chain (`reraise/`,
            // `last_exc_value/>r`, `handle_possible_exception` guard
            // emission) sees a non-null `last_exc_value` and routes
            // through the GuardException path.
            //
            // `execute_residual_call` cleared `BH_LAST_EXC_VALUE` on read;
            // restore it so the eval-loop walker-skip path
            // (`eval.rs:3285-3308`) can detect the pending exception and
            // route into the bytecode-interpreter's exception handler
            // via `PyError::from_exc_object` — matching RPython's
            // metainterp framestack scan after a raising residual call
            // (`pyjitpl.py:2156-2168 handle_possible_exception` +
            // `pyjitpl.py:3380 finishframe_exception`).
            ctx.last_exc_value = Some(ctx.trace_ctx.const_ref(bh_exc));
            ctx.last_exc_value_concrete =
                ConcreteValue::Ref(bh_exc as usize as pyre_object::PyObjectRef);
            // `pyjitpl.py:2768-2777 execute_raised(..., constant=False)`:
            // a residual exception has not had its class proven by a guard yet.
            ctx.fbw_mode.class_of_last_exc_is_const = false;
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(bh_exc));
            // `execute_raised` records the raise into `last_exc_value`
            // (above) only.  The shared `bh_*` residual helper also
            // published into the backend `_store_exception` cells
            // (`publish_residual_call_exception`), but RPython tracing never
            // touches them — they belong to compiled / blackhole execution.
            // Drain them so an aborted trace's snapshot-side raise cannot
            // leak into the live frame's re-run, where compiled
            // `GUARD_NO_EXCEPTION` would read it as a spurious pending
            // exception (the standing exception lives in `last_exc_value`
            // for the walk's `reraise` / `catch_exception` consumers).
            if let Some(cb) = crate::callbacks::try_get() {
                (cb.drain_backend_jit_exc)();
            }
        }
    }
    Ok(ResidualExecOutcome::Executed(exec_result))
}

/// `pyjitpl.py:3671-3681 MetaInterp.direct_call_release_gil` port.
/// Sub-case of the forces-virtual-or-virtualizable branch
/// (`pyjitpl.py:2063` `if effectinfo.is_call_release_gil()`): when the
/// descr's `call_release_gil_target` is a non-NULL `(realfuncaddr,
/// saveerr)` pair, the recorded trace op is `CALL_RELEASE_GIL_*`
/// with a re-shaped arglist:
///
/// ```text
///     realfuncaddr, saveerr = effectinfo.call_release_gil_target
///     funcbox = ConstInt(adr2int(realfuncaddr))
///     savebox = ConstInt(saveerr)
///     opnum   = rop.call_release_gil_for_descr(calldescr)
///     return self.history.record_nospec(
///         opnum, [savebox, funcbox] + argboxes[1:], ..., calldescr)
/// ```
///
/// `argboxes[0]` (the original funcbox) is replaced by the descr's real
/// target address, with `savebox` (`saveerr`) prepended.  The pyre-jit-
/// trace `allboxes` from [`build_allboxes`] starts with `funcptr` at
/// index 0 and the user-side arguments from index 1 onwards, matching
/// upstream's `argboxes[0] = funcbox` convention, so the slice rebuild
/// is `[savebox, funcbox_real] + allboxes[1..]`.
///
/// Mirror of `majit-metainterp/src/pyjitpl.rs:10437-10477
/// direct_call_release_gil` for the pyre-jit-trace dispatcher layer.
/// The two-frame-layer parity (majit `do_residual_call` and
/// pyre-jit-trace `dispatch_residual_call_*`) both implement the same
/// `pyjitpl.py:3671-3681` sub-case independently because the layers
/// receive different argument shapes.  `descr` is consumed (move) into
/// `record_op_with_descr` so the caller must `clone()` it before
/// calling if it needs the original after this returns.
///
/// Also emits the two guards the outer forces branch demands
/// (`pyjitpl.py:2079 GUARD_NOT_FORCED` unconditionally,
/// `pyjitpl.py:2082 GUARD_NO_EXCEPTION` when
/// `check_can_raise(False)` is true) — keeping guard emission inside
/// this helper means the dispatcher early-returns after a single call.
///
/// **`'r'` bank not supported.**  RPython
/// `resoperation.py:1238 call_release_gil_for_descr` has no
/// `CALL_RELEASE_GIL_R` arm (commented out as `# no such thing`),
/// and `:1462 is_call_release_gil` excludes `CALL_RELEASE_GIL_R`
/// from the predicate.  This helper panics on `dst_bank == 'r'` —
/// the closest behaviour to upstream's missing branch is fail-fast,
/// since silently routing to a non-existent OpCode would record an
/// IR op the optimizer / backend cannot consume.  Generic codewriter
/// `emit_residual_call` sites do not manufacture release-gil EIs via
/// `effect_info_for_call_flavor`; release-gil support is limited to
/// explicit via-target lowering that resolves the real call target
/// before materializing the final calldescr.  The panic is defensive
/// against a future producer that introduces a `'r'`-result release-gil
/// callee without first wiring an upstream `CALL_RELEASE_GIL_R` opcode.
///
/// `'i'` / `'f'` / `'v'` are the three result kinds upstream's
/// `call_release_gil_for_descr` accepts (`resoperation.py:1240-1248`).
/// All three are decoded here so the **opcode selection** matches
/// upstream's three-way result-kind table even though only
/// `dispatch_residual_call_iRd_kind` / `_iIRd_kind` currently route
/// `'i'` and `'r'` (the latter rejected per the panic above).
///
/// **Float / Void coverage is opcode-only, not full reuse.**  A
/// future float / void residual-call dispatcher would still have
/// to extend its own callsite to (a) widen `dst_bank` validation,
/// (b) add the corresponding writeback path to
/// `registers_f` / no-writeback, and (c) thread Float-typed
/// `argbox_types` through `build_allboxes` for the `'f'` arg-list
/// case.  This helper produces the right `OpCode::CallReleaseGil*`
/// once those landed; it does not by itself complete the dispatcher.
/// `pyjitpl.py:2003-2005 do_residual_call` parity:
///
/// ```python
/// if effectinfo.oopspecindex == effectinfo.OS_NOT_IN_TRACE:
///     return self.metainterp.do_not_in_trace_call(allboxes, descr)
/// ```
///
/// Upstream's `do_not_in_trace_call` (pyjitpl.py:3683-3697) executes the
/// callee concretely and raises `SwitchToBlackhole(ABORT_ESCAPE,
/// raising_exception=True)` if it raised, otherwise returns `None` so
/// no IR op is recorded.
///
/// The pyre trace-walker has no concrete-execution callback for
/// jitcode-walked residual_call bytecodes yet — concrete execution
/// happens in the metainterp layer (`pyjitpl.rs:9631-9659
/// do_not_in_trace_call`) which dispatches `BC_CALL_*` not
/// `BC_RESIDUAL_CALL_*`. Therefore an `OS_NOT_IN_TRACE` callee that
/// reached this dispatcher cannot be safely treated as a regular
/// residual call: upstream records no IR for the normal case, and
/// aborts to blackhole only when the concrete call raises. Until that
/// concrete callback is threaded into `WalkContext`, the walker reports
/// a typed error instead of inventing either outcome.
///
/// `effect_info_for_call_flavor` (`flatten.rs:431` audit table) never
/// sets `oopspecindex`, so this branch is unreachable from production
/// today. A future producer that begins populating `oopspecindex`
/// should replace this guard with a real `do_not_in_trace_call`
/// callback returning `Ok(None)` on normal completion and
/// `SwitchToBlackhole(ABORT_ESCAPE, raising_exception=True)` only on
/// raise.
#[inline]
pub(crate) fn do_not_in_trace_call_result(
    ei: &majit_ir::EffectInfo,
    pc: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if ei.oopspecindex == OopSpecIndex::NotInTrace {
        return Err(DispatchError::NotInTraceRequiresConcreteExecution { pc });
    }
    Ok(None)
}

/// IR-recording portion of `pyjitpl.py:3327-3335
/// vable_and_vrefs_before_residual_call`.  Records
/// `FORCE_TOKEN + SETFIELD_GC(vable_token_descr)` whenever the
/// jitdriver has a standard virtualizable registered for the current
/// frame.  RPython structure:
///
/// ```text
/// def vable_and_vrefs_before_residual_call(self):
///     self.vrefs_before_residual_call()                # heap mutation
///     vinfo = self.jitdriver_sd.virtualizable_info
///     if vinfo is not None:
///         virtualizable_box = self.virtualizable_boxes[-1]
///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
///         vinfo.tracing_before_residual_call(virtualizable) # heap mutation
///         force_token = self.history.record0(rop.FORCE_TOKEN, ...)  # IR
///         self.history.record2(rop.SETFIELD_GC, ..., descr=...)     # IR
/// ```
///
/// In pyre, the IR-recording role and the runtime heap-mutation role
/// are split.  This helper carries the IR portion only; the heap
/// halves of the vable token protocol
/// (`vinfo.tracing_before_residual_call(virtualizable)` /
/// `vinfo.tracing_after_residual_call(virtualizable)`) live with
/// the walk that executes the callee:
/// [`try_execute_residual_call_via_executor`] brackets the concrete
/// `execute_residual_call` with both halves of
/// `vable_and_vrefs_before_residual_call` /
/// `vable_after_residual_call` and surfaces
/// [`DispatchError::VableEscapedDuringResidualCall`] on a detected
/// force (pyjitpl.py:3365 ABORT_ESCAPE parity).
///
/// This helper records ONLY the IR portion here and never
/// touches the token; the heap-half token protocol is bracketed by
/// [`try_execute_residual_call_via_executor`], keeping the
/// `*token_ptr == 0` assertion in `tracing_before_residual_call`
/// intact.
///
/// `vrefs_before_residual_call` / `vrefs_after_residual_call`
/// (`pyjitpl.py:3341-3372`) are unported.  `PyreSym` does carry
/// `virtualref_boxes` (`state.rs`), so what is missing is the bracket
/// itself: the pre-call `vrefinfo.tracing_before_residual_call` loop
/// and the post-call `stop_tracking_virtualref`.  Unreachable today —
/// the codewriter emits no `jit.virtual_ref` producers
/// (`jit/call.rs`), leaving `virtualref_boxes` empty so both upstream
/// loops iterate zero times.
pub(crate) fn walker_vable_and_vrefs_before_residual_call(ctx: &mut TraceCtx) {
    // pyjitpl.py:3326-3327: vinfo = self.jitdriver_sd.virtualizable_info;
    //                       if vinfo is not None:
    let Some(vable_ref) = ctx.standard_virtualizable_box() else {
        return;
    };
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    // pyjitpl.py:3332-3335: force_token + SETFIELD_GC vable_token_descr
    let force_token = ctx.force_token();
    ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
}

/// Convenience wrapper for [`walker_vable_and_vrefs_before_residual_call`].
/// Kept as a thin pass-through so the dispatcher call sites stay
/// readable; collapses to direct `walker_*` once the dispatchers
/// inline.
pub(crate) fn maybe_walker_vable_and_vrefs_before_residual_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
) {
    walker_vable_and_vrefs_before_residual_call(ctx.trace_ctx);
}

/// Write a residual_call's recorded result OpRef into the dst register
/// chosen by `dst_bank`. Centralizes the result writeback so the
/// dispatchers can perform it BEFORE recording the
/// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION` guards, matching
/// `pyjitpl.py:1950 _opimpl_residual_call*` ordering: the result
/// must populate `registers_*[dst]` before
/// `handle_possible_exception()` captures the guard's `fail_args`,
/// otherwise a raising call surfaces NONE in the slot the resume
/// snapshot reads.
pub(crate) fn write_residual_call_result_to_dst<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    dst_bank: char,
    result: OpRef,
) -> Result<(), DispatchError> {
    // concrete_of_opref shadow write: route the shadow write through `concrete_of_opref`
    // so a CallPure* descr whose argboxes are all constant (do_residual_call
    // path that lands a constant result via the executor.execute_varargs
    // stamp) propagates concrete to the dst slot.  Falls back to Null when
    // the result has no recorded concrete (matches the pre-#75.F shape for
    // every non-elidable call).
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'r' => {
            write_ref_reg(ctx, pc, dst, result, concrete_for_shadow)?;
        }
        'i' => {
            write_int_reg(ctx, pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = result;
        }
        // Void variants (`pyjitpl.py:1348/1351/1355 opimpl_residual_call_*_v`):
        // the operand layout has no `>X` dst byte and no register slot to
        // populate. The cached / recorded OpRef is dropped on the floor
        // upstream too (the `_call*` body discards the call result for
        // void).
        'v' => {}
        _ => unreachable!("dst_bank validated by caller"),
    }
    Ok(())
}

pub(crate) fn residual_call_helper_kind_in_body(
    body_code: &[u8],
    d: &DecodedOp,
    callee_descr_refs: &[DescrRef],
) -> Option<majit_ir::PyreHelperKind> {
    let descr_index = residual_call_descr_index_in_body(body_code, d)?;
    callee_descr_refs
        .get(descr_index)
        .and_then(|descr| descr.as_call_descr())
        .map(|cd| cd.get_extra_info().pyre_helper)
}

/// Return the per-function descriptor-pool index carried by a residual call
/// in a callee jitcode body.  The layouts mirror the residual dispatchers:
/// the descriptor follows the one or two variable-length argument lists.
pub(crate) fn residual_call_descr_index_in_body(body_code: &[u8], d: &DecodedOp) -> Option<usize> {
    let descr_offset = match d.key {
        "residual_call_r_r/iRd>r" | "residual_call_r_i/iRd>i" | "residual_call_r_v/iRd" => {
            let r_len_pc = d.pc + 2;
            let r_len = *body_code.get(r_len_pc)? as usize;
            1 + 1 + r_len
        }
        "residual_call_ir_r/iIRd>r" | "residual_call_ir_i/iIRd>i" | "residual_call_ir_v/iIRd" => {
            let i_len_pc = d.pc + 2;
            let i_width = 1 + *body_code.get(i_len_pc)? as usize;
            let r_len_pc = d.pc + 1 + 1 + i_width;
            let r_width = 1 + *body_code.get(r_len_pc)? as usize;
            1 + i_width + r_width
        }
        _ => return None,
    };
    Some(decode_descr_index(body_code, d, descr_offset))
}

/// `BINARY_OP Add` has the generic residual shape in a per-function jitcode,
/// but the walker replaces a statically tagged plain add with `IntAddOvf` or
/// `FloatAdd` before the generic residual executor (and its nested-residual
/// decline) is reached when every incoming callee argument is int or float.
/// Non-numeric operands stay an impure residual, so admitting them here would
/// trigger the nested-residual 6421 abort storm.  Accept only the constant
/// `Add` tag with numeric arguments; every in-place tag and every dynamic or
/// different binary operation remains conservative.
pub(crate) fn residual_call_is_specialized_plain_int_add(
    body_code: &[u8],
    args_all_numeric: bool,
    d: &DecodedOp,
    num_regs_i: usize,
    constants_i: &[i64],
    callee_descr_refs: &[DescrRef],
) -> bool {
    if !args_all_numeric
        || !matches!(
            d.key,
            "residual_call_ir_r/iIRd>r" | "residual_call_ir_i/iIRd>i" | "residual_call_ir_v/iIRd"
        )
        || residual_call_helper_kind_in_body(body_code, d, callee_descr_refs)
            != Some(majit_ir::PyreHelperKind::BinaryOp)
    {
        return false;
    }
    // `iIR`: funcptr i-reg, then the I-list.  The first I-list item is the
    // BINARY_OP tag.  It must be in the callee's immutable constants window;
    // a runtime tag could select an in-place or user-defined operation.
    let Some(&i_len) = body_code.get(d.pc + 2) else {
        return false;
    };
    if i_len == 0 {
        return false;
    }
    let Some(&tag_reg) = body_code.get(d.pc + 3) else {
        return false;
    };
    let Some(&tag) = (tag_reg as usize)
        .checked_sub(num_regs_i)
        .and_then(|constant_index| constants_i.get(constant_index))
    else {
        return false;
    };
    matches!(
        pyre_interpreter::runtime_ops::binary_op_from_tag(tag),
        Some(pyre_interpreter::bytecode::BinaryOperator::Add)
    )
}

pub(crate) fn dispatch_residual_call_iRd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py:1940-1941) opens every residual call
    // with metainterp.clear_exception(), so a caught exception's
    // last_exc_value never survives past the next call — the
    // opimpl_catch_exception assert (pyjitpl.py:504) relies on it.
    // Clear at the arm entry so declined/folded paths uphold the same
    // invariant as the concrete-execution success arm.
    ctx.last_exc_value = None;
    ctx.last_exc_value_concrete = ConcreteValue::Null;
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (r_args, arg_width) = read_ref_var_list(code, op, 1, ctx)?;
    // #62: env-gated recognition probe (no-op unless PYRE_DIAG_INLINE_RECOG
    // set; full-body-walk authoritative path only).  First slice of the
    // call-inlining feature — confirms callable->JitCode recognition before
    // sub-walk wiring.
    if ctx.is_authoritative_executor && std::env::var("PYRE_DIAG_INLINE_RECOG").is_ok() {
        let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
        diagnose_inline_recognition(&arg_concretes, op.pc);
    }
    let descr_offset = 1 + arg_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    // RPython `do_residual_or_indirect_call` always receives a
    // CallDescr (pyjitpl.py:1995). Codewriter emits only CallDescrs
    // for residual_call slots; surface a typed error if a test fixture
    // (or future deviation) routes a non-CallDescr here.
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    // Void shape `_r_v/iRd` (`pyjitpl.py:1348 opimpl_residual_call_r_v =
    // _opimpl_residual_call1`) has no trailing `>X` dst byte. The
    // result OpRef is discarded by `write_residual_call_result_to_dst`'s
    // `'v'` arm, so `dst` is irrelevant on the void path; reading the
    // byte would walk past the operand list.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    let ei = call_descr.get_extra_info();
    // Residual-call entry mirrors `execute_varargs`: even when the walker
    // folds the call or leaves it recorded symbolically, stale handled
    // exceptions from earlier opcodes are not visible to the following
    // linear `catch_exception/L`.
    clear_walk_exception(ctx);

    // #62 slice (3c): attempt full-body-walk inline of a user-function call
    // (dev-gated PYRE_FBW_INLINE).  Eligible exact-positional closure-free
    // calls sub-walk the callee body in place of the residual; ineligible
    // calls (including every non-`call_fn` helper, gated on `pyre_helper`)
    // fall through with no IR emitted.
    if let Some(inlined) = try_walker_inline_user_call(
        ctx,
        op,
        code,
        1,
        funcptr,
        &r_args,
        call_descr,
        ei.pyre_helper,
        dst_bank,
        dst,
    )? {
        return Ok(inlined);
    }

    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
    {
        if let Some(inlined) = try_walker_inline_exception_string_override(
            ctx, op, code, funcptr, &r_args, call_descr, dst,
        )? {
            return Ok(inlined);
        }
    }

    // #62: a self-recursive call the inline path declined (e.g. the
    // branchy `fib`) gets a direct `CALL_ASSEMBLER` to its own loop token
    // (dev-gated PYRE_FBW_REC_CA) instead of the heavyweight func-entry
    // residency residual.  Independent of inline eligibility, matching the
    // trait's pending-token assembler branch (`trace_opcode.rs:6138`).
    if let Some(ca) = try_walker_call_assembler_self_recursive(
        ctx,
        op,
        code,
        funcptr,
        &r_args,
        call_descr,
        ei.pyre_helper,
        dst_bank,
        dst,
    )? {
        return Ok(ca);
    }

    // `_r_*` shape: argboxes = R-list only; argbox_types = [Ref; n].
    let argbox_types: Vec<Type> = vec![Type::Ref; r_args.len()];
    let mut allboxes = build_allboxes(funcptr, &r_args, &argbox_types, call_descr.arg_types());
    replace_movable_load_global_namespace_with_frame_globals(ctx, ei, &mut allboxes);
    if let Err(e) = ensure_residual_call_args_bound(&allboxes, op.pc) {
        if fbw_debug_abort_enabled() {
            let len_pc = op.pc + 1 + 1;
            let n = code[len_pc] as usize;
            let regs: Vec<u8> = code[len_pc + 1..len_pc + 1 + n].to_vec();
            let funcaddr = ctx.trace_ctx.box_value(funcptr).and_then(|v| match v {
                majit_ir::Value::Int(n) => Some(n as u64),
                _ => None,
            });
            eprintln!(
                "[fbw-unbound] pc={} regs={:?} r_args={:?} func={:?} pyre_helper={:?}",
                op.pc,
                regs,
                r_args,
                funcaddr.map(|a| format!("{a:#x}")),
                ei.pyre_helper,
            );
        }
        return Err(e);
    }

    // Optional diagnostic for iRd-shape residual calls.  The STORE_SUBSCR
    // specialization keys on a fn-pointer match against `bh_store_subscr_fn`
    // plus `r_args.len() == 3` with `dst_bank == 'v'`; logging raw addresses
    // here makes mismatches visible without affecting production when the
    // env var is unset.
    if crate::probe_subscr_enabled() {
        let funcptr_addr = ctx.trace_ctx.box_value(funcptr).and_then(|v| match v {
            majit_ir::Value::Int(n) => Some(n as u64),
            _ => None,
        });
        let arg_addrs: Vec<Option<u64>> = r_args
            .iter()
            .map(|&op| {
                ctx.trace_ctx.box_value(op).and_then(|v| match v {
                    majit_ir::Value::Ref(r) => Some(r.as_usize() as u64),
                    _ => None,
                })
            })
            .collect();
        eprintln!(
            "[PYRE_PROBE_SUBSCR] dispatch_residual_call_iRd_kind pc={} dst_bank={} r_args.len={} funcptr_addr={:?} arg_addrs={:?}",
            op.pc,
            dst_bank,
            r_args.len(),
            funcptr_addr.map(|a| format!("{:#x}", a)),
            arg_addrs
                .iter()
                .map(|o| o.map(|a| format!("{:#x}", a)))
                .collect::<Vec<_>>(),
        );
    }

    // STORE_SUBSCR strategy-aware specialization.  Fires when funcptr
    // matches the registered `store_subscr_fn` address, r_args carries the
    // 3-arg `[obj_reg, key_reg, value_reg]` shape codewriter emits
    // (`codewriter.rs:7042 build_store_subscr_fn_residual_call_r_v_insn`),
    // dst_bank is `'v'` (STORE_SUBSCR returns void), and all 3 concrete
    // shadow slots are populated.  On success, records the specialized
    // IR shape (guard_class + guard_strategy + setarrayitem-family) via
    // the trait-equivalent `generated_store_subscr_value` helper (now
    // generic over `WalkerFrameOps`, with `WalkContext` impl).
    //
    // Production dispatch supplies the expected address via
    // `WalkContext.store_subscr_fn_addr`; tests and diagnostics may use
    // `PYRE_WALKER_STORE_SUBSCR_FNADDR=<hex>`.  Without either address,
    // the gate decays to no-op and dispatcher falls through to the generic
    // residual-call path.
    let specialization =
        try_walker_store_subscr_specialization(ctx, code, op, funcptr, &r_args, dst_bank);
    // Drain the snapshot-capture failure the `WalkerFrameOps`
    // `generate_guard` impl latched (its `()` trait signature has no error
    // channel): a guard recorded without a resume snapshot must abort the
    // walk, whether the specialization completed or declined mid-way.
    if let Some(e) = ctx.pending_guard_snapshot_error.take() {
        return Err(e);
    }
    if let Some(outcome) = specialization {
        return Ok((outcome, op.next_pc));
    }

    // StoreName/StoreGlobal IntMutableCell in-place store fold: module-scope
    // dual of the LoadName/LoadGlobal cell fold.  Fires when the target slot
    // holds a stabilised immovable `IntMutableCell` and the store value is a
    // provably-plain-int box; emits `QUASIIMMUT_FIELD` + `setfield_gc_i(cell,
    // intvalue)`, eliding the boxing + residual dict setitem.  Same
    // handler-free gate as the LoadName fold (the fold elides a can-raise
    // residual a `catch_exception/L` could resume into).
    //
    // Default ON (`PYRE_FBW_STORENAME_FOLD=0` opts out).  Two staleness bugs
    // fixed before the flip: (1) the fold now eagerly applies the concrete
    // `cell.intvalue` write (journaled in [`FBW_CELL_STORE_JOURNAL`]) —
    // without it the walk's remaining concrete execution read the pre-store
    // global and the next LOAD fold's cache-hit sanity check tripped;
    // (2) `int_mutable_cell_value_descr` is a singleton `Arc` so the
    // optimizer's `cached_fields` (keyed by `descr_identity`) connects the
    // store's lazy `setfield_gc_i` to the LOAD's `getfield_gc_i` —
    // per-call fresh Arcs let `force_lazy_sets_for_guard` flush the store
    // BELOW an emitted load of the same cell (the nested module-loop
    // `i = i + 1; while i < n` read the pre-increment value and ran one
    // extra iteration).
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && r_args.len() == 3
        && matches!(
            ei.pyre_helper,
            majit_ir::PyreHelperKind::StoreName | majit_ir::PyreHelperKind::StoreGlobal
        )
        && !jitcode_has_exception_handler(code)
        && std::env::var("PYRE_FBW_STORENAME_FOLD").as_deref() != Ok("0")
    {
        if let (Some(&frame_opref), Some(&name_opref), Some(&value_opref)) =
            (r_args.first(), r_args.get(1), r_args.get(2))
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if try_walker_store_name_cell_fold(ctx, op.pc, frame_ptr, w_name_ptr, value_opref)?
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // pyjitpl.py:2003-2005 OS_NOT_IN_TRACE guard — see helper docstring
    // for the convergence rationale.
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py:2011-2014 OS_JIT_FORCE_VIRTUAL fail-loud — walker
    // can't reproduce `_do_jit_force_virtual` without a concrete
    // `vref_ptr` resolver; surface a typed error rather than silently
    // recording `CALL_MAY_FORCE_*`.
    do_jit_force_virtual_guard(ei, op.pc)?;

    // #62: `is_true(box_bool(t))` -> `t` fold.  A `POP_JUMP_IF_*` lowers to an
    // `is_true` residual (`residual_call_r_i`, Int result) whose sole Ref arg
    // is the boxed bool a preceding COMPARE specialization produced.  Folding
    // it to the raw truth Int elides the may-force unbox (and lets the dead box
    // + value-stack store DCE), matching the trait path's branch-on-raw-compare
    // behaviour.  bool->int is value-preserving so the fold is sound.  The
    // lookup is read-only (it does not remove the entry); OpRef SSA-uniqueness
    // (`recorder.rs`) guarantees the box opref never re-binds within one walk,
    // so a stale mis-fold is impossible and physical removal is unnecessary.
    // Full-body walks only: `BOOL_BOX_TRUTH` is reset at FBW walk entry;
    // an arm walk consulting it could read a stale OpRef key from an
    // earlier FBW walk's recorder.
    if ctx.is_authoritative_executor && dst_bank == 'i' && r_args.len() == 1 {
        if let Some(truth) = bool_box_truth_lookup(r_args[0]) {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, truth)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // #124: a TO_BOOL / POP_JUMP truth residual on a provably-int box
        // (e.g. the `(i % 7)` in `(i % 7) and (i + 3)`) folds to a pure
        // `int_is_true`, eliding the may-force call whose force/exc guards
        // mis-resume the kept short-circuit stack.
        if ei.pyre_helper == majit_ir::PyreHelperKind::Truth {
            if let Some(truth) = try_walker_specialize_truth_int(ctx, op.pc, r_args[0])? {
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, truth)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }

    // #62: specialize STORE_SUBSCR `list[int] = value` (int / float storage,
    // in-bounds, type-matching) to the walker-native `setarrayitem_raw` form,
    // eliding the `CALL_MAY_FORCE` that would force the virtualizable every
    // iteration.  Falls through to the generic residual otherwise (SAFE).
    // Full-body walks only: the eager store rides `FBW_STORE_JOURNAL`,
    // whose commit/rollback epilogues run on FBW walk ends.
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && ei.pyre_helper == majit_ir::PyreHelperKind::StoreSubscr
    {
        if try_walker_specialize_store_subscr(ctx, op.pc, &r_args)?.is_some() {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // #171 setslice inline: `target[const_slice] = source` for a
        // same-length, step-1, Integer↔Integer slice — fold the assignment
        // into per-element getarrayitem/setarrayitem on the int_items blocks so
        // a virtualizable BUILD_LIST source temp is consumed without forcing.
        // Gated on `PYRE_NEWLIST_VIRT`; declines to the opaque residual
        // otherwise (SAFE — always byte-correct).
        if newlist_virt_enabled() && try_walker_specialize_setslice(ctx, op.pc, &r_args)?.is_some()
        {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        if ctx.trace_ctx.is_bridge_trace && fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-store-fallthrough] bridge STORE_SUBSCR fell to GENERIC residual at pc={} \
                 (specialization declined — unjournaled concrete store)",
                op.pc
            );
        }
    }

    // Range GET_ITER: virtualize exact machine-word `range` into the same
    // `W_IntRangeIterator` shape PyPy's inlined `descr_iter` would trace.
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::GetIter {
        if let Some(iter_op) = try_walker_specialize_get_iter(ctx, op.pc, &r_args, dst, dst_bank)? {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, iter_op)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }

    // Range FOR_ITER is a C-level iterator advance.  Re-emit its field
    // updates so the opaque ForIterNext residual cannot invalidate optheap;
    // other iterator families retain the residual and its Python semantics.
    // The specialization supplies the same Ref result that the residual would,
    // including NULL for exhaustion, so the codewriter's trailing
    // GuardNonnull remains the only loop-exit guard.
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::ForIterNext {
        if let Some(item_op) =
            try_walker_specialize_for_iter_next(ctx, op.pc, &r_args, dst, dst_bank)?
        {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, item_op)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }

    // #195 / #73: virtualize an arity-2 plain-int BUILD_TUPLE
    // (`newtuple_from_array`) as a `spec_ii` `new_with_vtable` +
    // `value0` / `value1`, so the backing array build and the partner
    // UNPACK_SEQUENCE reads DCE to a pure-int loop.  Falls through to the
    // opaque residual for any other shape (SAFE — never declined).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::NewtupleFromArray
        && try_walker_specialize_newtuple(ctx, op.pc, &r_args, dst, dst_bank)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171: virtualize a non-escaping BUILD_LIST (`newlist_from_array`) by
    // decomposing it into the `opimpl_newlist` shape (`pyjitpl.py:779`) —
    // `new_with_vtable` + `new_array` + `setarrayitem_gc` + `setfield_gc` —
    // choosing the storage strategy from the concrete element shadows exactly
    // like `w_list_new` / `list_strategy_for`, so the traced object matches what
    // the blackhole rebuilds on deopt.  Gated on `PYRE_NEWLIST_VIRT`
    // (default-on).  Falls through to the opaque residual for any shape it
    // cannot reproduce faithfully (empty list, non-const array length, an
    // element without a concrete Ref shadow) — SAFE, never declined.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::NewlistFromArray
        && newlist_virt_enabled()
        && try_walker_specialize_newlist(ctx, op.pc, &r_args, dst, dst_bank)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171: specialize `lst.append(x)` so its array ops reach the trace,
    // replacing the opaque `bh_call_fn` residual (orthodox descent of the
    // real `w_list_append` body; see `try_walker_orthodox_list_append`).  The
    // call arrives as `CallFn` with `dst_bank == 'r'` (the None result is a
    // Ref, not void) and `r_args = [bound-method, PY_NULL, value]`.  Falls
    // through to the residual for any non-matching shape (SAFE).  The eager append rides
    // `FBW_APPEND_JOURNAL`, whose commit/rollback epilogues run on FBW walk
    // ends (same lifecycle as the STORE_SUBSCR store journal).
    //
    // Restrict to the top full-body frame: inside an inlined callee sub-walk
    // (`fbw_mode.inline_subwalk`) the fold's gating guards collapse
    // their resume to the caller's CALL boundary (`entry_py_pc` /
    // `outer_active_boxes`), which re-executes the whole caller iteration on a
    // guard failure — doubling any caller side effect sequenced before the
    // inlined call (e.g. a `STORE_ATTR` ahead of an inlined `push(lst, x)`).
    // An inlined append falls back to the generic residual, which resumes
    // *past* the call (after_residual_call) and so re-runs nothing extra.
    //
    // Both loop and function-entry (no-loop) traces are eligible.  A
    // no-loop helper compiled from entry (e.g. `def push(a, v): a.append(v)`
    // called in a hot loop) traces with `header_pc == 0`; its spare-capacity
    // guard's resume reconstructs the receiver from the call-site coordinate
    // published below (`collect_outer_active_boxes` at the CALL py_pc), which
    // preserves the receiver local across the fold's mid-statement guards in
    // both trace kinds.  The earlier loop-only restriction was a carryover
    // from the retired hand-rolled fold (#227), whose function-entry exit
    // layout dropped the receiver; the orthodox descent's resume coordinate
    // does not, verified by a two-list alternating-receiver append stress
    // (any wrong receiver box corrupts the cross-checked lists) and the
    // parity suite folding function-entry helper appends on both backends.
    // #171 ORTHODOX descent: descend the real `w_list_append` body,
    // recording its array ops native.
    // A decline (`None`) falls through to the generic residual below; an
    // un-lowered in-body helper aborts the trace (graceful interpreter
    // fallback).  Gated to top full-body frames, not inside a sub-walk.
    if ctx.is_authoritative_executor
        && !ctx.fbw_mode.inline_subwalk
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_orthodox_list_append(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171 ORTHODOX descent for the LIST_APPEND opcode (comprehension append,
    // e.g. `[f(x) for x in xs]`).  The codewriter lowers LIST_APPEND to a void
    // `jit_list_append(list, value)` residual tagged `ListAppendValue` (the
    // list is the peeked receiver, the value the popped operand — no
    // bound-method callable), so it arrives with `dst_bank == 'v'`.  Fold it
    // through the same `w_list_append` descent as the CallFn method-call form.
    // Gated to top full-body frames, not inside a sub-walk (same
    // caller-side-effect doubling concern as the CallFn form).
    if ctx.is_authoritative_executor
        && !ctx.fbw_mode.inline_subwalk
        && dst_bank == 'v'
        && ei.pyre_helper == majit_ir::PyreHelperKind::ListAppendValue
    {
        // Fold, or ABORT — never fall through to the generic residual.  The
        // fold's `orthodox_list_append_commit` journals the append
        // (`fbw_append_journal_push`) so `fbw_store_journal_rollback` can rewind
        // it on abort; the generic dispatcher concrete-executes
        // `jit_list_append` UNjournaled, so a later abort + interpreter replay
        // would apply the SAME append twice (a silent double).  The decline
        // point in `try_walker_orthodox_list_append_opcode` is side-effect-free
        // (it declines BEFORE emitting any IR), so surfacing the abort here is
        // safe.  Mirrors the pre-#171 `emit_abort_permanent` lowering.
        if try_walker_orthodox_list_append_opcode(ctx, code, op, &r_args, dst)?.is_some() {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        return Err(DispatchError::UnfoldableListAppendResidualUnsupported { pc: op.pc });
    }

    // `len(x)` on an exact canonical list: inline the strategy-guarded
    // length read (guard_value callable + guard_class + exact w_class +
    // guard_value strategy + length getfield + wrapint) instead of the
    // opaque `bh_call_fn(len_builtin, NULL, x)` residual — the shape the
    // meta-tracer produces upstream (descroperation.py:294 `_len` →
    // `W_ListObject.length()`).  Read-only like the SUBSCR fold, so no
    // sub-walk restriction; any non-matching shape falls through to the
    // generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_builtin_len(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // B3 (`PYRE_FBW_RAISE`, default OFF): a `raise Type(args)` of a canonical
    // builtin exception class arrives as two residuals — a `CallFn` that
    // constructs the exception, and a `RaiseVarargs`
    // (`normalize_raise_varargs_jit`) that publishes it.  The construct fold
    // (`try_walker_trace_exception_new`) emits the inline virtualizable
    // `NewWithVtable` + SetField shape instead of the opaque `bh_call_fn`
    // constructor call (mirrors the retired trait-side constructor fold); the
    // raise fold (`try_walker_trace_raise_builtin`) then skips the residual
    // publish for a freshly-built exception with no `from` cause, emitting
    // `__context__` as a `SetfieldGc` on the still-virtual exception.
    // Together they drop the two may-force calls (construct + normalize) so
    // the exception virtualizes and DCEs.  Any non-matching shape falls
    // through to the generic residual (SAFE).
    if ctx.is_authoritative_executor
        && fbw_raise_enabled()
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_trace_exception_new(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && fbw_raise_enabled()
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs
        && r_args.is_empty()
        && ctx.fbw_mode.current_exception_seed.is_some()
    {
        let seed = ctx.fbw_mode.current_exception_seed.unwrap();
        let concrete = ctx.fbw_mode.current_exception_seed_concrete;
        if !concrete.is_null() && unsafe { pyre_object::is_exception(concrete) } {
            // `RAISE_VARARGS 0` may use the normalizing nullary helper rather
            // than the raw current-exception helper.  A bridge seed is already
            // a live BaseException, so the helper's successful result is the
            // pending fieldbox itself; retaining that OpRef keeps the value
            // loop-variant in the bridge namespace.
            ctx.trace_ctx.set_opref_concrete(
                seed,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
            );
            write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', seed)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }
    if ctx.is_authoritative_executor
        && fbw_raise_enabled()
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs
        && try_walker_trace_raise_builtin(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    // B3 piece 3 (`PYRE_FBW_RAISE`): lower the PUSH_EXC_INFO / POP_EXCEPT
    // exc-info-stack residuals to GETFIELD_GC_R / SETFIELD_GC on the EC's
    // `sys_exc_value` slot, mirroring the trait's `push_exc_info` /
    // `pop_except` overrides (`trace_opcode.rs:11119`).  Recognised by the
    // codewriter-stamped `pyre_helper` tag (not a funcptr address — the
    // residual calls the cross-crate `cpu.{get,set}_current_exception_fn`
    // wrappers).  A balanced PUSH save + POP restore on the same descr-
    // identity field is dead-store-eliminated by the heap optimizer, so a
    // non-escaping exception (built + raised + caught in one trace) stays
    // virtual and DCEs — eliding the per-iteration `set_current_exception`
    // CALL that otherwise forces the exception to materialize.
    if ctx.is_authoritative_executor
        && fbw_raise_enabled()
        && matches!(
            ei.pyre_helper,
            majit_ir::PyreHelperKind::GetCurrentException
                | majit_ir::PyreHelperKind::SetCurrentException
        )
        && try_walker_lower_exc_info_residual(
            ctx,
            code,
            op,
            ei.pyre_helper,
            &r_args,
            dst_bank,
            dst,
        )?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // pyjitpl.py:2063 forces-branch sub-case: when the descr's
    // `call_release_gil_target` is a non-NULL `(realfuncaddr, saveerr)`
    // pair, route through `direct_call_release_gil` which records
    // `CALL_RELEASE_GIL_*` with the upstream-shape arglist
    // `[savebox, funcbox] + argboxes[1:]` (pyjitpl.py:3675-3681).  All
    // other forces-branch paths (CALL_MAY_FORCE_*, the loopinvariant
    // sub-case below, the elidable branch, the default branch) come
    // out of `select_residual_call_opcode`.
    if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iRd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        // pyjitpl.py:2087-2110 EF_LOOPINVARIANT short-circuit. The
        // cached path emits no IR op and no guard, so result-before-
        // guard ordering is moot — write the dst eagerly.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iRd_kind");
        // The abort gate is static (EI flags) and must run BEFORE the
        // concrete executor below: `do_residual_call` (pyjitpl.py:2019)
        // executes the helper only on a path that keeps recording — it has
        // no execute-then-abandon shape.  Aborting after execution would
        // leave the helper's heap/exception effects standing while the
        // declined retrace re-runs the same bytecode, double-applying them.
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py:2017 `vable_and_vrefs_before_residual_call` — fires
        // unconditionally on the forces branch.  Records FORCE_TOKEN +
        // SETFIELD_GC IR for the active virtualizable; the runtime heap
        // mutations on `vinfo.tracing_before_residual_call` and
        // `vrefinfo.tracing_before_residual_call` (`pyjitpl.py:3318-3330`)
        // sit on the trait-driven leg only — see
        // [`walker_vable_and_vrefs_before_residual_call`] docstring for
        // the IR-vs-heap split rationale.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx);
        }

        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py:1346-1400 `_record_helper_pure` parity: for
        // `CallPure*` whose every argbox carries a known `box_value`,
        // execute the helper now and stamp `recorded` with the result so
        // downstream walker chain (sub-jitcode bodies that consume the
        // result via `concrete_of_opref`) folds end-to-end.  No-op when
        // any argbox is symbolic, when the EI can raise, or for non-pure
        // call opcodes.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);

        // pyjitpl.py:1346/1349/1354 `_opimpl_residual_call{1,2,3}` parity
        // for the non-elidable shapes (Task #390 sub-slice 3).  PyPy
        // concrete-executes EVERY residual call regardless of EI — the
        // `exc` flag only selects the *guard* shape downstream
        // (`handle_possible_exception` → `GUARD_EXCEPTION` vs
        // `GUARD_NO_EXCEPTION`), not whether the helper runs.  Without
        // this, walker-recorded non-elidable helpers
        // (`store_subscr_fn`, `set_current_exception`, …) would skip
        // their heap mutation because `eval.rs:3285-3308`'s walker-skip
        // path bypasses `execute_opcode_step` → SIGBUS on the next read
        // of the un-mutated container (M4 walker unactivated taxonomy).
        // Task #390 sub-slice 4: PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  Pure/elidable calls never reach
        // this dispatcher (they fold via the pure-call executor).
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                fbw_abort_nested_unjournaled_residual(ctx, op.pc)?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iRd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        // pyjitpl.py:2659 `_record_helper_varargs` parity: every
        // recorded varargs op invalidates the heapcache via
        // `heapcache.invalidate_caches_varargs(opnum, descr,
        // argboxes)`.  Pyre's `record_op_with_descr` does NOT
        // auto-invalidate, so call it explicitly here.  Forces
        // branch (`select_residual_call_opcode` returned a
        // `CallMayForce*`) thus matches `pyjitpl.py:2072` which uses
        // `opnum1 = CALL_MAY_FORCE_*`; non-forces branches
        // (`CallLoopinvariant*`/`CallPure*`/`Call*`) match the
        // `_record_helper_varargs` invocation that runs inside
        // upstream's `executor.execute_varargs(opnum, ...)`.
        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        // pyjitpl.py:1950-1954 execute_varargs: `make_result_of_lastop(op)`
        // runs BEFORE `handle_possible_exception()` precisely "because we need
        // the box to show up in get_list_of_active_boxes()".  Write the dst
        // for every non-void result REGARDLESS of whether the helper raised,
        // so the GUARD_NOT_FORCED fail_args snapshot reads the recorded OpRef
        // in the slot the resume position points at — otherwise a raising call
        // surfaces NONE in fail_args for the `>X` slot.  On a raised call the
        // OpRef carries a Null concrete shadow (never read on the exception
        // path); only the *caching* of a raised result is skipped below,
        // matching upstream's `not last_exc_value` pure-cache gate.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        // pyjitpl.py:2079 `metainterp.generate_guard(rop.GUARD_NOT_FORCED)`
        // — unconditionally on the forces-virtual-or-virtualizable branch.
        // Walker omits the `vable_after_residual_call(funcbox)`
        // short-circuit (`pyjitpl.py:2078`) entirely — the trait-dispatch
        // leg detects vable escape via
        // `state.rs MIFrame::vable_after_residual_call`
        // (`trace_opcode.rs:2237-2263`) and aborts to blackhole through
        // `PyError::runtime_error("ABORT_ESCAPE: ...")` before walker IR
        // diff would run.
        if emit_guard_not_forced {
            // #73: maintain the `-live-` AFTER anchor.  A
            // residual-call guard reads its resume point at `self.pc` (the
            // `-live-` trailing the call, `pyjitpl.py:195`).  `op.next_pc` is
            // the first byte after the residual_call opcode, which the
            // `[funcptr, Call, -live-]` layout (jitcode.rs:565) makes the
            // trailing `-live-` byte.  Side-data only.
            ctx.live_after_jit_pc = op.next_pc;
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        // pyjitpl.py:2082 `metainterp.handle_possible_exception()` —
        // emits `GUARD_EXCEPTION(exc_type)` when the recording-time
        // helper raised (pinning the class for guard recovery), else
        // `GUARD_NO_EXCEPTION`.  The capture ports
        // `capture_resumedata(after_residual_call=True)`
        // (`pyjitpl.py:2599-2603`, keyed on the guard opcode) so the
        // optimizer's `store_final_boxes_in_guard` finds a
        // `rd_resume_position` advanced *past* the call.
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
                // `pyjitpl.py:2156-2168 handle_possible_exception` routes
                // the raising branch through `finishframe_exception()`
                // immediately after emitting `GUARD_EXCEPTION`, so the
                // remaining bytes of the arm never run.  Surface the
                // outcome to `walk_loop` as `SubRaise`: at top-level it
                // emits the outer `FINISH(exc)` and Terminates the trace;
                // at sub-walk depth it propagates up to the caller's
                // `inline_call_*` handler.  Continuing past this point
                // would record dead arm IR (e.g. the arm's tail
                // `*_return`) onto an exception path and confuse the
                // optimizer's guard-fail snapshot.
                let exc = ctx
                    .last_exc_value
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete;
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                // Request that this residual call's no-exception-guard resume
                // route through the call's OWN post-call catch
                // (`GuardCaptureScope::residual_call_catch_resume`).  The
                // snapshot helper carries the call's jitcode offset only when
                // the CALL pc is actually covered by the code's exception table
                // (checked in `walker_capture_snapshot_for_last_guard_impl`);
                // an uncovered residual keeps the generic fallthrough resume.
                // See the scope field's doc.
                walker_capture_snapshot_for_last_guard_scoped(
                    ctx,
                    op.pc,
                    GuardCaptureScope {
                        residual_call_catch_resume: true,
                        ..GuardCaptureScope::default()
                    },
                )?;
            }
        }

        // pyjitpl.py:2109 `heapcache.call_loopinvariant_now_known`:
        // populate the cache so a subsequent matching call short-
        // circuits via the lookup above.  No-op for non-loopinvariant
        // EI per `loopinvariant_now_known`'s extraeffect check.
        //
        // Skip on `resid_raised`: caching a `recorded` OpRef with no
        // stamped concrete would propagate the un-stamped value into a
        // subsequent loop iteration's `loopinvariant_lookup` hit,
        // bypassing the actual helper call.
        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

#[allow(non_snake_case)]
pub(crate) fn dispatch_residual_call_iIRd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py:1940-1941) clear_exception at every
    // residual-call entry; see dispatch_residual_call_iRd_kind.
    let saved_last_exc_value = ctx.last_exc_value;
    let saved_last_exc_value_concrete = ctx.last_exc_value_concrete;
    let preserve_last_exc_for_handler =
        saved_last_exc_value.is_some() && reads_last_exc_before_next_catch(code, op.next_pc);
    if !preserve_last_exc_for_handler {
        clear_walk_exception(ctx);
    }
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (i_args, i_width) = read_int_var_list(code, op, 1, ctx)?;
    let (r_args, r_width) = read_ref_var_list(code, op, 1 + i_width, ctx)?;
    let descr_offset = 1 + i_width + r_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let mut descr = read_descr(code, op, descr_offset, ctx)?;
    let original_call_descr =
        descr
            .as_call_descr()
            .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
                pc: op.pc,
                descr_index,
            })?;
    let descr_key = descr.index();
    // Void shape `_ir_v/iIRd` (`pyjitpl.py:1351 opimpl_residual_call_ir_v =
    // _opimpl_residual_call2`) has no `>X` dst byte; see
    // `dispatch_residual_call_iRd_kind` for the void operand-layout note.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // Flat argboxes = i_args ++ r_args (`boxes2` argcode order).
    // Parallel argbox_types stamps each entry with its source bank so
    // `_build_allboxes`'s type-filter loops can permute correctly.
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len());
    let mut argbox_types: Vec<Type> = Vec::with_capacity(i_args.len() + r_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    let mut allboxes = build_allboxes(
        funcptr,
        &argboxes,
        &argbox_types,
        original_call_descr.arg_types(),
    );

    // STORE_ATTR fold (mapdict.py:1591-1653): recognize an existing unboxed
    // integer slot and replace only the generic setattr residual's helper,
    // arguments, and effect.  The transformed CallN continues through the
    // ordinary record + concrete-execute path below; unsupported receivers,
    // descriptors, custom hooks, absent/boxed/float slots, and type-changing
    // values retain the original CallMayForceN unchanged.
    if ctx.is_authoritative_executor
        && original_call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::StoreAttr
        && fbw_storeattr_fold_enabled()
    {
        if let (Some(&obj_opref), Some(&value_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), r_args.get(2), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if let Some(specialization) = try_walker_specialize_store_attr(
                    ctx,
                    op.pc,
                    obj_opref,
                    value_opref,
                    w_code_ptr,
                    namei as usize,
                    original_call_descr.get_extra_info(),
                )? {
                    match specialization {
                        WalkerStoreAttrSpecialization::Residual(
                            specialized_descr,
                            specialized_allboxes,
                        ) => {
                            descr = specialized_descr;
                            allboxes = specialized_allboxes;
                        }
                        WalkerStoreAttrSpecialization::Direct => {
                            return Ok((DispatchOutcome::Continue, op.next_pc));
                        }
                    }
                }
            }
        }
    }

    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;

    let ei = call_descr.get_extra_info();
    // pyjitpl.py:2003-2005 OS_NOT_IN_TRACE guard — see helper docstring
    // for the convergence rationale.
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py:2011-2014 OS_JIT_FORCE_VIRTUAL fail-loud — see
    // `dispatch_residual_call_iRd_kind` for the rationale.
    do_jit_force_virtual_guard(ei, op.pc)?;

    // Method-form `CALL` helpers lower through the mixed int/ref residual
    // shape (`bh_call_fn_N(callable, null_or_self, args...)` carries the call
    // arity in the Int list).  Share the same user-function inline gate as the
    // plain Ref-only residual, but read the concrete Ref shadows from the
    // shifted R-list offset.
    if let Some(inlined) = try_walker_inline_user_call(
        ctx,
        op,
        code,
        1 + i_width,
        funcptr,
        &r_args,
        call_descr,
        ei.pyre_helper,
        dst_bank,
        dst,
    )? {
        return Ok(inlined);
    }

    // LoadConst fold: the LOAD_CONST helper (oopspec `LoadConst`, set
    // codewriter-side at flatten.rs
    // `build_residual_call_ir_r_single_ref_plain_insn_from_operands`)
    // re-materializes `co_consts[idx]` on every call.  When both the const
    // index (`i_args[0]`) and the code pointer (`r_args[0]`, the promoted
    // `frame.pycode`) are concrete, fold to the constant ref the call would
    // have produced — the indexed entry is loop-invariant — and suppress the
    // residual.  Falls through to the generic record when either operand is
    // not concrete (the residual stays correct in that case).
    if ei.pyre_helper == majit_ir::PyreHelperKind::LoadConst {
        if let (Some(&idx_opref), Some(&code_opref)) = (i_args.first(), r_args.first()) {
            if let (
                Some(majit_ir::Value::Int(consti)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(idx_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                // Materialize the constant identically to the runtime
                // `bh_load_const_fn` helper (call_jit.rs): a code constant reads
                // the one shared wrapper off the virtualizable `pycode`'s
                // `co_consts_w[index]`, other constants realize directly.
                let w_const = unsafe {
                    let w_code = pyre_interpreter::pycode::w_code_co_const(
                        w_code_ptr as pyre_object::PyObjectRef,
                        consti as usize,
                    );
                    if !w_code.is_null() {
                        w_code
                    } else {
                        let code = &*(pyre_interpreter::w_code_get_ptr(
                            w_code_ptr as pyre_object::PyObjectRef,
                        )
                            as *const pyre_interpreter::CodeObject);
                        pyre_interpreter::pyframe::load_const_from_code(code, consti as usize)
                    }
                };
                let const_box = ctx.trace_ctx.const_ref(w_const as i64);
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, const_box)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }

    // LoadGlobal fold (#62): mirror the trait LOAD_GLOBAL cell fast-path so
    // the optimizer hoists the module-global lookup out of the loop instead of
    // keeping the opaque CanRaise residual every iteration.  Requires namei
    // (i_args[0]), namespace (r_args[0]), promoted pycode (r_args[1]) and the
    // live frame (r_args[2]) concrete; the cell-strategy module-dict fast path
    // (name present in the executing frame's globals) and the builtins-cell
    // fast path (name absent from globals, resolved via `frame.get_builtin()`)
    // are both foldable.
    //
    // Handler-bearing bodies: `load_global_fn` is `CallFlavor::Plain` (can
    // raise NameError), so a residual lowering keeps a `GUARD_NO_EXCEPTION`
    // that a `catch_exception/L` in the same body may resume into.  The fold
    // emits the lookup as `ElidableCannotRaise` (the name resolves at trace
    // time and the version watchers fail the loop on any rebind/shadow), so a
    // SUCCESSFUL fold provably can't raise NameError and dropping that guard
    // for this load is sound — the handler can never be entered from it.  We
    // therefore attempt the fold even in handler-bearing bodies and keep the
    // residual (with its guard) only when the fold DECLINES.  The `B3`/builtin
    // raise+catch path (`PYRE_FBW_BUILTIN_FOLD`) needs this so the
    // `raise ValueError`/`except ValueError` class loads fold to const.
    //
    // Default ON since the Phase 5 flip (`PYRE_FBW_LOADGLOBAL_FOLD=0` opts
    // out): the fold is correct (`try_walker_load_global_cell_fold`
    // resolves the `co_names` index the same way `bh_load_global_fn` does)
    // and reaches production parity for global-function-call loops when
    // combined with the user-call inlining path.  The handler-bearing
    // reachability is additionally gated `PYRE_FBW_BUILTIN_FOLD` (default ON)
    // so the legacy handler-free behavior is recoverable.
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::LoadGlobal {
        if let (Some(&namei_opref), Some(&ns_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1))
        {
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(ns_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(ns_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                let name_idx = (namei as usize) >> 1;
                if !mark_trace_reads_module_global_from_code(
                    ctx.trace_ctx,
                    ns_ptr as pyre_object::PyObjectRef,
                    w_code_ptr,
                    name_idx,
                ) {
                    ctx.trace_ctx.reads_module_global = true;
                }
            } else {
                ctx.trace_ctx.reads_module_global = true;
            }
        } else {
            ctx.trace_ctx.reads_module_global = true;
        }
    }
    if ctx.is_authoritative_executor
        && ei.pyre_helper == majit_ir::PyreHelperKind::LoadGlobal
        && std::env::var("PYRE_FBW_LOADGLOBAL_FOLD").as_deref() != Ok("0")
        && (!jitcode_has_exception_handler(code) || fbw_builtin_fold_enabled())
    {
        if let (Some(&namei_opref), Some(&ns_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1))
        {
            // The live frame operand (r_args[2]) is needed for the builtins
            // fallback (`frame.get_builtin()`); it may be absent/unseeded
            // (an inlined callee's `portal_frame_reg`), in which case only
            // the module-dict cell path is attempted.
            let frame_ptr = r_args
                .get(2)
                .and_then(|&f| ctx.trace_ctx.box_value(f))
                .and_then(|v| match v {
                    majit_ir::Value::Ref(majit_ir::GcRef(p)) => Some(p),
                    _ => None,
                })
                .unwrap_or(0);
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(ns_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(ns_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                if try_walker_load_global_cell_fold(
                    ctx, op.pc, dst, dst_bank, ns_ptr, w_code_ptr, frame_ptr, namei,
                )? {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // LoadName fold: module-scope LOAD_NAME mirror of the LoadGlobal fold
    // above.  The residual is `bh_load_name_fn(frame, w_name, namei)`, so
    // r_args = [frame, w_name].  Same handler-free gate (the fold elides a
    // CanRaise residual a `catch_exception` could otherwise resume into);
    // `try_walker_load_name_cell_fold` gates module scope at runtime and
    // routes non-module frames back to this residual.
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::LoadName {
        if let (Some(&frame_opref), Some(&name_opref)) = (r_args.first(), r_args.get(1)) {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if !mark_trace_reads_module_global_from_frame_name(
                    ctx.trace_ctx,
                    frame_ptr,
                    w_name_ptr,
                ) {
                    ctx.trace_ctx.reads_module_global = true;
                }
            } else {
                ctx.trace_ctx.reads_module_global = true;
            }
        } else {
            ctx.trace_ctx.reads_module_global = true;
        }
    }
    if ctx.is_authoritative_executor
        && ei.pyre_helper == majit_ir::PyreHelperKind::LoadName
        && !jitcode_has_exception_handler(code)
        && std::env::var("PYRE_FBW_LOADNAME_FOLD").as_deref() != Ok("0")
    {
        if let (Some(&frame_opref), Some(&name_opref)) = (r_args.first(), r_args.get(1)) {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if try_walker_load_name_cell_fold(ctx, op.pc, dst, dst_bank, frame_ptr, w_name_ptr)?
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // LoadAttr fold (`mapdict.py:1479-1537 LOAD_ATTR_caching`): fold a
    // monomorphic plain instance-attribute read to `guard_class` +
    // `guard_value(map)` + `getfield(storage)` + `getarrayitem(C_index)`,
    // eliding the opaque `getattr_fn` MRO-walk residual.  The residual is
    // `load_attr_fn(obj, code, name_idx)`, so `r_args = [obj, code]` and
    // `i_args = [name_idx]`.  A successful fold provably cannot raise (the map
    // guard proves the attribute is present on this shape), so it is attempted
    // even in handler-bearing bodies; every unfoldable shape falls through to
    // the residual (which keeps its exception guard).
    if ctx.is_authoritative_executor
        && ei.pyre_helper == majit_ir::PyreHelperKind::LoadAttr
        && fbw_loadattr_fold_enabled()
    {
        if let (Some(&obj_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if try_walker_specialize_load_attr(
                    ctx,
                    op.pc,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }
    if ctx.is_authoritative_executor
        && ei.pyre_helper == majit_ir::PyreHelperKind::LoadAttr
        && fbw_loadmethod_fold_enabled()
        && next_op_is_load_method_self_for_attr(code, op, ctx, dst)
    {
        if let (Some(&obj_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if try_walker_specialize_load_method_attr(
                    ctx,
                    op.pc,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }
    if ctx.is_authoritative_executor
        && ei.pyre_helper == majit_ir::PyreHelperKind::LoadMethodSelf
        && fbw_loadmethod_fold_enabled()
    {
        if let (Some(&namei_opref), Some(&obj_opref), Some(&attr_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1), r_args.get(2))
        {
            let r_len_pc = op.pc + 1 + 1 + i_width;
            let attr_reg = code
                .get(r_len_pc + 1 + 1)
                .copied()
                .map(usize::from)
                .unwrap_or(usize::MAX);
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                if try_walker_fold_load_method_self(
                    ctx,
                    op.pc,
                    obj_opref,
                    attr_opref,
                    attr_reg,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    replace_movable_load_global_namespace_with_frame_globals(ctx, ei, &mut allboxes);

    // Defer the arg-bound check past the short-circuiting LoadConst /
    // LoadGlobal folds above: each resolves the call to a constant from
    // `i_args`/`r_args` without recording it, so an unbound *trailing* arg
    // is irrelevant when the call folds away.  In particular an inlined
    // callee's `load_global` passes its OWN unseeded `portal_frame_reg`
    // (Path-1, #68); the fold elides that call, so the frame box never
    // needs binding.  Only a call that survives to a genuine record
    // (BoxInt exec, generic residual below) requires every box bound.
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    // BoxInt fold (#62): `box_int_fn(raw)` allocates a fresh `PyLong`.  The
    // opaque CanRaise residual the generic leg would record blocks the
    // optimizer (no DCE of an unused/round-tripped box).  Emit the
    // virtualizable `new_with_vtable` + `setfield_gc` form (`wrapint`,
    // identical to the BINARY_OP result box) so a following unbox
    // (`getfield_gc_pure`) forwards through the setfield and the box DCEs
    // when it never escapes.  The concrete shadow carries the authentic
    // boxed pointer so downstream specializations still see a concrete int.
    if ei.pyre_helper == majit_ir::PyreHelperKind::BoxInt && dst_bank == 'r' {
        if let Some(&raw_arg) = i_args.first() {
            if let Some(boxed_ptr) = walker_execute_may_force_boxed(ctx, &allboxes, call_descr) {
                let intval =
                    unsafe { pyre_object::w_int_get_value(boxed_ptr as pyre_object::PyObjectRef) };
                let boxed = walker_box_int(ctx, op.pc, raw_arg, intval)?;
                ctx.trace_ctx
                    .set_opref_concrete(boxed, box_int_concrete(intval, boxed_ptr));
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, boxed)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }

    // #57: speculative int specialization for the BINARY_OP / COMPARE_OP
    // helper (oopspec `BinaryOp` / `CompareOp`, set codewriter-side at
    // flatten.rs `build_residual_call_ir_r_insn_from_operands`).  When both
    // operands are concrete `W_IntObject`, re-emit the guard_class + unbox
    // + int_OP (+ rebox / bool-box) sequence instead of an opaque
    // CALL_MAY_FORCE, matching the retired trait-side int binop / compare
    // paths.  Falls through to the generic record for
    // non-int operands / deferred operators.
    if matches!(
        ei.pyre_helper,
        majit_ir::PyreHelperKind::BinaryOp | majit_ir::PyreHelperKind::CompareOp
    ) {
        if let Some(&tag_opref) = i_args.first() {
            if let Some(majit_ir::Value::Int(op_tag)) = ctx.trace_ctx.box_value(tag_opref) {
                let specialized = if ei.pyre_helper == majit_ir::PyreHelperKind::BinaryOp {
                    let is_subscr = matches!(
                        pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
                        Some(pyre_interpreter::bytecode::BinaryOperator::Subscr)
                    );
                    if is_subscr {
                        // BINARY_SUBSCR list[int] getitem (int/float storage);
                        // falls through to the generic may-force leg otherwise.
                        try_walker_specialize_subscr(
                            ctx, op.pc, &r_args, &allboxes, call_descr, dst, dst_bank,
                        )?
                    } else {
                        // int specialization first; float (incl. mixed int/float)
                        // as a fallback so two-int operands keep int arithmetic.
                        match try_walker_specialize_binary_op_int(
                            ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                        )? {
                            Some(()) => Some(()),
                            // W_LongObject (bigint) operands: take the long
                            // fast path before falling to float so two-long
                            // operands keep bigint arithmetic.
                            None => match try_walker_specialize_binary_op_long(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )? {
                                Some(()) => Some(()),
                                // Two-long true-divide → float fast path
                                // (CallPureF + wrapfloat), before the generic
                                // float leg which only handles float operands.
                                None => match try_walker_specialize_truediv_op_long(
                                    ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst,
                                    dst_bank,
                                )? {
                                    Some(()) => Some(()),
                                    None => try_walker_specialize_binary_op_float(
                                        ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst,
                                        dst_bank,
                                    )?,
                                },
                            },
                        }
                    }
                } else if op_tag == 10 && fbw_raise_enabled() && ctx.is_authoritative_executor {
                    // B3 (`PYRE_FBW_RAISE`): `op_tag == 10` is CHECK_EXC_MATCH
                    // (`bh_compare_fn(exc, match_type, 10)`,
                    // `call_jit.rs:4299`).  Fold the match concretely to a
                    // const bool (the immortal TRUE/FALSE singleton) so the
                    // exception stays virtual and DCEs, eliding the may-force
                    // compare + its truth-extract residual.  Declines (falls
                    // through to the int/float compare attempts, which also
                    // decline for Ref operands → generic residual) when an
                    // operand has no concrete shadow or the match target is
                    // not a valid exception class.
                    let keep_last_exc_for_handler =
                        reads_last_exc_before_next_catch(code, op.next_pc);
                    let folded =
                        try_walker_fold_check_exc_match(ctx, op.pc, &r_args, dst, dst_bank)?;
                    if folded.is_some() && keep_last_exc_for_handler {
                        ctx.last_exc_value = saved_last_exc_value;
                        ctx.last_exc_value_concrete = saved_last_exc_value_concrete;
                    }
                    folded
                } else {
                    // int compare first; then long (two-bigint operands keep
                    // bigint comparison); float (incl. mixed int/float) last.
                    match try_walker_specialize_compare_op_int(
                        ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                    )? {
                        Some(()) => Some(()),
                        None => match try_walker_specialize_compare_op_long(
                            ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                        )? {
                            Some(()) => Some(()),
                            None => try_walker_specialize_compare_op_float(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )?,
                        },
                    }
                };
                if specialized.is_some() {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                if ei.pyre_helper == majit_ir::PyreHelperKind::BinaryOp {
                    if let Some(inlined) = try_walker_inline_user_binop(
                        ctx, op, code, op_tag, &r_args, call_descr, dst, dst_bank,
                    )? {
                        return Ok(inlined);
                    }
                }
                if ei.pyre_helper == majit_ir::PyreHelperKind::CompareOp {
                    if let Some(inlined) = try_walker_inline_user_compareop(
                        ctx, op, code, op_tag, &r_args, call_descr, dst, dst_bank,
                    )? {
                        return Ok(inlined);
                    }
                }
            }
        }
    }

    // UNPACK_SEQUENCE fold (#73): read an arity-2 specialised int tuple's
    // elements directly so the unpacked items stay unboxed ints (the trait
    // path's value0/value1 fold).  A non-foldable shape falls through to the
    // opaque residual below — correct, no decline.
    if matches!(
        ei.pyre_helper,
        majit_ir::PyreHelperKind::UnpackSequence | majit_ir::PyreHelperKind::UnpackItem
    ) && try_walker_specialize_unpack(
        ctx,
        op.pc,
        ei.pyre_helper,
        &i_args,
        &r_args,
        &allboxes,
        call_descr,
        dst,
        dst_bank,
    )?
    .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // pyjitpl.py:2063 forces-branch sub-case: route release-gil through
    // `direct_call_release_gil`.  Mirrors `dispatch_residual_call_iRd_kind`.
    if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        // pyjitpl.py:2087-2110 EF_LOOPINVARIANT short-circuit; no IR
        // op, no guard, ordering moot.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iIRd_kind");
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py:2017 `vable_and_vrefs_before_residual_call` —
        // records FORCE_TOKEN + SETFIELD_GC IR; runtime heap mutations
        // and the after-call helpers stay on the trait-driven leg.
        // See `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx);
        }

        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py:1346-1400 `_record_helper_pure` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream walk.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);

        // Non-elidable concrete-execute parity (Task #390 sub-slice 3)
        // — see `dispatch_residual_call_iRd_kind` for the full citation.
        // Task #390 sub-slice 4: PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  Pure/elidable calls never reach
        // this dispatcher (they fold via the pure-call executor).
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                fbw_abort_nested_unjournaled_residual(ctx, op.pc)?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iIRd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        // pyjitpl.py:2659 `_record_helper_varargs` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.  Same invalidation semantics; only the
        // arglist construction differs (boxes2 = i_args ++ r_args).
        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        // pyjitpl.py:1950 _opimpl_residual_call*: result writeback runs
        // BEFORE handle_possible_exception().  See
        // `dispatch_residual_call_iRd_kind` for the full citation.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
                // pyjitpl.py:2156-2168 `handle_possible_exception`
                // routes the raising branch through
                // `finishframe_exception()` immediately after emitting
                // GUARD_EXCEPTION — see iRd_kind for the full rationale.
                let exc = ctx
                    .last_exc_value
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete;
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                // The mixed int/ref residual-call shape must use the same
                // exception-region resume as the one-ref shape above.  In
                // particular, UNPACK_SEQUENCE and UNPACK_EX pass their arity
                // as Int arguments and the sequence as a Ref argument.  If
                // their validation raises after a guard failure, resume at the
                // call's own catch so the enclosing handler sees the error;
                // the generic fallthrough may already be outside the covered
                // exception-table range.
                walker_capture_snapshot_for_last_guard_scoped(
                    ctx,
                    op.pc,
                    GuardCaptureScope {
                        residual_call_catch_resume: true,
                        ..GuardCaptureScope::default()
                    },
                )?;
            }
        }

        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `residual_call` shape `iIRFd>X` dispatcher — `_irf_*` arglist with
/// int + ref + float lists before the descr. RPython parity:
/// `pyjitpl.py:1342-1346 _opimpl_residual_call3` (`@arguments` argspec
/// `"box", "boxes3", "descr", "orgpc"`) → same
/// `do_residual_or_indirect_call` body as `_call1` / `_call2`. The
/// `boxes3` argcode (`pyjitpl.py:3760-3776`) decodes three adjacent
/// count-prefixed lists into one concatenated `argboxes` array
/// `[i_args..., r_args..., f_args...]`. `_build_allboxes`
/// (`pyjitpl.py:1960-1993`, ported to [`build_allboxes`]) permutes
/// those to match `descr.get_arg_types()` ABI ordering.
///
/// Operand layout `iIRFd>X`:
///   1B funcptr (i) + 1B i-list count + N×1B i-regs + 1B r-list count
///   + M×1B r-regs + 1B f-list count + K×1B f-regs + 2B descr + 1B
///   `>X` dst.
///
/// EffectInfo classification + guard emission match
/// `dispatch_residual_call_iIRd_kind`; all sub-cases (release-gil,
/// loop-invariant, default) route through the same helpers
/// ([`select_residual_call_opcode`], [`direct_call_release_gil`],
/// [`loopinvariant_lookup`] / [`loopinvariant_now_known`]).
#[allow(non_snake_case)]
pub(crate) fn dispatch_residual_call_iIRFd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py:1940-1941) clear_exception at every
    // residual-call entry; see dispatch_residual_call_iRd_kind.
    ctx.last_exc_value = None;
    ctx.last_exc_value_concrete = ConcreteValue::Null;
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (i_args, i_width) = read_int_var_list(code, op, 1, ctx)?;
    let (r_args, r_width) = read_ref_var_list(code, op, 1 + i_width, ctx)?;
    let (f_args, f_width) = read_float_var_list(code, op, 1 + i_width + r_width, ctx)?;
    let descr_offset = 1 + i_width + r_width + f_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    // Void shape `_irf_v/iIRFd` (`pyjitpl.py:1355 opimpl_residual_call_irf_v =
    // _opimpl_residual_call3`) has no `>X` dst byte; see
    // `dispatch_residual_call_iRd_kind` for the void operand-layout note.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // Flat argboxes = i_args ++ r_args ++ f_args (`boxes3` argcode order).
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len() + f_args.len());
    let mut argbox_types: Vec<Type> =
        Vec::with_capacity(i_args.len() + r_args.len() + f_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    argboxes.extend_from_slice(&f_args);
    argbox_types.extend(std::iter::repeat(Type::Float).take(f_args.len()));
    let allboxes = build_allboxes(funcptr, &argboxes, &argbox_types, call_descr.arg_types());
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    let ei = call_descr.get_extra_info();
    clear_walk_exception(ctx);
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py:2011-2014 OS_JIT_FORCE_VIRTUAL fail-loud — see
    // `dispatch_residual_call_iRd_kind` for the rationale.
    do_jit_force_virtual_guard(ei, op.pc)?;

    if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRFd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iIRFd_kind");
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py:2017 `vable_and_vrefs_before_residual_call` —
        // records FORCE_TOKEN + SETFIELD_GC IR; runtime heap mutations
        // and the after-call helpers stay on the trait-driven leg.
        // See `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx);
        }

        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py:1346-1400 `_record_helper_pure` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream walk.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);
        // `boxes3`-shaped may-force residual (`CallMayForce{R,I,F,N}`):
        // execute concretely under the authoritative walk and stamp the
        // result, identically to the `iRd` / `iIRd` siblings.
        // `do_residual_call` (`pyjitpl.py:1342-1346 _opimpl_residual_call3`)
        // is arglist-shape-independent, so the float-arg shape needs the same
        // execution path — e.g. a float-returning helper such as `math.sqrt`
        // records here as `iIRFd>f` (empty i/r lists), and its result must be
        // made concrete so the downstream float math can specialize.  Void
        // float-stores (`irf_v`) are caught inside the helper's
        // `result_type() == Void` arm and deferred (#61), so the compiled
        // loop's re-run does not double-apply the store.

        // Non-elidable concrete-execute parity (Task #390 sub-slice 3)
        // — see `dispatch_residual_call_iRd_kind` for the full citation.
        // Task #390 sub-slice 4: PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  Pure/elidable calls never reach
        // this dispatcher (they fold via the pure-call executor).
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                fbw_abort_nested_unjournaled_residual(ctx, op.pc)?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iIRFd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
                // pyjitpl.py:2156-2168 `handle_possible_exception`
                // routes the raising branch through
                // `finishframe_exception()` immediately after emitting
                // GUARD_EXCEPTION — see iRd_kind for the full rationale.
                let exc = ctx
                    .last_exc_value
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete;
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            }
        }

        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}
