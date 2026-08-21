//! Bridge / sub-walk driving.
//!
//! **Parity:** trace-side counterpart of `MetaInterp`'s bridge / sub-trace
//! handling (`pyjitpl.py` + `compile.py`).
//!
//! `dispatch_via_miframe` entry plus the bridge-root / recipe-parent /
//! middle-frame / outer-frame continuation drivers that recurse the
//! walker into inlined and bridged frames.

use super::*;

/// Record the catching frame's own `PyTraceback` node at a bridge handler
/// entry.
///
/// `pyopcode.py handle_operation_error` runs
/// `pytraceback.record_application_traceback` BEFORE `lookup_exceptiontable`
/// routes to the handler, so the frame that catches contributes a node whether
/// or not it is the frame that raised.  Upstream meta-traces that interpreter
/// loop, so the attach lands in the bridge on its own; pyre synthesizes the
/// loop and has to emit it at each catch entry.
///
/// The five in-trace handler-entry paths already did this; these two bridge
/// arms did not, and the frame they lost is always the OUTERMOST one: on the
/// bridge leg the raising callee ran as real frames (a residual call), so those
/// frames attach their own nodes through the interpreted raise machinery, while
/// the catching frame is the compiled one whose node exists only if the trace
/// records it.
///
/// A delivery that reaches the parent trace's own handler-entry record as well
/// would get two adjacent nodes for this frame; `record_caught_blackhole_
/// traceback` drops the later one.
fn record_bridge_handler_entry_traceback<Sym: WalkSym>(
    wc: &mut WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: ConcreteValue,
    position: usize,
) -> Result<(), DispatchError> {
    // The handler is part of the trace, so once this bridge runs compiled it
    // catches the exception itself and the frame never surfaces an error the
    // interpreter's `handle_exception` could record a node from — hence the
    // runtime emit when the IR-virtual prepend declines.
    //
    // The concrete leg of each record mutates the live exception; both
    // recorders journal their own attach, so a walk that is later discarded
    // does not leave the node behind for the metainterp's own delivery to
    // record on top of.
    let emit_runtime = !record_prepend_application_traceback(wc, exc, exc_concrete, position)?;
    record_inline_application_traceback(wc, exc, exc_concrete, position, true, emit_runtime);
    record_top_level_application_traceback(wc, exc, exc_concrete, position, true, emit_runtime);
    Ok(())
}

/// `executioncontext.py leave` for a frame the bridge resumed into
/// rather than entered.
///
/// A carrier frame's `enter` — and the `virtual_ref` scope it opened — belongs
/// to the parent trace; `rebuild_state_after_failure` restores the still-open
/// pairs (`pyjitpl.py self.virtualref_boxes = virtualref_boxes`).
/// Upstream's resume continues inside that frame's `execute_frame`, so its
/// `finally: ec.leave(...)` still runs and closes the scope.  Pyre's carrier
/// sub-walk enters the callee body directly, with nothing standing in for that
/// `finally`, so the close is emitted here as each carrier frame returns.
///
/// Without it the bridge finishes with `ec.topframeref` still naming the
/// resumed frame's vref — one that, in compiled code, carries a live
/// FORCE_TOKEN and a null `forced` (`virtualize.py optimize_VIRTUAL_REF`).  Any
/// later `gettopframe` in the same trace then forces a vref whose scope no
/// guard still encodes, and the force yields null.
///
/// The frame box comes from the restored pair rather than from anything this
/// trace built: it is the box the parent's `enter` used, which is what
/// `opimpl_virtual_ref_finish`'s identity assert compares against.
pub(crate) fn carrier_ec_leave<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    root_sym: &Sym,
    got_exception: bool,
) {
    let Some((callee_frame, concrete_frame)) = ctx.innermost_virtualref_virtual() else {
        return;
    };
    let concrete_ec =
        root_sym.concrete_execution_context() as *mut pyre_interpreter::PyExecutionContext;
    if concrete_frame == 0 || concrete_ec.is_null() {
        return;
    }
    // `frame.execution_context`, the same read `walker_ec_enter`'s counterpart
    // performs at the inlined-call push.
    let callee_ec = ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[root_sym.frame()],
        crate::descr::pyframe_execution_context_descr(),
    );
    // Every `GetfieldGcR` the enter/leave pair records carries its concrete
    // value (`history.py *FrontendOp(pos, value)`); without it
    // `concrete_of_opref` reports this result symbolic on the residual-call
    // and snapshot paths.  The value is the same EC the leave below acts on.
    ctx.set_opref_concrete(
        callee_ec,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_ec as usize)),
    );
    super::inline_call::walker_ec_leave(
        ctx,
        callee_frame,
        callee_ec,
        concrete_frame as *mut pyre_interpreter::PyFrame,
        concrete_ec,
        got_exception,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_via_miframe<Sym: WalkSym>(
    trace_ctx: &mut TraceCtx,
    sym: &mut Sym,
    concrete_frame_addr: usize,
    orgpc: usize,
    session: &std::cell::RefCell<WalkSession>,
    jitcode_code: &[u8],
    position: usize,
    descr_refs: &[DescrRef],
    raw_descrs: RawDescrPool,
    is_authoritative_executor: bool,
    sub_jitcode_lookup: &SubJitCodeLookup,
    is_top_level: bool,
    // PyPy `pyjitpl.py MIFrame.__init__` analog: the
    // top-level jitcode's per-bank register count.  `dispatch_via_miframe`
    // allocates fresh `Vec<OpRef>`s sized to `top_num_regs_* +
    // top_constants_*.len()` — replacing the prior TODO that
    // reused `sym.registers_r` (a Python locals/stack mirror) as the
    // MIFrame register file.  The codewriter-compiled arm jitcode
    // expects `R[0]_r = handler = MIFrame self ptr`, which the
    // `argboxes_*` parameters supply via the `setup_call` analog
    // below.
    top_num_regs_r: usize,
    top_num_regs_i: usize,
    top_num_regs_f: usize,
    // Top-level jitcode's per-bank constant pool — seeded into
    // register slots `[num_regs_*, num_regs_* + constants_*.len())`
    // per `pyjitpl.py copy_constants`.
    top_constants_r: &[i64],
    top_constants_i: &[i64],
    top_constants_f: &[i64],
    // PyPy `pyjitpl.py setup_call(argboxes)` analog.
    // `argboxes_*[i]` is written to `registers_*[i]` before walking.
    // Production callers supply `argboxes_r = [const_ref(miframe_ptr)]`
    // so the codewriter-compiled arm finds the MIFrame self ptr at
    // `R[0]_r`.
    argboxes_r: &[OpRef],
    argboxes_i: &[OpRef],
    argboxes_f: &[OpRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sym_ptr = sym as *mut Sym;
    let entry_py_pc = EntryPyPc::Py(orgpc as u32);
    if is_top_level {
        let mut walk_session = session.borrow_mut();
        walk_session.recording_frame_ptr = sym.live_vable_frame_addr();
        walk_session.recording_jitcode_index = if sym.jitcode().is_null() {
            -1
        } else {
            unsafe { (*sym.jitcode()).index as i32 }
        };
        walk_session.recording_opcode_position = position;
    }

    // Phase 7: this IS the full-body walk over the outer `sym.jitcode`,
    // so guard snapshots can resolve a per-guard resume coordinate from
    // `op_pc`.  Set `fbw_mode.snapshot_sym` for the walk's lifetime;
    // `walker_capture_snapshot_for_last_guard` and
    // `fbw_foriter_body_from_op_pc` read it.  This is the PRODUCTION
    // default tracer: `trace.rs` enters `full_body_walk_trace` for every
    // traced key (unless a prior walk structurally declined it), so
    // `fbw_mode.snapshot_sym` is non-null on every default-JIT run.
    // Recover the portal EC red off `sym.frame` before the first opcode is
    // dispatched (thus before any guard is recorded), caching it into
    // `sym.execution_context`.  A bridge-from-guard sym whose ec color collides
    // with a real frame slot is left `OpRef::NONE` by `setup_bridge_sym`, which
    // defers recovery to `ensure_execution_context`.  The walker's
    // snapshot-capture path runs `collect_outer_active_boxes` AFTER the guard,
    // so recovering there would record the getfield after the guard that
    // references it (use-before-def).  Seed here — the trait's pre-guard
    // cache-once analog — so every guard snapshot reads a real EC OpRef.
    seed_execution_context_for_walk(sym, trace_ctx);
    seed_standing_exception_for_walk(sym, trace_ctx);

    // RPython parity: `metainterp.last_exc_value` (pyjitpl.py)
    // is the standing exception OpRef. Walker's `WalkContext::last_exc_value`
    // mirrors this as `Option<OpRef>` — `None` means "no active
    // exception", matching RPython's `assert self.metainterp.last_exc_value`
    // (pyjitpl.py).
    let initial_last_exc_value = if sym.last_exc_box().is_none() {
        None
    } else {
        Some(sym.last_exc_box())
    };

    // PyPy `pyjitpl.py MIFrame.__init__` analog: allocate
    // fresh per-bank register vectors sized to `top_num_regs_* +
    // top_constants_*.len()`.  This replaces the prior TODO
    // that reused `sym.registers_r` (a Python locals/stack mirror,
    // whose `[0]` slot is Python local 0) as the MIFrame register
    // file.  The codewriter-compiled arm jitcode emits getfield
    // chains rooted at `R[0] = handler = MIFrame self ptr`; the
    // `argboxes_r` parameter supplies that handler ptr below via the
    // `setup_call` analog.
    let total_r = top_num_regs_r + top_constants_r.len();
    let total_i = top_num_regs_i + top_constants_i.len();
    let total_f = top_num_regs_f + top_constants_f.len();
    let mut top_regs_r = vec![OpRef::NONE; total_r];
    let mut top_regs_i = vec![OpRef::NONE; total_i];
    let mut top_regs_f = vec![OpRef::NONE; total_f];
    let mut top_concrete_r = vec![ConcreteValue::Null; total_r];
    let mut top_concrete_i = vec![ConcreteValue::Null; total_i];

    // PyPy `pyjitpl.py copy_constants` analog: seed each
    // constant into the upper slot range `[num_regs_*, total_*)`.
    // `box_value` resolves these via `TraceCtx::constants` so
    // downstream getfield chains see the constant's `Value::*`.
    for (i, &v) in top_constants_i.iter().enumerate() {
        top_regs_i[top_num_regs_i + i] = trace_ctx.const_int(v);
        top_concrete_i[top_num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, &v) in top_constants_r.iter().enumerate() {
        top_regs_r[top_num_regs_r + i] = trace_ctx.const_ref(v);
        if v != 0 {
            top_concrete_r[top_num_regs_r + i] = ConcreteValue::Ref(v as pyre_object::PyObjectRef);
        }
    }
    for (i, &v) in top_constants_f.iter().enumerate() {
        top_regs_f[top_num_regs_f + i] = trace_ctx.const_float(v);
    }

    // PyPy `pyjitpl.py setup_call(argboxes)` analog: write
    // each argbox into the leading register slot.  The concrete
    // shadow is derived from `box_value(box)` — for `ConstRef(ptr)`
    // (the common case: argbox=miframe self ptr), this is
    // `Some(Value::Ref(GcRef(ptr)))` resolved via the constant pool;
    // for non-const argboxes it consults the `opref_concrete` stamp
    // table.
    //
    // CodeRabbit Major (PR #89): reject oversized argbox lists up
    // front instead of silently truncating with a per-loop `break`.
    // The `_*_arity_mismatch` DispatchError shapes already exist for
    // the inline-call paths (`InlineCall*ArityMismatch`); reuse them
    // here so a caller/shape mismatch surfaces as a typed failure
    // rather than a partially seeded frame.
    if argboxes_r.len() > top_num_regs_r {
        return Err(DispatchError::InlineCallArityMismatch {
            pc: position,
            provided: argboxes_r.len(),
            callee_num_regs_r: top_num_regs_r,
        });
    }
    if argboxes_i.len() > top_num_regs_i {
        return Err(DispatchError::InlineCallIntArityMismatch {
            pc: position,
            provided: argboxes_i.len(),
            callee_num_regs_i: top_num_regs_i,
        });
    }
    if argboxes_f.len() > top_num_regs_f {
        return Err(DispatchError::InlineCallFloatArityMismatch {
            pc: position,
            provided: argboxes_f.len(),
            callee_num_regs_f: top_num_regs_f,
        });
    }
    for (i, &box_ref) in argboxes_r.iter().enumerate() {
        top_regs_r[i] = box_ref;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = trace_ctx.box_value(box_ref) {
            top_concrete_r[i] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }
    for (i, &box_ref) in argboxes_i.iter().enumerate() {
        top_regs_i[i] = box_ref;
        if let Some(majit_ir::Value::Int(v)) = trace_ctx.box_value(box_ref) {
            top_concrete_i[i] = ConcreteValue::Int(v);
        }
    }
    for (i, &box_ref) in argboxes_f.iter().enumerate() {
        top_regs_f[i] = box_ref;
    }

    // Seed last_exc_value_concrete from
    // sym.last_exc_value (the live PyObjectRef supplied by the adapter caller). Null when
    // no active exception, matching `initial_last_exc_value == None`.
    let initial_last_exc_value_concrete = if sym.last_exc_value().is_null() {
        ConcreteValue::Null
    } else {
        ConcreteValue::Ref(sym.last_exc_value())
    };

    // Exception-edge bridge routing: an exception-guard
    // bridge with a standing exception resumes at the no-exception fallthrough
    // `-live-`, NOT the `except` handler.  Mirror the blackhole
    // `handle_exception_in_frame` backward case: route the walk entry to the
    // in-frame catch target so the handler body (`except KeyError: return -1`)
    // is recorded as the bridge instead of the NULL raised-call fallthrough.
    // `call_jit.rs trace_and_compile_from_bridge` only lets a bridge walk begin
    // with this precondition holding when it has already ROUTED the exc-edge
    // (published the standing exception into `BH_LAST_EXC_VALUE` and declined to
    // hand the guard to the blackhole).  So whenever the precondition holds the
    // walk MUST resume at an `except` handler — falling through to the
    // no-exception continuation would record the return of the NULL raised-call
    // result (`Finish(NULL)` → "call failed").
    let exc_edge_precondition = trace_ctx.is_bridge_trace
        && trace_ctx.bridge_source_is_exception_guard()
        && !sym.last_exc_box().is_none()
        && !sym.last_exc_value().is_null();
    // `finishframe_exception` (`pyjitpl.py`) reads the frame's
    // `catch_exception` and jumps to the handler with no further condition on
    // where the handler leads.  A handler that RETURNS out of the frame instead
    // of rejoining a loop is the ordinary `finishframe` case
    // (`pyjitpl.py:2503-2525`): the frame is popped and the result either lands
    // in the caller's last op or, with the framestack empty, becomes
    // `DoneWithThisFrame`.  Route on the catch alone.
    let exc_edge_catch_target = if exc_edge_precondition {
        find_catch_for_exc_resume(jitcode_code, position)
    } else {
        None
    };
    if exc_edge_precondition && exc_edge_catch_target.is_none() {
        // Routed by `call_jit` with no `catch_exception` in this frame at all,
        // which the routing precondition above is supposed to exclude.  Abort
        // BEFORE any recording so the guard failure resumes via the blackhole,
        // exactly as an unrouted one does.
        return Err(DispatchError::ExcEdgeNoInFrameCatch { pc: position });
    }
    let exc_edge_concrete = sym.last_exc_value();
    // typeptr at offset 0 (`_store_exception` invariant): the expected class the
    // bridge-entry GUARD_EXCEPTION checks the restored pending exception against.
    let exc_edge_class = if exc_edge_catch_target.is_some() && !exc_edge_concrete.is_null() {
        // One machine word, so read it at pointer width: an i64 read on a
        // 32-bit target pulls the adjacent header word into the high half, and
        // the guard then compares against a class value the pending-exception
        // cell (`jit_exc_raise`, pointer-width) can never hold.  That read is
        // why exception-edge routing was once wasm-off: the guard it emitted
        // could not pass, so every raising iteration deopted one chain link
        // deeper and `guard_failures` tracked the iteration count instead of
        // converging.  A backend that skips SAVE_EXCEPTION / SAVE_EXC_CLASS /
        // RESTORE_EXCEPTION rather than lowering them fails the same way, one
        // step later: the handler then reads a null caught exception.
        unsafe { *(exc_edge_concrete as *const usize) as i64 }
    } else {
        0
    };

    let result = {
        let mut wc = WalkContext {
            callee_shadow: None,
            inline_callee_consts: None,
            fbw_mode: FbwWalkMode {
                snapshot_sym: sym_ptr,
                current_exception_seed: (trace_ctx.is_bridge_trace
                    && !sym.last_exc_box().is_none())
                .then_some(sym.last_exc_box()),
                current_exception_seed_concrete: if trace_ctx.is_bridge_trace {
                    sym.last_exc_value()
                } else {
                    pyre_object::PY_NULL
                },
                class_of_last_exc_is_const: sym.class_of_last_exc_is_const(),
                // A guard-failure bridge resumes at the opcode boundary, so
                // its first `jit_merge_point` crossing at this python-pc is
                // the same op it is resuming INTO, not a loop crossing. The
                // merge-point arm skips exactly that first crossing. Seeded
                // only for bridge walks; a loop compile leaves it `None`.
                bridge_entry_merge_pc: match (trace_ctx.is_bridge_trace, entry_py_pc) {
                    (true, EntryPyPc::Py(pc)) => Some(pc as usize),
                    _ => None,
                },
                ..Default::default()
            },
            session,
            registers_r: &mut top_regs_r,
            registers_i: &mut top_regs_i,
            registers_f: &mut top_regs_f,
            concrete_registers_r: &mut top_concrete_r,
            concrete_registers_i: &mut top_concrete_i,
            descr_refs: &descr_refs,
            raw_descrs,
            is_authoritative_executor,
            trace_ctx,
            is_top_level,
            sub_jitcode_lookup,
            last_exc_value: initial_last_exc_value,
            last_exc_value_concrete: initial_last_exc_value_concrete,
            entry_py_pc,
            outer_resume_marker_jit_pc: None,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
            // This entry (test/fixture) hard-codes
            // `outer_jitcode_index = 0` and an empty `outer_active_boxes`
            // rather than seeding them from `sym.jitcode` /
            // `collect_outer_active_boxes` like the retired per-opcode
            // arm entry did.  A guard captured via
            // `walker_capture_snapshot_for_last_guard` would attach
            // resume data pointing at the wrong frame.
            pending_guard_snapshot_error: None,
            vstack_boxes: Vec::new(),
            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_ceiling: u32::MAX,
            vstack_reorder_saved: None,
            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
        };
        // #73: seed the walk-level operand-stack box mirror
        // at entry.  The mirror is only enabled when the outer sym owns the
        // virtualizable shadow (the production full-body loop trace) — the
        // synthetic/test entries leave it disabled (`vstack_valid = false`).
        // Seed at the FIRST-walked jitcode pc (`position`), not `entry_py_pc`,
        // so the first `step_vstack_mirror` is a no-op (no spurious
        // entry-boundary reconcile of the not-yet-executed first opcode).
        // Pure side-data: the snapshot read stays LEGACY until a later slice
        // makes it authoritative.
        // Exception-edge bridge: enter at the in-frame `except` handler target
        // instead of the no-exception fallthrough.  `vstack_enter_exception_
        // handler` re-seeds the operand-stack mirror at handler-entry depth and
        // places the caught exception box on the new TOS (the same reconstruction
        // the mid-walk SubRaise catch routing uses at handler entry); `wc.
        // last_exc_value` is already the standing exception box, so the handler's
        // `last_exc_value/>r` reads it.  On the normal path, seed the mirror at
        // the first-walked pc as before.
        // Carrier-boundary raise seed (`finishframe_exception`): a depth-2
        // inlined callee's sub-walk raised and the root frame's handler covers
        // the CALL (set by `drive_bridge_carrier_walk`).  Consumed once here.
        let carrier_raise_seed = crate::jitcode_dispatch::take_carrier_raise_seed();
        // Set by the carrier seed's no-handler arm: the root frame lets the
        // exception through, so the trace ends at bridge entry and the walk is
        // skipped entirely.
        let mut carrier_raise_escapes = false;
        let walk_position = if let Some(catch_target) = exc_edge_catch_target {
            // RPython `pyjitpl.py:3125-3173` exception-guard resumption, emitted
            // at the bridge-entry frame state so the GUARD_EXCEPTION captures a
            // fresh resume snapshot (the call-site prologue cannot — no frame is
            // reconstructed there):
            //   SAVE_EXC_CLASS / SAVE_EXCEPTION (no-arg record0 — read the
            //   pending exc cell at runtime) → RESTORE_EXCEPTION → GUARD_EXCEPTION
            //   (`handle_possible_exception`; deopts on a class mismatch, and
            //   consumes/clears the pending exception cell on match).
            //
            // RPython splits this across two calls because resume-data replay
            // runs between them: `_prepare_exception_resumption` records the
            // two SAVEs at the trace start (`pyjitpl.py:3148` asserts the trace
            // is still empty), and `prepare_resume_from_failure` records
            // RESTORE_EXCEPTION only after the resume operations. That gap is
            // the whole point of the SAVE/RESTORE pair — a bare
            // GUARD_NO_EXCEPTION at the bridge start is removable by the
            // optimizer (`pyjitpl.py:3132-3138`). pyre emits nothing between
            // them today, so `remove_bridge_exception`
            // (`majit-gc/src/rewrite.rs`, rewrite.py) strips the
            // consecutive triple and leaves the guard. Once pyre replays
            // resume data here, this must split into the same two phases
            // rather than stay one block.
            let class_op = wc.trace_ctx.save_exc_class();
            let value_op = wc.trace_ctx.save_exception();
            wc.trace_ctx.restore_exception(class_op, value_op);
            // `RefFrontendOp(pos, gcref)` parity (`history.py`): SAVE_EXCEPTION
            // returns `exc_value_box`, whose `getref()` is the concrete restored
            // exception pointer at trace-recording time (`pyjitpl.py:3163
            // execute_ll_raised`).  Stamp that concrete onto `value_op` so the box
            // stays symbolic (emits at runtime, class protected by the
            // GUARD_EXCEPTION below) yet carries a trace-time value: the handler's
            // `CHECK_EXC_MATCH` residual (`ll_issubclass(exc, KeyError)`) is then
            // concrete-executable and folds, instead of declining to a symbolic
            // result whose downstream `POP_JUMP_IF_FALSE` has no branch direction
            // (the residual-call executor keys concreteness on `box_value`, which
            // reads the frontend value slot — not the register `reg_shadow`).
            wc.trace_ctx.set_opref_concrete(
                value_op,
                majit_ir::Value::Ref(majit_ir::GcRef(exc_edge_concrete as usize)),
            );
            let exc_class_const = wc.trace_ctx.const_int(exc_edge_class);
            wc.trace_ctx
                .record_guard(OpCode::GuardException, &[exc_class_const], 0);
            // `handle_possible_exception` captures resume data at the MIFrame's
            // CURRENT pc — already past the residual call (`pyjitpl.py:2610
            // capture_resumedata`, default `resumepc`).  `position` here IS
            // that post-call resume coordinate (decoded from the failing
            // guard), so capture WITHOUT the after-residual advance and carry
            // `position` verbatim (`GuardCaptureScope::carried_resume_jit_pc`):
            // the twin lookups compensate CALL-START keys and would advance an
            // already-advanced coordinate a second time — on this shape onto
            // the physically-following `except` handler block, so the entry
            // guard's own bridge would resume INSIDE the handler (every
            // no-raise iteration then runs the handler body).
            walker_capture_snapshot_for_last_guard_impl(
                &mut wc,
                position,
                false,
                GuardCaptureScope {
                    carried_resume_jit_pc: Some(position),
                    ..Default::default()
                },
            )?;
            // `execute_ll_raised` parity: the standing exception the handler
            // reads (`last_exc_value/>r`) is the SAVE_EXCEPTION box — the
            // runtime-restored value, NOT a baked constant — so a value-using
            // handler (`except E as e`) sees the actual exception.
            wc.last_exc_value = Some(value_op);
            wc.last_exc_value_concrete = ConcreteValue::Ref(exc_edge_concrete);
            wc.fbw_mode.class_of_last_exc_is_const = true;
            // The inlined callees this route unwound clear out of, innermost
            // first, BEFORE the catching frame's own node: both recorders
            // prepend, so emission order is the chain read outermost-first.
            record_exc_edge_discarded_tracebacks(
                &mut wc,
                value_op,
                ConcreteValue::Ref(exc_edge_concrete),
            );
            record_bridge_handler_entry_traceback(
                &mut wc,
                value_op,
                ConcreteValue::Ref(exc_edge_concrete),
                position,
            )?;
            // Reconstruct the handler-entry operand stack + push the exc box on
            // the new TOS (mirrors the mid-walk SubRaise catch routing).
            vstack_enter_exception_handler(&mut wc, catch_target, value_op);
            // The exception is now caught by this frame's handler; drain the
            // standing residual-call exception flag so a later trace attempt's
            // `seed_standing_exception_for_walk` does not re-pick this
            // already-caught exception (mirrors `blackhole.rs route_to_catch`).
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
            catch_target
        } else if let Some(seed) = carrier_raise_seed {
            // Carrier-boundary raise: the inlined callee's sub-walk already
            // RECORDED the exception (a NewWithVtable of a known const class from
            // its inline RAISE), so `seed.exc` is a live trace box carrying its
            // own class — no SAVE/RESTORE/GUARD_EXCEPTION is needed (that path
            // exists for a runtime exception restored from the pending cell).
            // Mirror the walk-level SubRaise routing (mod.rs `finishframe_
            // exception`): seed `last_exc_value` + the handler-entry operand
            // stack (exc on TOS), then enter at the handler.
            wc.last_exc_value = Some(seed.exc);
            wc.last_exc_value_concrete = seed.exc_concrete;
            // The raise is a const-class exception (the callee's inline RAISE
            // recorded a NewWithVtable of a known class), so its class is
            // constant — same as the `exc_edge_catch_target` branch and the
            // walk-level SubRaise routing (`opimpl_raise`).  Without this, the
            // writeback below stamps the stale pre-bridge value onto the sym and
            // a reraise in the handler mis-reads the standing exception.
            wc.fbw_mode.class_of_last_exc_is_const = true;
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
            if let Some(catch_target) = seed.catch_target {
                record_bridge_handler_entry_traceback(
                    &mut wc,
                    seed.exc,
                    seed.exc_concrete,
                    position,
                )?;
                vstack_enter_exception_handler(&mut wc, catch_target, seed.exc);
                catch_target
            } else {
                // No handler in the root frame: `finishframe_exception` ran out
                // of frames to scan and reaches
                // `compile_exit_frame_with_exception`.  Same shape as the
                // walk-level top-level arm — record this frame's node for the
                // recording pass only (`emit_runtime = false`: at runtime the
                // interpreter records it when the trace hands the exception
                // back), publish the raise coordinate the interpreter reads it
                // from, and stash the exception as the FINISH payload.  The
                // remaining Python frames unwind interpreted, exactly as they
                // do when the raise surfaces from a residual call.
                //
                // The store-back belongs to that same shape and is what makes
                // the unwind readable: this frame keeps running in the
                // interpreter, which reads its locals out of
                // `locals_cells_stack_w`, while the walk held them in the
                // virtualizable boxes.  `virtualizable.py write_boxes`
                // writes them on every force with no way to decline, and
                // `record_top_level_application_traceback` above only performs
                // it concretely, for the recording pass — so without the
                // emitted store-back the compiled bridge leaves the frame
                // holding whatever its entry wrote, and a `tb_frame.f_locals`
                // or `sys._getframe()` on the way out reads every
                // post-entry local as unbound.
                if !recording_raise_keeps_existing_traceback(&mut wc, position) {
                    record_top_level_application_traceback(
                        &mut wc,
                        seed.exc,
                        seed.exc_concrete,
                        position,
                        true,
                        false,
                    );
                }
                fbw_publish_exit_last_instr(&mut wc, position);
                fbw_force_virtualizable_before_return(&mut wc);
                fbw_terminate_with_raise(seed.exc, seed.exc_concrete);
                carrier_raise_escapes = true;
                position
            }
        } else {
            // `_prepare_exception_resumption` null-exception arm +
            // `prepare_resume_from_failure` (pyjitpl.py): every exception-guard
            // bridge re-checks its entry flavor.  With no pending exception at
            // walk time, `clear_exception()` + `handle_possible_exception()`
            // record GUARD_NO_EXCEPTION at the bridge start, so the OTHER
            // failure flavor — a pending exception whose class the source
            // guard's expected class does not match — deopts to the blackhole
            // at bridge entry instead of running the recorded no-exception
            // continuation on a NULL raised-call result.
            if wc.trace_ctx.is_bridge_trace && wc.trace_ctx.bridge_source_is_exception_guard() {
                // `_prepare_exception_resumption` records SAVE_EXC_CLASS +
                // SAVE_EXCEPTION for the exception-guard descr flavor whether or
                // not the deadframe carried an exception — `exc_class = 0` and a
                // null value on this arm — and `prepare_resume_from_failure`
                // records RESTORE_EXCEPTION just before the guard.  That pair is
                // what keeps the optimizer from deleting the guard: resume data
                // can put a removable op ahead of it (here
                // `seed_execution_context_for_walk`'s GetfieldGcR, a heap-CSE
                // candidate), and both `optimize_GUARD_NO_EXCEPTION` ports
                // (rewrite.rs, pure.rs) drop a GUARD_NO_EXCEPTION whose
                // predecessor was removed.  RESTORE_EXCEPTION is never folded, so
                // it holds that flag clear.  `remove_bridge_exception` strips the
                // trio again once it is consecutive and unused, so keeping the
                // guard costs nothing at runtime.  Without it the bridge is
                // entered with a pending exception at the same source guard and
                // runs the no-exception continuation on a NULL raised-call
                // result — the shape upstream issue #2132 describes.
                let class_op = wc.trace_ctx.save_exc_class();
                let value_op = wc.trace_ctx.save_exception();
                wc.trace_ctx.restore_exception(class_op, value_op);
                wc.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                // `position` is already the post-call resume coordinate —
                // capture without the after-residual advance and carry it
                // verbatim (see the routed arm above).
                walker_capture_snapshot_for_last_guard_impl(
                    &mut wc,
                    position,
                    false,
                    GuardCaptureScope {
                        carried_resume_jit_pc: Some(position),
                        ..Default::default()
                    },
                )?;
            }
            seed_vstack_mirror(&mut wc, sym, position);
            position
        };
        let outcome = if carrier_raise_escapes {
            Ok((DispatchOutcome::Terminate, walk_position))
        } else {
            walk(jitcode_code, walk_position, &mut wc)
        };
        if matches!(
            &outcome,
            Err(DispatchError::BranchGuardUnrestorableKeptStackPermanent { .. })
                | Err(DispatchError::BranchGuardKeptStackUnsupported { .. })
        ) && wc.vstack_valid
            && wc.vstack_depth <= wc.vstack_boxes.len()
        {
            // `MIFrame.registers_r` is the authoritative source upstream when
            // an abort converts the metainterp framestack to blackholes
            // (`blackhole.py:1711-1727`).  Preserve pyre's equivalent mirror
            // before `wc` drops; the epilogue checks this coordinate against
            // the decoded abort resume pc before it mutates the live frame.
            fbw_branch_abort_stack_latch(
                wc.vstack_cur_pypc as usize,
                wc.vstack_boxes[..wc.vstack_depth].to_vec(),
            );
        }
        // Read final last_exc_value before wc drops so the borrow
        // checker can release sym for the writeback below.
        let final_last_exc = wc.last_exc_value;
        let final_class_of_last_exc_is_const = wc.fbw_mode.class_of_last_exc_is_const;
        drop(wc);
        // Full `sym.last_exc_*` state writeback parity.
        //
        // RPython `pyjitpl.py opimpl_raise` sets THREE pieces
        // of metainterp state when a raise fires:
        //   self.metainterp.class_of_last_exc_is_const = True
        //   self.metainterp.last_exc_value = exc_value_box.getref(rclass.OBJECTPTR)
        //   self.metainterp.last_exc_box = exc_value_box
        //
        // Of these, the walker can produce:
        //   - `last_exc_box`: the symbolic OpRef. Mirrored from
        //     `wc.last_exc_value` (RPython's metainterp.last_exc_value
        //     and last_exc_box are different fields — concrete pointer
        //     vs Box — but the walker tracks only the symbolic one,
        //     which lines up with `sym.last_exc_box`).
        //   - `class_of_last_exc_is_const`: true after a raise/r or a
        //     SubRaise routed into a catch handler. RPython sets this
        //     in `opimpl_raise` (line 1694) AND `execute_ll_raised`
        //     (pyjitpl.py with `constant=...` parameter — set
        //     after GUARD_CLASS / GUARD_EXCEPTION). Walker's raise/r
        //     arm always sets `wc.last_exc_value = Some(exc)` so
        //     mirroring `Some` → const=true is RPython-orthodox.
        //
        // This adapter does not currently write back `sym.last_exc_value`
        // (the concrete `PyObjectRef`); it retains only the symbolic
        // `final_last_exc` here.
        if let Some(exc) = final_last_exc {
            sym.set_last_exc_box(exc);
            sym.set_class_of_last_exc_is_const(final_class_of_last_exc_is_const);
        }
        outcome
    };
    result
}

/// Build the paused root portal frame for a multi-frame bridge-carrier
/// sub-walk (#215 item 2).  The root resumes at `root_pc` once the
/// reconstructed deepest callee returns; the callee's in-callee guards must
/// snapshot this frame on the walk framestack so a guard-failure resume
/// rebuilds both Python frames.  Mirror of [`compute_inline_caller_frame`], but
/// the root register banks come straight from the bridge-seeded `root_sym`
/// rather than a live caller [`WalkContext`] (the root walk has not started —
/// this resumes mid-flight).
/// `blackhole.py `_copy_data_from_miframe`: the concrete register image a
/// paused caller resumes from when a descendant abort converts the framestack
/// into blackhole interpreters.
///
/// [`capture_inline_parent_blackhole`] builds this for a caller the walker is
/// still standing in, by reading its live concrete shadow banks. A frame
/// rebuilt from a guard's resume data has no such bank — its per-color boxes
/// are all this side holds. They carry the same intrinsic concrete upstream
/// reads (`history.py` `*FrontendOp(pos, value)`; `box.getint()` /
/// `box.getref_base()` at :1718/:1724), so resolve them through
/// `concrete_of_opref` instead.
///
/// `OpRef::NONE` is upstream's `if box is not None` skip. A live box whose
/// concrete is unknown declines the whole capture: an MIFrame that silently
/// left one live register at its default resumes on a stale value.
fn capture_reconstructed_parent_blackhole(
    ctx: &TraceCtx,
    resume_pc: usize,
    int_boxes: &[(usize, OpRef)],
    ref_boxes: &[(usize, OpRef)],
    float_boxes: &[(usize, OpRef)],
) -> Option<InlineParentBlackhole> {
    let mut int_values = Vec::with_capacity(int_boxes.len());
    for &(color, opref) in int_boxes {
        if opref == OpRef::NONE {
            continue;
        }
        let Some(majit_ir::Value::Int(value)) = ctx.concrete_of_opref(opref) else {
            return None;
        };
        int_values.push((color, value));
    }
    let mut ref_values = Vec::with_capacity(ref_boxes.len());
    for &(color, opref) in ref_boxes {
        if opref == OpRef::NONE {
            continue;
        }
        let Some(majit_ir::Value::Ref(value)) = ctx.concrete_of_opref(opref) else {
            return None;
        };
        ref_values.push((color, value.as_usize() as pyre_object::PyObjectRef));
    }
    // Floats keep their OpRefs — `build_multi_frame_miframe` resolves them
    // while the trace context is still live, the same as the walker capture.
    let float_values = float_boxes
        .iter()
        .copied()
        .filter(|&(_, opref)| opref != OpRef::NONE)
        .collect();
    Some(InlineParentBlackhole {
        resume_pc,
        int_values,
        ref_values,
        float_values,
    })
}

pub(crate) fn compute_bridge_root_parent_frame<Sym: WalkSym>(
    root_sym: &Sym,
    trace_ctx: &mut TraceCtx,
    root_pc: usize,
) -> Option<InlineParentFrame> {
    if root_sym.jitcode().is_null() {
        return None;
    }
    let jitcode_index = unsafe { (*root_sym.jitcode()).index as u32 };
    // `root_pc` (`resume_data.frames[0].pc`) is already the post-call resume
    // point — the slot the inner frame's result lands in — so its Python
    // coordinate is a direct backtranslation (no `semantic_fallthrough_pc`).
    // Null the not-yet-produced call-result slot before collecting the active
    // boxes (the reconstructed callee supplies it on `SubReturn`), mirroring
    // `compute_inline_caller_frame`.  Operate on a clone so `root_sym` stays a
    // shared borrow.
    //
    // `collect_outer_active_boxes` reads the Ref bank by abstract register
    // color (`_get_list_of_active_boxes`, pyjitpl.py), so it needs the
    // color-indexed `f.registers_r` (`consume_boxes`, resume.py), NOT the
    // slot-indexed semantic mirror `setup_bridge_sym` left in
    // `sym.registers_r`.  The mirror leaves an operand live across the resumed
    // call (e.g. `t1` in `return fib(n-1)+fib(n-2)`) at `OpRef::NONE` under its
    // color, which resolves to a NULL const and aborts the second residual
    // call.  Prefer the persisted color decode; fall back to `registers_r` for
    // non-bridge callers (`bridge_registers_r == None`).
    let mut regs_r = root_sym
        .bridge_registers_r()
        .cloned()
        .unwrap_or_else(|| root_sym.registers_r().to_vec());
    let result_color = unsafe { &(*root_sym.jitcode()).payload }
        .result_color_trivia_for_jitcode_pc(root_pc)
        .map(|c| c as usize)
        .filter(|&c| c != u16::MAX as usize);
    if let Some(result_color) = result_color {
        if result_color < regs_r.len() {
            regs_r[result_color] = trace_ctx.const_ref(pyre_object::PY_NULL as i64);
        }
    }
    let root_word = ((root_pc as i32) != majit_ir::resumedata::NO_JITCODE_PC
        && (root_pc as i32) >= 0)
        .then_some(root_pc);
    let root_liveness_word = match root_word {
        Some(w) => w as i32,
        None => majit_ir::resumedata::NO_JITCODE_PC,
    };
    let boxes = collect_outer_active_boxes(
        root_sym,
        trace_ctx,
        root_sym.registers_i(),
        &regs_r,
        root_sym.registers_f(),
        jitcode_index,
        false,
        // Key the query off the same carried root-frame word the snapshot and
        // decode side read from `frames[0].jitcode_pc`, so both resolve the
        // identical liveness window.
        root_liveness_word,
        root_liveness_word,
        OuterActiveBoxesEntryTwin::Trivia,
        "bridge_root_parent",
        None,
        &[],
        None,
    );
    // The concrete image this paused root resumes from when a descendant
    // sub-walk aborts and the drain converts the chain instead of rewinding to
    // the guard. The Ref bank skips the call-result color for the same reason
    // it was NULLed above: the callee's blackhole writes it on return.
    let blackhole = crate::state::try_frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
        jitcode_index as i32,
        root_liveness_word,
    )
    .and_then(|live| {
        let pairs = |colors: &[u32], regs: &[OpRef]| -> Vec<(usize, OpRef)> {
            colors
                .iter()
                .map(|&color| color as usize)
                .map(|color| (color, regs.get(color).copied().unwrap_or(OpRef::NONE)))
                .collect()
        };
        let ref_pairs: Vec<(usize, OpRef)> = pairs(&live.ref_, &regs_r)
            .into_iter()
            .filter(|&(color, _)| Some(color) != result_color)
            .collect();
        capture_reconstructed_parent_blackhole(
            trace_ctx,
            root_pc,
            &pairs(&live.int, root_sym.registers_i()),
            &ref_pairs,
            &pairs(&live.float, root_sym.registers_f()),
        )
    });

    Some(InlineParentFrame {
        jitcode_index,
        call_jitcode_pc: None,
        call_stack_overrides: Vec::new(),
        blackhole,
        resume_coord: ParentResumeCoord::Backxlat(root_pc),
        // Parent-frame words are never branch-tagged; negative tags belong to
        // a branch guard's own top-frame word.
        resume_marker_jit_pc: root_word,
        boxes,
    })
}

/// Issue #215 item 2: drive the reconstructed deepest
/// callee frame of a multi-frame bridge as an INLINE SUB-WALK
/// (`is_top_level = false`) rooted on the caller-visible portal `root_sym`.
///
/// The callee resumes at `entry` (its translated recipe Python pc) with
/// its registers seeded by `argboxes_r` (portal reds + in-flight operand-stack
/// temps from `setup_reconstructed_callee_frame`) and its locals carried in the
/// already-emitted frame vable.  Because the walk is a sub-walk, the callee's
/// `ref_return` surfaces `SubReturn { result }` (`pyjitpl.py finishframe`)
/// instead of the top-level `Finish` that pyre's own-portal model cannot close
/// here — the original #215 item-2 wall.
///
/// The root portal is installed as `fbw_mode.snapshot_sym` and pushed onto the
/// walk framestack for the sub-walk's lifetime, so an in-callee guard
/// snapshots both the callee frame and the paused root
/// (`walker_capture_multi_frame_inline_snapshot`).
///
/// Diagnostic today: returns the sub-walk outcome; the caller logs it
/// and aborts (trace discarded).  Threading `SubReturn` into the root operand
/// stack + walking the root forward to a terminator is not wired yet.
/// `None` signals a setup failure (terminal descrs unwired / no root frame).
pub(crate) fn call_dst_reg_for_residual_return(code: &[u8], entry: usize) -> Option<usize> {
    for op in crate::jitcode_runtime::decoded_ops(code) {
        if op.next_pc == entry {
            return (op.opname.starts_with("residual_call") && op.argcodes.ends_with(">r"))
                .then(|| code.get(entry - 1).map(|&b| b as usize))
                .flatten();
        }
    }
    None
}

pub(crate) fn recipe_parent_frame_from_recipe(
    ctx: &mut TraceCtx,
    recipe: &majit_metainterp::ReconstructRecipe,
    root_ec: *const pyre_interpreter::PyExecutionContext,
) -> Option<InlineParentFrame> {
    let pjc = crate::state::pyjitcode_for_jitcode_index(recipe.jitcode_index)?;
    if !pjc.is_populated() || pjc.code_ptr.is_null() {
        return None;
    }
    let entry =
        if crate::state::frame_pc_is_resolved_offset_at(recipe.jitcode_index, recipe.jitcode_pc) {
            recipe.jitcode_pc as usize
        } else {
            return None;
        };
    let call_jit_pc = crate::jitcode_runtime::decoded_ops(pjc.jitcode.code.as_slice())
        .find(|op| op.next_pc == entry && op.opname.starts_with("residual_call"))
        .map(|op| op.pc);
    let resume_marker_jit_pc =
        call_jit_pc.and_then(|pc| super::resume_snapshot::inline_call_return_marker(&pjc, pc));

    // Reconstruct this paused parent frame's vable + ec (the same
    // `emit_new_pyframe_inline_with_params` the deepest-callee setup uses) so
    // the paused-frame snapshot resolves the portal reds
    // [frame, ec] (`interp_jit.py`) to real boxes rather than reading the
    // slot-indexed `registers_r` at the portal-red color positions.  Only
    // `pending.sym.frame` / `pending.sym.execution_context` are consumed here;
    // the `argboxes_r` register seeding is for the forward drive, not the
    // snapshot.
    let (pending, _argboxes_r) =
        crate::state::setup_reconstructed_callee_frame(ctx, recipe, root_ec, Vec::new())?;
    let frame_box = pending.sym.frame();
    let ec_box = pending.sym.execution_context();

    let (frame_reg, ec_reg) = crate::state::portal_red_regs_at(recipe.jitcode_index);
    let (frame_reg, ec_reg) = (u32::from(frame_reg), u32::from(ec_reg));
    let sentinel = u32::from(u16::MAX);
    let result_color = pjc
        .result_color_trivia_for_jitcode_pc(recipe.jitcode_pc as usize)
        .map(|c| c as usize)
        .filter(|&c| c != u16::MAX as usize);

    let banks = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
        recipe.jitcode_index,
        recipe.jitcode_pc,
    );
    let stack_only = recipe.valuestackdepth.saturating_sub(recipe.nlocals);
    let maps =
        crate::state::bridge_semantic_maps_from_jitcode_pc(recipe.jitcode_index, recipe.jitcode_pc);
    let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
    let mut boxes = Vec::with_capacity(banks.total_len());
    // Per-color `(color, box)` pairs for the blackhole capture below, collected
    // alongside the liveness-ordered box list the snapshot consumes.
    let mut bh_int = Vec::with_capacity(banks.int.len());
    let mut bh_ref = Vec::with_capacity(banks.ref_.len());
    let mut bh_float = Vec::with_capacity(banks.float.len());
    for &color in &banks.int {
        let opref = recipe
            .registers_i
            .get(color as usize)
            .copied()
            .unwrap_or(OpRef::NONE);
        boxes.push(opref);
        bh_int.push((color as usize, opref));
    }
    // Ref bank, in liveness-color order — mirror the retired MIFrame encoder
    // `get_list_of_active_boxes` (trace_opcode.rs) box-for-box:
    //   * the not-yet-produced call-result color is NULL-seeded (`in_a_call`);
    //   * a force-alived portal-red SCRATCH color (no live semantic slot at this
    //     pc) routes to the reconstructed frame's `frame`/`ec` box, NOT the
    //     slot-indexed register file;
    //   * every other live color reads its semantic `locals_cells_stack_w` slot
    //     from the slot-indexed `registers_r` (the reconstruct decode).
    for &color in &banks.ref_ {
        let c = color as usize;
        if result_color == Some(c) {
            boxes.push(null_ref);
            continue;
        }
        let semantic_idx = crate::state::semantic_ref_slot_for_reg_color(
            recipe.nlocals,
            stack_only,
            &maps.pcdep_entries,
            c,
        );
        let is_portal_red_scratch = semantic_idx.is_none()
            && ((color == frame_reg && frame_reg != sentinel)
                || (color == ec_reg && ec_reg != sentinel));
        let opref = if is_portal_red_scratch {
            if color == frame_reg {
                frame_box
            } else {
                ec_box
            }
        } else {
            let slot = semantic_idx.or_else(|| (c < recipe.valuestackdepth).then_some(c))?;
            recipe.registers_r.get(slot).copied().unwrap_or(OpRef::NONE)
        };
        boxes.push(opref);
        bh_ref.push((c, opref));
    }
    for &color in &banks.float {
        let opref = recipe
            .registers_f
            .get(color as usize)
            .copied()
            .unwrap_or(OpRef::NONE);
        boxes.push(opref);
        bh_float.push((color as usize, opref));
    }
    if boxes.iter().any(|b| b.is_none()) {
        // Name which bank left the hole.  `ReconstructRecipe`'s bank vectors are
        // SEMANTIC-slot indexed (`trace_ctx.rs`), and every semantic slot in
        // a `locals_cells_stack_w` array is a boxed ref — so the recipe decode
        // fills the ref bank only, and the two loops above read `OpRef::NONE`
        // for every live int/float COLOR.  A callee with any live unboxed
        // register at its resume pc can therefore never be a carrier; that is a
        // coverage bound, not a shape the walk could not have handled.
        crate::jitcode_dispatch::census_record(
            if bh_int.iter().any(|&(_, b)| b.is_none())
                || bh_float.iter().any(|&(_, b)| b.is_none())
            {
                "P2Parent::UnboxedBankHole"
            } else {
                "P2Parent::RefBankHole"
            },
        );
        return None;
    }

    // Same capture as the bridge root's, off the recipe's decoded boxes. The
    // reconstructed `frame`/`ec` reds are freshly emitted here, so a portal-red
    // scratch color has no concrete yet and declines — leaving the drain's
    // pre-existing rollback rather than a half-filled blackhole frame.
    let blackhole = capture_reconstructed_parent_blackhole(
        ctx,
        recipe.jitcode_pc as usize,
        &bh_int,
        &bh_ref,
        &bh_float,
    );

    Some(InlineParentFrame {
        jitcode_index: recipe.jitcode_index as u32,
        call_jitcode_pc: call_jit_pc,
        call_stack_overrides: Vec::new(),
        blackhole,
        // The recipe's resolved word was `trivia_normalized_py_pc_for_jitcode_pc(jitcode_index,
        // jitcode_pc)` by construction, exactly the bridge-root flavor.
        resume_coord: ParentResumeCoord::Backxlat(recipe.jitcode_pc as usize),
        resume_marker_jit_pc,
        boxes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_bridge_frame_subwalk<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<WalkSession>,
    root_sym: &Sym,
    root_pc: usize,
    callee_pjc: &std::sync::Arc<crate::PyJitCode>,
    callee_code_key: usize,
    callee_w_globals: usize,
    entry: usize,
    argboxes_r: &[OpRef],
    argboxes_i: &[OpRef],
    argboxes_f: &[OpRef],
    local_oprefs: &[OpRef],
    local_concretes: &[majit_ir::Value],
    resumed_stack_oprefs: &[OpRef],
    resumed_stack_concretes: &[majit_ir::Value],
    concrete_callee_frame: usize,
    child_result: Option<OpRef>,
    paused_parent_recipes: &[majit_metainterp::ReconstructRecipe],
) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
    use majit_metainterp::jitcode::RuntimeBhDescr;

    // Terminal descrs must be wired on MetaInterpStaticData before the walk can
    // produce a compilable FINISH (mirror `dispatch_perfn_frame`).  The walk
    // itself no longer carries them: the compile consumer selects the descr
    // from `finish_arg_types` (`pyjitpl.rs done_with_this_frame_descr_from_types`).
    {
        let sd = ctx.metainterp_sd();
        if sd.done_with_this_frame_descr_void.is_none()
            || sd.done_with_this_frame_descr_int.is_none()
            || sd.done_with_this_frame_descr_ref.is_none()
            || sd.done_with_this_frame_descr_float.is_none()
            || sd.exit_frame_with_exception_descr_ref.is_none()
        {
            return None;
        }
    }

    // Per-fn descr pool + sub-jitcode lookup off the callee body's own runtime
    // pool (mirror `dispatch_perfn_frame`).  `callee_pjc` is an `Arc` that
    // outlives the walk, so extend the descr-slice borrow to `'static` for the
    // `'static`-bodied `SubJitCodeBody` lookup.
    let perfn_descrs: &'static [RuntimeBhDescr] =
        unsafe { &*(callee_pjc.jitcode.exec.descrs.as_slice() as *const [RuntimeBhDescr]) };
    let perfn_descr_refs: Vec<majit_ir::DescrRef> = perfn_descrs
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            RuntimeBhDescr::Descr(bh) => crate::descr::make_descr_from_bh(bh),
            RuntimeBhDescr::JitCode(_)
            | RuntimeBhDescr::Call(_)
            | RuntimeBhDescr::AssemblerToken(_) => crate::descr::make_jitcode_descr(i),
        })
        .collect();
    let sub_jitcode_lookup = |idx: usize| -> Option<SubJitCodeBody> {
        perfn_descrs
            .get(idx)
            .and_then(|d| d.as_jitcode())
            .map(|jc| SubJitCodeBody {
                code: jc.code.as_slice(),
                num_regs_r: jc.num_regs_r() as usize,
                num_regs_i: jc.num_regs_i() as usize,
                num_regs_f: jc.num_regs_f() as usize,
                constants_i: jc.constants_i.as_slice(),
                constants_r: jc.constants_r.as_slice(),
                constants_f: jc.constants_f.as_slice(),
            })
    };

    // Allocate the callee register banks sized to `num_regs_* + constants_*`,
    // seed the constant pool into the upper slots and `argboxes_r` into the
    // leading slots (mirror `dispatch_via_miframe`).
    let jc = &callee_pjc.jitcode;
    let num_regs_r = jc.num_regs_r() as usize;
    let num_regs_i = jc.num_regs_i() as usize;
    let num_regs_f = jc.num_regs_f() as usize;
    let total_r = num_regs_r + jc.constants_r.len();
    let total_i = num_regs_i + jc.constants_i.len();
    let total_f = num_regs_f + jc.constants_f.len();
    let mut regs_r = vec![OpRef::NONE; total_r];
    let mut regs_i = vec![OpRef::NONE; total_i];
    let mut regs_f = vec![OpRef::NONE; total_f];
    let mut concrete_r = vec![ConcreteValue::Null; total_r];
    let mut concrete_i = vec![ConcreteValue::Null; total_i];
    for (i, &v) in jc.constants_i.iter().enumerate() {
        regs_i[num_regs_i + i] = ctx.const_int(v);
        concrete_i[num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, &v) in jc.constants_r.iter().enumerate() {
        regs_r[num_regs_r + i] = ctx.const_ref(v);
        if v != 0 {
            concrete_r[num_regs_r + i] = ConcreteValue::Ref(v as pyre_object::PyObjectRef);
        }
    }
    for (i, &v) in jc.constants_f.iter().enumerate() {
        regs_f[num_regs_f + i] = ctx.const_float(v);
    }
    if argboxes_r.len() > num_regs_r {
        return Some(Err(DispatchError::InlineCallArityMismatch {
            pc: entry,
            provided: argboxes_r.len(),
            callee_num_regs_r: num_regs_r,
        }));
    }
    for (i, &box_ref) in argboxes_r.iter().enumerate() {
        regs_r[i] = box_ref;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = ctx.box_value(box_ref) {
            concrete_r[i] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }
    // `consume_boxes` refills all three banks (resume.py
    // rebuild_from_resumedata), and the recipe's int/float banks are already
    // COLOR-indexed, so they seed the leading slots directly.  A live color the
    // callee's register file cannot hold means the recipe and this body
    // disagree about the jitcode; decline rather than drop the register.
    if argboxes_i.len() > num_regs_i || argboxes_f.len() > num_regs_f {
        crate::jitcode_dispatch::census_record("P2Drain::UnboxedBankArity");
        return None;
    }
    for (i, &box_ref) in argboxes_i.iter().enumerate() {
        if box_ref.is_none() {
            continue;
        }
        regs_i[i] = box_ref;
        if let Some(majit_ir::Value::Int(v)) = ctx.box_value(box_ref) {
            concrete_i[i] = ConcreteValue::Int(v);
        }
    }
    for (i, &box_ref) in argboxes_f.iter().enumerate() {
        if box_ref.is_none() {
            continue;
        }
        regs_f[i] = box_ref;
    }
    if let Some(result) = child_result {
        let call_dst_reg = call_dst_reg_for_residual_return(jc.code.as_slice(), entry)?;
        if call_dst_reg >= regs_r.len() {
            return Some(Err(DispatchError::InlineCallArityMismatch {
                pc: entry,
                provided: call_dst_reg + 1,
                callee_num_regs_r: regs_r.len(),
            }));
        }
        regs_r[call_dst_reg] = result;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = ctx.box_value(result) {
            concrete_r[call_dst_reg] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }

    // Paused root portal frame for the multi-frame guard snapshot.
    let root_frame = compute_bridge_root_parent_frame(root_sym, ctx, root_pc)?;
    let outer_jitcode_index = root_frame.jitcode_index;
    let outer_active_boxes = root_frame.boxes.clone();

    let callee_code = jc.code.as_slice();
    let lookup_ref: &SubJitCodeLookup = &sub_jitcode_lookup;
    // `InlineCalleeConsts.w_code` is the callee frame's `pycode` red — a
    // `W_Code` object.  `callee_code_key` is the raw compiled-code pointer the
    // recipe carries, which is the JIT-side key, not that object; the live
    // wrapper is what `recover_inline_callee_globals` already reads
    // `w_globals` out of.  Passing the key through made every consumer that
    // type-checks it decline (the inlined callee then contributed no
    // `PyTraceback` node) and every consumer that does not re-read it as a
    // `W_Code` of the wrong type.
    let consts = InlineCalleeConsts {
        w_globals: callee_w_globals,
        w_code: crate::state::recover_inline_callee_code(callee_code_key as *const ()) as usize,
        jitcode_index: jc.try_index().map_or(-1, |index| index as i32),
    };
    if consts.w_code == 0 {
        return None;
    }
    let callee_w_code = consts.w_code;

    // Install the ROOT sym as the snapshot sym (NOT the callee's) so in-callee
    // guards snapshot the paused root.
    let root_sym_ptr = root_sym as *const Sym;

    let mut parent_guards = Vec::new();
    let mut parent_for_current = root_frame.clone();
    for parent_recipe in paused_parent_recipes {
        // `InlineFrame.w_code` is the portal green (`W_Code`) used by
        // `fbw_inline_recursion_count`, not the raw `CodeObject*` carried by
        // a reconstruction recipe.  Forward inlining pushes the wrapper too;
        // preserving the same identity here makes reconstructed and newly
        // entered frames one recursion chain, as in `MIFrame.greenkey`.
        let parent_w_code =
            crate::state::recover_inline_callee_code(parent_recipe.code_ptr) as usize;
        if parent_w_code == 0 {
            return None;
        }
        let guard_parent = parent_for_current.clone();
        parent_guards.push(InlineFrameGuard::enter(
            session,
            parent_w_code,
            Some(guard_parent),
        ));
        parent_for_current = recipe_parent_frame_from_recipe(
            ctx,
            parent_recipe,
            root_sym.concrete_execution_context(),
        )?;
    }

    let outcome = {
        let mut sub_wc = WalkContext {
            callee_shadow: Some(super::CalleeLocalsShadow {
                code_ptr: callee_pjc.code_ptr,
                // `resume.py:1042-1057` rebuilds one concrete frame for every
                // resumed MIFrame.  Keep that identity on this callee's own
                // walk context so residual execution can enter precisely this
                // frame on the ExecutionContext chain.  It deliberately does
                // not use `InlineConcreteFrameGuard`: that TLS also selects
                // the standard-virtualizable heap-sync target, which remains
                // the bridge root for a reconstructed carrier.
                concrete_frame: concrete_callee_frame,
                ..Default::default()
            }),
            inline_callee_consts: Some(consts),
            fbw_mode: FbwWalkMode {
                snapshot_sym: root_sym_ptr,
                inline_subwalk: true,
                carrier_resume: true,
                current_exception_seed: (!root_sym.last_exc_box().is_none())
                    .then_some(root_sym.last_exc_box()),
                current_exception_seed_concrete: root_sym.last_exc_value(),
                class_of_last_exc_is_const: root_sym.class_of_last_exc_is_const(),
                ..Default::default()
            },
            session,
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut regs_f,
            concrete_registers_r: &mut concrete_r,
            concrete_registers_i: &mut concrete_i,
            descr_refs: &perfn_descr_refs,
            raw_descrs: RawDescrPool::PerFn(perfn_descrs),
            // The carrier sub-walk IS the bridge-resume metainterp: after a
            // guard failure `handle_guard_failure` rebuilds the frame state and
            // drives it forward through the SAME `self.interpret()` the initial
            // trace uses (`pyjitpl.py _handle_guard_failure` →
            // `prepare_resume_from_failure` → `interpret`, cf.
            // `_compile_and_run_once:2899`).  There is no second-class executor
            // mode: the resume walk concrete-executes every residual call
            // (`do_residual_call` → `execute_varargs`, `pyjitpl.py`) exactly
            // like the initial trace, which is what lets a nested self-recursive
            // call fold to a live `CALL_ASSEMBLER`.  The residual it reaches was
            // never run pre-deopt (the deopt cut the trace there), so this is its
            // first and only concrete execution.
            is_authoritative_executor: true,
            pending_guard_snapshot_error: None,
            vstack_boxes: Vec::new(),
            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_ceiling: u32::MAX,
            vstack_reorder_saved: None,
            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
            trace_ctx: ctx,
            is_top_level: false,
            sub_jitcode_lookup: lookup_ref,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            // The outer Python frame is the root, paused at `root_pc`.
            entry_py_pc: EntryPyPc::Jit(root_pc),
            outer_resume_marker_jit_pc: root_frame.resume_marker_jit_pc,
            outer_jitcode_index,
            outer_active_boxes,
        };
        let _inline_frame =
            InlineFrameGuard::enter(session, callee_w_code, Some(parent_for_current));
        // No `InlineConcreteFrameGuard` here.  A forward-inline sub-walk owns
        // the callee frame it publishes, so retargeting `last_instr` /
        // `valuestackdepth` onto it is the whole point.  A bridge-resume
        // sub-walk does not: its callee frame is the resume-decoded portal
        // register, and publishing it makes `LiveLastInstrGuard` and
        // `current_inline_vable_target` retarget onto that frame while
        // `setfield_vable_via_metainterp`'s `INLINE_CONCRETE_FRAME.is_null()`
        // arm stops syncing the walk's own virtualizable.  The live frame's
        // `last_instr` / `valuestackdepth` then stay at the last resume point,
        // so the blackhole reads an operand stack that was never published.
        // Nested self-recursive calls inside the resumed callee fold straight to
        // a recursive-portal CALL_ASSEMBLER (the bridge is the deopt
        // continuation, not a fresh unroll).
        // Seed the reconstructed callee's local slot concretes into its frame-
        // owned shadow.  The resume is mid-body,
        // so the locals were stored to the frame vable before the guard fired;
        // the map is empty until seeded.  A concrete local lets a callee
        // `getarrayitem_vable(frame, slot)` read fold to its value, so a nested
        // self-recursive call's int arg is known (`arg_is_int`) and the call
        // folds to `CALL_ASSEMBLER` instead of declining.
        for (slot, &opref) in local_oprefs.iter().enumerate() {
            sub_wc
                .callee_shadow
                .as_mut()
                .unwrap()
                .set_opref(slot as i64, opref);
        }
        for (slot, &v) in local_concretes.iter().enumerate() {
            sub_wc.callee_shadow.as_mut().unwrap().set_concrete(
                callee_pjc.metadata.portal_frame_reg,
                slot as i64,
                v,
            );
        }
        // The decoded multi-frame recipe carries this callee's semantic
        // operand stack after its locals/cells prefix.  Preserve that red
        // frame state on the same per-frame shadow and seed the walk mirror
        // from it; dropping these slots collapses bridge resume onto the root
        // frame and later CALL operands reconstruct as NULL.
        let stack_base = local_oprefs.len();
        for (s, &opref) in resumed_stack_oprefs.iter().enumerate() {
            sub_wc
                .callee_shadow
                .as_mut()
                .unwrap()
                .set_opref((stack_base + s) as i64, opref);
        }
        for (s, &value) in resumed_stack_concretes.iter().enumerate() {
            sub_wc.callee_shadow.as_mut().unwrap().set_concrete(
                callee_pjc.metadata.portal_frame_reg,
                (stack_base + s) as i64,
                value,
            );
        }
        if let Some(frame) = ActiveResumeFrame::current(session, root_sym_ptr)
            && frame.0.jitcode.code.as_ptr() == callee_code.as_ptr()
            && frame.0.jitcode.code.len() == callee_code.len()
            && let Some((py_pc, _code_ptr, depth)) = frame.vstack_coordinate_for_jitcode_pc(entry)
            && depth == resumed_stack_oprefs.len()
        {
            sub_wc.vstack_boxes = resumed_stack_oprefs.to_vec();
            sub_wc.vstack_depth = depth;
            sub_wc.vstack_cur_pypc = py_pc;
            sub_wc.vstack_valid = true;
        }
        // `resume.py rebuild_from_resumedata` seeds the rebuilt MIFrame's
        // operand stack directly from the frame recipe.  `ActiveResumeFrame`
        // is only an optional richer coordinate source; a carrier recipe is
        // already the authoritative per-frame state and must not be discarded
        // when that helper object is absent.  Otherwise the first may-force
        // call invalidates the heapcache, a later CALL reads a color-reloaded
        // box with no runtime value, and an effectful sub-walk replays from the
        // guard (the framed-pickle stream is consumed twice).
        if !sub_wc.vstack_valid {
            sub_wc.vstack_boxes = resumed_stack_oprefs.to_vec();
            sub_wc.vstack_depth = resumed_stack_oprefs.len();
            sub_wc.vstack_cur_pypc =
                crate::py_coord::resume_py_pc_for_jitcode_word(consts.jitcode_index, entry as i32)
                    as u32;
            sub_wc.vstack_valid = true;
        }
        let outcome = walk(callee_code, entry, &mut sub_wc);
        // `pyjitpl.py handle_guard_failure` wraps `_handle_guard_failure`
        // in `except SwitchToBlackhole as stb:
        // self.run_blackhole_interp_to_cancel_tracing(stb)` (:2930-2931), which
        // converts the frames `interpret()` reached and runs them forward
        // (`blackhole.py:1799`); `_handle_guard_failure` itself ends
        // `assert False, "should always raise"` (:2956).  An aborted bridge is
        // never rewound to its guard upstream.  This sub-walk has already
        // concrete-executed the reconstructed callee's residual calls (the
        // `is_authoritative_executor` contract above), so the caller's rollback
        // would replay them.  Capture the frames while this `WalkContext` still
        // owns their banks — `drive_bridge_carrier_walk`'s abort tail adopts
        // the image, and a decline there leaves the pre-existing rollback.
        if let Err(ref error) = outcome {
            let _ = latch_abort_blackhole(&sub_wc, error.stop_pc(), "bridge1361");
        }
        outcome
    };
    drop(parent_guards);
    Some(outcome)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_bridge_carrier_subwalk<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<WalkSession>,
    root_sym: &Sym,
    root_pc: usize,
    callee_pjc: &std::sync::Arc<crate::PyJitCode>,
    callee_code_key: usize,
    callee_w_globals: usize,
    entry: usize,
    argboxes_r: &[OpRef],
    argboxes_i: &[OpRef],
    argboxes_f: &[OpRef],
    local_oprefs: &[OpRef],
    local_concretes: &[majit_ir::Value],
    resumed_stack_oprefs: &[OpRef],
    resumed_stack_concretes: &[majit_ir::Value],
    concrete_callee_frame: usize,
    paused_parent_recipes: &[majit_metainterp::ReconstructRecipe],
) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
    drive_bridge_frame_subwalk(
        ctx,
        session,
        root_sym,
        root_pc,
        callee_pjc,
        callee_code_key,
        callee_w_globals,
        entry,
        argboxes_r,
        argboxes_i,
        argboxes_f,
        local_oprefs,
        local_concretes,
        resumed_stack_oprefs,
        resumed_stack_concretes,
        concrete_callee_frame,
        None,
        paused_parent_recipes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_bridge_middle_frame<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<WalkSession>,
    root_sym: &Sym,
    root_pc: usize,
    middle_pjc: &std::sync::Arc<crate::PyJitCode>,
    middle_code_key: usize,
    middle_w_globals: usize,
    entry: usize,
    argboxes_r: &[OpRef],
    argboxes_i: &[OpRef],
    argboxes_f: &[OpRef],
    local_oprefs: &[OpRef],
    local_concretes: &[majit_ir::Value],
    resumed_stack_oprefs: &[OpRef],
    resumed_stack_concretes: &[majit_ir::Value],
    concrete_callee_frame: usize,
    paused_parent_recipes: &[majit_metainterp::ReconstructRecipe],
    child_result: OpRef,
) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
    drive_bridge_frame_subwalk(
        ctx,
        session,
        root_sym,
        root_pc,
        middle_pjc,
        middle_code_key,
        middle_w_globals,
        entry,
        argboxes_r,
        argboxes_i,
        argboxes_f,
        local_oprefs,
        local_concretes,
        resumed_stack_oprefs,
        resumed_stack_concretes,
        concrete_callee_frame,
        Some(child_result),
        paused_parent_recipes,
    )
}
