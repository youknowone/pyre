//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! `trace_bytecode` drives the authoritative full-body walk
//! (`full_body_walk_trace`): the walker-as-tracer that walks the per-CodeObject
//! JitCode body, combining symbolic IR recording
//! with the per-step concrete frame snapshot.  Any location the walk declines
//! re-interprets without JIT (the trait `PyreMetaInterp` interpret loop is
//! retired, gap-10 of issue #73 Phase 6).

use majit_metainterp::{MetaInterp, TraceAction, TraceCtx};
use pyre_interpreter::CodeObject;

use crate::state::{PyreMeta, PyreSym};

struct ObjectSlotRoot {
    slot: *mut *mut u8,
    registered: bool,
}

impl ObjectSlotRoot {
    fn new(value: &mut pyre_object::PyObjectRef) -> Self {
        let slot = value as *mut pyre_object::PyObjectRef as *mut *mut u8;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for ObjectSlotRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

thread_local! {
    /// pyjitpl.py:3048-3091 `raise_continue_running_normally` seam: set
    /// when the authoritative full-body walk committed its end-of-walk
    /// frame state into the trace's concrete frame snapshot
    /// (`flush_walk_end_state_to_frame`).  The portal call sites consume
    /// it via [`take_walk_end_flush_committed`] to decide whether the
    /// returned `FrameBox` carries adoptable end state for the LIVE
    /// frame (no-replay) or still holds the entry state (legacy replay).
    static WALK_END_FLUSH_COMMITTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// A no-handler exception produced by a committed rebuilt callee.  The
    /// portal consumes it as `LoopResult::Done(Err(..))`; keeping it separate
    /// from `ContinueRunningNormally` mirrors `_exit_frame_with_exception`.
    static WALK_END_PROPAGATED_EXCEPTION: std::cell::RefCell<Option<pyre_interpreter::PyError>> =
        const { std::cell::RefCell::new(None) };
    /// True at portal trace sites that can consume
    /// `WALK_END_PROPAGATED_EXCEPTION`. Bridge tracing leaves this false and
    /// conservatively retains its legacy preflight.
    static WALK_END_PROPAGATE_ALLOWED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Take-and-reset the walk-end flush flag for the trace that just
/// returned from [`trace_bytecode`].
pub fn take_walk_end_flush_committed() -> bool {
    WALK_END_FLUSH_COMMITTED.with(|c| c.replace(false))
}

pub fn take_walk_end_propagated_exception() -> Option<pyre_interpreter::PyError> {
    WALK_END_PROPAGATED_EXCEPTION.with(|c| c.borrow_mut().take())
}

/// Copy the walk-accumulated `TraceCtx.reads_module_global` flag into the
/// trace's `PyreMeta.namespace_dependent` at every `trace_bytecode` return
/// path.  This is the sole authority for the finalized value; `build_meta`
/// seeds it `false` at trace start (the walk hasn't run yet) and the
/// entry-bridge fold ORs the live flag mid-walk, so `false` there is the OR
/// identity.  On mid-walk-compile close paths the tracing ctx was already
/// taken, so `trace_ctx()` is `None` and this no-ops exactly as before — the
/// value was folded into the entry bridge's meta by then.
fn finish_trace_namespace_dependency(meta: &mut MetaInterp<PyreMeta>) {
    let namespace_dependent = meta
        .trace_ctx()
        .map(|ctx| ctx.reads_module_global)
        .unwrap_or(false);
    if let Some(trace_meta) = meta.trace_meta_mut() {
        trace_meta.namespace_dependent = namespace_dependent;
    }
}

thread_local! {
    /// Green keys whose full-body walk failed on a structural walker
    /// limitation (the recurring `DispatchError` classes listed in
    /// `full_body_walk_trace`).  A retrace of such a key skips the walk and
    /// re-interprets without JIT instead of re-walking and re-failing every
    /// hot encounter; the location stays trace-eligible (not
    /// `DONT_TRACE_HERE`), so upstream's invariant that an abort never marks a
    /// location untraceable (pyjitpl.py:2392 aborted_tracing) holds.  A walker
    /// improvement that closes one of those `DispatchError` classes shrinks
    /// this set.
    static FBW_DECLINED_KEYS: std::cell::RefCell<std::collections::HashSet<u64>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

fn fbw_declined(key: u64) -> bool {
    FBW_DECLINED_KEYS.with(|s| s.borrow().contains(&key))
}

fn fbw_decline(key: u64) {
    FBW_DECLINED_KEYS.with(|s| {
        s.borrow_mut().insert(key);
    });
}

fn midbody_post_marker_is_effect_free(code: &CodeObject, start_pc: usize) -> bool {
    (start_pc..code.instructions.len()).all(|pc| {
        let Some((instruction, _)) = pyre_interpreter::decode_instruction_at(code, pc) else {
            return false;
        };
        matches!(
            instruction,
            pyre_interpreter::Instruction::Cache
                | pyre_interpreter::Instruction::ExtendedArg
                | pyre_interpreter::Instruction::Resume { .. }
                | pyre_interpreter::Instruction::Nop
                | pyre_interpreter::Instruction::NotTaken
                | pyre_interpreter::Instruction::LoadConst { .. }
                | pyre_interpreter::Instruction::LoadCommonConstant { .. }
                | pyre_interpreter::Instruction::LoadSmallInt { .. }
                | pyre_interpreter::Instruction::LoadFast { .. }
                | pyre_interpreter::Instruction::LoadFastBorrow { .. }
                | pyre_interpreter::Instruction::LoadFastCheck { .. }
                | pyre_interpreter::Instruction::LoadFastBorrowLoadFastBorrow { .. }
                | pyre_interpreter::Instruction::LoadFastLoadFast { .. }
                | pyre_interpreter::Instruction::StoreFast { .. }
                | pyre_interpreter::Instruction::StoreFastLoadFast { .. }
                | pyre_interpreter::Instruction::StoreFastStoreFast { .. }
                | pyre_interpreter::Instruction::PopTop
                | pyre_interpreter::Instruction::Copy { .. }
                | pyre_interpreter::Instruction::Swap { .. }
                | pyre_interpreter::Instruction::BinaryOp { .. }
                | pyre_interpreter::Instruction::CompareOp { .. }
                | pyre_interpreter::Instruction::IsOp { .. }
                | pyre_interpreter::Instruction::JumpForward { .. }
                | pyre_interpreter::Instruction::JumpBackward { .. }
                | pyre_interpreter::Instruction::JumpBackwardNoInterrupt { .. }
                | pyre_interpreter::Instruction::PopJumpIfFalse { .. }
                | pyre_interpreter::Instruction::PopJumpIfTrue { .. }
                | pyre_interpreter::Instruction::PopJumpIfNone { .. }
                | pyre_interpreter::Instruction::PopJumpIfNotNone { .. }
                | pyre_interpreter::Instruction::MatchMapping
                | pyre_interpreter::Instruction::MatchSequence
                | pyre_interpreter::Instruction::GetLen
                | pyre_interpreter::Instruction::UnpackSequence { .. }
                | pyre_interpreter::Instruction::ReturnValue
        )
    })
}

fn exception_delivery_stack_is_sourceable(
    handler_depth: u32,
    array_len: usize,
    stack_base: usize,
) -> bool {
    handler_depth == 0 && array_len >= stack_base + 1
}

fn try_commit_midbody_abort(
    ctx: &TraceCtx,
    cf_addr: usize,
    payload: &crate::jitcode_dispatch::MidBodyPayload,
) -> bool {
    if !crate::state::can_flush_walk_end_state_after_outer_call(
        ctx,
        cf_addr,
        payload.call_py_pc,
        payload.post_call_py_pc,
        payload.call_stack_len,
    ) {
        return false;
    }
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(payload.w_code) as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return false;
    }
    let code = unsafe { &*raw };
    // Only portal trace sites currently carry `_exit_frame_with_exception`
    // out of the walk. Bridge sites keep the former effect-free preflight;
    // this is checked before running the rebuilt callee, so they never strand
    // an effectful Err into replay.
    if !WALK_END_PROPAGATE_ALLOWED.with(|c| c.get())
        && (!code.exceptiontable.is_empty()
            || !midbody_post_marker_is_effect_free(code, payload.callee_py_pc))
    {
        return false;
    }
    if cf_addr == 0 {
        return false;
    }
    let ec = unsafe { (*(cf_addr as *const pyre_interpreter::PyFrame)).execution_context }
        as *mut pyre_interpreter::PyExecutionContext;
    if ec.is_null() {
        return false;
    }
    let propagate_allowed = WALK_END_PROPAGATE_ALLOWED.with(|c| c.get());
    let outer = unsafe { &mut *(cf_addr as *mut pyre_interpreter::PyFrame) };
    let outer_stack_base = outer.nlocals() + outer.ncells();
    let outer_code = unsafe { &*pyre_interpreter::pyframe_get_pycode(outer) };
    let outer_handler = pyre_interpreter::pycode::lookup_exceptiontable(
        &outer_code.exceptiontable,
        (payload.call_py_pc as u32) * 2,
    );
    if propagate_allowed {
        // E-G2: this specialization reconstructs only the exact empty
        // operand-stack level used by statement-position calls. A handler
        // preserving any operand below the call remains on legacy replay.
        if let Some((_target, depth, _lasti)) = outer_handler {
            if !exception_delivery_stack_is_sourceable(
                depth,
                outer.locals_w().as_slice().len(),
                outer_stack_base,
            ) {
                return false;
            }
        }
        // G7: materialize every outer local before the rebuilt callee can run.
        // `can_flush_walk_end_state_after_outer_call` already proved all
        // shadow entries sourceable, so no post-effect decline remains.
        if !crate::state::write_back_outer_locals(ctx, cf_addr) {
            return false;
        }
    }
    let mut w_code = payload.w_code;
    let mut w_globals = payload.w_globals;
    let mut x_arg = payload.x_arg;
    let _w_code_root = ObjectSlotRoot::new(&mut w_code);
    let _w_globals_root = ObjectSlotRoot::new(&mut w_globals);
    let _x_arg_root = ObjectSlotRoot::new(&mut x_arg);
    let frame = match pyre_interpreter::PyFrame::try_new_for_call_with_closure_and_globals_obj(
        w_code as *const (),
        &[x_arg],
        std::ptr::null_mut(),
        w_globals,
        ec,
        pyre_object::PY_NULL,
        pyre_interpreter::pyframe::FrameLocalsArrayAllocation::OldGenGc,
    ) {
        Ok(frame) => frame,
        Err(_) => return false,
    };
    let mut frame = pyre_interpreter::pyframe::FrameBox::new(frame);
    frame.fix_array_ptrs();
    let _frame_locals_root = pyre_interpreter::pyframe::FrameLocalsRoot::new(frame.as_mut_ptr());

    let Some(crate::jitcode_dispatch::InlineAbortCarrier::MidBody(current)) =
        crate::jitcode_dispatch::fbw_abort_carrier_clone()
    else {
        return false;
    };
    if current.live_locals.len() != code.varnames.len() {
        return false;
    }
    for slot in &mut frame.locals_w_mut().as_mut_slice()[..code.varnames.len()] {
        *slot = pyre_object::PY_NULL;
    }
    // `_copy_data_from_miframe` restores Ref registers before any scalar
    // boxing allocation; once installed, the rooted frame array owns them.
    for (slot, value) in current.live_locals.iter().enumerate() {
        if let Some(crate::state::ConcreteValue::Ref(value)) = value {
            frame.locals_w_mut().as_mut_slice()[slot] = *value;
        }
    }
    let stack_base = code.varnames.len();
    for (rel, value) in current.live_stack.iter().enumerate() {
        let crate::state::ConcreteValue::Ref(value) = value else {
            return false;
        };
        frame.locals_w_mut().as_mut_slice()[stack_base + rel] = *value;
    }
    for (slot, value) in current.live_locals.iter().enumerate() {
        frame.locals_w_mut().as_mut_slice()[slot] = match value {
            None => pyre_object::PY_NULL,
            Some(crate::state::ConcreteValue::Ref(value)) => *value,
            Some(crate::state::ConcreteValue::Int(value)) => pyre_object::w_int_new(*value),
            Some(crate::state::ConcreteValue::Float(value)) => {
                pyre_object::floatobject::w_float_new(*value)
            }
            Some(crate::state::ConcreteValue::Null | crate::state::ConcreteValue::Bool(_)) => {
                return false;
            }
        };
    }
    frame.valuestackdepth = stack_base + current.live_stack.len();
    frame.last_instr = current.callee_py_pc as isize - 1;
    let sys_exc_value_pre = unsafe { (*ec).sys_exc_value };
    match frame.execute_frame(None, None) {
        Ok(mut retval) => {
            crate::jitcode_dispatch::fbw_abort_carrier_set_return(retval);
            let _retval_root = ObjectSlotRoot::new(&mut retval);
            crate::state::flush_walk_end_state_after_outer_call(
                ctx,
                cf_addr,
                current.call_py_pc,
                current.post_call_py_pc,
                current.call_stack_len,
                retval,
            )
        }
        Err(mut operr) => {
            // `_resume_mainloop(current_exc)` returns the exception to the
            // caller frame. Restore the caller's pre-CALL handled-exception
            // state first; PUSH_EXC_INFO/POP_EXCEPT will manage it from the
            // selected handler onward.
            unsafe { (*ec).sys_exc_value = sys_exc_value_pre };
            if !propagate_allowed {
                return false;
            }
            let outer = unsafe { &mut *(cf_addr as *mut pyre_interpreter::PyFrame) };
            outer.last_instr = current.call_py_pc as isize;
            outer.valuestackdepth = outer_stack_base;
            let mut next_instr = current.call_py_pc;
            if pyre_interpreter::eval::handle_exception(outer, &mut operr, &mut next_instr) {
                outer.last_instr = next_instr as isize - 1;
            } else {
                WALK_END_PROPAGATED_EXCEPTION.with(|c| *c.borrow_mut() = Some(operr));
            }
            true
        }
    }
}

/// True when `start_pc` is the target of a `JumpBackward` in `code` — i.e. a
/// loop header, the origin of a loop-header trace rather than a function-entry
/// trace.
fn start_pc_is_loop_header(code: &pyre_interpreter::CodeObject, start_pc: usize) -> bool {
    use pyre_interpreter::Instruction as I;
    let mut arg_state = pyre_interpreter::OpArgState::default();
    for (pc, unit) in code.instructions.iter().copied().enumerate() {
        let (instr, op_arg) = arg_state.get(unit);
        let delta = match instr {
            I::JumpBackward { delta } | I::JumpBackwardNoInterrupt { delta } => delta,
            _ => continue,
        };
        if pyre_interpreter::jump_target_backward_decoded(code, pc + 1, delta, op_arg) == start_pc {
            return true;
        }
    }
    false
}

/// Trace an entire loop body starting at `start_pc`.
///
/// Drives the authoritative full-body walk (`full_body_walk_trace`): the
/// walker walks the per-CodeObject JitCode body, recording symbolic IR against
/// the per-step concrete frame snapshot.  A location the walk declines
/// re-interprets without JIT (the trait `PyreMetaInterp` interpret loop is
/// retired, gap-10 of issue #73 Phase 6).
pub fn trace_bytecode(
    meta: &mut MetaInterp<PyreMeta>,
    sym: &mut PyreSym,
    _code: &CodeObject,
    start_pc: usize,
    mut concrete_frame: pyre_interpreter::pyframe::FrameBox,
    live_frame_addr: usize,
    allow_propagate_out: bool,
) -> (TraceAction, pyre_interpreter::pyframe::FrameBox) {
    // `llmodel.py:557` parity — install pyre's `Cpu` impl so the
    // optimizer's `protect_speculative_string` / `bh_strlen` /
    // `bh_strgetitem` family routes through `W_UnicodeObject`-shaped
    // `str_descr` / `unicode_descr` (`pyre_cpu` module).
    meta.set_cpu(crate::pyre_cpu::shared());

    // A stale flag from a prior trace on this thread must not leak into
    // this trace's adoption decision.
    WALK_END_FLUSH_COMMITTED.with(|c| c.set(false));
    WALK_END_PROPAGATED_EXCEPTION.with(|c| *c.borrow_mut() = None);
    WALK_END_PROPAGATE_ALLOWED.with(|c| c.set(allow_propagate_out));
    // `TraceCtx.reads_module_global` needs no reset here: a fresh TraceCtx is
    // built per trace (zero-init `false`), unlike the walk-end TLS flags above.
    // Likewise clear any no-replay finish payload a prior trace left
    // unconsumed.  The FBW walk re-clears this in `run_perfn_walk`; the
    // trait leg (`trace_step_result_to_action`) has no such epilogue, so
    // reset here covers both before either can stash a capture.
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    // Likewise drop any cross-frame-resume abort request a prior aborted
    // trace left unconsumed (`metainterp::interpret` clears it on the normal
    // path; this guards the paths that exit before the poll runs).
    let _ = crate::state::take_trace_abort_requested();

    let ctx = meta
        .trace_ctx()
        .expect("trace_bytecode invariant: meta.tracing must be Some during merge_point closure");
    // A multi-frame bridge carrier overrides the trace-start
    // pc with the OUTERMOST (`frames[0]`) resume pc. The passed `start_pc` is
    // the INNERMOST frame's pc (`decode_and_restore_guard_failure` returns
    // `jit_state.next_instr()`), which belongs to the deepest reconstructed
    // callee — NOT the root. A carrier resume now re-interprets without JIT (the
    // trait leg that reconstructed + pushed the callees is retired, gap-10), so
    // the walker dispatch below routes it to a plain abort; the root pc is the
    // relevant trace-start either way.
    let carrier = ctx.take_bridge_inline_carrier();
    let start_pc = if let Some(ref c) = carrier {
        c.root_pc
    } else {
        start_pc
    };
    // RPython MetaInterp._interpret() parity: the walker (sole tracer)
    // executes as it records over a concrete `PyFrame` snapshot
    // (`snapshot_for_tracing`); the interpreter does not run during tracing.
    // The snapshot copies frame-LOCAL state (abort-safety) while sharing
    // `w_globals`; vable-statics capture reads pointer-valued fields from the
    // live frame (`live_vable_frame_addr` below), not the snapshot copy.
    //
    // The former snapshot double-apply (inline-frame SHARED-heap STOREs
    // leaking during tracing and re-applying on the compiled loop's re-run)
    // is resolved by gap 10: the concrete executor is deleted so STOREs are
    // record-only, and `flush_walk_end_state_to_frame`
    // (`raise_continue_running_normally` parity) advances the real frame so
    // the interpreter resumes AFTER the walked region, not from its start.
    concrete_frame.set_last_instr_from_next_instr(start_pc);
    let w_code = concrete_frame.pycode;
    // Issue #73 walker-as-tracer foundation probe (read-only).
    // `PYRE_DUMP_PERFN_JITCODE=1` dumps the per-CodeObject JitCode body
    // — the byte stream the walker-as-tracer must learn to walk so that
    // `miframe.pc == jitcode_pc` and `pc_map` can retire.  See
    // `project_issue73_architecture_walker_as_tracer_2026_05_28`.
    if std::env::var_os("PYRE_DUMP_PERFN_JITCODE").is_some() {
        dump_perfn_jitcode_for_trace(w_code, start_pc);
    }
    let cf_addr = &*concrete_frame as *const pyre_interpreter::pyframe::PyFrame as usize;
    // The snapshot stands in for concrete stepping only; vable-statics
    // capture must read pointer-valued fields (`debugdata` / `lastblock`)
    // from the live frame the compiled loop will run on.  See the
    // `live_vable_frame_addr` field doc (state.rs).  Set before the
    // full-body-walk leg below so the walker path (the production tracer
    // post-#73) sees it as well as the trait `interpret()` leg.
    //
    // gap 10 slice 2b: set this BEFORE `init_symbolic` so the root vable
    // identity (seed_virtualizable_boxes) is baked against the live frame
    // address, not the discarded snapshot's.
    sym.live_vable_frame_addr = live_frame_addr;
    // pyjitpl.py:65 MIFrame.__init__: sym fields populated once at frame
    // construction. Callee (inline) frames are set up by perform_call
    // (trace_opcode.rs:3323-3424) and don't call init_symbolic; this path
    // handles the root frame push.
    sym.init_symbolic(ctx, cf_addr);
    // Issue #215 item 2: drive the multiframe bridge-carrier resume via the
    // full-body walker (reconstruct the in-flight callee framestack + walk
    // innermost-first) instead of aborting to a no-JIT re-interpret below.
    if let Some(ref carrier) = carrier {
        // A multi-frame bridge resume is driven by the orthodox framestack
        // trampoline (`rebuild_from_resumedata` resume.py:1042-1057 + the
        // continuous interpret loop): reconstruct the resumed callee framestack
        // and walk it forward. Without it a present carrier falls through to the
        // degenerate `Trait::CarrierAbort` below, which never compiles the bridge
        // and re-aborts on every guard failure. The `PYRE_P2_DRAIN`
        // sub-walk+inject shape is a separate unsound deviation, kept gated off.
        if crate::state::p2_drain_enabled() {
            let action = drive_bridge_carrier_walk(ctx, sym, w_code, start_pc, cf_addr, carrier);
            finish_trace_namespace_dependency(meta);
            return (action, concrete_frame);
        }
        let action = drive_bridge_framestack_walk(ctx, sym, w_code, start_pc, cf_addr, carrier);
        finish_trace_namespace_dependency(meta);
        return (action, concrete_frame);
    }
    // Issue #73 walker-as-tracer foundation probe (slice #1, gated).
    // `PYRE_WALK_PERFN_JITCODE=1` attempts to walk the per-CodeObject
    // JitCode body via `dispatch_via_miframe` from the resume entry pc,
    // logs how far the symbolic walk gets (terminator outcome vs first
    // `DispatchError` stop), then aborts the trace.  Default-off → zero
    // production change.  Produces the Path A (payload-seeding) gap
    // inventory on a live bench now that walk-capability gaps #1/#2/#3
    // are closed.  See
    // `project_issue73_architecture_walker_as_tracer_2026_05_28`.
    // Both walker entries below are gated on `carrier.is_none()`: a
    // multi-frame bridge resume carries reconstructed inline-callee
    // recipes that only the trait path assembles+pushes (the carrier
    // drain before `interpret()` below — `rebuild_from_resumedata`
    // resume.py:1049-1056).  The walker has no multi-Python-frame
    // reconstruction yet (#68); entering it would walk the outer root
    // frame at `root_pc` instead of the deepest resumed callee.
    if carrier.is_none() && std::env::var_os("PYRE_WALK_PERFN_JITCODE").is_some() {
        probe_walk_perfn_jitcode(ctx, sym, w_code, start_pc, cf_addr);
        finish_trace_namespace_dependency(meta);
        return (TraceAction::Abort, concrete_frame);
    }
    // Issue #73 Phase 5 production flip: the per-CodeObject JitCode body is
    // traced via the authoritative full-body walk — the walker-as-tracer
    // path that makes `miframe.pc == jitcode_pc` and lets `pc_map` retire.
    // `PYRE_FULL_BODY_WALK=0` opts back into the trait
    // `metainterp.interpret` loop below (transition escape hatch; the trait
    // tracer is deleted in Phase 6).
    //
    // A green key in `FBW_DECLINED_KEYS` had a prior walk fail on a
    // structural walker limitation (the recurring error classes in
    // `full_body_walk_trace`); the retrace routes through the trait
    // tracer below instead of permanently blacklisting the location
    // (`DONT_TRACE_HERE`).  Tracing aborts must not mark a location
    // untraceable — upstream aborts or switches to the blackhole and the
    // location stays eligible (pyjitpl.py:2392 aborted_tracing /
    // blackhole switch); the trait leg is pyre's transitional stand-in
    // until the walker covers those shapes (deleted with the trait in
    // Phase 6).
    if carrier.is_none()
        && std::env::var_os("PYRE_FULL_BODY_WALK").as_deref() != Some(std::ffi::OsStr::new("0"))
        && !fbw_declined(crate::driver::make_green_key(w_code, start_pc))
    {
        let action = full_body_walk_trace(ctx, sym, w_code, start_pc, cf_addr);
        finish_trace_namespace_dependency(meta);
        return (action, concrete_frame);
    }
    // gap-10: the trait tracer (`PyreMetaInterp` / `owned_concrete_frame`
    // interpret loop) is retired.  Any path the walker did not trace above — an
    // `fbw_declined` key whose walk hit a structural limit, a
    // `PYRE_FULL_BODY_WALK=0` opt-out, or a multi-frame bridge `carrier` resume
    // (reconstructed only by the deleted trait leg) — re-interprets without JIT
    // for this key.  The location stays trace-eligible (no `DONT_TRACE_HERE`);
    // the next hot encounter re-walks.
    crate::jitcode_dispatch::census_record(if carrier.is_some() {
        "Trait::CarrierAbort"
    } else {
        "Trait::DeclinedAbort"
    });
    finish_trace_namespace_dependency(meta);
    (TraceAction::Abort, concrete_frame)
}

/// Issue #73 walker-as-tracer foundation probe (slice #1).
///
/// Attempts to walk the per-CodeObject JitCode body via
/// [`crate::jitcode_dispatch::dispatch_via_miframe`] from the resume
/// entry pc (`pc_map[start_pc]`) and logs how far the symbolic walk
/// gets: a terminator outcome (`Finish` / `CloseLoop` / `SubReturn`)
/// or the first `DispatchError` stop with its pc.
///
/// Diagnostic-only: the caller aborts the trace immediately after this
/// returns, so any IR / merge-point / heap-cache mutation the walk
/// records is discarded with the aborted trace.  The recorder is also
/// rolled back via `cut_trace` to keep the discarded trace tidy.
///
/// Purpose: with walk-capability gaps #1/#2/#3 closed (decode table +
/// vable array ops + jit_merge_point/last_exception/abort handlers),
/// this surfaces the next blocker for the full-body walk — the Path A
/// payload-seeding gap (an op reading a register slot the entry never
/// seeded, e.g. a `goto_if_not` over a non-concrete Int produced by an
/// unfolded `residual_call`).  See
/// `project_issue73_architecture_walker_as_tracer_2026_05_28`.
///
/// Whether the loop whose header merge point is `header_mp_pc` (the loop a
/// header entry at `entry` is about to trace) is NESTED inside another loop
/// still active at `entry`.
///
/// A backward `goto/L` (target `< pc`) is a loop back-edge spanning
/// `[target, goto_pc]`.  The loop being traced is nested iff some back-edge
/// `[t, g]` ENCLOSES it: it starts before the entry (`t < entry`) and closes
/// AFTER the whole inner loop (`g > header_mp_pc`).  A PRECEDING sibling
/// loop's back-edge closes before `entry` (`g < header_mp_pc`) so it does not
/// match; the inner loop's OWN back-edge is excluded because its target lands
/// at its own header (the first merge point at or after `t` is `header_mp_pc`
/// itself, not an earlier loop).
fn header_entry_is_nested(code: &[u8], entry: usize, header_mp_pc: usize) -> bool {
    let first_mp_at_or_after = |t: usize| {
        crate::jitcode_runtime::decoded_ops(code)
            .filter(|op| op.opname == "jit_merge_point")
            .map(|op| op.pc)
            .filter(|&pc| pc >= t)
            .min()
    };
    crate::jitcode_runtime::decoded_ops(code)
        .filter(|op| op.opname == "goto")
        .any(|op| {
            let target = crate::jitcode_dispatch::read_label(code, &op, 0);
            target < op.pc // backward branch = loop back-edge
                && target < entry
                && op.pc > header_mp_pc
                && first_mp_at_or_after(target).is_some_and(|mp| mp < header_mp_pc)
        })
}

/// Decode the loop-header `jit_merge_point` that governs the resume
/// coordinate `entry` and return its green-ref (`gr`) and red (`rr`)
/// register lists.
///
/// These name the jitcode register colors the loop body reads its
/// loop-invariant pycode (`gr`) and frame/ec (`rr`) from.  A mid-loop walk
/// entering PAST the merge point never executes it, so those colors are
/// left `OpRef::NONE` unless explicitly seeded — the 51d.1 / B1 blocker.
///
/// Operand layout `cIRFIRF`: jdindex(`c`, 1 byte) followed by six
/// count-prefixed register lists `gi, gr, gf, ri, rr, rf`.  Returns `None`
/// when no preceding merge point exists (straight-line resume) or the
/// operand stream is truncated.
///
/// `body_resume` selects which merge point governs `entry`.  RPython binds
/// each merge point's greenboxes 1:1 from the op it is ABOUT TO execute
/// (pyjitpl.py:1537 `opimpl_jit_merge_point(greenboxes, ...)`), because the
/// MIFrame walks every op forward.  pyre resumes PAST that op at a resume
/// marker, so it must reconstruct the same op's register colors:
///  - A HEADER entry (`body_resume=false`, a fresh loop trace) sits at the
///    loop header's leading `-live-` marker, immediately BEFORE the merge
///    point the walk is about to reach — so the governing op is the FIRST
///    merge point at or after `entry`.  Picking the largest one BEFORE
///    `entry` would select a PRECEDING sibling loop's merge point (a
///    function with two loops), seeding that loop's pycode/frame/ec colors
///    and leaving this loop's distinct colors `OpRef::NONE` — the
///    born-between-loops abort.
///  - A body-guard bridge resume (`body_resume=true`) enters PAST its loop's
///    merge point, so the governing op is the LAST merge point at or before
///    `entry`.
///
/// EXCEPTION: a header entry into a loop that is NESTED inside another loop
/// still active at `entry` (a `for` inside a `while`) keeps the pre-fix
/// behavior — the enclosing loop's (earlier) merge point is selected, which
/// leaves the inner loop's own green color `OpRef::NONE` so the inner-loop
/// trace declines at its merge point instead of compiling.  pyre's walker
/// miscompiles the bridges an inner nested loop closes into (wrong result),
/// so declining (and running the inner loop interpreted) is the safe shape
/// until that separate nested-loop limitation is fixed.  Only the top-level
/// sibling-loop case is newly enabled by the forward selection.
pub(crate) fn loop_header_merge_point_regs(
    code: &[u8],
    entry: usize,
    body_resume: bool,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let merge_point_pcs = || {
        crate::jitcode_runtime::decoded_ops(code)
            .filter(|op| op.opname == "jit_merge_point")
            .map(|op| op.pc)
    };
    let mp_pc = if body_resume {
        merge_point_pcs()
            .filter(|&pc| pc <= entry)
            .max()
            .or_else(|| merge_point_pcs().filter(|&pc| pc >= entry).min())
    } else {
        let forward = merge_point_pcs().filter(|&pc| pc >= entry).min();
        match forward {
            Some(f) if !header_entry_is_nested(code, entry, f) => Some(f),
            _ => merge_point_pcs()
                .filter(|&pc| pc <= entry)
                .max()
                .or(forward),
        }
    }?;
    let mut cursor = mp_pc + 1 + 1; // opcode byte + jdindex (`c`)
    let mut lists: [Vec<u8>; 6] = Default::default();
    for slot in lists.iter_mut() {
        let count = *code.get(cursor)? as usize;
        cursor += 1;
        for _ in 0..count {
            slot.push(*code.get(cursor)?);
            cursor += 1;
        }
    }
    let [_gi, gr, _gf, _ri, rr, _rf] = lists;
    Some((gr, rr))
}

type PerfnWalkResult = Result<
    (crate::jitcode_dispatch::DispatchOutcome, usize),
    crate::jitcode_dispatch::DispatchError,
>;

/// Shared per-CodeObject full-body walk used by both the read-only
/// diagnostic probe ([`probe_walk_perfn_jitcode`], `authoritative=false`,
/// trace discarded) and the production full-body tracer
/// ([`full_body_walk_trace`], `authoritative=true`, trace kept).
///
/// Returns `(entry, code_len, walk_result)` or `None` when the
/// per-CodeObject setup is unavailable.  The caller owns the post-walk
/// disposition: the probe captures a trace position beforehand and
/// `cut_trace`s + logs; the production path maps `walk_result` to a
/// `TraceAction` and keeps the recording.
/// Per-frame jitcode dispatch shared by the root full-body walk
/// ([`run_perfn_walk`]) and the multiframe bridge-carrier drain
/// ([`drive_bridge_carrier_walk`]).  Resolves the five terminal descrs off
/// `MetaInterpStaticData`, builds the per-CodeObject descr pool + sub-jitcode
/// lookup off `pjc.jitcode.exec.descrs`, and runs `dispatch_via_miframe` from
/// `entry` with the caller-seeded `argboxes_r`.  Returns
/// `(code_len, walk_result)`; `None` when the terminal descrs are unwired.
fn dispatch_perfn_frame(
    mi: &mut crate::state::MIFrame,
    session: &std::cell::RefCell<crate::jitcode_dispatch::WalkSession>,
    pjc: &std::sync::Arc<crate::PyJitCode>,
    entry: usize,
    argboxes_r: &[majit_ir::OpRef],
    argboxes_i: &[majit_ir::OpRef],
    authoritative: bool,
) -> Option<(usize, PerfnWalkResult)> {
    // Resolve the five terminal descrs off MetaInterpStaticData so the
    // walk's Finish / exit-with-exception records carry production descr
    // identities.  A missing one means setup never ran — log and bail
    // rather than feed placeholder descrs.
    let (done_void, done_int, done_ref, done_float, exit_exc_ref) = {
        let sd = mi.ctx().metainterp_sd();
        match (
            sd.done_with_this_frame_descr_void.clone(),
            sd.done_with_this_frame_descr_int.clone(),
            sd.done_with_this_frame_descr_ref.clone(),
            sd.done_with_this_frame_descr_float.clone(),
            sd.exit_frame_with_exception_descr_ref.clone(),
        ) {
            (Some(v), Some(i), Some(r), Some(f), Some(e)) => (v, i, r, f, e),
            _ => {
                eprintln!("[walk-perfn] terminal descrs not wired; skipping walk");
                return None;
            }
        }
    };

    // Per-fn descr-pool plumbing: the per-CodeObject body resolves `d`/`j`
    // descr operands through its OWN runtime pool (`pjc.jitcode.exec.descrs`,
    // `Vec<RuntimeBhDescr>`), NOT the global `all_descr_refs()`.  Build the
    // index-parallel adapted `descr_refs` and resolve `inline_call` callee
    // jitcodes through the same pool.
    use majit_metainterp::jitcode::RuntimeBhDescr;
    // The per-CodeObject JitCode lives in the process-global jitcode registry
    // (installed by `install_jitcodes` before tracing); `pjc` is an `Arc` clone
    // of that data, so the descr pool (and the callee jitcode bodies it
    // references) outlive this walk.  Extend the borrow to `'static` so the
    // `'static`-bodied `SubJitCodeBody` from `sub_jitcode_lookup` type-checks —
    // mirrors the production arm-entry borrow extension at `trace_opcode.rs`.
    let perfn_descrs: &'static [RuntimeBhDescr] =
        unsafe { &*(pjc.jitcode.exec.descrs.as_slice() as *const [RuntimeBhDescr]) };
    let perfn_descr_refs: Vec<majit_ir::DescrRef> = perfn_descrs
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            RuntimeBhDescr::Descr(bh) => crate::descr::make_descr_from_bh(bh),
            // `inline_call`'s `d` operand resolves the callee through
            // `JitCodeDescr::jitcode_index()` → `sub_jitcode_lookup`.  Key the
            // descr by its own pool slot `i` so the per-fn lookup below
            // re-reads `exec.descrs[i].as_jitcode()`.  `Call` /
            // `AssemblerToken` pool entries belong to the `BC_CALL_*` /
            // `BC_CALL_ASSEMBLER_*` op families, whose walker handlers read the
            // target straight from `RawDescrPool::PerFn`, not through this
            // adapted `DescrRef` slot; the jitcode-descr stand-in is a
            // fail-loud tripwire for a mis-routed slot.
            RuntimeBhDescr::JitCode(_) => crate::descr::make_jitcode_descr(i),
            RuntimeBhDescr::Call(_) | RuntimeBhDescr::AssemblerToken(_) => {
                crate::descr::make_jitcode_descr(i)
            }
        })
        .collect();

    let sub_jitcode_lookup = |idx: usize| -> Option<crate::jitcode_dispatch::SubJitCodeBody> {
        perfn_descrs
            .get(idx)
            .and_then(|d| d.as_jitcode())
            .map(|jc| crate::jitcode_dispatch::SubJitCodeBody {
                code: jc.code.as_slice(),
                num_regs_r: jc.num_regs_r() as usize,
                num_regs_i: jc.num_regs_i() as usize,
                num_regs_f: jc.num_regs_f() as usize,
                constants_i: jc.constants_i.as_slice(),
                constants_r: jc.constants_r.as_slice(),
                constants_f: jc.constants_f.as_slice(),
            })
    };

    let code = pjc.jitcode.code.as_slice();
    let code_len = code.len();
    let walk_result = crate::jitcode_dispatch::dispatch_via_miframe(
        mi,
        session,
        code,
        entry,
        &perfn_descr_refs,
        crate::jitcode_dispatch::RawDescrPool::PerFn(perfn_descrs),
        // Authoritative concrete execution: `false` for a read-only probe
        // (trace discarded → re-executing would corrupt live state); `true`
        // for the production full-body tracer (the walk IS the execution).
        authoritative,
        &sub_jitcode_lookup,
        done_ref,
        done_int,
        done_float,
        done_void,
        exit_exc_ref,
        true,
        pjc.jitcode.num_regs_r() as usize,
        pjc.jitcode.num_regs_i() as usize,
        pjc.jitcode.num_regs_f() as usize,
        pjc.jitcode.constants_r.as_slice(),
        pjc.jitcode.constants_i.as_slice(),
        pjc.jitcode.constants_f.as_slice(),
        argboxes_r,
        argboxes_i,
        &[],
    );
    Some((code_len, walk_result))
}

/// Select a reconstructed frame's walk-entry JitCode offset: prefer the
/// guard-carried `jitcode_pc` decoded from the resume frame only when it belongs
/// to the same JitCode body that will drive the walk. Pyre permits multiple
/// JitCode bodies per code object, so the carried offset is invalid in another
/// body's coordinate space. Upstream `resume.py:1050-1051` uses the same
/// snapshot-selected jitcode for frame construction and its PC. Fall back to
/// the runtime `resume_jitcode_pc_for` derivation supplied by `derived`.
/// Gated by `PYRE_M73_ENTRY_CARRY` (off → derivation only). Under
/// `PYRE_M73_ENTRY_DECLINE`, entry-carry failures decline instead of deriving.
fn select_recipe_entry(
    jitcode_index: i32,
    body_index: i32,
    py_pc: usize,
    carried_jitcode_pc: i32,
    derived: impl Fn() -> Option<usize>,
) -> Option<usize> {
    let carried = (carried_jitcode_pc != majit_ir::resumedata::NO_JITCODE_PC
        && jitcode_index == body_index)
        .then(|| {
            crate::state::resolve_bridge_walk_entry_at(
                jitcode_index,
                py_pc as i32,
                carried_jitcode_pc,
            )
        })
        .flatten();
    if crate::jitcode_dispatch::m73_entry_carry_enabled() {
        if crate::jitcode_dispatch::m73_entry_decline_enabled() {
            // p5-s3: carried-only. A failed carried resolution aborts/declines
            // at the caller (each `select_recipe_entry` caller routes `None`
            // to a graceful abort), instead of re-deriving from py_pc.
            carried
        } else {
            carried.or_else(derived)
        }
    } else {
        derived()
    }
}

/// Issue #215 item 2 (P2 drain): drive a multiframe bridge-carrier resume via
/// the full-body walker instead of aborting to a no-JIT re-interpret.
///
/// The carrier reconstructs the in-flight callee framestack
/// (`rebuild_from_resumedata`, resume.py:1042-1057); each callee is rebuilt as
/// a virtualizable the walker can drive (`setup_reconstructed_callee_frame`),
/// then walked innermost-first via [`dispatch_perfn_frame`], threading each
/// frame's return into its parent before the parent walks, until the root
/// walks forward to a terminator.
///
/// Increment 1 (diagnostic): walk only the DEEPEST reconstructed callee
/// (`recipes` is outermost-first, so the last entry is the guard-failing
/// frame), log the outcome, discard the trace, and abort — validates the
/// reconstructed-frame walk plumbing before result-threading + the root walk
/// are wired.  Gated behind `PYRE_P2_DRAIN` (default off → unchanged behavior).
/// Thread a reconstructed callee's `SubReturn` value into the root portal's
/// operand-stack result slot so the subsequent root walk (`run_perfn_walk`'s
/// `bridge_stack_oprefs` seeding) reads it as the call result at `root_pc`.
///
/// The result lands at the codewriter-precomputed result color for the call's
/// return pc (`result_color_at_pc_at`), mapped to its `bridge_stack_oprefs`
/// stack slot (`color - nlocals`).  Returns `false` (caller declines the
/// compile) when the color is unresolved or sits below the operand stack.
fn inject_root_call_result(sym: &mut PyreSym, root_pc: usize, result: majit_ir::OpRef) -> bool {
    if sym.jitcode.is_null() {
        return false;
    }
    let jitcode_index = unsafe { (*sym.jitcode).index as i32 };
    let Some(result_color) = crate::state::result_color_at_pc_at(jitcode_index, root_pc) else {
        return false;
    };
    let nlocals = sym.nlocals;
    if result_color < nlocals {
        return false;
    }
    let slot = result_color - nlocals;
    let bridge = sym.bridge_stack_oprefs.get_or_insert_with(Vec::new);
    if bridge.len() <= slot {
        bridge.resize(slot + 1, majit_ir::OpRef::NONE);
    }
    bridge[slot] = result;
    true
}

fn drive_bridge_carrier_walk(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    root_pc: usize,
    cf_addr: usize,
    carrier: &majit_metainterp::BridgeInlineCarrier,
) -> TraceAction {
    let session = std::cell::RefCell::new(crate::jitcode_dispatch::WalkSession::default());
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();

    let root_ec = sym.concrete_execution_context;
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        let pcs: Vec<usize> = carrier.recipes.iter().map(|r| r.pc).collect();
        eprintln!(
            "[p2-shape] root_pc={root_pc} n_recipes={} recipe_pcs={pcs:?}",
            carrier.recipes.len()
        );
    }
    let Some(recipe) = carrier.recipes.last() else {
        crate::jitcode_dispatch::census_record("P2Drain::NoRecipes");
        return TraceAction::Abort;
    };

    let pre_pos = ctx.get_trace_position();
    // `setup_reconstructed_callee_frame` emits the callee frame vable into the
    // trace and returns `argboxes_r` seeding the portal reds + in-flight
    // operand-stack temps; the `_pending` callee sym/concrete frame is unused on
    // the sub-walk path (the sub-walk drives the callee body off `argboxes_r` +
    // the emitted frame vable, not a callee MIFrame).
    let Some((_pending, argboxes_r)) =
        crate::state::setup_reconstructed_callee_frame(ctx, recipe, root_ec, Vec::new())
    else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Drain::SetupFailed");
        return TraceAction::Abort;
    };
    let Some(callee_pjc) = crate::state::pyjitcode_for_code(recipe.code_ptr) else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Drain::NoCalleePjc");
        return TraceAction::Abort;
    };
    let entry = select_recipe_entry(
        recipe.jitcode_index,
        callee_pjc.jitcode.index() as i32,
        recipe.pc,
        recipe.jitcode_pc,
        || callee_pjc.resume_jitcode_pc_for(recipe.pc),
    );
    let Some(entry) = entry else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Drain::NoCalleeEntry");
        return TraceAction::Abort;
    };
    let callee_w_globals = crate::state::recover_inline_callee_globals(recipe.code_ptr) as usize;
    // The reconstructed callee's local slot concretes (`recipe.concrete_r` is
    // parallel to `registers_r`; locals occupy `[0, nlocals)`), seeded into the
    // sub-walk's local-concrete shadow so a nested self-recursive call's int arg
    // is known.
    let nlocals = recipe.nlocals.min(recipe.concrete_r.len());
    let local_concretes = &recipe.concrete_r[..nlocals];

    // Increment 2b-i: drive the deepest callee as an inline SUB-WALK rooted on
    // the portal `sym` (is_top_level=false), so its `ref_return` surfaces
    // `SubReturn` instead of the top-level `Finish` pyre's own-portal model
    // rejects.  Diagnostic: log the outcome, then abort (trace discarded).
    let walk = crate::jitcode_dispatch::drive_bridge_carrier_subwalk(
        ctx,
        &session,
        sym,
        root_pc,
        carrier.root_jitcode_pc,
        &callee_pjc,
        recipe.code_ptr as usize,
        callee_w_globals,
        entry,
        &argboxes_r,
        local_concretes,
    );
    // 2b-ii: on a clean single-recipe `SubReturn`, thread the callee result
    // into the root's operand-stack result slot and walk the ROOT top-level to
    // compile the bridge (the recorded callee continuation + the root
    // continuation form one bridge body).  Gated on `PYRE_P2_COMPILE` (requires
    // the authoritative sub-walk that produced the `SubReturn`); other shapes /
    // outcomes log + abort (trace discarded).
    let subwalk_result = match &walk {
        Some(Ok((crate::jitcode_dispatch::DispatchOutcome::SubReturn { result: Some(r) }, _))) => {
            Some(*r)
        }
        _ => None,
    };
    if let Some(result) = subwalk_result {
        if carrier.recipes.len() == 1 && std::env::var_os("PYRE_P2_COMPILE").is_some() {
            if inject_root_call_result(sym, root_pc, result) {
                crate::jitcode_dispatch::census_record("P2Drain::CompileRoot");
                return full_body_walk_trace(ctx, sym, w_code, root_pc, cf_addr);
            }
            crate::jitcode_dispatch::census_record("P2Drain::ResultSlotUnresolved");
        }
    }

    match &walk {
        Some(Ok((outcome, end_pc))) => {
            eprintln!(
                "[p2-drain] callee sub-walk OK recipe.pc={} entry={entry} end_pc={end_pc} outcome={outcome:?}",
                recipe.pc
            );
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkOk");
        }
        Some(Err(e)) => {
            eprintln!(
                "[p2-drain] callee sub-walk STOP recipe.pc={} entry={entry} err={e:?}",
                recipe.pc
            );
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkStop");
        }
        None => {
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkSetupNone");
        }
    }

    ctx.cut_trace(pre_pos);
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();
    TraceAction::Abort
}

/// Shape A orthodox multi-frame bridge resume: the default driver for a
/// single-recipe (depth-1) carrier.
///
/// A driver-managed framestack trampoline that mirrors RPython's
/// `rebuild_from_resumedata` (resume.py:1042-1057) + continuous interpret loop:
///
///   1. The reconstructed callee framestack is held in the DRIVER (owned
///      register banks per frame), not in `WalkContext` — the walker's register
///      banks are borrowed slices (`&'frame mut [OpRef]`), so a `Vec<Frame>`
///      field is borrow-check infeasible; the framestack lives here.
///   2. Each frame is driven FORWARD from its resume pc via a single-frame
///      `walk()` (`drive_bridge_carrier_subwalk` shape). Because the frame
///      traces forward, its own recursive calls fold to a live `CALL_ASSEMBLER`
///      (the self-recursive fold, `try_walker_call_assembler_self_recursive`) —
///      NOT unrolled into frame reconstruction. The framestack walk therefore
///      supersedes bounded unroll for the resumed recursion.
///   3. A frame's `SubReturn` is delivered into its PARENT frame's dst register
///      via `make_result_of_lastop` (`pyjitpl.py:258-275`) — the parent then
///      resumes forward from its own resume pc with the child result live in
///      its register. Innermost→outermost; the outermost callee's result lands
///      in the ROOT portal frame, which resumes at `root_pc`.
///
/// This dissolves the (n-1)-vs-`fib(n-1)` provenance bug (the return is a live
/// `CALL_ASSEMBLER` result, not a reconstructed-frame arg slot) and the
/// missing-`CALL_ASSEMBLER` bug (recursion stays a call boundary).
///
/// A single-recipe carrier drives the outer continuation via
/// [`drive_outer_continuation_and_map`] and compiles the whole cross-frame
/// bridge. `recipes.len() != 1` (depth≥2, #343) and any setup/continuation
/// failure fall through to `SafeAbortReconstruction`, which cuts the whole
/// reconstruction and re-interprets with the correct result (no SEGV). The
/// [`drive_bridge_carrier_walk`] sub-walk+inject shape (`PYRE_P2_DRAIN`) is a
/// separate unsound deviation, kept gated off.
fn drive_bridge_framestack_walk(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    root_pc: usize,
    cf_addr: usize,
    carrier: &majit_metainterp::BridgeInlineCarrier,
) -> TraceAction {
    let session = std::cell::RefCell::new(crate::jitcode_dispatch::WalkSession::default());
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();

    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        let pcs: Vec<usize> = carrier.recipes.iter().map(|r| r.pc).collect();
        eprintln!(
            "[p2-framestack] root_pc={root_pc} n_recipes={} recipe_pcs={pcs:?}",
            carrier.recipes.len()
        );
    }

    let Some(recipe) = carrier.recipes.last() else {
        crate::jitcode_dispatch::census_record("P2Framestack::NoRecipes");
        return TraceAction::Abort;
    };

    let root_ec = sym.concrete_execution_context;
    let pre_pos = ctx.get_trace_position();
    // Reconstruct the deepest resumed callee frame vable + its `argboxes_r`
    // portal reds (mirror of the `drive_bridge_carrier_walk` setup).
    let Some((_pending, argboxes_r)) =
        crate::state::setup_reconstructed_callee_frame(ctx, recipe, root_ec, Vec::new())
    else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Framestack::SetupFailed");
        return TraceAction::Abort;
    };
    let Some(callee_pjc) = crate::state::pyjitcode_for_code(recipe.code_ptr) else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Framestack::NoCalleePjc");
        return TraceAction::Abort;
    };
    let entry = select_recipe_entry(
        recipe.jitcode_index,
        callee_pjc.jitcode.index() as i32,
        recipe.pc,
        recipe.jitcode_pc,
        || callee_pjc.resume_jitcode_pc_for(recipe.pc),
    );
    let Some(entry) = entry else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Framestack::NoCalleeEntry");
        return TraceAction::Abort;
    };
    let callee_w_globals = crate::state::recover_inline_callee_globals(recipe.code_ptr) as usize;
    let nlocals = recipe.nlocals.min(recipe.concrete_r.len());
    let local_concretes = &recipe.concrete_r[..nlocals];

    let pos_after_setup = ctx.get_trace_position();
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        let root_entry = crate::state::pyjitcode_for_code(w_code)
            .and_then(|pjc| pjc.resume_jitcode_pc_for(root_pc));
        eprintln!(
            "[p2-fs] callee_entry(jit)={entry} callee.pc(py)={} root_pc(py)={root_pc} root_entry(jit)={root_entry:?} pos_pre={pre_pos:?} pos_after_setup={pos_after_setup:?}",
            recipe.pc
        );
    }

    // Drive the deepest reconstructed callee FORWARD from its resume pc. The
    // sub-walk runs with `fbw_mode.carrier_resume` set, so a nested
    // self-recursive call folds to a live `CALL_ASSEMBLER` instead of
    // re-unrolling the call tree.
    let walk = crate::jitcode_dispatch::drive_bridge_carrier_subwalk(
        ctx,
        &session,
        sym,
        root_pc,
        carrier.root_jitcode_pc,
        &callee_pjc,
        recipe.code_ptr as usize,
        callee_w_globals,
        entry,
        &argboxes_r,
        local_concretes,
    );
    let subwalk_result = match &walk {
        Some(Ok((crate::jitcode_dispatch::DispatchOutcome::SubReturn { result: Some(r) }, _))) => {
            Some(*r)
        }
        _ => None,
    };
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        let pos_after_subwalk = ctx.get_trace_position();
        eprintln!(
            "[p2-fs] subwalk outcome={:?} result={subwalk_result:?} pos_after_subwalk={pos_after_subwalk:?}",
            walk.as_ref()
                .map(|r| r.as_ref().map(|(o, pc)| (format!("{o:?}"), *pc)))
        );
        // Dump the ops the sub-walk recorded into `ctx` (pre_pos..now) to confirm
        // the returned SubReturn result traces to the boxed ADD of the two live
        // CALL_ASSEMBLER (the recursive fib(n-1)/fib(n-2) results), not a
        // reconstructed-frame arg-slot read.
        ctx.dump_trace_ops_diag("framestack-subwalk-end");
    }

    // The sub-walk drives the deepest reconstructed callee frame (WITH its
    // emitted vable) forward and records into `ctx`: it emits the two live
    // `CALL_ASSEMBLER` for the callee recursion ([p2-ca] EMIT=2) and its
    // in-callee guards encode resume snapshots against the paused root
    // (`fbw_mode.snapshot_sym`, snapshot_data_len>0), returning a live
    // `SubReturn` result. The vable is load-bearing: local reads lower to
    // `getarrayitem_vable`, which aborts `VableBoxNotSeeded` on an unseeded base
    // — the orthodox resume rebuilds the frame virtualizable
    // (`rebuild_from_resumedata` resume.py:1042 fills the jitcode registers; the
    // Python locals live in the rebuilt vable).
    //
    // #41 continuous cross-frame walk: after the deepest callee sub-walk returns
    // its result, continue the OUTER (root portal) frame forward from its resume
    // pc WITHOUT cutting — appending to the sub-walk's `ctx` so the sub-walk's
    // live `CALL_ASSEMBLER` continuation stays in the compiled bridge. The callee
    // result is delivered into the outer's call-dst register
    // (`make_result_of_lastop`), never a resume color, so the outer resumes with
    // the 1st-call result live and records its 2nd call + ADD + return.
    if let Some(result) = subwalk_result {
        // A single-recipe (depth-1) reconstruction continues the OUTER frame
        // forward and compiles the whole cross-frame bridge. `recipes.len() != 1`
        // (depth≥2) and any continuation-setup failure fall through to the clean
        // `SafeAbortReconstruction` below (correct no-JIT re-interpret).
        if carrier.recipes.len() == 1 {
            if let Some(action) = drive_outer_continuation_and_map(
                ctx,
                &session,
                sym,
                w_code,
                root_pc,
                carrier.root_jitcode_pc,
                cf_addr,
                result,
                pre_pos,
            ) {
                return action;
            }
        }
    }

    let _ = subwalk_result;
    ctx.cut_trace(pre_pos);
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();
    crate::jitcode_dispatch::census_record("P2Framestack::SafeAbortReconstruction");
    TraceAction::Abort
}

/// #41: set up + drive the outer (root portal) continuation and map its outcome
/// to a `TraceAction`.  Returns `Some(action)` when the continuation produced a
/// compilable bridge (or a definite terminal decision), `None` when setup could
/// not proceed so the caller falls through to its clean abort (which cuts the
/// whole reconstruction).  Delivery is by physical call-dst register, decoded
/// from the residual-call op whose `next_pc` is the outer resume entry.
fn drive_outer_continuation_and_map(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<crate::jitcode_dispatch::WalkSession>,
    sym: &mut PyreSym,
    w_code: *const (),
    root_pc: usize,
    root_jitcode_pc: i32,
    _cf_addr: usize,
    result: majit_ir::OpRef,
    pre_pos: majit_metainterp::recorder::TracePosition,
) -> Option<TraceAction> {
    let root_pjc = crate::state::pyjitcode_for_code(w_code)?;
    let entry = select_recipe_entry(
        root_pjc.jitcode.index() as i32,
        root_pjc.jitcode.index() as i32,
        root_pc,
        root_jitcode_pc,
        || root_pjc.resume_jitcode_pc_for(root_pc),
    )?;
    // Decode the call-dst register: the op whose `next_pc == entry` is the
    // residual call the outer resumes after; its `>r` dst is the last operand
    // byte (`code[entry-1]`).
    let code = root_pjc.jitcode.code.as_slice();
    let call_dst_reg = {
        let mut found = None;
        for op in crate::jitcode_runtime::decoded_ops(code) {
            if op.next_pc == entry {
                if op.opname.starts_with("residual_call") && op.argcodes.ends_with(">r") {
                    found = code.get(entry - 1).map(|&b| b as usize);
                }
                break;
            }
        }
        found
    }?;
    let frame_reg = {
        let r = root_pjc.metadata.portal_frame_reg;
        if r != u16::MAX { r as usize } else { 1 }
    };
    let frame_box = ctx
        .standard_virtualizable_box()
        .unwrap_or_else(|| ctx.const_ref(_cf_addr as i64));
    // `w_code` is the root frame's PyCode wrapper; `recover_inline_callee_globals`
    // keys the `code_ptr → live wrapper` registry by RAW code identity, so resolve
    // the raw pointer first.
    let root_code_ptr =
        unsafe { pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef) };
    let root_w_globals = crate::state::recover_inline_callee_globals(root_code_ptr) as usize;

    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        eprintln!(
            "[p2-fs] outer-continuation entry(jit)={entry} call_dst_reg={call_dst_reg} frame_reg={frame_reg} result={result:?} frame_box={frame_box:?}"
        );
    }

    let outcome = crate::jitcode_dispatch::drive_outer_frame_continuation(
        ctx,
        session,
        sym,
        &root_pjc,
        w_code as usize,
        root_w_globals,
        root_pc,
        root_jitcode_pc,
        entry,
        frame_box,
        frame_reg,
        result,
        call_dst_reg,
    );

    if crate::state::take_trace_abort_requested() {
        crate::jitcode_dispatch::census_record("P2Framestack::OuterTraceAbortRequested");
        ctx.cut_trace(pre_pos);
        return Some(TraceAction::Abort);
    }
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        ctx.dump_trace_ops_diag("framestack-outer-walk-end");
        eprintln!(
            "[p2-fs] outer outcome={:?}",
            outcome
                .as_ref()
                .map(|r| r.as_ref().map(|(o, pc)| (format!("{o:?}"), *pc)))
        );
    }

    match outcome {
        Some(Ok((crate::jitcode_dispatch::DispatchOutcome::Terminate, _end_pc))) => {
            match crate::jitcode_dispatch::fbw_finish_payload_take() {
                Some((_, majit_ir::Type::Void)) => {
                    let key = crate::driver::make_green_key(w_code, root_pc);
                    ctx.set_green_key(key, (w_code as usize, root_pc));
                    Some(TraceAction::Finish {
                        finish_args: vec![],
                        finish_arg_types: vec![],
                        exit_with_exception: false,
                    })
                }
                Some((finish_value, finish_type)) => {
                    let key = crate::driver::make_green_key(w_code, root_pc);
                    ctx.set_green_key(key, (w_code as usize, root_pc));
                    crate::jitcode_dispatch::census_record("P2Framestack::OuterFinish");
                    if std::env::var_os("PYRE_P2_DIAG").is_some() {
                        eprintln!(
                            "[p2-fs] COMPILE Finish finish_value={finish_value:?} type={finish_type:?}"
                        );
                    }
                    Some(TraceAction::Finish {
                        finish_args: vec![finish_value],
                        finish_arg_types: vec![finish_type],
                        exit_with_exception: false,
                    })
                }
                None => {
                    crate::jitcode_dispatch::census_record("P2Framestack::OuterNoFinishPayload");
                    if std::env::var_os("PYRE_P2_DIAG").is_some() {
                        eprintln!("[p2-fs] outer Terminate but NO finish payload -> abort");
                    }
                    ctx.cut_trace(pre_pos);
                    Some(TraceAction::Abort)
                }
            }
        }
        other => {
            if std::env::var_os("PYRE_P2_DIAG").is_some() {
                eprintln!("[p2-fs] outer non-terminate outcome, aborting: {other:?}");
            }
            crate::jitcode_dispatch::census_record("P2Framestack::OuterNonTerminate");
            ctx.cut_trace(pre_pos);
            Some(TraceAction::Abort)
        }
    }
}

fn run_perfn_walk(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
    authoritative: bool,
) -> Option<(usize, usize, PerfnWalkResult)> {
    let session = std::cell::RefCell::new(crate::jitcode_dispatch::WalkSession::default());
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        eprintln!("[walk-perfn] no per-CodeObject PyJitCode for code={w_code:?}");
        return None;
    };
    // The green stays in Python-bytecode coordinates for merge-point matching;
    // the codewrite-time trace-entry sidecar carries its JitCode coordinate for
    // plain-portal function entries and loop headers. A bridge starts at its
    // guard resume py_pc, outside that sidecar by construction.
    let is_plain_portal = !ctx.is_bridge_trace;
    let is_loop_header =
        !pjc.code_ptr.is_null() && start_pc_is_loop_header(unsafe { &*pjc.code_ptr }, start_pc);
    let is_entry_green = start_pc == 0 || is_loop_header;
    let uses_entry_sidecar = is_plain_portal && is_entry_green;
    let sidecar_entry = pjc.merge_entry_for(start_pc);
    let carry = crate::jitcode_dispatch::m73_entry_carry_enabled();
    let pc_map_entry = if carry && sym.bridge_walk_entry_pc.is_some() {
        // Guard resume with a carried jitcode coordinate: the walk enters at
        // the carried offset (override below); the entry-marker derivation is
        // unused, so a py_pc the tables cannot encode must not decline the walk.
        sym.bridge_walk_entry_pc
    } else if carry && uses_entry_sidecar {
        sidecar_entry
    } else {
        // Bridge resume: `start_pc` is the guard's py_pc, not a loop-header
        // green — outside the sidecar by construction. The carried coordinate
        // for this leg is `sym.bridge_walk_entry_pc` (used below when present);
        // under `PYRE_M73_ENTRY_DECLINE` the residual derivation is
        // decline-converted, and only opt-out states reach it.
        if carry && crate::jitcode_dispatch::m73_entry_decline_enabled() {
            // p5-s3: under entry-carry this leg is certified unreached
            // (EntryDerivedTaken = 0 corpus-wide); route it to the existing
            // `fbw_decline` below instead of the py_pc translation.
            None
        } else {
            pjc.resume_jitcode_pc_for(start_pc)
        }
    };
    let Some(pc_map_entry) = pc_map_entry else {
        // The frozen pc_map of this already-built body does not encode
        // `start_pc` as a resume coordinate, so the same body walked from
        // the same entry recurs identically on every retrace.  Decline the
        // key (route its retraces to the trait tracer via FBW_DECLINED_KEYS)
        // instead of re-walking and re-aborting each iteration; mirrors the
        // `built_as_portal=false` structural decline below.
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[walk-perfn] no jitcode entry for start_pc={start_pc} (pc_map_len={}); declining walk",
                pjc.metadata.first_jit_pc_by_py_pc.len()
            );
        }
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return None;
    };
    // A kept-stack branch-guard bridge resumes at the guard's OWN mid-opcode
    // jitcode offset (`setup_bridge_sym` resolved it into
    // `sym.bridge_walk_entry_pc`, the same coordinate the blackhole
    // `setposition`s to) — NOT the opcode-entry marker `pc_map[start_pc]`.
    // Resuming at the entry marker re-executes the whole opcode from the top,
    // reading abstract-register colors that were live at entry but dead
    // (recolored / already consumed) at the guard, which the guard's resume
    // data never preserved. See the field doc on `PyreSym::bridge_walk_entry_pc`.
    let entry = sym.bridge_walk_entry_pc.unwrap_or(pc_map_entry);
    // The full-body walk drives a PORTAL trace, so the body must carry the
    // portal entry INPUT SHAPE (`FrameInputs::Portal`: `[frame, ec]` red inputs
    // + the frame-vable locals prologue). Every drained per-code jitcode is
    // Portal-shaped (`built_as_portal` records the
    // input shape, independent of true-portal-ness), so this decline narrows to
    // the only remaining shapeless case: a skeleton jitcode with no portal
    // input shape (pyjitcode.rs `skeleton`).
    if !pjc.metadata.built_as_portal {
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort] start_pc={start_pc} jitcode has no portal input shape \
                 (built_as_portal=false); declining walk"
            );
        }
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return None;
    }

    let is_bridge_trace = ctx.is_bridge_trace;
    let mut mi = crate::state::MIFrame::from_sym(ctx, sym, cf_addr, start_pc, start_pc);

    // setup_call argbox: seed r0 = the standard virtualizable identity box
    // (`virtualizable_boxes[-1]`, the `InputArgRef(SYM_FRAME_IDX)` that
    // `init_symbolic` seeded) — the SAME OpRef production's arm entry uses
    // (`dispatch_via_miframe_at_opcode_entry` seeds r0 = `sym.frame`, and
    // `sym.frame == OpRef::input_arg_typed(SYM_FRAME_IDX, Ref)`).  A fresh
    // `const_ref(cf_addr)` would be a DIFFERENT OpRef than the identity box,
    // so `concrete_of_opref`'s standard-vable resolution (trace_ctx.rs:1842,
    // keyed on `== standard_virtualizable_box()`) would miss and every vable
    // read would fall through to the nonstandard GETFIELD_GC leg.  Falls back
    // to `const_ref` only when no virtualizable is bound.
    //
    // NOTE (51d.1 root cause): seeding r0 is NECESSARY but not sufficient for
    // the mid-loop resume entry (pc=107, after the loop-header
    // `jit_merge_point` @ pc=94).  The loop body reads its vable from a
    // post-merge LOOP-INPUT register (the merge-point reds), NOT from r0; that
    // register is left `OpRef::NONE` because the probe enters past the merge
    // point and never binds the reds.  `concrete_of_opref(NONE)` returns the
    // `GcRef(usize::MAX)` sentinel → `is_nonstandard_virtualizable` takes the
    // nonstandard leg → `getarrayitem_vable` returns `Value::Void` even though
    // the virtualizable SHADOW entry is correct.  Closing that needs the live
    // loop-input registers seeded at walk entry (task #53), not just r0.
    let frame_box = mi
        .ctx()
        .standard_virtualizable_box()
        .unwrap_or_else(|| mi.ctx().const_ref(cf_addr as i64));
    // 51d.1 (B1 blocker): seed the loop's live INPUT registers so the
    // post-merge-point loop body resolves its loop-invariant reads.  The
    // walk enters PAST the loop-header `jit_merge_point`, which would
    // otherwise leave those colors `OpRef::NONE` (→ sentinel concrete →
    // nonstandard-virtualizable Void leg on the first `getarrayitem_vable`).
    // Decode the merge point's green-ref (`gr` = [pycode]) and red (`rr` =
    // [frame, ec], portal jitdriver convention) register lists and seed each
    // named color.  Int greens (`gi` = next_instr, is_being_profiled) live
    // in the int CONSTANT region and are seeded by `copy_constants` inside
    // `dispatch_via_miframe`, so they need no entry seed.  `frame` is the
    // standard-vable identity box (so the body's vable reads hit the
    // standard fast path); `pycode`/`ec` are const-refs to the live
    // pointers.  `argboxes_r[i] -> top_regs_r[i]` is the seed channel.
    let ec_box = mi.ctx().const_ref(sym.concrete_execution_context as i64);
    let pycode_box = mi.ctx().const_ref(w_code as i64);
    let portal_frame_reg = pjc.metadata.portal_frame_reg;
    let portal_ec_reg = pjc.metadata.portal_ec_reg;
    let argboxes_r: Vec<majit_ir::OpRef> = {
        let mut v = vec![majit_ir::OpRef::NONE; 1];
        let mut seed = |reg: u8, val: majit_ir::OpRef| {
            let reg = reg as usize;
            if reg >= v.len() {
                v.resize(reg + 1, majit_ir::OpRef::NONE);
            }
            v[reg] = val;
        };
        // Colors holding the red virtualizable identity (frame) + ec — the
        // standard virtualizable. The #124 operand-stack override below must
        // not overwrite these (a kept temp never lives in a red-input color).
        let mut reserved_red_colors: Vec<u8> = Vec::new();
        match loop_header_merge_point_regs(pjc.jitcode.code.as_slice(), entry, is_bridge_trace) {
            Some((gr, rr)) => {
                if let Some(&r) = gr.first() {
                    seed(r, pycode_box);
                }
                if let Some(&r) = rr.first() {
                    seed(r, frame_box);
                    reserved_red_colors.push(r);
                }
                if let Some(&r) = rr.get(1) {
                    seed(r, ec_box);
                    reserved_red_colors.push(r);
                }
            }
            // Straight-line entry, no governing loop header (e.g. a
            // non-looping function like `fib` or a leaf method): seed the
            // portal red args `[frame, ec]` at the AUTHORITATIVE
            // post-regalloc colors the codewriter recorded
            // (`metadata.portal_frame_reg` / `portal_ec_reg`), the same
            // colors the loop-header `jit_merge_point` `rr` list carries.
            // The earlier positional `[pycode=r0, frame=r1, ec=r2]`
            // convention only coincided with regalloc for an nlocals==1
            // function (fib: frame=r1); a 2-local leaf method (`value()`)
            // places frame at r2 / ec at r3, so the positional seed put
            // `ec_box` in the frame color and every `getfield/getarrayitem
            // _vable` of a local took the nonstandard-virtualizable leg
            // (internal promote `GuardValue` + force store-back, no resume
            // snapshot → `NonStandardVableFinishPortalUnsupported` abort).
            // pycode (the jitdriver's green ref) is read from the frame's
            // `pycode` field via `getfield_vable`, so it needs no register
            // seed once `frame` resolves to the standard virtualizable; the
            // r0 seed is retained as a defensive best-effort (overwritten by
            // the entry prologue's first dst in practice).
            //
            None => {
                seed(0, pycode_box);
                let frame_color = if portal_frame_reg != u16::MAX {
                    portal_frame_reg as u8
                } else {
                    1
                };
                let ec_color = if portal_ec_reg != u16::MAX {
                    portal_ec_reg as u8
                } else {
                    2
                };
                seed(frame_color, frame_box);
                seed(ec_color, ec_box);
                reserved_red_colors.push(frame_color);
                reserved_red_colors.push(ec_color);
            }
        }
        // Loop-trace entry seeds no operand-stack colors.  The codewriter's
        // `Instruction::ForIter` handler emits `getarrayitem_vable_r` to reload
        // the iterator from the virtualizable on every iteration, so the
        // residual consumes that in-loop read rather than an entry register.
        //
        // #124: a bridge enters mid-body, where the loop-header merge-point
        // colors seeded above (the loop's green pycode / red frame+ec) are
        // reused for live operand-stack temps — the kept conditional-
        // expression / short-circuit / chained-compare value.  Leaving e.g.
        // the pycode green at the kept temp's color feeds a stale code object
        // into its binary op (`unsupported operand type(s) for +: 'code' and
        // 'int'`).  Seed the guard's live abstract-register colors from the
        // resume-data OpRefs setup_bridge_sym resolved.  Locals (read through
        // the vable) and frame/ec (at their own colors) keep the seeding above.
        //
        // A kept operand-stack temp never occupies a red-input color, so skip
        // `reserved_red_colors`: seeding a temp over the frame color overwrites
        // the standard virtualizable identity and forces every later `vable_*`
        // op onto the nonstandard leg (NonStandardVableFinishPortalUnsupported
        // abort).
        if is_bridge_trace {
            if sym.bridge_walk_entry_pc.is_some() {
                // Kept-stack branch guard resumed at the guard's own jitcode
                // offset (`entry` above).  The live registers there are the
                // guard-time abstract-register colors the resume data decoded
                // into `bridge_registers_r` (color-indexed, `consume_boxes`
                // parity, resume.py:1055) — the SAME bank the blackhole's
                // `init_register_files` + resume fill would hold.  Seed each
                // non-NONE color directly; the `nlocals + depth` slot→color
                // shortcut below is wrong here because a kept temp's abstract
                // color is not `nlocals + depth` under free register coloring.
                if let Some(ref bridge_regs_r) = sym.bridge_registers_r {
                    for (color, &opref) in bridge_regs_r.iter().enumerate() {
                        if opref.is_none() {
                            continue;
                        }
                        let color = color as u8;
                        if reserved_red_colors.contains(&color) {
                            // `bridge_registers_r` is the authoritative guard-time
                            // register coloring.  Free register allocation reuses
                            // the portal EC color for a live operand at PCs where
                            // the trace has no live EC read (the same collision the
                            // guard-failure resume handles at
                            // jitcode_dispatch.rs:6994 via `semantic_idx.is_none()`
                            // — otherwise `fib(n-1) + fib(n-2)` resumes the left
                            // operand as the EC and SIGSEGVs).  When the bridge
                            // names a genuine operand here (an opref other than the
                            // pre-seeded `ec_box`/`frame_box`), skipping strands the
                            // stale `ConstPtr(ec)` in the color the resumed body
                            // reads as its operand.  Seed the real operand.
                            //
                            // The FRAME color (`reserved_red_colors[0]`) keeps its
                            // skip unconditionally: its `frame_box` is the standard
                            // virtualizable identity and overwriting it forces every
                            // later `vable_*` op onto the nonstandard leg (#124
                            // `NonStandardVableFinishPortalUnsupported`).  The EC
                            // color carries no such identity — the EC stays
                            // recoverable from the frame — so reseeding it is safe.
                            let is_frame_color =
                                reserved_red_colors.first().copied() == Some(color);
                            let bridge_names_operand =
                                !is_frame_color && opref != ec_box && opref != frame_box;
                            if pyre_object::tagged_int::CAN_BE_TAGGED && bridge_names_operand {
                                seed(color, opref);
                            }
                            continue;
                        }
                        seed(color, opref);
                    }
                }
            } else if let Some(ref bridge_stack) = sym.bridge_stack_oprefs {
                // Non-branch-guard resume at the opcode-entry
                // marker: the walk re-executes the opcode from the top, reading
                // its operand-stack inputs POSITIONALLY — `registers_r[nlocals +
                // stack_idx]` (trace_opcode.rs:628 `stack_slot_reg_idx`) — so
                // the slot-indexed `bridge_stack` tail seeds color `nlocals + i`.
                //
                // The reserved-red skip is per-PC-wrong here.  `portal_frame_reg`
                // / `portal_ec_reg` are a SINGLE global pair naming where the
                // reds live at PORTAL ENTRY, not at an arbitrary interior resume
                // PC; under free register coloring the reds sit at the
                // operand-stack base `[nlocals, nlocals + 1]`, exactly the colors
                // a shallow live operand stack occupies.  A live (non-NONE)
                // `bridge_stack[i]` is the authoritative per-PC witness that slot
                // `nlocals + i` holds a real operand-stack value at THIS PC — and
                // a single color cannot simultaneously hold that value and the
                // red, so the red is provably not live at color `nlocals + i`
                // here (its identity is recovered through the frame field, e.g.
                // `ensure_execution_context` / the standard-vable box, not this
                // register).  Seeding the temp is therefore correct and the skip
                // is what dropped it: on `re/_parser.py:append` (nlocals=2) the
                // live callable at slot 2 = color 2 = `portal_frame_reg` was
                // skipped, so `residual_call` read the stale entry frame/ec seed
                // as its callable → SIGSEGV in dispatch_callable /
                // `exception_is_valid_obj_as_class_w` (#389 with-gate probe).
                //
                // A NONE `bridge_stack[i]` (dead/empty slot) leaves the color's
                // red seed intact — the red genuinely still owns the color there.
                let nl = sym.nlocals;
                for (i, &opref) in bridge_stack.iter().enumerate() {
                    if !opref.is_none() {
                        let color = (nl + i) as u8;
                        seed(color, opref);
                    }
                }
            }
        }
        v
    };

    // Int-bank seed for a kept-stack branch-guard bridge: the guard reads its
    // condition from an Int register (the `b < 9` compare result) that ran
    // BEFORE the guard, so resuming at the guard offset requires it from the
    // resume data. `setup_bridge_sym` decoded the Int bank color-indexed into
    // `sym.registers_i` (concrete already stamped there); pass it positionally
    // so `dispatch_via_miframe` writes `top_regs_i[color] = value`. Empty for a
    // non-branch-guard resume (`bridge_walk_entry_pc == None`), where the walk
    // enters at the opcode boundary with no live mid-opcode Int temps.
    let argboxes_i: Vec<majit_ir::OpRef> = if sym.bridge_walk_entry_pc.is_some() {
        // Clamp to the jitcode's Int register count: `sym.registers_i` may carry
        // trailing scratch/constant colors beyond `num_regs_i`, and
        // `dispatch_via_miframe` rejects an argbox list longer than the callee
        // bank (`InlineCallIntArityMismatch`). Only the leading `num_regs_i`
        // colors are real Int registers the walk reads.
        let num_regs_i = pjc.jitcode.num_regs_i() as usize;
        let mut v = sym.registers_i.clone();
        v.truncate(num_regs_i);
        v
    } else {
        Vec::new()
    };

    let Some((code_len, mut walk_result)) = dispatch_perfn_frame(
        &mut mi,
        &session,
        &pjc,
        entry,
        &argboxes_r,
        &argboxes_i,
        authoritative,
    ) else {
        return None;
    };

    // Full-body-walk loop close: the walker's `jit_merge_point` handler
    // produces RPython-style reds (`jump_args = [frame, ec]`, len 2 for the
    // portal jitdriver), but pyre's runtime closes loops against the
    // EXPLICIT scalar inputarg vector
    // `[frame, ec, next_instr, code, valuestackdepth, debugdata, lastblock,
    //  namespace, locals..., stack...]` (len >= NUM_SCALAR_INPUTARGS).
    // `validate_close_with_jump_args` (state.rs) rejects the reds shape, so
    // rebuild the explicit vector via `close_loop_args_at`, mirroring the
    // trait path's `reached_loop_header` (trace_opcode.rs close path).  The
    // loop-carried local/stack OpRefs come from the virtualizable shadow in
    // the TraceCtx (`virtualizable_box_at`, maintained by the authoritative
    // walk's vable ops), NOT from the walk's private register file, so the
    // shadow is live here even though `sym.registers_r` is not.
    //
    // Authoritative only: `close_loop_args_at` records SameAs ops and flushes
    // the virtualizable shadow to the concrete frame heap, which the
    // read-only probe (trace discarded) must not do.
    if authoritative {
        if let Ok((
            crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                jump_args,
                loop_header_pc,
                loop_header_marker_jit_pc,
            },
            _end_pc,
        )) = &mut walk_result
        {
            let loop_header_pc = *loop_header_pc;
            // `close_loop_args_at` reads `self.orgpc` for the last_instr anchor; the merge point
            // closes at the loop header, so anchor orgpc there.
            mi.orgpc = loop_header_pc;
            *jump_args =
                mi.close_loop_args_at(ctx, Some(loop_header_pc), *loop_header_marker_jit_pc);
        }
        // pyjitpl.py:3048-3091 raise_continue_running_normally parity: a
        // walk that ends at a merge point hands the interpreter (and the
        // compiled loop's heap-reloading preamble) the END-of-walk frame
        // state, so the walked iteration — whose residual calls executed
        // concretely — is not re-run.  After `close_loop_args_at` (whose
        // jump-arg derivation reads the pre-walk frame) is the one safe
        // commit point.  All-or-nothing inside the helper; a `false`
        // return keeps the legacy replay.
        //
        // Commit preconditions:
        //   - no unjournaled effect (a symbolically recorded residual
        //     call only the replay applies);
        //   - the frame flush resolves every live slot (all-or-nothing);
        // then the committed flag routes the portal to adopt the end
        // state instead of replaying.  The store-journal epilogue below
        // settles the walk's eager list stores either way (commit keeps
        // them, non-commit rolls them back for the replay).
        // `PYRE_FBW_END_FLUSH=0` opts out for bisection.
        if std::env::var_os("PYRE_FBW_END_FLUSH").as_deref() != Some(std::ffi::OsStr::new("0")) {
            if let Ok((outcome, _end_pc)) = &walk_result {
                let header_pc = match outcome {
                    crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                        loop_header_pc, ..
                    } => Some(*loop_header_pc),
                    crate::jitcode_dispatch::DispatchOutcome::CompileTracePending {
                        loop_header_pc,
                    } => Some(*loop_header_pc),
                    _ => None,
                };
                if let Some(header_pc) = header_pc {
                    if crate::jitcode_dispatch::fbw_has_unjournaled_effect() {
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-end-flush] declined at header_pc={header_pc} \
                                 (unjournaled effect) — legacy replay kept"
                            );
                        }
                    } else if crate::state::flush_walk_end_state_to_frame(ctx, cf_addr, header_pc) {
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-end-flush] COMMIT header_pc={header_pc} bridge={} \
                                 journal_len={} outcome={outcome:?}",
                                ctx.is_bridge_trace,
                                crate::jitcode_dispatch::fbw_store_journal_len(),
                            );
                        }
                        WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
                    } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-end-flush] declined at header_pc={header_pc} (shadow slot \
                             without concrete / depth / lastblock) — legacy replay kept"
                        );
                    }
                }
            }
        }

        // Inline-callee forward abort, or an `abort_permanent` marker abort
        // (DELETE_FAST and the other emit_abort_permanent opcodes).  The
        // marker's contract is "resume
        // the interpreter AT this unsupported opcode and run it" — codewriter
        // stores `last_instr = py_pc - 1` for the blackhole.  On the
        // full-body walk that recorded write is discarded with the aborted
        // trace, while the walk already executed the region's residual side
        // effects concretely, so the legacy `ContinueRunningNormally` replays
        // them from entry → double-execution (e.g. a `del`-bearing method
        // whose prior STORE_ATTR ran once during the walk, then again on
        // replay).  Flush the abort-point frame (locals + last_instr) so the
        // portal resumes at the unsupported opcode instead of replaying.
        // The marker-only fallback uses the same no-unjournaled-effect
        // predicate as the CloseLoop end-flush above.  A latched inline-callee
        // forward abort has already distinguished an outside mark from a mark
        // inside its discarded attempt.  `PYRE_FBW_ABORT_FLUSH=0` opts out.
        if std::env::var_os("PYRE_FBW_ABORT_FLUSH").as_deref() != Some(std::ffi::OsStr::new("0")) {
            let call_forward_abort = match &walk_result {
                Err(crate::jitcode_dispatch::DispatchError::AbortPermanentMarkerReached { pc }) => {
                    Some((*pc, true))
                }
                Err(
                    crate::jitcode_dispatch::DispatchError::LoopBearingCalleeInlineUnsupported {
                        pc,
                    },
                ) => Some((*pc, false)),
                _ => None,
            };
            if let Some((abort_jit_pc, is_marker_abort)) = call_forward_abort {
                // gh#467: a supported abort fired inside a TOP-level inline
                // sub-walk whose callee executed no concrete effect
                // (`try_walker_inline_user_call` latched the carrier only under
                // that gate).  The nested-unjournaled-decline class means the
                // residual did not execute; its callee attempt can be discarded
                // with any inside-only unjournaled mark.  Flush the OUTER frame
                // at the CALL that entered the callee and resume the interpreter
                // forward — re-executing the whole call from scratch — instead
                // of the legacy replay from loop entry, which double-applies the
                // non-journaled pre-CALL store.  The abort's `abort_jit_pc` is a
                // CALLEE coordinate with no meaning in the outer py_pc tables,
                // so the outer CALL py_pc and operand stack come from the latch.
                // Convergence of `run_blackhole_interp_to_cancel_tracing`
                // (`pyjitpl.py:2949`), minus the inner-frame rebuild (#126/#215).
                let carrier = crate::jitcode_dispatch::fbw_abort_carrier_clone();
                match carrier.as_ref() {
                    Some(crate::jitcode_dispatch::InlineAbortCarrier::Entry {
                        call_py_pc,
                        call_stack,
                    }) => {
                        if crate::state::flush_walk_end_state_at_outer_call(
                            ctx,
                            cf_addr,
                            *call_py_pc,
                            call_stack,
                        ) {
                            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                                eprintln!(
                                    "[fbw-abort-flush] gh#467 CALL-forward COMMIT \
                                     abort_jit_pc={abort_jit_pc} call_py_pc={call_py_pc} \
                                     stack_depth={}",
                                    call_stack.len()
                                );
                            }
                            WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
                        } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] gh#467 CALL-forward declined at \
                                 call_py_pc={call_py_pc} (depth mismatch / unresolved local / \
                                 lastblock) — legacy replay kept"
                            );
                        }
                    }
                    Some(crate::jitcode_dispatch::InlineAbortCarrier::MidBody(payload))
                        if (is_marker_abort
                            && payload.abort_kind
                                == crate::jitcode_dispatch::MidBodyAbortKind::Marker)
                            || (!is_marker_abort
                                && payload.abort_kind
                                    == crate::jitcode_dispatch::MidBodyAbortKind::Structural) =>
                    {
                        if try_commit_midbody_abort(ctx, cf_addr, payload) {
                            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                                eprintln!(
                                    "[fbw-abort-flush] gh#467 callee-rebuild COMMIT \
                                     abort_jit_pc={abort_jit_pc} callee_py_pc={} \
                                     call_py_pc={} post_call_py_pc={}",
                                    payload.callee_py_pc,
                                    payload.call_py_pc,
                                    payload.post_call_py_pc,
                                );
                            }
                            WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
                        } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] gh#467 callee-rebuild declined at \
                                 callee_py_pc={} — legacy replay kept",
                                payload.callee_py_pc,
                            );
                        }
                    }
                    None if is_marker_abort => {
                        if crate::jitcode_dispatch::fbw_has_unjournaled_effect()
                            || session.borrow().abort_in_subwalk
                        {
                            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                                eprintln!(
                                    "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                                     (unjournaled effect or inline sub-walk) — legacy replay kept"
                                );
                            }
                        } else if let Some(resume_py_pc) =
                            crate::jitcode_dispatch::fbw_abort_resume_py_pc(sym, abort_jit_pc)
                        {
                            if crate::state::flush_walk_end_state_to_frame(
                                ctx,
                                cf_addr,
                                resume_py_pc,
                            ) {
                                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                                    eprintln!(
                                        "[fbw-abort-flush] COMMIT abort_jit_pc={abort_jit_pc} \
                                         resume_py_pc={resume_py_pc}"
                                    );
                                }
                                WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
                            } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                                eprintln!(
                                    "[fbw-abort-flush] declined at resume_py_pc={resume_py_pc} \
                                     (shadow slot without concrete / depth / lastblock) — legacy replay kept"
                                );
                            }
                        }
                    }
                    _ if crate::jitcode_dispatch::fbw_debug_abort_enabled() => {
                        eprintln!(
                            "[fbw-abort-flush] gh#467 CALL-forward declined at \
                             abort_jit_pc={abort_jit_pc} (no carrier) — legacy replay kept"
                        );
                    }
                    _ => {}
                }
                if carrier.is_some() {
                    crate::jitcode_dispatch::fbw_abort_carrier_clear();
                }
            }
            if let Err(
                crate::jitcode_dispatch::DispatchError::LoopBearingCalleeInlineUnsupported { pc },
            ) = &walk_result
            {
                let abort_jit_pc = *pc;
                if !crate::jitcode_dispatch::fbw_executed_nonpure_residual() {
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                             (no executed non-pure residual) — legacy replay kept"
                        );
                    }
                } else if crate::jitcode_dispatch::fbw_has_unjournaled_effect() {
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                             (unjournaled effect) — legacy replay kept"
                        );
                    }
                } else if let Some(resume_py_pc) =
                    crate::jitcode_dispatch::fbw_abort_outer_resume_take()
                {
                    // Flush while the overrides stay rooted in
                    // FBW_ABORT_OUTER_STACK_OVERRIDES (the flush boxes Int/Float
                    // locals — an allocation that can move the nursery-resident
                    // override refs; the area walker forwards them in place),
                    // then clear the cell.
                    let committed = crate::jitcode_dispatch::fbw_abort_outer_stack_overrides_with(
                        |stack_overrides| {
                            crate::state::flush_walk_end_state_to_frame_with_stack_overrides(
                                ctx,
                                cf_addr,
                                resume_py_pc,
                                stack_overrides,
                            )
                        },
                    );
                    crate::jitcode_dispatch::fbw_abort_outer_stack_overrides_clear();
                    if committed {
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] COMMIT abort_jit_pc={abort_jit_pc} \
                                 resume_py_pc={resume_py_pc} (nested inline decline)"
                            );
                        }
                        WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
                    } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-abort-flush] declined at resume_py_pc={resume_py_pc} \
                             (shadow slot without concrete / depth / lastblock) — legacy replay kept"
                        );
                    }
                } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                         (no outer caller resume pc) — legacy replay kept"
                    );
                }
            }
        }

        // #32 S2: a kept-stack branch guard whose not-taken arm cannot be
        // restored for the COMPILED trace aborts (`AbortPermanent` for the
        // unrestorable-Ref shape, decline + `Abort` for the depth>1
        // invalid-mirror shape), but the authoritative walk's symbolic shadow
        // IS complete at the abort pc (the hazard is about the JIT resume
        // snapshot, not the interpreter-side shadow).  Flush that end state to
        // the live frame so the interpreter resumes at the abort pc with the
        // walked iterations already counted, instead of discarding the walk
        // and dropping an in-flight FOR_ITER item via the conservative #30
        // header-guard drop (or, for the `Unsupported` shape, re-executing the
        // walk's residual effects from loop entry).  Same
        // no-unjournaled-effect / no-sub-walk predicate and same all-or-nothing
        // `flush_walk_end_state_to_frame` gate as the CloseLoop / marker legs;
        // when the flush declines (a slot the shadow cannot resolve) the legacy
        // drop stands (the residual S3 case).  `PYRE_FBW_BRANCH_FLUSH=0` opts
        // out.
        if std::env::var_os("PYRE_FBW_BRANCH_FLUSH").as_deref() != Some(std::ffi::OsStr::new("0")) {
            let kept_stack_abort_pc = match &walk_result {
                Err(
                    crate::jitcode_dispatch::DispatchError::BranchGuardUnrestorableKeptStackPermanent {
                        pc,
                    },
                ) => Some((*pc, false)),
                Err(crate::jitcode_dispatch::DispatchError::BranchGuardKeptStackUnsupported {
                    pc,
                }) => Some((*pc, true)),
                _ => None,
            };
            if let Some((pc, is_unsupported)) = kept_stack_abort_pc {
                let abort_jit_pc = pc;
                if crate::jitcode_dispatch::fbw_has_unjournaled_effect()
                    || session.borrow().abort_in_subwalk
                {
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-branch-flush] declined at abort_jit_pc={abort_jit_pc} \
                             (unjournaled effect or inline sub-walk) — legacy drop kept"
                        );
                    }
                } else if let Some(resume_py_pc) =
                    crate::jitcode_dispatch::fbw_abort_resume_py_pc(sym, abort_jit_pc)
                {
                    // Two kept-stack branch aborts reach this leg (`is_unsupported`
                    // came from the `kept_stack_abort_pc` match).  Both resume at a
                    // FOR_ITER header whose walk already advanced the iterator; they
                    // differ in whether the consumed item's body ran.
                    let committed = if is_unsupported {
                        // `BranchGuardKeptStackUnsupported` aborts AFTER the body
                        // ran for the consumed item (the depth>1 kept operand stack
                        // the COMPILED trace could not restore); the walk's
                        // symbolic shadow already counted that iteration.  Resume
                        // at the FOR_ITER header on the flushed end state WITHOUT
                        // re-delivering the in-flight item — delivering it would
                        // re-run the body and double-count the value.  A plain
                        // end-state flush, gated the same as the marker leg.
                        crate::state::flush_walk_end_state_to_frame(ctx, cf_addr, resume_py_pc)
                    } else {
                        // Shape A — a `BranchGuardUnrestorableKeptStackPermanent`
                        // abort resumes AT a FOR_ITER header whose consumed item is
                        // in flight (`body_pc == resume_py_pc + 1`, the opcode
                        // there really is a FOR_ITER): the walk advanced the
                        // iterator but the item is not yet on the flushed (header)
                        // stack, so deliver it (push + reposition to the body) so
                        // the body runs once.  Commit ONLY when an item is
                        // delivered — a Permanent abort not at such a header keeps
                        // the legacy drop byte-identically (the residual S3 case),
                        // so every other abort shape (and the whole flag-OFF path)
                        // is untouched.
                        let push = crate::jitcode_dispatch::fbw_foriter_inflight_take_for_resume(
                            cf_addr,
                            resume_py_pc,
                        );
                        push.is_some()
                            && crate::state::flush_walk_end_state_to_frame_with_item(
                                ctx,
                                cf_addr,
                                resume_py_pc,
                                push,
                            )
                    };
                    if committed {
                        // The flush owns the iteration count; drop any remaining
                        // in-flight items so the legacy deliver cannot re-apply
                        // one (exactly-once).
                        crate::jitcode_dispatch::fbw_foriter_inflight_clear();
                        WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-branch-flush] COMMIT abort_jit_pc={abort_jit_pc} \
                                 resume_py_pc={resume_py_pc} (delivered in-flight FOR_ITER item)"
                            );
                        }
                    } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-branch-flush] declined at resume_py_pc={resume_py_pc} \
                             (shadow slot without concrete / depth / lastblock) — legacy drop kept"
                        );
                    }
                }
            }
        }
    }

    // No-replay portal exit for a loop-free function trace: a `Terminate`
    // walk whose top-level `*_return` captured a concrete result hands that
    // result to the portal directly (`eval.rs` consumes the stash) instead
    // of re-running the freshly compiled trace for the SAME invocation —
    // the walk already executed every residual call concretely, consuming
    // any side-effecting callee (e.g. a tokenizer's `get`), so a re-run
    // would re-read the mutated heap and deopt.  Declined when an
    // unjournaled effect (a symbolically recorded residual only the legacy
    // replay applies) is present: drop the capture so the portal degrades
    // to `ContinueRunningNormally`.  This shares its predicate with the
    // store-journal commit below so the two decisions never disagree.
    //
    // A guard-failure BRIDGE `Terminate` walk takes the same shortcut when
    // the bridge tracer armed it (`fbw_bridge_noreplay_armed`): the caller
    // hands the captured concrete result forward as `DoneWithThisFrame`
    // rather than rewinding the live frame to the guard pc and re-running the
    // region through the `ContinueRunningNormally` re-entry — which would
    // execute every residual a second time and double-apply any
    // callee-internal side effect (#177).  The tracer arms it for any
    // single-frame resume: the general guard path consumes the kept stash as
    // a terminal `BridgeResolution`, and the CALL_ASSEMBLER callback hands it
    // to its back-to-back blackhole hook, so a committed journal never
    // strands into a guard-state re-run; the three decisions (this
    // predicate, the journal commit below, and the caller's
    // consume-vs-rewind) stay in agreement.  A multiframe resume is never
    // armed, so it stays on the legacy rewind-and-replay path.
    let terminate_no_replay = crate::jitcode_dispatch::fbw_no_replay_exit_enabled()
        && (!is_bridge_trace || crate::jitcode_dispatch::fbw_bridge_noreplay_armed())
        && matches!(
            &walk_result,
            Ok((crate::jitcode_dispatch::DispatchOutcome::Terminate, _))
        )
        && crate::jitcode_dispatch::fbw_finish_concrete_peek().is_some()
        && !crate::jitcode_dispatch::fbw_has_unjournaled_effect();
    if !terminate_no_replay {
        crate::jitcode_dispatch::fbw_finish_concrete_reset();
    }

    // Store-journal epilogue, on EVERY walk exit (commit, declined
    // commit, walk error): a committed walk keeps its eagerly executed
    // list stores (drop the undo log); any other exit returns control to
    // a replay-from-start path, which re-executes the walked region and
    // must find the pre-walk heap — roll the stores back.  A
    // `terminate_no_replay` exit also keeps the stores: the portal returns
    // the walk's result without replaying, exactly like the loop-flush
    // commit.
    if is_bridge_trace && crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        let outcome_kind = match &walk_result {
            Ok((crate::jitcode_dispatch::DispatchOutcome::Continue, _)) => "Continue",
            Ok((crate::jitcode_dispatch::DispatchOutcome::Terminate, _)) => "Terminate",
            Ok((crate::jitcode_dispatch::DispatchOutcome::SubReturn { .. }, _)) => "SubReturn",
            Ok((crate::jitcode_dispatch::DispatchOutcome::SubRaise { .. }, _)) => "SubRaise",
            Ok((crate::jitcode_dispatch::DispatchOutcome::SwitchToBlackhole { .. }, _)) => {
                "SwitchToBlackhole"
            }
            Ok((crate::jitcode_dispatch::DispatchOutcome::CloseLoop { .. }, _)) => "CloseLoop",
            Ok((crate::jitcode_dispatch::DispatchOutcome::CompileTracePending { .. }, _)) => {
                "CompileTracePending"
            }
            Ok((_, _)) => "OtherOk",
            Err(_) => "Err",
        };
        eprintln!(
            "[fbw-bridge-epilogue] committed={} store_journal_len={} unjournaled={} outcome={}",
            WALK_END_FLUSH_COMMITTED.with(|c| c.get()),
            crate::jitcode_dispatch::fbw_store_journal_len(),
            crate::jitcode_dispatch::fbw_has_unjournaled_effect(),
            outcome_kind,
        );
    }
    let committed = WALK_END_FLUSH_COMMITTED.with(|c| c.get()) || terminate_no_replay;
    let journal = crate::jitcode_dispatch::fbw_store_journal_len();
    if committed {
        crate::jitcode_dispatch::fbw_store_journal_commit();
    } else {
        crate::jitcode_dispatch::fbw_store_journal_rollback();
    }
    if authoritative && std::env::var_os("PYRE_FBW_CENSUS").is_some() {
        let mut end = match &walk_result {
            Ok((outcome, _)) => format!("{outcome:?}"),
            Err(error) => format!("{error:?}"),
        };
        if let Some(at) = end.find(|c: char| matches!(c, '(' | '{' | ' ')) {
            end.truncate(at);
        }
        let (unj_val, unj_sym) = crate::jitcode_dispatch::fbw_unjournaled_kinds();
        let (exec_v, exec_mf, exec_pl) = crate::jitcode_dispatch::fbw_executed_residual_counts();
        eprintln!(
            "[fbw-census] end={end} committed={committed} bridge={} unj_val={unj_val} \
             unj_sym={unj_sym} exec_v={exec_v} exec_mf={exec_mf} exec_pl={exec_pl} journal={journal}",
            ctx.is_bridge_trace,
        );
    }

    Some((entry, code_len, walk_result))
}

/// Issue #73 walker-as-tracer foundation probe (slice #1, read-only).
///
/// Runs the per-CodeObject full-body walk via [`run_perfn_walk`] in
/// non-authoritative mode, logs how far the symbolic walk got (terminator
/// outcome vs first `DispatchError` stop), then rolls the recorder back so
/// the diagnostic leaves no partial trace.  `PYRE_PROBE_AUTHORITATIVE=1`
/// opts into authoritative execution for diagnosis ONLY (verifying the
/// walk advances past the loop `goto_if_not`); it corrupts the live
/// frame/iterator state because the probe still discards the trace, so it
/// must never be set outside a throwaway run.
fn probe_walk_perfn_jitcode(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
) {
    let authoritative = std::env::var_os("PYRE_PROBE_AUTHORITATIVE").is_some();
    // Capture the trace position BEFORE the walk so `cut_trace` rolls back
    // every op the diagnostic recorded (the walk discards its trace).
    let pre_pos = ctx.get_trace_position();
    let Some((entry, code_len, walk_result)) =
        run_perfn_walk(ctx, sym, w_code, start_pc, cf_addr, authoritative)
    else {
        return;
    };
    match &walk_result {
        Ok((outcome, end_pc)) => eprintln!(
            "[walk-perfn] entry={entry} code_len={code_len} OK end_pc={end_pc} outcome={outcome:?}"
        ),
        Err(e) => {
            eprintln!("[walk-perfn] entry={entry} code_len={code_len} STOP err={e:?}");
        }
    }

    // Roll the recorder back so the aborted trace leaves no partial ops.
    ctx.cut_trace(pre_pos);
    // The probe discards its trace; clear the walk-local bool-box-truth map and
    // stashed Finish payload an authoritative probe walk may have recorded so
    // they cannot leak into the next walk (the production tracer clears these
    // at entry, but the probe never runs through that path).
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();
}

/// True when a loop body in `w_code` contains an `abort_permanent` marker.
///
/// An `abort_permanent` inside a loop body (e.g. the `SWAP` an `a < b < c`
/// chained comparison lowers to, or any other unported in-loop opcode)
/// corrupts the authoritative full-body walk: the unsupported op breaks the
/// loop-input register seeding, so the walk mis-evaluates the loop guard,
/// exits the loop on the first pass, and concretely executes the post-loop
/// tail — double-running its side effects and leaving the frame positioned
/// past the loop (#125).  The walk's reactive `abort_permanent` decline
/// never fires because the corrupted guard exits before reaching the
/// marker.  The scan is scoped to ops at/after the first `jit_merge_point`
/// (the inner loop header) so a prologue-only marker (e.g. `COPY_FREE_VARS`
/// ahead of a clean hot loop) does not over-decline.
fn loop_body_has_abort_permanent(w_code: *const ()) -> bool {
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        return false;
    };
    let code = pjc.jitcode.code.as_slice();
    let mut seen_merge_point = false;
    for op in crate::jitcode_runtime::decoded_ops(code) {
        if op.opname == "jit_merge_point" {
            seen_merge_point = true;
        } else if seen_merge_point && op.opname == "abort_permanent" {
            return true;
        }
    }
    false
}

/// True when the hot loop body in `w_code` inline-calls — transitively — a
/// user function whose per-fn jitcode body carries an `abort_permanent`
/// marker.
///
/// [`loop_body_has_abort_permanent`] only scans the top-level per-CodeObject
/// jitcode, so an `abort_permanent` reached through an inlined callee slips
/// past it.  That gap causes a walk-time double-apply: a non-journaled
/// concrete heap store (dict/attr/set item, list `extend`, …) in the loop
/// body executes concretely, then an inline-eligible user CALL later in the
/// same body is inline-attempted; the callee sub-walk hits `abort_permanent`
/// and routes the whole walk to abort; the epilogue rolls back the store
/// journal and REPLAYS FROM LOOP ENTRY, so the non-journaled store — which
/// the journal never captured — re-executes and the loop over-counts (e.g.
/// `300001` instead of `300000`).  Declining the walk up front, before it
/// executes anything, avoids the double-apply: the location re-interprets
/// without JIT (correct, at interpreter speed).
///
/// This is an OVER-DECLINING stopgap, and static: it mirrors the inline path's
/// static eligibility gates, but can still decline on a function merely
/// referenced by the loop (present in `co_names`/locals), not just one the
/// executed path actually calls.  Call-site-dependent gates such as the
/// passed-argument count and recursion depth cannot be resolved by this scan,
/// so a callee that would fail one of those gates can also over-decline.  A hot
/// loop that calls an otherwise inline-eligible helper whose body contains an
/// unported op (`match`, `async`, chained-compare `SWAP`, …) — even on a rarely
/// taken path — now runs interpreted in full, not just the aborting call.  The
/// orthodox mechanism has no up-front scan at all: an unsupported op raises
/// `SwitchToBlackhole` mid-trace and
/// `run_blackhole_interp_to_cancel_tracing` (pyjitpl.py:2949) converts the live
/// framestack and continues FORWARD in the blackhole interpreter, so nothing
/// replays and nothing double-applies.  This decline holds until that
/// forward-resume convergence (#126/#215) lets an inlined-callee abort resume
/// the outer walk in place instead of rolling back to loop entry.
///
/// The scan resolves candidate callees CONCRETELY from the live frame (the
/// walk has not run yet, so no store has executed).  Two seed sources:
/// - the frame's module globals — every referenced name in the CodeObject's
///   `co_names` is looked up in `w_globals`;
/// - the ROOT frame's fastlocals + closure cells — a helper passed as an
///   argument/local (`h([i, i])`) or held in a closure cell resolves here.
///
/// Each plain-function value first passes the inline path's static closure,
/// positional-parameter, jitcode-body, and Ref-register-capacity gates.  Its
/// per-fn jitcode body is then scanned end-to-end for `abort_permanent` (the
/// marker can sit at any pc, ahead of the callee's own merge point).
/// Non-aborting eligible callees are enqueued and their own referenced
/// functions scanned transitively through THEIR globals, guarded by a
/// scan-local visited set.  The root `w_code` is pre-marked visited — its own
/// loop-body marker is already handled by [`loop_body_has_abort_permanent`].
///
/// Frame-local seeding is ROOT-frame only; a deeper (not-yet-pushed) callee's
/// locals are not available up front.  Callees reached via attribute access,
/// container elements, or another call's return value, and callees local to a
/// deeper frame, stay unresolvable before the walk — those rely on the
/// deferred #126/#215 forward-resume convergence rather than this stopgap.
fn loop_inlines_abort_permanent_callee(w_code: *const (), cf_addr: usize) -> bool {
    // Gate: only scan when the top-level loop body (ops after the first
    // `jit_merge_point`) contains a `residual_call*` op.  Every inline-eligible
    // user call lowers to a residual_call, so a call-free loop cannot
    // inline-abort — skipping it avoids resolving globals for the common case.
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        return false;
    };
    let mut seen_merge_point = false;
    let mut has_residual_call = false;
    for op in crate::jitcode_runtime::decoded_ops(pjc.jitcode.code.as_slice()) {
        if op.opname == "jit_merge_point" {
            seen_merge_point = true;
        } else if seen_merge_point && op.opname.starts_with("residual_call") {
            has_residual_call = true;
            break;
        }
    }
    if !has_residual_call || cf_addr == 0 {
        return false;
    }

    // Process one concrete candidate value shared by both seed paths (globals
    // and frame slots): gate it to a plain user function, scan its whole
    // jitcode body for `abort_permanent`, and otherwise enqueue it for
    // transitive resolution through its own globals.  Returns `true` iff the
    // candidate's body carries the marker.
    //
    // SAFETY: `cand` is a live concrete `PyObjectRef` read from the frame or a
    // module dict before the walk mutates anything.
    unsafe fn consider_candidate(
        cand: pyre_object::PyObjectRef,
        function_type_addr: usize,
        visited: &mut std::collections::HashSet<*const ()>,
        queue: &mut std::collections::VecDeque<(*const (), pyre_object::PyObjectRef)>,
    ) -> bool {
        // A tagged immediate int is never a FUNCTION_TYPE callee; skip it
        // before the `ob_type` deref below (which reads an even-aligned heap
        // pointer). Reaches here via the globals-dict scan and via a cell
        // whose contents are a tagged int.
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(cand) {
            return false;
        }
        // Only plain user functions inline (mirrors the inline path's exact
        // FUNCTION_TYPE gate); builtins carry no CodeObject.
        if cand.is_null() || (*cand).ob_type as *const () as usize != function_type_addr {
            return false;
        }
        let callee_w_code = pyre_interpreter::function_get_code(cand);
        if callee_w_code.is_null() {
            return false;
        }
        // A FUNCTION_TYPE object can wrap a BuiltinCode, not a CodeObject:
        // `make_builtin_function*` (gateway.rs:701) puts such a function into
        // module globals (e.g. `from sys import getsizeof`).  Feeding its
        // BuiltinCode to `sub_jitcode_body_for_code` / `w_code_get_ptr` casts it
        // as a PyCode and derefs garbage, so reject it before the scan — a
        // builtin carries no traceable body and never inlines.
        if pyre_interpreter::is_builtin_code(callee_w_code as pyre_object::PyObjectRef) {
            return false;
        }
        let Some((callee_w_code, nparams, has_closure)) =
            crate::jitcode_dispatch::resolve_inlinable_callee(cand)
        else {
            return false;
        };
        if has_closure || nparams == 0 {
            return false;
        }
        let Some(body) = crate::state::sub_jitcode_body_for_code(callee_w_code) else {
            return false;
        };
        if nparams > body.num_regs_r || !visited.insert(callee_w_code) {
            return false;
        }
        for op in crate::jitcode_runtime::decoded_ops(body.code) {
            if op.opname == "abort_permanent" {
                return true;
            }
        }
        // Transitive: resolve this callee's own referenced functions in its own
        // globals.
        let callee_globals = pyre_interpreter::function_get_globals_obj(cand);
        if !callee_globals.is_null() {
            queue.push_back((callee_w_code, callee_globals));
        }
        false
    }

    // SAFETY: `cf_addr` is the live `PyFrame` pointer the portal passed to the
    // walk; its `w_globals` is the module dict and its locals/cells region is
    // initialised.  All callee resolution reads live concrete objects before
    // the walk mutates anything.
    unsafe {
        let cf = &*(cf_addr as *const pyre_interpreter::pyframe::PyFrame);
        let root_globals = cf.w_globals;
        if root_globals.is_null() {
            return false;
        }
        let function_type_addr = &pyre_interpreter::FUNCTION_TYPE as *const _ as usize;
        let mut visited: std::collections::HashSet<*const ()> = std::collections::HashSet::new();
        // The root's own loop-body `abort_permanent` is handled upstream.
        visited.insert(w_code);
        // BFS over (code wrapper ptr, globals in which its `co_names` resolve).
        let mut queue: std::collections::VecDeque<(*const (), pyre_object::PyObjectRef)> =
            std::collections::VecDeque::new();
        queue.push_back((w_code, root_globals));

        // Seed from the root frame's fastlocals + closure cells: a helper
        // passed as an argument/local or held in a cell is not in `co_names`,
        // so resolve it directly from the frame's initialised locals/cells
        // region.  Stop at `stack_base()` — operand-stack slots beyond it are
        // uninitialised.
        let slots = cf.locals_w().as_slice();
        let bound = cf.stack_base().min(slots.len());
        for &slot in &slots[..bound] {
            if slot.is_null() {
                continue;
            }
            // A tagged immediate int is neither a cell nor a FUNCTION_TYPE
            // callee; skip it before `is_cell(slot)` derefs its `ob_type`.
            if pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(slot)
            {
                continue;
            }
            // A closure cell holds the function indirectly; unwrap it.
            let value = if pyre_object::is_cell(slot) {
                pyre_object::w_cell_get(slot)
            } else {
                slot
            };
            if consider_candidate(value, function_type_addr, &mut visited, &mut queue) {
                return true;
            }
        }

        while let Some((code_ptr, globals)) = queue.pop_front() {
            let raw = pyre_interpreter::w_code_get_ptr(code_ptr as pyre_object::PyObjectRef)
                as *const CodeObject;
            if raw.is_null() {
                continue;
            }
            for name in (*raw).names.iter() {
                let Some(cand) =
                    pyre_object::dictmultiobject::w_dict_getitem_str(globals, name.as_str())
                else {
                    continue;
                };
                if consider_candidate(cand, function_type_addr, &mut visited, &mut queue) {
                    return true;
                }
            }
        }
    }
    false
}

/// Issue #73 production full-body tracer (Phase 5 flip, gated).
///
/// `PYRE_FULL_BODY_WALK=1` drives the per-CodeObject JitCode body via
/// [`run_perfn_walk`] in authoritative mode AS the production trace — the
/// walk IS the concrete execution, so unlike the probe it keeps the
/// recorded trace.  Maps the walk outcome to a [`TraceAction`] for the
/// caller to compile.
///
/// Conservative mapping (first slice): only `CloseLoop` — the validated
/// end-to-end case (the four loop benches close under authoritative) — is
/// mapped to a real `CloseLoopWithArgs`; every other outcome (`Terminate`
/// finish-arg recovery, `SubReturn`/`SubRaise`, `SwitchToBlackhole`, any
/// `DispatchError`) aborts the trace so the portal falls back to the trait
/// tracer.  Default-off → the trait `metainterp.interpret` path is
/// untouched.  The remaining flip blocker is guard-snapshot/resume
/// correctness, which this harness exists to validate.
fn full_body_walk_trace(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
) -> TraceAction {
    // #125: decline up front when a loop body carries an `abort_permanent`
    // marker.  The authoritative walk would otherwise mis-seed the loop
    // guard, exit early, and concretely double-execute the post-loop tail;
    // routing to the trait tracer (which handles the unported op) is the
    // same outcome the reactive in-walk `abort_permanent` decline reaches,
    // minus the frame corruption.
    if loop_body_has_abort_permanent(w_code) {
        // Tag the decline so `PYRE_FBW_DEBUG_ABORT` census attributes it to the
        // up-front `abort_permanent` scan, not the trait retry fall-through
        // (`Trait::DeclinedAbort`).  Without this the real declining class is
        // invisible to the census.
        crate::jitcode_dispatch::census_record("FullBodyWalk::LoopBodyAbortPermanent");
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return TraceAction::Abort;
    }
    // Sibling defense to the above, transitively through inlined callees: a
    // non-journaled concrete store in the loop body followed by an
    // inline-eligible CALL whose callee body carries `abort_permanent` would
    // abort the walk, roll back the store journal, and replay from loop entry
    // — re-executing the non-journaled store.  Decline up front, before the
    // walk runs anything.  (See `loop_inlines_abort_permanent_callee`.)
    if loop_inlines_abort_permanent_callee(w_code, cf_addr) {
        crate::jitcode_dispatch::census_record("FullBodyWalk::CalleeAbortPermanent");
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return TraceAction::Abort;
    }
    // Mirror the trait path (trace_bytecode pre-interpret): register the
    // initial merge point with typed input-arg boxes so the trace head
    // carries the portal's entry signature (`inputarg_types()`).  Without
    // it the compiled loop's entry args don't match what the portal
    // supplies, so the portal cannot enter the compiled loop and re-traces
    // every iteration (the observed spin).
    // Clear the walk-local bool-box-truth map left by a prior aborted walk so
    // it cannot leak into this one.
    crate::jitcode_dispatch::bool_box_truth_reset();
    // Slice b (PYRE_FBW_CALL_ASSEMBLER): clear any Finish payload a prior
    // aborted walk's top-level `*_return` arm may have stashed, so a stale
    // value cannot leak into this walk's `Terminate` handling.
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_executed_nonpure_residual_reset();
    crate::jitcode_dispatch::fbw_executed_body_residual_reset();
    crate::jitcode_dispatch::fbw_abort_outer_resume_py_pc_reset();
    // Clear the prior walk's store journal + unjournaled-effect flag so
    // dropped (aborted) entries cannot be applied by this walk's commit.
    crate::jitcode_dispatch::fbw_store_journal_reset();
    // A bridge resumes mid-loop from a guard failure; its input args are the
    // guard's resumedata, already seeded into the bridge sym by
    // `setup_bridge_sym`.  PyPy's `interpret()` (rebuild_state_after_failure →
    // continue) registers NO merge point at the resume pc: the bridge walks
    // forward until it reaches an existing compiled loop header and closes as
    // a bridge there.  Registering a merge point at `start_pc` would instead
    // treat the resume pc as a fresh loop header (the portal entry signature),
    // which only a MAIN trace should do.  So skip it for bridges.
    if !ctx.is_bridge_trace {
        let start_key = crate::driver::make_green_key(w_code, start_pc);
        let input_types = ctx.inputarg_types();
        let input_args: Vec<majit_metainterp::GreenBox> = input_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| {
                majit_metainterp::GreenBox::new(majit_ir::OpRef::input_arg_typed(i as u32, tp), tp)
            })
            .collect();
        ctx.add_merge_point(start_key, input_args, start_pc);
    }
    let walk_result = run_perfn_walk(ctx, sym, w_code, start_pc, cf_addr, true);
    // A guard snapshot emitted during the walk may have hit a resume
    // coordinate the jitcode pc_map cannot encode (#124/#130) and requested
    // an abort (`state::request_trace_abort`).  The walker does not poll the
    // flag mid-walk, so honor it here before mapping the outcome — otherwise a
    // walk that reaches a terminator would compile a trace carrying the bad
    // guard.  Discarding the trace matches the trait leg's `interpret()` poll.
    if crate::state::take_trace_abort_requested() {
        crate::jitcode_dispatch::census_record("TraceAbortRequested");
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort] start_pc={start_pc} unencodable cross-frame resume coordinate (#124/#130)"
            );
        }
        return TraceAction::Abort;
    }
    if ctx.is_bridge_trace && std::env::var_os("PYRE_P2_DIAG").is_some() {
        ctx.dump_trace_ops_diag("carrier-root-walk-end");
    }
    match walk_result {
        Some((_entry, _code_len, Ok((outcome, _end_pc)))) => match outcome {
            crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                jump_args,
                loop_header_pc,
                ..
            } => {
                // Mirror trace_bytecode's post-interpret CloseLoop green-key
                // handling: a loop header other than start_pc retargets the
                // green key to the true merge point (cut-to-inner-loop);
                // start_pc closes at the trace head.
                if loop_header_pc != start_pc {
                    let target_key = crate::driver::make_green_key(w_code, loop_header_pc);
                    ctx.set_green_key(target_key, (w_code as usize, loop_header_pc));
                    ctx.header_pc = loop_header_pc;
                    ctx.cut_inner_green_key = Some(target_key);
                } else {
                    let key = crate::driver::make_green_key(w_code, start_pc);
                    ctx.set_green_key(key, (w_code as usize, start_pc));
                    ctx.header_pc = start_pc;
                }
                TraceAction::CloseLoopWithArgs {
                    jump_args,
                    loop_header_pc: Some(loop_header_pc),
                }
            }
            crate::jitcode_dispatch::DispatchOutcome::Terminate => {
                // A loop-free portal exit: the top-level `*_return` reached
                // `done_with_this_frame` with no back-edge.  Under the
                // PYRE_FBW_CALL_ASSEMBLER gate the return arm routed through
                // `fbw_terminate_with_finish`, which re-boxed the result to
                // Type::Ref, recorded the vable store-back + GUARD_NOT_FORCED_2,
                // and stashed the finish payload.  Build the portal-exit FINISH
                // from it so the compile pipeline records FINISH from
                // `finish_args` (mirror of the trait `StepResult::Return`
                // path, trace_opcode.rs).  Ungated → no payload → `Abort`
                // exactly as before the slice.
                match crate::jitcode_dispatch::fbw_finish_payload_take() {
                    // A top-level `void_return/` stashes a `Type::Void`-marked
                    // payload: the portal exits with no value, so build a
                    // FINISH with empty args.  The compile pipeline maps an
                    // empty `finish_arg_types` to `done_with_this_frame_descr_void`
                    // (pyjitpl.rs `done_with_this_frame_descr_from_types`),
                    // matching the trait tracer's `BC_VOID_RETURN` action.
                    Some((_, majit_ir::Type::Void)) => TraceAction::Finish {
                        finish_args: vec![],
                        finish_arg_types: vec![],
                        exit_with_exception: false,
                    },
                    Some((finish_value, finish_type)) => TraceAction::Finish {
                        finish_args: vec![finish_value],
                        finish_arg_types: vec![finish_type],
                        exit_with_exception: false,
                    },
                    None => {
                        crate::jitcode_dispatch::census_record("Terminate::NoFinishPayload");
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort] start_pc={start_pc} Terminate without finish payload (ungated portal exit)"
                            );
                        }
                        TraceAction::Abort
                    }
                }
            }
            crate::jitcode_dispatch::DispatchOutcome::CompileTracePending { .. } => {
                // pyjitpl.py:3095 raise_if_successful parity: the walker's
                // in-walk `compile_trace` already compiled+installed the
                // trace as a (entry) bridge jumping into an existing loop;
                // hand the dedicated action back so the driver neither
                // compiles nor aborts this session again — the trait-leg
                // equivalent is `trace_step_result_to_action`'s
                // `compile_trace_success_pending()` branch.
                TraceAction::CompileTrace
            }
            other => {
                crate::jitcode_dispatch::census_record("Outcome::Other");
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!("[fbw-abort] start_pc={start_pc} outcome={other:?}");
                }
                TraceAction::Abort
            }
        },
        Some((_entry, _code_len, Err(e))) => {
            // Structural walker limitations recur identically on every
            // retrace of this location (the same jitcode walked from the same
            // entry produces the same error), so route the key's retraces to
            // the trait tracer (`FBW_DECLINED_KEYS` → the trait leg of
            // `trace_bytecode`) instead of thrashing futile deep re-walks —
            // each of which executes the body's residual calls concretely
            // before failing at the unsupported resume / exception / closure
            // shape.  Permanently blacklisting (`AbortPermanent` →
            // `DONT_TRACE_HERE`) is wrong here: it leaves the location
            // interpreting forever (a try-protected raise in a hot loop
            // deopt-storms past any timeout), and upstream never marks a
            // location untraceable on an abort (pyjitpl.py:2392
            // aborted_tracing).  These are the multi-session-blocked
            // shapes (resume snapshot #124, exception-handler resume #51c,
            // closure NULL-self #60, unported raise marker, a residual
            // arg register the walk never binds); other errors retain the
            // plain `Abort` without declining so a capability that lands
            // mid-run can still pick the location up.
            use crate::jitcode_dispatch::DispatchError as DE;
            crate::jitcode_dispatch::census_record(e.variant_name());
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!("[fbw-abort] start_pc={start_pc} Err={e:?}");
            }
            match e {
                // A kept-stack branch guard whose not-taken arm reads an
                // unrestorable boxed Ref register miscompiles identically in
                // the trait leg, so re-routing there (the `fbw_decline` +
                // recoverable `Abort` path below) would crash there too.
                // Mark the location `DONT_TRACE_HERE` so it interprets
                // permanently — correct, matching the pre-#416/#420 decline.
                DE::BranchGuardUnrestorableKeptStackPermanent { .. } => TraceAction::AbortPermanent,
                // #57 (Finding #1): a non-journalable in-place container mutation
                // in a FOR_ITER body cannot be rolled back on abort, so this
                // location can never trace soundly — interpret it permanently
                // (the loop runs correctly under the interpreter).
                DE::InplaceContainerMutationUnsupported { .. } => TraceAction::AbortPermanent,
                DE::AbortPermanentMarkerReached { .. }
                | DE::GuardSnapshotVableUntyped { .. }
                | DE::MayForceNullRefArgUnsupported { .. }
                | DE::BranchGuardKeptStackUnsupported { .. }
                | DE::NonStandardVableFinishPortalUnsupported { .. }
                | DE::LoopBearingCalleeInlineUnsupported { .. }
                | DE::UnfoldableListAppendResidualUnsupported { .. }
                | DE::ResidualCallArgUnbound { .. } => {
                    fbw_decline(crate::driver::make_green_key(w_code, start_pc));
                    TraceAction::Abort
                }
                // #68 multiframe (`PYRE_FBW_INLINE_MULTIFRAME`): a data-dependent
                // `goto_if_not` whose branch input is not concrete at trace-time
                // recurs identically on every retrace of this entry (the same
                // jitcode walked from the same start_pc reaches the same
                // non-concrete branch operand).  Relaxing the inline predicate
                // lets a portal trace (e.g. a callee independently traced as its
                // own origin) walk PAST its prior `LoopBearing` decline and reach
                // such a branch, which would otherwise re-trace unbounded (each
                // re-walk executes the body's residual calls before failing) —
                // a slowdown worse than the trait leg.  Decline it permanently to
                // the trait leg, mirroring the default path's behavior for the
                // same location.  Gated on the flag so the default path's plain
                // `Abort` (a capability landing mid-run can still pick it up) is
                // byte-identical.
                DE::GotoIfNotValueNotConcrete { .. }
                    if crate::jitcode_dispatch::fbw_inline_multiframe_enabled() =>
                {
                    fbw_decline(crate::driver::make_green_key(w_code, start_pc));
                    TraceAction::Abort
                }
                _ => TraceAction::Abort,
            }
        }
        None => {
            crate::jitcode_dispatch::census_record("RunPerfnWalkNone");
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!("[fbw-abort] start_pc={start_pc} run_perfn_walk returned None");
            }
            TraceAction::Abort
        }
    }
}

/// Issue #73 walker-as-tracer foundation probe.
///
/// Dumps the per-CodeObject JitCode body that the walker-as-tracer must
/// walk for `miframe.pc == jitcode_pc` (the precondition for retiring
/// `pc_map`).  The per-CodeObject JitCode is built BEFORE this point by
/// `register_portal_jitdriver` → `make_jitcodes`
/// (`pyre/pyre-jit/src/eval.rs:3924`), so it is available here.
///
/// Read-only: logs the body op stream + entry offset, mutates nothing.
fn dump_perfn_jitcode_for_trace(w_code: *const (), start_pc: usize) {
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        eprintln!("[perfn-jitcode] no per-CodeObject PyJitCode for code={w_code:?}");
        return;
    };
    let code = pjc.jitcode.code.as_slice();
    let entry = pjc.resume_jitcode_pc_for(start_pc);
    eprintln!(
        "[perfn-jitcode] code_len={} pc_map_len={} start_pc={} entry_jitcode_pc={:?} \
         num_regs_r={} num_regs_i={} num_regs_f={} portal_frame_reg={} portal_ec_reg={} \
         built_as_portal={}",
        code.len(),
        pjc.metadata.first_jit_pc_by_py_pc.len(),
        start_pc,
        entry,
        pjc.jitcode.num_regs_r(),
        pjc.jitcode.num_regs_i(),
        pjc.jitcode.num_regs_f(),
        pjc.metadata.portal_frame_reg,
        pjc.metadata.portal_ec_reg,
        pjc.metadata.built_as_portal,
    );
    let cap = std::env::var("PYRE_DUMP_PERFN_JITCODE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 1)
        .unwrap_or(80);
    let mut count = 0usize;
    let mut last_next = 0usize;
    let mut histogram: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for op in crate::jitcode_runtime::decoded_ops(code) {
        if count < cap {
            eprintln!(
                "[perfn-jitcode]   pc={:>4} next={:>4} {}/{} bytes={:?}",
                op.pc,
                op.next_pc,
                op.opname,
                op.argcodes,
                &code[op.pc + 1..op.next_pc.min(code.len())]
            );
        }
        *histogram.entry(op.key.to_string()).or_default() += 1;
        count += 1;
        last_next = op.next_pc;
    }
    let clean = last_next == code.len();
    eprintln!("[perfn-jitcode] TOTAL ops={count} last_next={last_next} clean_eof={clean}");
    for (key, n) in &histogram {
        eprintln!("[perfn-jitcode] HIST {n:>4} {key}");
    }
    if !clean && last_next < code.len() {
        let stop_byte = code[last_next];
        eprintln!(
            "[perfn-jitcode] STOP at pc={last_next}: byte=0x{stop_byte:02x} opname={:?}",
            crate::jitcode_runtime::opname_for_byte(stop_byte),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::pyjitpl::semantic_fallthrough_pc;
    use pyre_interpreter::bytecode::Instruction;
    use pyre_interpreter::compile_exec;
    use pyre_interpreter::decode_instruction_at;

    #[test]
    fn test_semantic_fallthrough_pc_skips_branch_trivia() {
        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let branch_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::PopJumpIfFalse { .. }, _))
                )
            })
            .expect("test bytecode should contain POP_JUMP_IF_FALSE");

        let fallthrough_pc = semantic_fallthrough_pc(&code, branch_pc);
        let fallthrough_instruction = decode_instruction_at(&code, fallthrough_pc)
            .map(|(instruction, _)| instruction)
            .expect("semantic fallthrough should decode");

        assert!(
            !matches!(
                fallthrough_instruction,
                Instruction::ExtendedArg
                    | Instruction::Resume { .. }
                    | Instruction::Nop
                    | Instruction::Cache
                    | Instruction::NotTaken
            ),
            "semantic fallthrough must skip bytecode trivia"
        );
    }

    fn named_function_code(source: &str, name: &str) -> pyre_interpreter::CodeObject {
        fn find_in(
            code: &pyre_interpreter::CodeObject,
            name: &str,
        ) -> Option<pyre_interpreter::CodeObject> {
            for constant in code.constants.iter() {
                if let pyre_interpreter::ConstantData::Code { code: inner } = constant {
                    if inner.obj_name.as_str() == name {
                        return Some((**inner).clone());
                    }
                    if let Some(found) = find_in(inner, name) {
                        return Some(found);
                    }
                }
            }
            None
        }
        let module = compile_exec(source).expect("test code should compile");
        find_in(&module, name)
            .unwrap_or_else(|| panic!("test source should contain function {name}"))
    }

    #[test]
    fn start_pc_is_loop_header_detects_while_target() {
        let src = "def f(a, b):\n    if a <= 0:\n        total = 0\n        i = 0\n        while i < b:\n            total = total + i\n            i = i + 1\n        return total\n    return f(a - 1, b)\n";
        let code = named_function_code(src, "f");

        // Locate the loop header (the JumpBackward target).
        let mut arg_state = pyre_interpreter::OpArgState::default();
        let mut loop_header: Option<usize> = None;
        for (pc, unit) in code.instructions.iter().copied().enumerate() {
            let (instr, op_arg) = arg_state.get(unit);
            if let pyre_interpreter::Instruction::JumpBackward { delta }
            | pyre_interpreter::Instruction::JumpBackwardNoInterrupt { delta } = instr
            {
                loop_header = Some(pyre_interpreter::jump_target_backward_decoded(
                    &code,
                    pc + 1,
                    delta,
                    op_arg,
                ));
                break;
            }
        }
        let loop_header = loop_header.expect("the while loop must emit a JumpBackward");
        assert!(
            super::start_pc_is_loop_header(&code, loop_header),
            "the JumpBackward target must be recognized as a loop header"
        );
        assert!(
            !super::start_pc_is_loop_header(&code, 0),
            "function entry must not be recognized as the loop header"
        );
    }

    #[test]
    fn forward_exception_delivery_requires_exact_empty_handler_stack() {
        assert!(super::exception_delivery_stack_is_sourceable(0, 8, 7));
        assert!(!super::exception_delivery_stack_is_sourceable(1, 9, 7));
        assert!(!super::exception_delivery_stack_is_sourceable(0, 7, 7));
    }
}
