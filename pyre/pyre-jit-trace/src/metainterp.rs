//! PyreMetaInterp — RPython MetaInterp (pyjitpl.py:2371) parity.
//!
//! Single interpret() loop with a single framestack for both root
//! and inline frames. CALL → push_inline_frame (RPython perform_call),
//! RETURN → finishframe_inline (RPython finishframe).

use majit_ir::{OpRef, Type};
use majit_metainterp::{TraceAction, TraceCtx};
use pyre_interpreter::CodeObject;
use pyre_interpreter::bytecode::Instruction;

use super::state::{
    ConcreteValue, MIFrame, PendingInlineFrame, PyreSym, ResumeFrameState, wrapfloat, wrapint,
};

/// RPython MIFrame (pyjitpl.py:65) — per-frame tracing state.
pub struct MetaInterpFrame {
    pub sym: *mut PyreSym,
    pub owned_sym: Option<Box<PyreSym>>,
    pub jitcode: *const (),
    pub pc: usize,
    /// Raw `(code_ptr, target_pc)` greenkey components. Used for
    /// element-wise recursion-depth comparison in
    /// `step_root_frame` — see the filter at the
    /// `max_unroll_recursion` check. This is `Option<(usize, usize)>`
    /// rather than `Option<u64>` so the comparison cannot be fooled
    /// by hash collisions, matching
    /// rpython/jit/metainterp/pyjitpl.py:1396-1401 element-wise
    /// `same_constant` iteration.
    pub greenkey: Option<(usize, usize)>,
    pub concrete_frame: usize,
    /// Header-bearing heap frame, stable across Vec reallocs.
    pub owned_concrete_frame: Option<pyre_interpreter::pyframe::FrameBox>,
    /// opencoder.py:819-834: accumulated parent frame chain.
    pub parent_frames: Vec<ResumeFrameState>,
    pub drop_frame_opref: Option<OpRef>,
    pub caller_result_stack_idx: Option<usize>,
    pub caller_result_type: Option<Type>,
    pub arg_state: pyre_interpreter::bytecode::OpArgState,
    /// The CALL bytecode's pc on THIS frame at the moment it pushed an
    /// inline callee.  `top.pc` is advanced to the post-CALL fallthrough
    /// before the callee is pushed (TODO: differs from PyPy where
    /// `self.last_instr` stays at the CALL during the call); when the
    /// callee raises and `finishframe_exception` walks up to this frame,
    /// the exception-table lookup must probe the CALL site, not the
    /// post-call fallthrough, to mirror PyPy
    /// `pyopcode.handle_operation_error` reading `self.last_instr` from
    /// the still-at-CALL frame.  `None` when no inline call is in progress
    /// from this frame.
    pub call_site_pc: Option<usize>,
    /// Symbolic callable + arg OpRefs of the CALL that pushed this
    /// (inline) frame, carried from `PendingInlineFrame`. Consumed by
    /// the inline back-edge CALL_ASSEMBLER path (`do_recursive_call`)
    /// to shape the guard resume via `push_call_replay_stack`.
    /// `OpRef::NONE` / empty on root frames.
    pub replay_callable: OpRef,
    pub replay_args: Vec<OpRef>,
}

impl MetaInterpFrame {
    /// Root frame borrows sym from caller; inline frame owns its sym.
    fn is_inline(&self) -> bool {
        self.owned_sym.is_some()
    }

    fn concrete_frame_addr(&self) -> usize {
        if let Some(ref cf) = self.owned_concrete_frame {
            &**cf as *const pyre_interpreter::pyframe::PyFrame as usize
        } else {
            self.concrete_frame
        }
    }
}

/// RPython MetaInterp (pyjitpl.py:2371).
pub struct PyreMetaInterp {
    pub framestack: Vec<MetaInterpFrame>,
    pub portal_call_depth: i32,
    pub jitcode: *const (),
    pub namespace: *mut pyre_interpreter::DictStorage,
    inline_trace_base: usize,
}

impl PyreMetaInterp {
    pub fn new(jitcode: *const (), namespace: *mut pyre_interpreter::DictStorage) -> Self {
        Self {
            framestack: Vec::new(),
            portal_call_depth: -1,
            jitcode,
            namespace,
            inline_trace_base: 0,
        }
    }

    /// RPython MetaInterp._interpret() — single loop, single framestack.
    pub fn interpret(&mut self, ctx: &mut TraceCtx) -> TraceAction {
        self.inline_trace_base = ctx.inline_trace_depth();

        loop {
            let Some(top) = self.framestack.last_mut() else {
                return TraceAction::Abort;
            };
            let code = unsafe {
                &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject)
            };
            // RPython _interpret() parity: all frames use cf.next_instr as PC.
            // Skip trivia (Cache, ExtendedArg, etc.) — RPython's trace loop
            // only processes real opcodes. Inline frame returns or branch
            // targets can land on Cache instructions; skip forward to the
            // next semantic opcode so orgpc is always a real opcode PC.
            let mut pc = top
                .owned_concrete_frame
                .as_ref()
                .map(|cf| cf.next_instr())
                .unwrap_or(top.pc);
            if pc >= code.instructions.len() {
                return TraceAction::Abort;
            }
            // Advance past Cache/ExtendedArg/Nop/NotTaken/Resume trivia.
            while pc < code.instructions.len() {
                match pyre_interpreter::decode_instruction_at(code, pc) {
                    Some((
                        pyre_interpreter::bytecode::Instruction::Cache
                        | pyre_interpreter::bytecode::Instruction::ExtendedArg
                        | pyre_interpreter::bytecode::Instruction::Nop
                        | pyre_interpreter::bytecode::Instruction::NotTaken,
                        _,
                    )) => {
                        pc += 1;
                        // Sync the concrete frame so subsequent steps see the
                        // corrected PC.
                        if let Some(ref mut cf) = top.owned_concrete_frame {
                            cf.set_last_instr_from_next_instr(pc);
                        }
                        top.pc = pc;
                    }
                    _ => break,
                }
            }
            if pc >= code.instructions.len() {
                return TraceAction::Abort;
            }

            let action = if top.is_inline() {
                self.step_inline_frame(ctx, pc)
            } else {
                self.step_root_frame(ctx, pc)
            };

            match action {
                LoopAction::Continue => {}
                LoopAction::Return(ta) => return ta,
            }
            // pyjitpl.py:2837-2843 _interpret parity:
            //     while True:
            //         self.framestack[-1].run_one_step()
            //         self.blackhole_if_trace_too_long()
            //
            // The full `blackhole_if_trace_too_long` bookkeeping
            // (find_biggest_function / disable_noninlinable_function /
            // prepare_trace_segmenting) lives on `MetaInterp` and runs
            // in the jitdriver when it sees `TraceAction::Abort`. Here
            // we only mirror the trigger: stop dispatch the moment the
            // recorder exceeds `trace_limit`, which RPython signals via
            // `raise SwitchToBlackhole(ABORT_TOO_LONG)` inside
            // `blackhole_if_trace_too_long`.
            if ctx.is_too_long() {
                return TraceAction::Abort;
            }
        }
    }

    // ── Root frame step ──────────────────────────────────────────

    fn step_root_frame(&mut self, ctx: &mut TraceCtx, pc: usize) -> LoopAction {
        let top = self.framestack.last_mut().unwrap();
        let code = unsafe {
            &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject)
        };
        let sym = unsafe { &mut *top.sym };
        let cf_addr = top.concrete_frame_addr();
        let fallthrough_pc = semantic_fallthrough_pc(code, pc);

        let mut fs = MIFrame::from_sym(ctx, sym, cf_addr, fallthrough_pc, pc);
        let action = fs.trace_code_step(code, pc);
        let pending = fs.pending_inline_frame.take();
        drop(fs);

        if let Some(pending) = pending {
            // pyjitpl.py:1415 `self.metainterp.perform_call(portal_code,
            // allboxes)` — push callee frame onto framestack, trace
            // into callee. The recursion-limit check lives at the
            // decision site (trace_opcode.rs `_opimpl_recursive_call`
            // parity block); over-limit callees never produce a
            // pending here.
            let top = self.framestack.last_mut().unwrap();
            let sym = unsafe { &mut *top.sym };
            let next_pc = sym
                .pending_next_instr
                .take()
                .unwrap_or_else(|| semantic_fallthrough_pc(code, pc));
            // Save the CALL-site pc on the caller before advancing
            // `top.pc` past it.  `finishframe_exception` uses this when an
            // inline callee raises, so the caller's exception_table is
            // probed at the CALL bytecode (matching PyPy
            // `pyopcode.handle_operation_error` reading `self.last_instr`).
            top.call_site_pc = Some(pc);
            top.pc = next_pc;
            let result_idx = sym
                .valuestackdepth
                .saturating_sub(sym.nlocals)
                .checked_sub(1);
            if let Some(ref mut cf) = top.owned_concrete_frame {
                cf.set_last_instr_from_next_instr(next_pc);
            }
            self.push_inline_frame(ctx, pending, result_idx);
            return LoopAction::Continue;
        }

        match action {
            TraceAction::Continue => {
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                let next_pc = sym
                    .pending_next_instr
                    .take()
                    .unwrap_or_else(|| semantic_fallthrough_pc(code, pc));
                top.pc = next_pc;
                // Sync concrete frame PC with symbolic PC.
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    cf.set_last_instr_from_next_instr(next_pc);
                }
                LoopAction::Continue
            }
            TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
                // pyjitpl.py:2970-2975 reached_loop_header:
                //     ptoken = self.get_procedure_token(greenboxes)
                //     if has_compiled_targets(ptoken):
                //         self.compile_trace(...)   # close the trace
                //     else:
                //         return  # continue tracing past this merge point
                //
                // Bridge traces skip loop headers that don't have compiled
                // code, continuing until they reach a COMPILED loop's header.
                // Without this, nested loop bridges close at the outer loop's
                // header instead of the inner loop's.
                // pyjitpl.py:2978 `if not self.partial_trace:` — only
                // bridge traces apply the close-loop skip when no
                // compiled target exists for the current greenkey.
                // Primary traces always close their first complete
                // iteration regardless of has_compiled_targets, so
                // gate on the explicit `is_bridge_trace` flag rather
                // than `has_compiled_targets_fn` presence (which is
                // installed at primary entry too as the auto-stamp
                // gate input).
                if ctx.is_bridge_trace {
                    if let Some(ref has_compiled) = ctx.has_compiled_targets_fn {
                        // pyjitpl.py:2979: ptoken = self.get_procedure_token(greenboxes)
                        // greenboxes = (code, pc) for the CURRENT frame's loop header.
                        let w_code = self.framestack.last().unwrap().jitcode;
                        let target_pc = match &action {
                            TraceAction::CloseLoopWithArgs {
                                loop_header_pc: Some(tp),
                                ..
                            } => *tp,
                            _ => pc,
                        };
                        let gk = crate::driver::make_green_key(w_code, target_pc);
                        if !has_compiled(gk) {
                            // No compiled targets for this loop — continue tracing.
                            // RPython: reached_loop_header returns without closing.
                            let top = self.framestack.last_mut().unwrap();
                            top.pc = target_pc;
                            if let Some(ref mut cf) = top.owned_concrete_frame {
                                cf.set_last_instr_from_next_instr(target_pc);
                            }
                            return LoopAction::Continue;
                        }
                    }
                }
                self.handle_close_loop(ctx, &action, pc);
                LoopAction::Return(action)
            }
            other => LoopAction::Return(other),
        }
    }

    // ── Inline frame step ────────────────────────────────────────

    fn step_inline_frame(&mut self, ctx: &mut TraceCtx, pc: usize) -> LoopAction {
        let top = self.framestack.last_mut().unwrap();
        let code = unsafe {
            &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject)
        };
        let sym = unsafe { &mut *top.sym };
        let cf_addr = top.concrete_frame_addr();
        let parent_frames = top.parent_frames.clone();
        let fallthrough_pc = semantic_fallthrough_pc(code, pc);

        let inline_action = {
            let mut fs = MIFrame::from_sym(ctx, sym, cf_addr, fallthrough_pc, pc);
            fs.parent_frames = parent_frames;
            let result = fs.trace_code_step_inline(code, pc);
            let pending = fs.pending_inline_frame.take();
            drop(fs);

            if let Some(pending) = pending {
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                let sfall = semantic_fallthrough_pc(code, pc);
                top.call_site_pc = Some(pc);
                top.pc = sym.pending_next_instr.take().unwrap_or(sfall);
                let result_idx = sym
                    .valuestackdepth
                    .saturating_sub(sym.nlocals)
                    .checked_sub(1);
                // Pop concrete call args from parent owned_concrete_frame
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    for _ in 0..pending.nargs {
                        cf.pop();
                    }
                    let _ = cf.pop(); // null_or_self
                    let _ = cf.pop(); // callable
                    cf.set_last_instr_from_next_instr(sfall);
                }
                self.push_inline_frame(ctx, pending, result_idx);
                return LoopAction::Continue;
            }
            result
        };

        match inline_action {
            super::state::InlineTraceStepAction::Trace(TraceAction::Continue) => {
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                top.pc = sym
                    .pending_next_instr
                    .take()
                    .unwrap_or_else(|| semantic_fallthrough_pc(code, pc));
                // Concrete execution step
                self.concrete_execute_step();
                // Fused-compare concrete catch-up. A fused COMPARE+POP_JUMP
                // (try_fused_compare_goto_if_not) records one branch guard and
                // advances the symbolic pc PAST the POP_JUMP in a single trace
                // step, but the concrete_execute_step above ran only the
                // COMPARE. The inline-frame pc is read from the concrete
                // frame's next_instr (interpret()), so leaving the concrete
                // frame at the POP_JUMP re-steps the already-consumed branch
                // (stack underflow during trace opcode → graceful abort,
                // blocking inline tracing through the compare). Drive the
                // concrete frame through the POP_JUMP — its concrete branch
                // lands at the same `next_target` the fused step chose — until
                // it reaches the symbolic pc. Trivia is positioned-past as
                // interpret()'s prelude does (Cache/ExtendedArg/Nop/NotTaken,
                // NOT Resume); only a real opcode strictly before the target
                // (the fused POP_JUMP) is executed.
                //
                // Gated to NON-recursive inline frames. The catch-up is
                // correct on its own (a non-recursive inlined callee with a
                // compare traces cleanly with it), but letting a *recursive*
                // inline chain proceed past the compare exposes a separate,
                // pre-existing bug in the recursion-unroll machinery (the
                // compiled trace fails to terminate). Until that is fixed,
                // recursive inline frames keep the old behavior — the fused
                // compare desyncs and the inline trace aborts gracefully, so
                // recursion still JITs via the CALL_ASSEMBLER self-recursion
                // path. `jitcode` appearing 2+ times on the framestack marks a
                // recursive chain.
                let top_jitcode = self.framestack.last().unwrap().jitcode;
                let is_recursive_chain = self
                    .framestack
                    .iter()
                    .filter(|f| f.jitcode == top_jitcode)
                    .count()
                    >= 2;
                // Skip the catch-up while any frame in the inline chain is
                // executing inside an active exception-handler (try) range. A
                // fused-compare catch-up lets an inlined callee proceed in the
                // trace; if a later op in that callee raises, the exception
                // unwinds into a CALLER's inlined except handler, and the
                // blackhole resume of an inlined-callee raise into a caller
                // catch-landing is malformed (the catch_exception target skips
                // the landing's last_exc_value op, so the bound exception value
                // reads NULL). Until that resume path is fixed, keep the prior
                // graceful-abort behavior for these: the inline trace aborts,
                // the callee stays a residual call, and the exception path runs
                // correctly. Hot loops without active try blocks (the catch-up's
                // intended target) are unaffected.
                //
                // TODO: fix the underlying blackhole/codewriter defect so this
                // gate can be removed and try-protected callees inline too. When
                // an inlined callee raises into a caller's inlined except, the
                // resume catch-landing's `catch_exception` target resolves PAST
                // the landing's `last_exc_value` op (op 90 never executes), so
                // the bound exception value reads NULL. RPython sidesteps this
                // because the raising block carries `exitswitch=c_last_exception`
                // (its exception edges are non-trivial graph links); pyre models
                // the catch as a byte-level `catch_exception/L`, NOT a graph
                // link, so the trivial-link / inline-resume passes skip it. The
                // proper fix emits/positions `last_exc_value` so the inlined
                // catch target lands on it, restoring inlining for these.
                let in_exception_handler = self.framestack.iter().any(|f| {
                    let Some(ref cf) = f.owned_concrete_frame else {
                        return false;
                    };
                    let ccode = unsafe { &*pyre_interpreter::pyframe_get_pycode(&**cf) };
                    // A suspended caller's concrete `next_instr` was advanced
                    // past the CALL before its callee was pushed, so probe the
                    // saved CALL site (`call_site_pc`) — mirroring
                    // `finishframe_exception`. Frames with no outstanding inline
                    // call fall back to `next_instr`. exceptiontable offsets are
                    // byte offsets; the pc is an instruction-word index (×2).
                    let lookup_pc = f.call_site_pc.unwrap_or_else(|| cf.next_instr());
                    let byte_off = (lookup_pc as u32).saturating_mul(2);
                    pyre_interpreter::exception_table::lookup_exceptiontable(
                        &ccode.exceptiontable,
                        byte_off,
                    )
                    .is_some()
                });
                if !is_recursive_chain && !in_exception_handler {
                    let mut guard = 0u32;
                    loop {
                        let top = self.framestack.last_mut().unwrap();
                        let target = top.pc;
                        let Some(cf) = top.owned_concrete_frame.as_mut() else {
                            break;
                        };
                        let ccode = unsafe { &*pyre_interpreter::pyframe_get_pycode(&**cf) };
                        let mut ni = cf.next_instr();
                        while ni < target {
                            match pyre_interpreter::decode_instruction_at(ccode, ni) {
                                Some((
                                    Instruction::Cache
                                    | Instruction::ExtendedArg
                                    | Instruction::Nop
                                    | Instruction::NotTaken,
                                    _,
                                )) => ni += 1,
                                _ => break,
                            }
                        }
                        cf.set_last_instr_from_next_instr(ni);
                        if ni >= target || guard >= 16 {
                            break;
                        }
                        self.concrete_execute_step();
                        guard += 1;
                    }
                }
                LoopAction::Continue
            }
            super::state::InlineTraceStepAction::Trace(TraceAction::Finish {
                finish_args,
                finish_arg_types,
                // pyjitpl.py:2489-2502 finishframe path: inline return/yield
                // never raises through this branch (exception exits take the
                // finishframe_exception route above), so the flag is ignored.
                exit_with_exception: _,
            }) => {
                self.finishframe_inline(ctx, &finish_args, &finish_arg_types);
                LoopAction::Continue
            }
            super::state::InlineTraceStepAction::Trace(
                TraceAction::Abort | TraceAction::AbortPermanent,
            ) => {
                // RPython finishframe_exception: if exception pending,
                // try multi-frame unwind before giving up.
                let has_exc = {
                    let top = self.framestack.last().unwrap();
                    let sym = unsafe { &*top.sym };
                    // pyjitpl.py:2506 finishframe_exception is reached
                    // only after opimpl_raise/handle_possible_exception has
                    // populated both last_exc_value and last_exc_box.  A
                    // concrete-only exception without a box cannot be encoded
                    // as compile_exit_frame_with_exception(valuebox).
                    !sym.last_exc_value.is_null() && sym.last_exc_box != OpRef::NONE
                };
                if has_exc {
                    // Single-frame finishframe_exception (called from the
                    // inline trace step) already consumed any saved
                    // reraise_lasti for the RERAISE-issuing frame (handler
                    // push or no-handler last_instr restoration).  Multi-
                    // frame escalation walks remaining frames whose handler
                    // dispatch carries no lasti — match PyPy's
                    // `pyopcode.handle_operation_error(reraise_lasti=-1)`
                    // default.
                    if let Some(action) = self.finishframe_exception(ctx, -1) {
                        return action;
                    }
                }
                ctx.truncate_inline_trace_positions(self.inline_trace_base);
                LoopAction::Return(TraceAction::Abort)
            }
            super::state::InlineTraceStepAction::Trace(TraceAction::RecursiveCallAssembler {
                green_key,
                target_pc,
            }) => self.opimpl_recursive_call_assembler(ctx, green_key, target_pc),
            super::state::InlineTraceStepAction::Trace(action) => LoopAction::Return(action),
            super::state::InlineTraceStepAction::PushFrame(pending) => {
                // trace_code_step_inline already took the pending frame
                let top = self.framestack.last_mut().unwrap();
                let code = unsafe {
                    &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                        as *const pyre_interpreter::CodeObject)
                };
                let sym = unsafe { &mut *top.sym };
                let sfall = semantic_fallthrough_pc(code, pc);
                top.call_site_pc = Some(pc);
                top.pc = sym.pending_next_instr.take().unwrap_or(sfall);
                let result_idx = sym
                    .valuestackdepth
                    .saturating_sub(sym.nlocals)
                    .checked_sub(1);
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    for _ in 0..pending.nargs {
                        cf.pop();
                    }
                    let _ = cf.pop(); // null_or_self
                    let _ = cf.pop(); // callable
                    cf.set_last_instr_from_next_instr(sfall);
                }
                self.push_inline_frame(ctx, pending, result_idx);
                LoopAction::Continue
            }
        }
    }

    // ── Frame management ─────────────────────────────────────────

    /// TODO: equivalent to RPython
    /// `MetaInterp.perform_call` (`rpython/jit/metainterp/pyjitpl.py`)
    /// which constructs and pushes the callee `MIFrame` directly.  Pyre
    /// splits trace-step output (returns `PendingInlineFrame`) from the
    /// frame-stack mutation (this method) so the trace handler stays
    /// borrow-isolated from the metainterp framestack.  The
    /// `PendingInlineFrame` struct itself has no upstream counterpart.
    pub(crate) fn push_inline_frame(
        &mut self,
        ctx: &mut TraceCtx,
        mut pending: PendingInlineFrame,
        caller_result_idx: Option<usize>,
    ) {
        let (driver, _) = crate::driver::driver_pair();
        driver.enter_inline_frame(pending.green_key_raw);
        ctx.push_inline_trace_position(pending.green_key);

        // `pyjitpl.py:181-193` analogue: tag the most recent parent frame
        // (the caller that just traced this call instruction) with the
        // stack slot where the result will land, so a guard captured
        // inside the callee can clear that slot before snapshotting.
        if let Some(parent) = pending.parent_frames.first_mut() {
            parent.pending_result_stack_idx = caller_result_idx;
            parent.pending_result_type = pending.caller_result_type;
        }

        let callee_code = pending.concrete_frame.pycode;
        let mut owned_sym = Box::new(pending.sym);
        let sym_ptr = owned_sym.as_mut() as *mut PyreSym;
        let owned_cf = pyre_interpreter::pyframe::FrameBox::new(pending.concrete_frame);
        let cf_addr = &*owned_cf as *const pyre_interpreter::pyframe::PyFrame as usize;

        // executioncontext.py:76 / pyjitpl.py:1789: virtual_ref for callee frame.
        if let Some(frame_opref) = pending.drop_frame_opref {
            if let Some(caller) = self.framestack.last() {
                let caller_sym = unsafe { &mut *caller.sym };
                super::state::opimpl_virtual_ref(ctx, caller_sym, frame_opref, cf_addr);
            }
        }

        let frame = MetaInterpFrame {
            sym: sym_ptr,
            owned_sym: Some(owned_sym),
            jitcode: callee_code,
            pc: 0,
            greenkey: Some(pending.green_key_raw),
            concrete_frame: cf_addr,
            owned_concrete_frame: Some(owned_cf),
            parent_frames: pending.parent_frames,
            drop_frame_opref: pending.drop_frame_opref,
            caller_result_stack_idx: caller_result_idx,
            caller_result_type: pending.caller_result_type,
            arg_state: pyre_interpreter::bytecode::OpArgState::default(),
            // Freshly pushed inline frames have no outstanding call of
            // their own; cleared when this frame becomes a caller via a
            // subsequent inline push.
            call_site_pc: None,
            replay_callable: pending.replay_callable,
            replay_args: pending.replay_args,
        };

        self.portal_call_depth += 1;
        self.framestack.push(frame);
    }

    /// RPython finishframe: pop inline frame, store result in parent.
    fn finishframe_inline(
        &mut self,
        ctx: &mut TraceCtx,
        finish_args: &[OpRef],
        finish_arg_types: &[Type],
    ) {
        let result_opref = finish_args.first().copied().unwrap_or(OpRef::NONE);
        let result_type = finish_arg_types.first().copied().unwrap_or(Type::Ref);

        // Concrete RETURN_VALUE execution on inline frame
        let concrete_result = self.concrete_execute_return();

        // Pop frame
        let popped = self.framestack.pop().unwrap();
        self.portal_call_depth -= 1;
        ctx.pop_inline_trace_position();

        // executioncontext.py:89 / pyjitpl.py:1819: virtual_ref_finish for callee frame.
        if let Some(frame_opref) = popped.drop_frame_opref {
            if let Some(parent) = self.framestack.last() {
                let parent_sym = unsafe { &mut *parent.sym };
                super::state::opimpl_virtual_ref_finish(ctx, parent_sym, frame_opref);
            }
        }

        // Drop callee frame in trace
        if let Some(frame_opref) = popped.drop_frame_opref {
            ctx.call_void(
                crate::callbacks::get().jit_drop_callee_frame,
                &[frame_opref],
            );
        }

        let (driver, _) = crate::driver::driver_pair();
        driver.leave_inline_frame();

        if self.framestack.is_empty() {
            return; // shouldn't happen: root never produces Finish
        }

        // make_result_of_lastop: store in parent
        let parent = self.framestack.last_mut().unwrap();
        // The inline callee returned successfully — clear the saved CALL
        // pc on the parent.  Mirrors PyPy `pyopcode.handle_bytecode`: after
        // a normal return, `self.last_instr` advances past the call so a
        // subsequent exception dispatched from this frame looks up the
        // table at the post-call instruction, not the (now-consumed) CALL.
        parent.call_site_pc = None;
        let parent_sym = unsafe { &mut *parent.sym };

        if let Some(result_idx) = popped.caller_result_stack_idx {
            // pyjitpl.py:258-275 `make_result_of_lastop` dispatches on
            // `resultbox.type`. Pyre's Python operand stack is a W_Root
            // array (`virtualizable.py:86-98`), so scalar results are
            // explicitly boxed before the parent stack slot is updated.
            let boxed = match result_type {
                Type::Int => wrapint(ctx, result_opref),
                Type::Float => wrapfloat(ctx, result_opref),
                Type::Ref => result_opref,
                Type::Void => {
                    debug_assert!(
                        false,
                        "void inline return cannot write caller_result_stack_idx"
                    );
                    return;
                }
            };
            let cv = if concrete_result.is_null() {
                ConcreteValue::Null
            } else {
                ConcreteValue::Ref(concrete_result)
            };
            super::trace_opcode::write_stack_slot(parent_sym, ctx, result_idx, boxed, cv);
        }

        // Keep concrete stack in sync only for inline parents.
        // Root frames are traced symbolically against a snapshot and are
        // not concrete-stepped instruction-by-instruction.
        if parent.is_inline() {
            if let Some(ref mut pcf) = parent.owned_concrete_frame {
                pcf.push(concrete_result);
            }
        }

        // No pending_concrete_push needed — concrete_stack[result_idx] already updated above.
    }

    /// pyjitpl.py:1565-1602 opimpl_jit_merge_point portal_call_depth>0:
    /// the inline callee reached its loop back-edge and the loop's green
    /// key has compiled code. Pop the inline frame without a result
    /// (finishframe(None, leave_portal_frame=False)), record a
    /// CALL_ASSEMBLER into the loop token from the parent frame
    /// (do_recursive_call assembler_call=True), bind the call result to
    /// the parent's pending result slot (make_result_of_lastop,
    /// pyjitpl.py:1586-1589), then resume dispatch on the parent
    /// (ChangeFrame).
    ///
    /// RPython's do_residual_call executes the portal runner concretely
    /// at this point (the callee's remaining iterations run during
    /// tracing). Pyre traces on a snapshot and re-runs the closed loop's
    /// iteration on restart, so the callee's remaining execution is
    /// DEFERRED to the restart — the same contract as deferred
    /// StoreSubscr/StoreAttr. The result's concrete half is therefore
    /// Null; a downstream consumer that needs it fails the
    /// missing-concrete checks and aborts the trace gracefully.
    /// Convergence path: trace on the live frame (no re-run), then the
    /// concrete continuation can run here as upstream does.
    fn opimpl_recursive_call_assembler(
        &mut self,
        ctx: &mut TraceCtx,
        green_key: u64,
        target_pc: usize,
    ) -> LoopAction {
        let token_number = {
            let (driver, _) = crate::driver::driver_pair();
            driver.get_loop_token_number(green_key)
        };
        // Re-verify with framestack-level data the close_loop_args signal
        // site cannot see: the callee must own a trace-visible frame
        // object (CALL_ASSEMBLER red arg) and carry the CALL-site OpRefs
        // for the guard-resume replay; the parent must be the root frame
        // with a recorded CALL site.
        let eligible = token_number.is_some()
            && self.framestack.len() == 2
            && {
                let top = self.framestack.last().unwrap();
                top.drop_frame_opref.is_some() && top.replay_callable != OpRef::NONE
            }
            && self.framestack[0].call_site_pc.is_some();
        if !eligible {
            // Fall back to the inline-loop-abort end state (the
            // trace_step_result_to_action drain): stop tracing the root
            // key so the callee loop compiles standalone.
            let root_green_key = ctx.root_green_key();
            let (driver, _) = crate::driver::driver_pair();
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .disable_noninlinable_function(root_green_key);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][recursive-call-assembler] ineligible, abort: key={} target_pc={} depth={} root={}",
                    green_key,
                    target_pc,
                    self.framestack.len(),
                    root_green_key,
                );
            }
            ctx.truncate_inline_trace_positions(self.inline_trace_base);
            return LoopAction::Return(TraceAction::Abort);
        }
        let token_number = token_number.unwrap();

        // Virtualizable write_boxes on the callee frame (pre-pop): the
        // compiled loop reads locals_cells_stack_w / last_instr /
        // valuestackdepth from the frame object at entry.
        {
            let top = self.framestack.last_mut().unwrap();
            let frame_opref = top.drop_frame_opref.unwrap();
            let sym = unsafe { &mut *top.sym };
            let vsd = top
                .owned_concrete_frame
                .as_ref()
                .map(|cf| cf.valuestackdepth)
                .unwrap_or(sym.valuestackdepth)
                .max(sym.nlocals);
            super::trace_opcode::gen_writeback_inline_frame_to_heap(
                ctx,
                sym,
                frame_opref,
                target_pc,
                vsd,
            );
        }

        // pyjitpl.py:1581 finishframe(None, leave_portal_frame=False).
        let popped = self.framestack.pop().unwrap();
        self.portal_call_depth -= 1;
        ctx.pop_inline_trace_position();
        {
            let (driver, _) = crate::driver::driver_pair();
            driver.leave_inline_frame();
        }
        let frame_opref = popped.drop_frame_opref.unwrap();

        // pyjitpl.py:1590 frame.do_recursive_call(assembler_call=True) on
        // the parent frame.
        let parent = self.framestack.last_mut().unwrap();
        let call_pc = parent.call_site_pc.take().unwrap();
        let code = unsafe {
            &*(pyre_interpreter::w_code_get_ptr(parent.jitcode as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject)
        };
        let cf_addr = parent.concrete_frame_addr();
        let parent_frames = parent.parent_frames.clone();
        let parent_sym = unsafe { &mut *parent.sym };
        let fallthrough_pc = semantic_fallthrough_pc(code, call_pc);
        let ca_result = {
            let mut fs = MIFrame::from_sym(ctx, parent_sym, cf_addr, fallthrough_pc, call_pc);
            fs.parent_frames = parent_frames;
            fs.do_recursive_call_assembler(
                token_number,
                frame_opref,
                popped.replay_callable,
                &popped.replay_args,
                call_pc,
            )
        };

        // pyjitpl.py:1592 leave_portal_frame — after the call, mirroring
        // finishframe_inline's vref close + callee-frame drop.
        let parent_sym = unsafe { &mut *self.framestack.last().unwrap().sym };
        super::state::opimpl_virtual_ref_finish(ctx, parent_sym, frame_opref);
        ctx.call_void(
            crate::callbacks::get().jit_drop_callee_frame,
            &[frame_opref],
        );

        match ca_result {
            Ok(result) => {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][recursive-call-assembler] emitted: key={} target_pc={} token={} call_pc={}",
                        green_key, target_pc, token_number, call_pc,
                    );
                }
                // make_result_of_lastop (pyjitpl.py:1586-1589). Concrete
                // half stays Null — see the method doc.
                if let Some(result_idx) = popped.caller_result_stack_idx {
                    super::trace_opcode::write_stack_slot(
                        parent_sym,
                        ctx,
                        result_idx,
                        result,
                        ConcreteValue::Null,
                    );
                }
                // ChangeFrame: resume dispatch on the parent.
                LoopAction::Continue
            }
            Err(_) => {
                ctx.truncate_inline_trace_positions(self.inline_trace_base);
                LoopAction::Return(TraceAction::Abort)
            }
        }
    }

    // ── Concrete execution helpers ───────────────────────────────

    /// Concretely execute the inline frame's next opcode on its
    /// `owned_concrete_frame` (the trace snapshot).
    ///
    /// KNOWN DIVERGENCE from RPython `execute_and_record` (live miscompile,
    /// see memory `cf-executor-into-walker-epic-2026-06-08`): this runs ONLY
    /// for inline frames (called from `step_inline_frame`); `step_root_frame`
    /// never concrete-steps.  For a residual SHARED-heap STORE (STORE_GLOBAL /
    /// STORE_SUBSCR / LIST_APPEND on a caller-shared object) `execute_opcode_step`
    /// mutates the live heap during recording, because `snapshot_for_tracing`
    /// shares `w_globals_obj` (pyframe.rs).  The compiled loop then re-runs the
    /// traced iteration from the loop header (the real frame never advanced),
    /// re-applying the store → one-time over-commit scaling with inline depth.
    /// RPython advances the single real frame as it records (so the store
    /// commits once and the compiled loop resumes AFTER the traced iterations),
    /// and blackhole-FORWARDS on abort instead of re-running.  Fix = the
    /// executor-into-walker cutover (Task #14); gate
    /// `pyre/bench/synth/inlined_callee_globals.py`.
    fn concrete_execute_step(&mut self) {
        let top = self.framestack.last_mut().unwrap();
        let Some(cf) = top.owned_concrete_frame.as_mut() else {
            return;
        };
        let cf = &mut **cf;
        let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(cf) };
        let ni = cf.next_instr();
        if ni >= code.instructions.len() {
            return;
        }

        let (opcode_pc, instruction, op_arg) =
            pyre_interpreter::decode_instruction_for_dispatch(code, ni)
                .expect("concrete_execute_step: bytecode corruption");
        cf.set_last_instr_from_next_instr(opcode_pc + 1);
        let next = cf.next_instr();

        // issue #143 (inline-frame double-mutation): defer subscript/attribute
        // stores instead of committing them to the shared heap here. This
        // speculative concrete step runs inside an inline callee frame whose
        // loop back-edge is aborted (close_loop_args inline gate); a store
        // committed here persists past the abort and is re-applied when the
        // interpreter re-runs the callee → wrong result. The root tracer
        // already defers these stores (trace_store_subscr emits IR only,
        // leaving its snapshot frame's concrete view stale until re-execution),
        // so mirroring that here keeps the two paths consistent — only pop the
        // operands to preserve the frame's stack.
        match instruction {
            Instruction::StoreSubscr => {
                cf.pop();
                cf.pop();
                cf.pop();
                return;
            }
            Instruction::StoreAttr { .. } => {
                cf.pop();
                cf.pop();
                return;
            }
            _ => {}
        }

        if let Instruction::Call { argc } = instruction {
            let nargs = argc.get(op_arg) as usize;
            let _ = super::state::execute_inline_residual_call(cf, nargs);
            return;
        }

        let _ = pyre_interpreter::execute_opcode_step(cf, code, instruction, op_arg, next);
    }

    fn concrete_execute_return(&mut self) -> pyre_object::PyObjectRef {
        let top = self.framestack.last_mut().unwrap();
        let Some(cf) = top.owned_concrete_frame.as_mut() else {
            return pyre_object::PY_NULL;
        };
        let cf = &mut **cf;
        let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(cf) };
        let ni = cf.next_instr();
        if ni >= code.instructions.len() {
            return pyre_object::PY_NULL;
        }

        let (opcode_pc, instruction, op_arg) =
            pyre_interpreter::decode_instruction_for_dispatch(code, ni)
                .expect("concrete_execute_return: bytecode corruption");
        cf.set_last_instr_from_next_instr(opcode_pc + 1);
        let next = cf.next_instr();

        match pyre_interpreter::execute_opcode_step(cf, code, instruction, op_arg, next) {
            Ok(pyre_interpreter::StepResult::Return(value)) => value,
            _ => pyre_object::PY_NULL,
        }
    }

    // ── Helpers ──────────────────────────────────────────────────

    fn handle_close_loop(&self, ctx: &mut TraceCtx, action: &TraceAction, pc: usize) {
        let w_code = self.framestack[0].jitcode;
        let target_pc = if let TraceAction::CloseLoopWithArgs {
            loop_header_pc: Some(tp),
            ..
        } = action
        {
            Some(*tp)
        } else if matches!(action, TraceAction::CloseLoop) {
            Some(pc)
        } else {
            None
        };
        if let Some(tp) = target_pc {
            ctx.set_green_key(
                crate::driver::make_green_key(w_code, tp),
                (w_code as usize, tp),
            );
        }
    }

    /// RPython pyjitpl.py:2506 finishframe_exception (multi-frame).
    ///
    /// Walks the framestack looking for an exception handler, popping
    /// frames that don't have one. Structurally matches RPython's
    /// `while self.framestack: ... self.popframe()` loop.
    ///
    /// `reraise_lasti` mirrors `pypy/interpreter/pyopcode.py:122
    /// handle_operation_error`'s like-named parameter: when non-negative it
    /// is the original raise-site offset extracted by a RERAISE bytecode.
    /// Per PyPy's single-call locality (`:181-184`), only the FIRST frame
    /// visited consumes it — either for the handler-entry lasti push
    /// (`:165-170`) or for the no-handler `last_instr` restoration
    /// (`:181-184`).  Subsequent frames see -1.
    ///
    /// Returns Some(LoopAction) if handled, None if all frames exhausted.
    fn finishframe_exception(
        &mut self,
        ctx: &mut TraceCtx,
        mut reraise_lasti: i32,
    ) -> Option<LoopAction> {
        // RPython pyjitpl.py:2506: while self.framestack:
        while let Some(top) = self.framestack.last() {
            let code = unsafe {
                &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject)
            };
            let sym = unsafe { &*top.sym };
            // `top.pc` was advanced past the CALL bytecode before any
            // inline callee was pushed (TODO: differs from RPython).  When a
            // callee raises and we walk up to this caller, the
            // exception-table lookup must probe the CALL site (the saved
            // `call_site_pc`) — mirroring PyPy
            // `pyopcode.handle_operation_error` reading `self.last_instr`
            // from the still-at-CALL frame (`pyopcode.py:151`).  Frames
            // with no outstanding inline call (the original raiser or any
            // frame whose call already returned) keep using `top.pc`.
            let lookup_pc = top.call_site_pc.unwrap_or(top.pc);

            // RPython: if opcode == op_catch_exception → handler found.
            // `lookup_exceptiontable` takes byte offsets and returns byte
            // offsets; pyre's JIT trace uses code-unit indices for `pc`
            // and handler PC, so multiply/divide by 2 at the boundary.
            if let Some((target_bytes, depth, lasti)) =
                pyre_interpreter::exception_table::lookup_exceptiontable(
                    &code.exceptiontable,
                    (lookup_pc * 2) as u32,
                )
            {
                let handler_pc = target_bytes as usize / 2;
                let handler_depth = depth as usize;
                let exc_opref = sym.last_exc_box;
                let exc_obj = sym.last_exc_value;

                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][finishframe_exception] found handler in frame depth={} handler_pc={}",
                        self.framestack.len(),
                        handler_pc
                    );
                }

                // Unwind symbolic + concrete state to handler.
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                let ncells = code.cellvars.len() + code.freevars.len();
                let nlocals = sym.nlocals;
                let target_stack_len = ncells + handler_depth;
                let post_truncate_vsd = nlocals + target_stack_len;
                let owns_shadow = sym.owns_virtualizable_shadow();
                let old_vsd = sym.valuestackdepth;

                sym.symbolic_stack_types.truncate(target_stack_len);
                sym.concrete_stack.truncate(target_stack_len);
                sym.valuestackdepth = post_truncate_vsd;
                // Unified register file: truncate the stack tail of
                // `registers_r` in lockstep with the types / concrete
                // mirrors.
                if sym.registers_r.len() > post_truncate_vsd {
                    sym.registers_r.truncate(post_truncate_vsd);
                }

                // opencoder.py:718 `_list_of_boxes_virtualizable`
                // parity, mirroring `trace_opcode::finishframe_exception`
                // (single-frame): the snapshot reads only
                // `virtualizable_boxes`, so the unwind must also clear
                // the truncated stack slots in the shadow to PY_NULL
                // (pyframe.py:411 `popvalue_maybe_none`).  When the
                // handler frame owns the shadow (loop portal frame
                // catching an exception raised in an inlined callee
                // that has just been popped, or any bridge with seeded
                // `bridge_local_oprefs`), this clear matches the
                // single-frame path so the subsequent
                // `write_stack_slot` exc push is the last write to its
                // slot.  If the handler frame does NOT own the shadow
                // (rare — non-portal frame deep in an inline chain),
                // the legacy `registers_r`-only unwind suffices.
                let static_offset = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                if owns_shadow && old_vsd > post_truncate_vsd {
                    let null_opref = ctx.const_ref(pyre_object::PY_NULL as i64);
                    let null_value =
                        majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::PY_NULL as usize));
                    for reg_idx in post_truncate_vsd..old_vsd {
                        let flat_idx = static_offset + reg_idx;
                        ctx.set_virtualizable_entry_at(flat_idx, null_opref, null_value);
                    }
                }

                let lasti_obj = if lasti {
                    // pyopcode.py:165-170 lasti push:
                    //   if reraise_lasti >= 0:
                    //       lasti_value = reraise_lasti
                    //   else:
                    //       lasti_value = intmask(self.last_instr)
                    //   self.pushvalue(self.space.newint(lasti_value))
                    //
                    // `self.last_instr` in PyPy is the call/raise site —
                    // the same offset we use for the table lookup
                    // (`lookup_pc`), NOT the post-call fallthrough.  Route
                    // through `lookup_pc` so a handler in a caller frame
                    // sees the CALL bytecode's offset.
                    let lasti_value: i64 = if reraise_lasti >= 0 {
                        reraise_lasti as i64
                    } else {
                        lookup_pc as i64
                    };
                    let lasti_obj = pyre_object::w_int_new(lasti_value);
                    let lasti_opref = ctx.const_ref(lasti_obj as i64);
                    let stack_idx = sym.valuestackdepth - sym.nlocals;
                    super::trace_opcode::write_stack_slot(
                        sym,
                        ctx,
                        stack_idx,
                        lasti_opref,
                        ConcreteValue::Ref(lasti_obj),
                    );
                    sym.valuestackdepth += 1;
                    Some(lasti_obj)
                } else {
                    None
                };
                {
                    let stack_idx = sym.valuestackdepth - sym.nlocals;
                    super::trace_opcode::write_stack_slot(
                        sym,
                        ctx,
                        stack_idx,
                        exc_opref,
                        ConcreteValue::Ref(exc_obj),
                    );
                    sym.valuestackdepth += 1;
                }
                sym.pending_next_instr = Some(handler_pc);
                // Handler dispatch unwound the outstanding inline call —
                // the frame is no longer in the middle of a CALL.  Clear
                // call_site_pc so a subsequent exception dispatched from
                // this frame's handler code uses the in-handler pc, not
                // the stale CALL pc.
                top.call_site_pc = None;

                // Sync concrete frame
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    let target_depth = cf.nlocals() + cf.ncells() + handler_depth;
                    while cf.valuestackdepth > target_depth {
                        cf.pop();
                    }
                    if let Some(lasti_obj) = lasti_obj {
                        cf.push(lasti_obj);
                    }
                    cf.push(exc_obj);
                    cf.set_last_instr_from_next_instr(handler_pc);
                }

                // pyjitpl.py:2518: frame.pc = target; raise ChangeFrame
                return Some(LoopAction::Continue);
            }

            // RPython: self.popframe() — no handler in this frame
            if top.is_inline() {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][finishframe_exception] pop inline frame, depth={}",
                        self.framestack.len()
                    );
                }
                // Propagate last_exc_value/last_exc_box to parent before popping
                let exc_value = sym.last_exc_value;
                let exc_box = sym.last_exc_box;
                let exc_const = sym.class_of_last_exc_is_const;

                // pyopcode.py:181-184 no-handler propagation, applied to
                // the about-to-be-popped frame's concrete shadow:
                //   if reraise_lasti >= 0:
                //       self.last_instr = reraise_lasti
                //   self.frame_finished_execution = True
                //
                // Per `:122` single-call locality, reraise_lasti is
                // consumed here for the RERAISE-issuing frame; further
                // iterations see -1 so parent frames behave as if their
                // own `handle_operation_error(reraise_lasti=-1)` was
                // called.
                if let Some(top_mut) = self.framestack.last_mut() {
                    if let Some(ref mut cf) = top_mut.owned_concrete_frame {
                        if reraise_lasti >= 0 {
                            cf.last_instr = reraise_lasti as isize;
                        }
                        cf.frame_finished_execution = true;
                    }
                }
                reraise_lasti = -1;

                // Pop the inline frame
                let popped = self.framestack.pop().unwrap();
                self.portal_call_depth -= 1;
                ctx.pop_inline_trace_position();

                // executioncontext.py:89 / pyjitpl.py:1819: virtual_ref_finish on exception leave.
                if let Some(frame_opref) = popped.drop_frame_opref {
                    if let Some(parent) = self.framestack.last() {
                        let parent_sym = unsafe { &mut *parent.sym };
                        super::state::opimpl_virtual_ref_finish(ctx, parent_sym, frame_opref);
                    }
                }

                // Drop callee frame in trace
                if let Some(frame_opref) = popped.drop_frame_opref {
                    ctx.call_void(
                        crate::callbacks::get().jit_drop_callee_frame,
                        &[frame_opref],
                    );
                }
                let (driver, _) = crate::driver::driver_pair();
                driver.leave_inline_frame();

                // Propagate exception state to parent sym
                if let Some(parent) = self.framestack.last() {
                    let parent_sym = unsafe { &mut *parent.sym };
                    parent_sym.last_exc_value = exc_value;
                    parent_sym.last_exc_box = exc_box;
                    parent_sym.class_of_last_exc_is_const = exc_const;
                }

                continue; // check parent frame
            }

            // Root frame with no handler → compile_exit_frame_with_exception
            // pyjitpl.py:3238-3242 compile_exit_frame_with_exception:
            //   self.store_token_in_vable()
            //   token = sd.exit_frame_with_exception_descr_ref
            //   self.history.record1(rop.FINISH, valuebox, None, descr=token)
            let exc_opref = sym.last_exc_box;
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[jit][finishframe_exception] root frame, no handler → FINISH");
            }
            // pyopcode.py:181-184 applied to the root concrete frame
            // before FINISH escapes the trace.  Mirror the same shape as
            // the inline-pop branch: if this root frame was the RERAISE
            // origin, restore `last_instr`; either way, set
            // `frame_finished_execution = True` so the dead frame surfaces
            // the same flag state the interpreter would.
            {
                let concrete_frame_ptr =
                    sym.concrete_vable_ptr as *mut pyre_interpreter::pyframe::PyFrame;
                if !concrete_frame_ptr.is_null() {
                    let cf = unsafe { &mut *concrete_frame_ptr };
                    if reraise_lasti >= 0 {
                        cf.last_instr = reraise_lasti as isize;
                    }
                    cf.frame_finished_execution = true;
                }
            }
            // pyjitpl.py:3239: self.store_token_in_vable()
            // Record SetfieldGc on vable_token + GUARD_NOT_FORCED_2 with
            // proper resumedata, matching the return/yield path in
            // trace_step_result_to_action.
            {
                let sym_mut = unsafe { &mut *top.sym };
                let concrete_frame = sym_mut.concrete_vable_ptr as usize;
                let pc = sym_mut.pending_next_instr.unwrap_or(0);
                let mut mi = super::state::MIFrame::from_sym(ctx, sym_mut, concrete_frame, pc, pc);
                mi.store_token_in_vable(ctx);
            }
            return Some(LoopAction::Return(TraceAction::Finish {
                finish_args: vec![exc_opref],
                finish_arg_types: vec![Type::Ref],
                // pyjitpl.py:3241 token = sd.exit_frame_with_exception_descr_ref
                exit_with_exception: true,
            }));
        }
        None
    }
}

/// Internal loop control — not exposed outside interpret().
enum LoopAction {
    Continue,
    Return(TraceAction),
}

/// The PC the tracer auto-advances to after `pc` (skipping
/// `ExtendedArg` / `Resume` / `Nop` / `Cache` / `NotTaken`), i.e. the
/// value the tracer stores in `MIFrame::fallthrough_pc`.  `pub` so the
/// codewriter's splice resume-coverage gate can compute a can-raise op's
/// `after_residual_call` resume PC the same way the runtime does
/// (trace_opcode.rs:3634 `resume_pc = self.fallthrough_pc`); the sparse
/// resume resolver reuses it so the can-raise fallthrough resume marker
/// keys off the SAME pc the runtime records in the guard's resume data.
pub fn semantic_fallthrough_pc(code: &CodeObject, pc: usize) -> usize {
    let mut next_pc = pc.saturating_add(1);
    loop {
        match pyre_interpreter::decode_instruction_at(code, next_pc) {
            Some((
                Instruction::ExtendedArg
                | Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken,
                _,
            )) => next_pc += 1,
            _ => return next_pc,
        }
    }
}
