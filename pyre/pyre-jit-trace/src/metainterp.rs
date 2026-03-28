//! PyreMetaInterp — RPython MetaInterp (pyjitpl.py:2371) parity.
//!
//! Single interpret() loop with a single framestack for both root
//! and inline frames. CALL → push_inline_frame (RPython perform_call),
//! RETURN → finishframe_inline (RPython finishframe).

use majit_ir::{OpRef, Type};
use majit_metainterp::{TraceAction, TraceCtx};
use pyre_bytecode::CodeObject;
use pyre_bytecode::bytecode::Instruction;

use super::state::{
    ConcreteValue, MIFrame, PendingInlineFrame, PyreSym, materialize_pending_inline_result,
    pending_inline_result_from_concrete,
};

/// RPython MIFrame (pyjitpl.py:65) — per-frame tracing state.
pub struct MetaInterpFrame {
    pub sym: *mut PyreSym,
    pub owned_sym: Option<Box<PyreSym>>,
    pub jitcode: *const CodeObject,
    pub pc: usize,
    pub greenkey: Option<u64>,
    pub concrete_frame: usize,
    /// Box for heap stability across Vec reallocs.
    pub owned_concrete_frame: Option<Box<pyre_interpreter::pyframe::PyFrame>>,
    pub parent_fail_args: Vec<OpRef>,
    pub parent_fail_arg_types: Vec<Type>,
    /// opencoder.py:806 parent frame pc for multi-frame snapshot.
    pub parent_resumepc: usize,
    pub drop_frame_opref: Option<OpRef>,
    pub caller_result_stack_idx: Option<usize>,
    pub arg_state: pyre_bytecode::bytecode::OpArgState,
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
    pub jitcode: *const CodeObject,
    pub namespace: *mut pyre_interpreter::PyNamespace,
    inline_call_guard: Option<pyre_interpreter::call::InlineCallOverrideGuard>,
    inline_trace_base: usize,
}

impl PyreMetaInterp {
    pub fn new(jitcode: *const CodeObject, namespace: *mut pyre_interpreter::PyNamespace) -> Self {
        Self {
            framestack: Vec::new(),
            portal_call_depth: -1,
            jitcode,
            namespace,
            inline_call_guard: None,
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
            let code = unsafe { &*top.jitcode };
            // RPython _interpret() parity: all frames use cf.next_instr as PC.
            // Skip trivia (Cache, ExtendedArg, etc.) — RPython's trace loop
            // only processes real opcodes. Inline frame returns or branch
            // targets can land on Cache instructions; skip forward to the
            // next semantic opcode so orgpc is always a real opcode PC.
            let mut pc = top
                .owned_concrete_frame
                .as_ref()
                .map(|cf| cf.next_instr)
                .unwrap_or(top.pc);
            if pc >= code.instructions.len() {
                return TraceAction::Abort;
            }
            // Advance past Cache/ExtendedArg/Nop/NotTaken/Resume trivia.
            while pc < code.instructions.len() {
                match pyre_interpreter::decode_instruction_at(code, pc) {
                    Some((
                        pyre_bytecode::bytecode::Instruction::Cache
                        | pyre_bytecode::bytecode::Instruction::ExtendedArg
                        | pyre_bytecode::bytecode::Instruction::Nop
                        | pyre_bytecode::bytecode::Instruction::NotTaken,
                        _,
                    )) => {
                        pc += 1;
                        // Sync the concrete frame so subsequent steps see the
                        // corrected PC.
                        if let Some(ref mut cf) = top.owned_concrete_frame {
                            cf.next_instr = pc;
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
        }
    }

    // ── Root frame step ──────────────────────────────────────────

    fn step_root_frame(&mut self, ctx: &mut TraceCtx, pc: usize) -> LoopAction {
        let top = self.framestack.last_mut().unwrap();
        let code = unsafe { &*top.jitcode };
        let sym = unsafe { &mut *top.sym };
        let cf_addr = top.concrete_frame_addr();
        let fallthrough_pc = semantic_fallthrough_pc(code, pc);

        let mut fs = MIFrame::from_sym(ctx, sym, cf_addr, fallthrough_pc, pc);
        let action = fs.trace_code_step(code, pc);
        let pending = fs.pending_inline_frame.take();
        drop(fs);

        if let Some(pending) = pending {
            // RPython perform_call: callee deferred via pending_inline_frame.
            let top = self.framestack.last_mut().unwrap();
            let sym = unsafe { &mut *top.sym };
            let next_pc = sym
                .pending_next_instr
                .take()
                .unwrap_or_else(|| semantic_fallthrough_pc(code, pc));
            top.pc = next_pc;
            let result_idx = sym.stack_only_depth().checked_sub(1);
            // Root frame: don't pop from owned_concrete_frame — interpreter
            // drives concrete execution separately. Only sync PC.
            if let Some(ref mut cf) = top.owned_concrete_frame {
                cf.next_instr = next_pc;
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
                    cf.next_instr = next_pc;
                }
                LoopAction::Continue
            }
            TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
                self.handle_close_loop(ctx, &action, pc);
                LoopAction::Return(action)
            }
            other => LoopAction::Return(other),
        }
    }

    // ── Inline frame step ────────────────────────────────────────

    fn step_inline_frame(&mut self, ctx: &mut TraceCtx, pc: usize) -> LoopAction {
        let top = self.framestack.last_mut().unwrap();
        let code = unsafe { &*top.jitcode };
        let sym = unsafe { &mut *top.sym };
        let cf_addr = top.concrete_frame_addr();
        let pfa = top.parent_fail_args.clone();
        let pfa_types = top.parent_fail_arg_types.clone();
        let pfa_resumepc = top.parent_resumepc;
        let fallthrough_pc = semantic_fallthrough_pc(code, pc);

        let inline_action = {
            let mut fs = MIFrame::from_sym(ctx, sym, cf_addr, fallthrough_pc, pc);
            fs.parent_fail_args = Some(pfa);
            fs.parent_fail_arg_types = Some(pfa_types);
            fs.parent_resumepc = pfa_resumepc;
            let result = fs.trace_code_step_inline(code, pc);
            let pending = fs.pending_inline_frame.take();
            drop(fs);

            if let Some(pending) = pending {
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                let sfall = semantic_fallthrough_pc(code, pc);
                top.pc = sym.pending_next_instr.take().unwrap_or(sfall);
                let result_idx = sym.stack_only_depth().checked_sub(1);
                // Pop concrete call args from parent owned_concrete_frame
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    for _ in 0..pending.nargs {
                        cf.pop();
                    }
                    let _ = cf.pop(); // null_or_self
                    let _ = cf.pop(); // callable
                    cf.next_instr = sfall;
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
                LoopAction::Continue
            }
            super::state::InlineTraceStepAction::Trace(TraceAction::Finish {
                finish_args,
                finish_arg_types,
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
                    !sym.last_exc_value.is_null()
                };
                if has_exc {
                    if let Some(action) = self.finishframe_exception(ctx) {
                        return action;
                    }
                }
                ctx.truncate_inline_trace_positions(self.inline_trace_base);
                LoopAction::Return(TraceAction::Abort)
            }
            super::state::InlineTraceStepAction::Trace(action) => LoopAction::Return(action),
            super::state::InlineTraceStepAction::PushFrame(pending) => {
                // trace_code_step_inline already took the pending frame
                let top = self.framestack.last_mut().unwrap();
                let code = unsafe { &*top.jitcode };
                let sym = unsafe { &mut *top.sym };
                let sfall = semantic_fallthrough_pc(code, pc);
                top.pc = sym.pending_next_instr.take().unwrap_or(sfall);
                let result_idx = sym.stack_only_depth().checked_sub(1);
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    for _ in 0..pending.nargs {
                        cf.pop();
                    }
                    let _ = cf.pop(); // null_or_self
                    let _ = cf.pop(); // callable
                    cf.next_instr = sfall;
                }
                self.push_inline_frame(ctx, pending, result_idx);
                LoopAction::Continue
            }
        }
    }

    // ── Frame management ─────────────────────────────────────────

    /// RPython perform_call: push PendingInlineFrame onto framestack.
    fn push_inline_frame(
        &mut self,
        ctx: &mut TraceCtx,
        pending: PendingInlineFrame,
        caller_result_idx: Option<usize>,
    ) {
        if self.inline_call_guard.is_none() {
            self.inline_call_guard = Some(pyre_interpreter::call::inline_call_override_guard());
        }

        let (driver, _) = crate::driver::driver_pair();
        driver.enter_inline_frame(pending.green_key);
        ctx.push_inline_trace_position(pending.green_key);

        let callee_code = pending.concrete_frame.code;
        let mut owned_sym = Box::new(pending.sym);
        let sym_ptr = owned_sym.as_mut() as *mut PyreSym;
        let mut owned_cf = Box::new(pending.concrete_frame);
        let cf_addr = &*owned_cf as *const pyre_interpreter::pyframe::PyFrame as usize;

        let frame = MetaInterpFrame {
            sym: sym_ptr,
            owned_sym: Some(owned_sym),
            jitcode: callee_code,
            pc: 0,
            greenkey: Some(pending.green_key),
            concrete_frame: cf_addr,
            owned_concrete_frame: Some(owned_cf),
            parent_fail_args: pending.parent_fail_args,
            parent_fail_arg_types: pending.parent_fail_arg_types,
            parent_resumepc: pending.parent_resumepc,
            drop_frame_opref: pending.drop_frame_opref,
            caller_result_stack_idx: caller_result_idx,
            arg_state: pyre_bytecode::bytecode::OpArgState::default(),
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

        // Drop callee frame in trace
        if let Some(frame_opref) = popped.drop_frame_opref {
            ctx.call_void(
                crate::callbacks::get().jit_drop_callee_frame,
                &[frame_opref],
            );
        }

        let (driver, _) = crate::driver::driver_pair();
        driver.leave_inline_frame();

        // Release inline call guard when all inline frames are gone
        if !self.framestack.iter().any(|f| f.is_inline()) {
            self.inline_call_guard = None;
        }

        if self.framestack.is_empty() {
            return; // shouldn't happen: root never produces Finish
        }

        // make_result_of_lastop: store in parent
        let parent = self.framestack.last_mut().unwrap();
        let parent_sym = unsafe { &mut *parent.sym };

        if let Some(result_idx) = popped.caller_result_stack_idx {
            // Update symbolic stack
            if let Some(slot) = parent_sym.symbolic_stack.get_mut(result_idx) {
                *slot = result_opref;
            }
            if result_idx >= parent_sym.symbolic_stack_types.len() {
                parent_sym
                    .symbolic_stack_types
                    .resize(result_idx + 1, Type::Ref);
            }
            parent_sym.symbolic_stack_types[result_idx] = result_type;
            parent_sym
                .transient_value_types
                .insert(result_opref, result_type);

            // Update concrete stack (THE CRITICAL FIX)
            let cv = ConcreteValue::from_pyobj(concrete_result);
            if result_idx < parent_sym.concrete_stack.len() {
                parent_sym.concrete_stack[result_idx] = cv;
            } else {
                parent_sym
                    .concrete_stack
                    .resize(result_idx + 1, ConcreteValue::Null);
                parent_sym.concrete_stack[result_idx] = cv;
            }
        }

        // Push concrete result to parent's owned PyFrame
        if let Some(ref mut pcf) = parent.owned_concrete_frame {
            pcf.push(concrete_result);
        }

        // No pending_concrete_push needed — concrete_stack[result_idx] already updated above.
    }

    // ── Concrete execution helpers ───────────────────────────────

    fn concrete_execute_step(&mut self) {
        let top = self.framestack.last_mut().unwrap();
        let Some(cf) = top.owned_concrete_frame.as_mut() else {
            return;
        };
        let cf = &mut **cf;
        let code = unsafe { &*cf.code };
        let ni = cf.next_instr;
        if ni >= code.instructions.len() {
            return;
        }

        let Some((instruction, op_arg)) = pyre_interpreter::decode_instruction_at(code, ni) else {
            return;
        };
        cf.next_instr = ni + 1;
        let next = cf.next_instr;

        if let Instruction::Call { argc } = instruction {
            let nargs = argc.get(op_arg) as usize;
            if pyre_interpreter::call::replay_pending_inline_call(cf, nargs) {
                return;
            }
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
        let code = unsafe { &*cf.code };
        let ni = cf.next_instr;
        if ni >= code.instructions.len() {
            return pyre_object::PY_NULL;
        }

        let Some((instruction, op_arg)) = pyre_interpreter::decode_instruction_at(code, ni) else {
            return pyre_object::PY_NULL;
        };
        cf.next_instr = ni + 1;
        let next = cf.next_instr;

        match pyre_interpreter::execute_opcode_step(cf, code, instruction, op_arg, next) {
            Ok(pyre_interpreter::StepResult::Return(value)) => materialize_pending_inline_result(
                pending_inline_result_from_concrete(Type::Ref, value),
            ),
            _ => pyre_object::PY_NULL,
        }
    }

    // ── Helpers ──────────────────────────────────────────────────

    fn handle_close_loop(&self, ctx: &mut TraceCtx, action: &TraceAction, pc: usize) {
        let code = unsafe { &*self.framestack[0].jitcode };
        if let TraceAction::CloseLoopWithArgs {
            loop_header_pc: Some(target_pc),
            ..
        } = action
        {
            ctx.set_green_key(crate::driver::make_green_key(
                code as *const CodeObject,
                *target_pc,
            ));
        } else if matches!(action, TraceAction::CloseLoop) {
            ctx.set_green_key(crate::driver::make_green_key(code as *const CodeObject, pc));
        }
    }

    /// RPython pyjitpl.py:2506 finishframe_exception (multi-frame).
    ///
    /// Walks the framestack looking for an exception handler, popping
    /// frames that don't have one. Structurally matches RPython's
    /// `while self.framestack: ... self.popframe()` loop.
    ///
    /// Returns Some(LoopAction) if handled, None if all frames exhausted.
    fn finishframe_exception(&mut self, ctx: &mut TraceCtx) -> Option<LoopAction> {
        // RPython pyjitpl.py:2506: while self.framestack:
        while let Some(top) = self.framestack.last() {
            let code = unsafe { &*top.jitcode };
            let sym = unsafe { &*top.sym };
            let pc = top.pc;

            // RPython: if opcode == op_catch_exception → handler found
            if let Some(entry) =
                pyre_bytecode::bytecode::find_exception_handler(&code.exceptiontable, pc as u32)
            {
                let handler_pc = entry.target as usize;
                let handler_depth = entry.depth as usize;
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
                let ncells = unsafe { (&*code).cellvars.len() + (&*code).freevars.len() };
                let nlocals = sym.nlocals;
                let target_stack_len = ncells + handler_depth;

                sym.symbolic_stack.truncate(target_stack_len);
                sym.symbolic_stack_types.truncate(target_stack_len);
                sym.concrete_stack.truncate(target_stack_len);
                sym.valuestackdepth = nlocals + target_stack_len;
                if entry.push_lasti {
                    sym.symbolic_stack.push(OpRef::NONE);
                    sym.symbolic_stack_types.push(Type::Ref);
                    sym.concrete_stack
                        .push(ConcreteValue::Ref(pyre_object::w_int_new(pc as i64)));
                    sym.valuestackdepth += 1;
                }
                sym.symbolic_stack.push(exc_opref);
                sym.symbolic_stack_types.push(Type::Ref);
                sym.concrete_stack.push(ConcreteValue::Ref(exc_obj));
                sym.valuestackdepth += 1;
                sym.pending_next_instr = Some(handler_pc);

                // Sync concrete frame
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    let target_depth = cf.nlocals() + cf.ncells() + handler_depth;
                    while cf.valuestackdepth > target_depth {
                        cf.pop();
                    }
                    if entry.push_lasti {
                        cf.push(pyre_object::w_int_new(pc as i64));
                    }
                    cf.push(exc_obj);
                    cf.next_instr = handler_pc;
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

                // Pop the inline frame
                let popped = self.framestack.pop().unwrap();
                self.portal_call_depth -= 1;
                ctx.pop_inline_trace_position();

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

                // Release inline call guard when all inline frames gone
                if !self.framestack.iter().any(|f| f.is_inline()) {
                    self.inline_call_guard = None;
                }
                continue; // check parent frame
            }

            // Root frame with no handler → compile_exit_frame_with_exception
            // pyjitpl.py:2532-2538: FINISH op with exception value.
            let exc_opref = sym.last_exc_box;
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[jit][finishframe_exception] root frame, no handler → FINISH");
            }
            return Some(LoopAction::Return(TraceAction::Finish {
                finish_args: vec![exc_opref],
                finish_arg_types: vec![Type::Ref],
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

pub(crate) fn semantic_fallthrough_pc(code: &CodeObject, pc: usize) -> usize {
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
