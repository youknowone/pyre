//! PyreMetaInterp — RPython MetaInterp (pyjitpl.py:2371) parity.
//!
//! The MetaInterp drives the tracing loop via interpret(), which
//! repeatedly calls MIFrame::trace_code_step() on the top frame.
//! Each step computes concrete results AND records symbolic IR.
//!
//! RPython control flow:
//!   MetaInterp._interpret():
//!     while True:
//!       framestack[-1].run_one_step()
//!
//! Currently the root frame delegates to MIFrame (state.rs) for
//! dispatch. Inline call tracing uses a separate framestack in
//! inline_trace_and_execute (state.rs). Future: unify via
//! perform_call() / finishframe().

use majit_ir::OpRef;
use majit_meta::{TraceAction, TraceCtx};
use pyre_bytecode::CodeObject;
use pyre_bytecode::bytecode::Instruction;

use super::state::{ConcreteValue, FrontendOp, MIFrame, PyreSym};

/// RPython ChangeFrame / DoneWithThisFrame* parity.
///
/// Returned by MetaInterpFrame::run_one_step() to signal frame transitions.
#[derive(Debug)]
pub enum StepAction {
    /// Continue executing in the current frame.
    Continue,
    /// Frame stack changed (call or return) — reload framestack top.
    /// RPython: raise ChangeFrame
    ChangeFrame,
    /// Outermost frame returned — trace complete.
    /// RPython: raise DoneWithThisFrameInt/Ref/Float(result)
    DoneWithThisFrame(FrontendOp),
    /// Back-edge detected — close loop and compile.
    CloseLoop {
        jump_args: Vec<FrontendOp>,
        loop_header_pc: usize,
    },
    /// Trace aborted (error, too long, unsupported).
    Abort,
}

/// RPython MIFrame (pyjitpl.py:65) parity.
///
/// Each frame in the MetaInterp's framestack tracks both symbolic
/// (PyreSym) and concrete (ConcreteValue arrays) state.
pub struct MetaInterpFrame {
    /// Symbolic + concrete state (borrowed from caller via raw pointer).
    /// RPython MIFrame owns its registers; here we borrow PyreSym.
    pub sym: *mut PyreSym,
    /// Code object being traced (RPython MIFrame.jitcode).
    pub jitcode: *const CodeObject,
    /// Program counter (RPython MIFrame.pc).
    pub pc: usize,
    /// Green key for recursive portal calls.
    pub greenkey: Option<u64>,
    /// Concrete frame pointer for MIFrame delegation.
    pub concrete_frame: usize,
}

/// RPython MetaInterp (pyjitpl.py:2371) parity.
///
/// Takes over execution during tracing. Owns the framestack and
/// drives the interpret loop.
pub struct PyreMetaInterp {
    /// Stack of execution frames (RPython MetaInterp.framestack).
    pub framestack: Vec<MetaInterpFrame>,
    /// Recursive portal call depth.
    pub portal_call_depth: i32,
    /// Root code object.
    pub jitcode: *const CodeObject,
    /// Namespace for global lookups.
    pub namespace: *mut pyre_runtime::PyNamespace,
}

impl PyreMetaInterp {
    /// Create a new MetaInterp for tracing.
    pub fn new(jitcode: *const CodeObject, namespace: *mut pyre_runtime::PyNamespace) -> Self {
        Self {
            framestack: Vec::new(),
            portal_call_depth: -1,
            jitcode,
            namespace,
        }
    }

    /// RPython MetaInterp._interpret() parity.
    ///
    /// Main tracing loop. Repeatedly calls run_one_step() on the top
    /// frame until a terminal action is reached (CloseLoop or Abort).
    ///
    /// Delegates each step to MIFrame::trace_code_step() which handles
    /// both concrete execution and symbolic IR recording.
    pub fn interpret(&mut self, ctx: &mut TraceCtx) -> TraceAction {
        loop {
            let Some(top) = self.framestack.last_mut() else {
                return TraceAction::Abort;
            };

            let code = unsafe { &*top.jitcode };
            let pc = top.pc;
            let sym = unsafe { &mut *top.sym };

            if pc >= code.instructions.len() {
                return TraceAction::Abort;
            }

            // RPython: framestack[-1].run_one_step()
            // Delegates to MIFrame::trace_code_step() which handles both
            // concrete execution and symbolic IR recording.
            let fallthrough_pc = semantic_fallthrough_pc(code, pc);
            let mut frame_state = MIFrame::from_sym(ctx, sym, top.concrete_frame, fallthrough_pc);
            let action = frame_state.trace_code_step(code, pc);

            match action {
                TraceAction::Continue => {
                    let next_pc = sym.pending_next_instr.take().unwrap_or(pc + 1);
                    top.pc = next_pc;
                }

                TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
                    if let TraceAction::CloseLoopWithArgs {
                        loop_header_pc: Some(target_pc),
                        ..
                    } = action
                    {
                        let key = crate::eval::make_green_key(code as *const CodeObject, target_pc);
                        ctx.set_green_key(key);
                    } else if matches!(action, TraceAction::CloseLoop) {
                        let key = crate::eval::make_green_key(code as *const CodeObject, pc);
                        ctx.set_green_key(key);
                    }
                    return action;
                }

                other => return other,
            }
        }
    }

    /// RPython MetaInterp.perform_call() parity.
    ///
    /// Create a new frame for the callee and push it onto the framestack.
    /// Currently unused — inline tracing uses inline_trace_and_execute.
    /// Will be activated when inline tracing migrates to framestack pattern.
    #[allow(dead_code)]
    pub fn perform_call(
        &mut self,
        callee_code: *const CodeObject,
        args: Vec<FrontendOp>,
        greenkey: Option<u64>,
    ) {
        let code = unsafe { &*callee_code };
        let nlocals = code.varnames.len();

        // RPython MIFrame.setup_call: distribute args to registers
        // Note: PyreSym is heap-allocated so the raw pointer stays valid.
        let mut sym = Box::new(PyreSym::new_uninit(OpRef::NONE));
        sym.concrete_locals = vec![ConcreteValue::Null; nlocals];
        sym.concrete_stack = Vec::new();
        sym.symbolic_locals = vec![OpRef::NONE; nlocals];
        sym.nlocals = nlocals;
        for (i, arg) in args.iter().enumerate() {
            if i < nlocals {
                sym.concrete_locals[i] = arg.concrete;
                sym.symbolic_locals[i] = arg.opref;
            }
        }

        let frame = MetaInterpFrame {
            sym: Box::into_raw(sym),
            jitcode: callee_code,
            pc: 0,
            greenkey,
            concrete_frame: 0,
        };

        self.portal_call_depth += 1;
        self.framestack.push(frame);
    }

    /// RPython MetaInterp.finishframe() parity.
    ///
    /// Pop the top frame and store the result in the parent frame.
    #[allow(dead_code)]
    pub fn finishframe(&mut self, result: FrontendOp) -> StepAction {
        if let Some(popped) = self.framestack.pop() {
            // Drop the heap-allocated PyreSym if it was created by perform_call.
            // Root frames created by trace_bytecode point into the caller's
            // stack and must NOT be freed here.
            // TODO: distinguish owned vs borrowed PyreSym
            let _ = popped;
        }
        self.portal_call_depth -= 1;

        if self.framestack.is_empty() {
            StepAction::DoneWithThisFrame(result)
        } else {
            let parent = self.framestack.last_mut().unwrap();
            let parent_sym = unsafe { &mut *parent.sym };
            parent_sym.concrete_stack.push(result.concrete);
            parent_sym.symbolic_stack.push(result.opref);
            StepAction::ChangeFrame
        }
    }

    /// RPython MetaInterp.capture_resumedata() parity (pyjitpl.py:2580).
    ///
    /// Collect live boxes from all frames in the framestack for guard recovery.
    #[allow(dead_code)]
    fn capture_resumedata(&self) -> Vec<OpRef> {
        let mut fail_args = Vec::new();
        for frame in &self.framestack {
            let sym = unsafe { &*frame.sym };
            fail_args.extend(sym.symbolic_locals.iter().copied());
            fail_args.extend(sym.symbolic_stack.iter().copied());
        }
        fail_args
    }
}

pub(crate) fn semantic_fallthrough_pc(code: &CodeObject, pc: usize) -> usize {
    let mut next_pc = pc.saturating_add(1);
    loop {
        match pyre_runtime::decode_instruction_at(code, next_pc) {
            Some((
                Instruction::ExtendedArg
                | Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken,
                _,
            )) => {
                next_pc += 1;
            }
            _ => return next_pc,
        }
    }
}
