//! PyreMetaInterp — RPython MetaInterp (pyjitpl.py:2371) parity.
//!
//! The MetaInterp takes over execution during tracing. It maintains a
//! framestack of PyreMetaFrames and runs bytecodes through its own
//! dispatch loop, simultaneously computing concrete results (via executor)
//! and recording IR operations (via TraceCtx).
//!
//! RPython control flow:
//!   MetaInterp._interpret():
//!     while True:
//!       framestack[-1].run_one_step()
//!
//! Frame transitions use StepAction enum (RPython ChangeFrame exception):
//!   perform_call() → push frame → StepAction::ChangeFrame
//!   finishframe()  → pop frame  → StepAction::ChangeFrame or DoneWithThisFrame

use majit_ir::OpRef;
use majit_meta::TraceCtx;
use pyre_bytecode::CodeObject;
use pyre_object::PyObjectRef;

use super::state::{ConcreteValue, TracedBox as FrontendOp, PyreSym};

/// RPython ChangeFrame / DoneWithThisFrame* parity.
///
/// Returned by PyreMetaFrame::run_one_step() to signal frame transitions.
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
pub struct PyreMetaFrame {
    /// Symbolic IR tracking state.
    pub sym: PyreSym,
    /// Concrete local variable values (RPython registers_i/r/f unified).
    pub concrete_locals: Vec<ConcreteValue>,
    /// Concrete operand stack values.
    pub concrete_stack: Vec<ConcreteValue>,
    /// Code object being traced (RPython MIFrame.jitcode).
    pub jitcode: *const CodeObject,
    /// Program counter (RPython MIFrame.pc).
    pub pc: usize,
    /// Green key for recursive portal calls.
    pub greenkey: Option<u64>,
    /// Resume data: parent frame snapshot index.
    pub parent_snapshot_idx: i32,
}

/// RPython MetaInterp (pyjitpl.py:2371) parity.
///
/// Takes over execution during tracing. Owns the framestack and
/// trace context.
pub struct PyreMetaInterp {
    /// Stack of execution frames (RPython MetaInterp.framestack).
    pub framestack: Vec<PyreMetaFrame>,
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
    /// frame until a terminal action is reached.
    pub fn interpret(&mut self, ctx: &mut TraceCtx) -> Result<StepAction, pyre_runtime::PyError> {
        loop {
            let top = self.framestack.last_mut().ok_or_else(|| {
                pyre_runtime::PyError::type_error("empty framestack during interpret")
            })?;

            match top.run_one_step(ctx)? {
                StepAction::Continue => {}
                StepAction::ChangeFrame => {
                    // Framestack was modified — continue with new top
                }
                action @ StepAction::DoneWithThisFrame(_) => return Ok(action),
                action @ StepAction::CloseLoop { .. } => return Ok(action),
                StepAction::Abort => return Ok(StepAction::Abort),
            }
        }
    }

    /// RPython MetaInterp.perform_call() parity.
    ///
    /// Create a new frame for the callee and push it onto the framestack.
    pub fn perform_call(
        &mut self,
        callee_code: *const CodeObject,
        args: Vec<FrontendOp>,
        greenkey: Option<u64>,
    ) {
        let code = unsafe { &*callee_code };
        let nlocals = code.varnames.len();

        // RPython MIFrame.setup_call: distribute args to registers
        let mut concrete_locals = vec![ConcreteValue::Null; nlocals];
        for (i, arg) in args.iter().enumerate() {
            if i < nlocals {
                concrete_locals[i] = arg.concrete;
            }
        }

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        // TODO: initialize sym from args

        let frame = PyreMetaFrame {
            sym,
            concrete_locals,
            concrete_stack: Vec::new(),
            jitcode: callee_code,
            pc: 0,
            greenkey,
            parent_snapshot_idx: -1,
        };

        self.portal_call_depth += 1;
        self.framestack.push(frame);
    }

    /// RPython MetaInterp.finishframe() parity.
    ///
    /// Pop the top frame and store the result in the parent frame.
    pub fn finishframe(&mut self, result: FrontendOp) -> StepAction {
        self.framestack.pop();
        self.portal_call_depth -= 1;

        if self.framestack.is_empty() {
            // Outermost return — compile trace
            StepAction::DoneWithThisFrame(result)
        } else {
            // RPython make_result_of_lastop: store in parent
            let parent = self.framestack.last_mut().unwrap();
            parent.concrete_stack.push(result.concrete);
            parent.sym.symbolic_stack.push(result.opref);
            StepAction::ChangeFrame
        }
    }
}

impl PyreMetaFrame {
    /// RPython MIFrame.run_one_step() parity.
    ///
    /// Decode and execute one bytecode instruction, recording IR and
    /// computing concrete results simultaneously.
    pub fn run_one_step(
        &mut self,
        _ctx: &mut TraceCtx,
    ) -> Result<StepAction, pyre_runtime::PyError> {
        let code = unsafe { &*self.jitcode };
        if self.pc >= code.instructions.len() {
            return Err(pyre_runtime::PyError::type_error(
                "fell off end of bytecode during tracing",
            ));
        }

        let Some((_instruction, _op_arg)) =
            pyre_runtime::decode_instruction_at(code, self.pc)
        else {
            return Ok(StepAction::Abort);
        };
        self.pc += 1;

        // TODO: dispatch_opcode — Phase 2 will implement all handlers
        // For now, return Continue to satisfy the type system
        Ok(StepAction::Continue)
    }
}
