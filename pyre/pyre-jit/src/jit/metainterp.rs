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
    /// Symbolic + concrete state (PyreSym owns concrete_locals/concrete_stack).
    pub sym: PyreSym,
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

            match top.run_one_step(ctx, self.namespace)? {
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
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
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

        let frame = PyreMetaFrame {
            sym,
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
            parent.sym.concrete_stack.push(result.concrete);
            parent.sym.symbolic_stack.push(result.opref);
            StepAction::ChangeFrame
        }
    }

    /// RPython MetaInterp.generate_guard() parity (pyjitpl.py:2552).
    ///
    /// Record a guard operation and capture resume data from the framestack.
    /// RPython MetaInterp.generate_guard() parity (pyjitpl.py:2552).
    ///
    /// Record a guard operation and capture resume data from the framestack.
    /// TODO: proper fail_descr integration when TraceCtx API is extended.
    pub fn generate_guard(&self, _ctx: &mut TraceCtx, _opcode: majit_ir::OpCode, _args: &[OpRef]) {
        // Placeholder — guard recording requires fail_descr creation
        // which depends on TraceCtx internal API. Will be wired up when
        // PyreMetaInterp replaces TraceFrameState as the active tracer.
        let _fail_args = self.capture_resumedata();
    }

    /// RPython MetaInterp.capture_resumedata() parity (pyjitpl.py:2580).
    ///
    /// Collect live boxes from all frames in the framestack for guard recovery.
    fn capture_resumedata(&self) -> Vec<OpRef> {
        let mut fail_args = Vec::new();
        for frame in &self.framestack {
            // RPython get_list_of_active_boxes: all live registers
            fail_args.extend(frame.sym.symbolic_locals.iter().copied());
            fail_args.extend(frame.sym.symbolic_stack.iter().copied());
        }
        fail_args
    }
}

impl PyreMetaFrame {
    /// Push a FrontendOp onto the operand stack.
    fn push(&mut self, op: FrontendOp) {
        self.sym.symbolic_stack.push(op.opref);
        self.sym.concrete_stack.push(op.concrete);
    }

    /// Pop a FrontendOp from the operand stack.
    fn pop(&mut self) -> FrontendOp {
        let opref = self.sym.symbolic_stack.pop().unwrap_or(OpRef::NONE);
        let concrete = self.sym.concrete_stack.pop().unwrap_or(ConcreteValue::Null);
        FrontendOp::new(opref, concrete)
    }

    /// Peek at the top of the operand stack without popping.
    fn peek(&self) -> FrontendOp {
        let opref = self
            .sym
            .symbolic_stack
            .last()
            .copied()
            .unwrap_or(OpRef::NONE);
        let concrete = self
            .sym
            .concrete_stack
            .last()
            .copied()
            .unwrap_or(ConcreteValue::Null);
        FrontendOp::new(opref, concrete)
    }

    /// RPython MetaInterp.execute_and_record() parity.
    ///
    /// Step 1: executor.execute() — compute concrete result
    /// Step 2: Check constant fold
    /// Step 3: history.record() — record IR operation
    fn execute_and_record(
        &self,
        ctx: &mut TraceCtx,
        opcode: majit_ir::OpCode,
        args: &[FrontendOp],
        descr: Option<majit_ir::DescrRef>,
    ) -> FrontendOp {
        // Step 1: Concrete execution (RPython executor.execute)
        let concrete_args: Vec<ConcreteValue> = args.iter().map(|a| a.concrete).collect();
        let concrete_result = super::executor::execute_opcode(opcode, &concrete_args);

        // Step 2: Constant fold (RPython: if pure and all_const → wrap_constant)
        // TODO: implement constant folding check

        // Step 3: Record IR (RPython history.record)
        let arg_oprefs: Vec<OpRef> = args.iter().map(|a| a.opref).collect();
        let result_opref = if let Some(d) = descr {
            ctx.record_op_with_descr(opcode, &arg_oprefs, d)
        } else {
            ctx.record_op(opcode, &arg_oprefs)
        };

        FrontendOp::new(result_opref, concrete_result)
    }

    /// RPython MIFrame.run_one_step() parity — concrete dispatch.
    ///
    /// Computes concrete results for each opcode. Symbolic IR recording
    /// is delegated to TraceFrameState::trace_code_step via the
    /// trace_bytecode loop.
    ///
    /// When this is used as the primary dispatch (replacing TraceFrameState),
    /// execute_and_record handles both concrete + symbolic in one call.
    pub fn run_one_step(
        &mut self,
        ctx: &mut TraceCtx,
        namespace: *mut pyre_runtime::PyNamespace,
    ) -> Result<StepAction, pyre_runtime::PyError> {
        use pyre_bytecode::bytecode::{BinaryOperator, ComparisonOperator, Instruction};

        let code = unsafe { &*self.jitcode };
        if self.pc >= code.instructions.len() {
            return Err(pyre_runtime::PyError::type_error(
                "fell off end of bytecode during tracing",
            ));
        }

        let Some((instruction, op_arg)) = pyre_runtime::decode_instruction_at(code, self.pc) else {
            return Ok(StepAction::Abort);
        };
        self.pc += 1;

        match instruction {
            // ── Trivia (no-ops) ──
            Instruction::Nop
            | Instruction::Cache
            | Instruction::NotTaken
            | Instruction::Resume { .. } => Ok(StepAction::Continue),

            // ── Constants ──
            // RPython: opimpl_int_const → self.execute(rop.SAME_AS_I, constbox)
            Instruction::LoadSmallInt { i } => {
                let value = i.get(op_arg) as i64;
                let opref = ctx.const_int(value);
                self.push(FrontendOp::new(opref, ConcreteValue::Int(value)));
                Ok(StepAction::Continue)
            }
            Instruction::LoadConst { consti } => {
                let idx = consti.get(op_arg);
                let concrete = super::state::load_const_concrete(&code.constants[idx]);
                let opref = ctx.const_int(concrete.to_pyobj() as i64);
                self.push(FrontendOp::new(opref, concrete));
                Ok(StepAction::Continue)
            }

            // ── Locals ──
            // RPython: registers[idx] already holds the box
            Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
                let idx = var_num.get(op_arg).as_usize();
                let concrete = self
                    .sym
                    .concrete_locals
                    .get(idx)
                    .copied()
                    .unwrap_or(ConcreteValue::Null);
                // RPython: box is already in registers[idx]
                let opref = self
                    .sym
                    .symbolic_locals
                    .get(idx)
                    .copied()
                    .unwrap_or(OpRef::NONE);
                self.push(FrontendOp::new(opref, concrete));
                Ok(StepAction::Continue)
            }
            Instruction::StoreFast { var_num } => {
                let idx = var_num.get(op_arg).as_usize();
                let val = self.pop();
                if idx < self.sym.concrete_locals.len() {
                    self.sym.concrete_locals[idx] = val.concrete;
                }
                if idx < self.sym.symbolic_locals.len() {
                    self.sym.symbolic_locals[idx] = val.opref;
                }
                Ok(StepAction::Continue)
            }

            // ── Stack ──
            Instruction::PopTop => {
                self.pop();
                Ok(StepAction::Continue)
            }

            // ── Binary ops ──
            // RPython: opimpl_int_add(b1, b2) → self.execute(rop.INT_ADD, b1, b2)
            Instruction::BinaryOp { op } => {
                let b = self.pop();
                let a = self.pop();
                let opcode = match op.get(op_arg) {
                    BinaryOperator::Add => majit_ir::OpCode::IntAddOvf,
                    BinaryOperator::Subtract => majit_ir::OpCode::IntSubOvf,
                    BinaryOperator::Multiply => majit_ir::OpCode::IntMulOvf,
                    BinaryOperator::Remainder => majit_ir::OpCode::IntMod,
                    BinaryOperator::FloorDivide => majit_ir::OpCode::IntFloorDiv,
                    _ => majit_ir::OpCode::IntAdd, // fallback
                };
                let result = self.execute_and_record(ctx, opcode, &[a, b], None);
                self.push(result);
                Ok(StepAction::Continue)
            }

            // ── Comparison ──
            Instruction::CompareOp { opname } => {
                let b = self.pop();
                let a = self.pop();
                let opcode = match opname.get(op_arg) {
                    ComparisonOperator::Less => majit_ir::OpCode::IntLt,
                    ComparisonOperator::LessOrEqual => majit_ir::OpCode::IntLe,
                    ComparisonOperator::Greater => majit_ir::OpCode::IntGt,
                    ComparisonOperator::GreaterOrEqual => majit_ir::OpCode::IntGe,
                    ComparisonOperator::Equal => majit_ir::OpCode::IntEq,
                    ComparisonOperator::NotEqual => majit_ir::OpCode::IntNe,
                };
                let result = self.execute_and_record(ctx, opcode, &[a, b], None);
                self.push(result);
                Ok(StepAction::Continue)
            }

            // ── Branches ──
            // RPython: opimpl_goto_if_not(box, target) → box.getint() for direction
            Instruction::PopJumpIfFalse { delta } => {
                let val = self.pop();
                let concrete_truth = val.concrete.is_truthy();
                // TODO: generate guard + record
                if !concrete_truth {
                    let target = self.pc + u32::from(delta.get(op_arg)) as usize;
                    self.pc = target;
                }
                Ok(StepAction::Continue)
            }
            Instruction::PopJumpIfTrue { delta } => {
                let val = self.pop();
                let concrete_truth = val.concrete.is_truthy();
                if concrete_truth {
                    let target = self.pc + u32::from(delta.get(op_arg)) as usize;
                    self.pc = target;
                }
                Ok(StepAction::Continue)
            }

            // ── Control flow ──
            Instruction::JumpForward { delta } => {
                let target = self.pc + u32::from(delta.get(op_arg)) as usize;
                self.pc = target;
                Ok(StepAction::Continue)
            }
            Instruction::JumpBackward { delta } => {
                let target = self.pc - u32::from(delta.get(op_arg)) as usize;
                // RPython: reached_loop_header → CloseLoop
                let jump_args = Vec::new(); // TODO: collect jump args
                Ok(StepAction::CloseLoop {
                    jump_args,
                    loop_header_pc: target,
                })
            }
            Instruction::ReturnValue => {
                let result = self.pop();
                Ok(StepAction::DoneWithThisFrame(result))
            }

            // ── Globals ──
            Instruction::LoadGlobal { namei } => {
                // RPython: residual GETFIELD on namespace
                let name_idx = (namei.get(op_arg) >> 1) as usize;
                let name = code.names.get(name_idx).map(|s| s.as_ref()).unwrap_or("");
                let _push_null = (namei.get(op_arg) & 1) != 0;
                // Push NULL for PUSH_NULL semantics
                if _push_null {
                    self.push(FrontendOp::new(OpRef::NONE, ConcreteValue::Null));
                }
                // Concrete: read from namespace
                let concrete = if !namespace.is_null() {
                    let ns = unsafe { &*namespace };
                    ns.get(name)
                        .map(|&v| ConcreteValue::from_pyobj(v))
                        .unwrap_or(ConcreteValue::Null)
                } else {
                    ConcreteValue::Null
                };
                self.push(FrontendOp::new(OpRef::NONE, concrete));
                Ok(StepAction::Continue)
            }

            // ── Call ──
            Instruction::Call { argc } => {
                let nargs = argc.get(op_arg) as usize;
                let mut args = Vec::with_capacity(nargs);
                for _ in 0..nargs {
                    args.push(self.pop());
                }
                args.reverse();
                let _null_or_self = self.pop();
                let callable = self.pop();

                // RPython: execute_varargs for residual call concrete result
                let concrete_result = match callable.concrete {
                    ConcreteValue::Ref(obj) if !obj.is_null() => unsafe {
                        if pyre_runtime::is_builtin_func(obj) {
                            let func = pyre_runtime::w_builtin_func_get(obj);
                            let concrete_args: Vec<PyObjectRef> =
                                args.iter().map(|a| a.concrete.to_pyobj()).collect();
                            ConcreteValue::from_pyobj(func(&concrete_args))
                        } else {
                            ConcreteValue::Null
                        }
                    },
                    _ => ConcreteValue::Null,
                };
                // TODO: proper IR recording for call
                self.push(FrontendOp::new(OpRef::NONE, concrete_result));
                Ok(StepAction::Continue)
            }

            // ── Collections ──
            Instruction::BuildList { count } => {
                let n = count.get(op_arg) as usize;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(self.pop());
                }
                items.reverse();
                let concrete_items: Vec<PyObjectRef> =
                    items.iter().map(|i| i.concrete.to_pyobj()).collect();
                let list = pyre_runtime::build_list_from_refs(&concrete_items);
                self.push(FrontendOp::new(
                    OpRef::NONE,
                    ConcreteValue::from_pyobj(list),
                ));
                Ok(StepAction::Continue)
            }
            Instruction::BuildTuple { count } => {
                let n = count.get(op_arg) as usize;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(self.pop());
                }
                items.reverse();
                let concrete_items: Vec<PyObjectRef> =
                    items.iter().map(|i| i.concrete.to_pyobj()).collect();
                let tuple = pyre_runtime::build_tuple_from_refs(&concrete_items);
                self.push(FrontendOp::new(
                    OpRef::NONE,
                    ConcreteValue::from_pyobj(tuple),
                ));
                Ok(StepAction::Continue)
            }

            // ── Subscript ──
            Instruction::StoreSubscr => {
                let key = self.pop();
                let obj = self.pop();
                let value = self.pop();
                let _ = pyre_objspace::space::py_setitem(
                    obj.concrete.to_pyobj(),
                    key.concrete.to_pyobj(),
                    value.concrete.to_pyobj(),
                );
                Ok(StepAction::Continue)
            }

            // ── Attribute ──
            Instruction::LoadAttr { namei } => {
                let name_idx = u32::from(namei.get(op_arg)) as usize;
                let name = code
                    .names
                    .get(name_idx >> 1)
                    .map(|s| s.as_ref())
                    .unwrap_or("");
                let obj = self.pop();
                let concrete = match pyre_objspace::space::py_getattr(obj.concrete.to_pyobj(), name)
                {
                    Ok(v) => ConcreteValue::from_pyobj(v),
                    Err(_) => ConcreteValue::Null,
                };
                self.push(FrontendOp::new(OpRef::NONE, concrete));
                Ok(StepAction::Continue)
            }

            // ── Stack manipulation ──
            Instruction::Copy { i } => {
                let depth = i.get(op_arg) as usize;
                let idx = self.sym.concrete_stack.len().saturating_sub(depth);
                let concrete = self
                    .sym
                    .concrete_stack
                    .get(idx)
                    .copied()
                    .unwrap_or(ConcreteValue::Null);
                let opref = self
                    .sym
                    .symbolic_stack
                    .get(idx)
                    .copied()
                    .unwrap_or(OpRef::NONE);
                self.push(FrontendOp::new(opref, concrete));
                Ok(StepAction::Continue)
            }
            Instruction::Swap { i } => {
                let depth = i.get(op_arg) as usize;
                let top = self.sym.concrete_stack.len() - 1;
                let other = self.sym.concrete_stack.len() - depth;
                self.sym.concrete_stack.swap(top, other);
                self.sym.symbolic_stack.swap(top, other);
                Ok(StepAction::Continue)
            }

            // ── Misc ──
            Instruction::PushNull => {
                self.push(FrontendOp::new(OpRef::NONE, ConcreteValue::Null));
                Ok(StepAction::Continue)
            }
            Instruction::JumpBackwardNoInterrupt { delta } => {
                let target = self.pc - u32::from(delta.get(op_arg)) as usize;
                self.pc = target;
                Ok(StepAction::Continue)
            }

            // ── Exception handling ──
            Instruction::PushExcInfo => {
                let exc = self.pop();
                self.push(FrontendOp::new(OpRef::NONE, ConcreteValue::Null)); // prev exc
                self.push(exc);
                Ok(StepAction::Continue)
            }
            Instruction::PopExcept => {
                self.pop(); // prev exc
                Ok(StepAction::Continue)
            }
            Instruction::CheckExcMatch => {
                let exc_type = self.pop();
                let _exc_value = self.peek();
                // TODO: proper type matching
                self.push(FrontendOp::new(OpRef::NONE, ConcreteValue::Int(1))); // always match
                Ok(StepAction::Continue)
            }
            Instruction::RaiseVarargs { argc } => {
                let n = argc.get(op_arg) as usize;
                for _ in 0..n {
                    self.pop();
                }
                Err(pyre_runtime::PyError::runtime_error("raise during tracing"))
            }

            // ── Iterator ──
            Instruction::GetIter => {
                // TOS is iterable, convert to iterator concrete
                // For now pass through — concrete value stays
                Ok(StepAction::Continue)
            }
            Instruction::ForIter { delta } => {
                let iter = self.peek();
                let continues = match iter.concrete {
                    ConcreteValue::Ref(obj) if !obj.is_null() => {
                        pyre_runtime::range_iter_continues(obj).unwrap_or(false)
                    }
                    _ => false,
                };
                if continues {
                    let next_val = match iter.concrete {
                        ConcreteValue::Ref(obj) => {
                            let v = pyre_runtime::range_iter_next_or_null(obj)
                                .unwrap_or(pyre_object::PY_NULL);
                            if v.is_null() {
                                ConcreteValue::Null
                            } else {
                                ConcreteValue::from_pyobj(v)
                            }
                        }
                        _ => ConcreteValue::Null,
                    };
                    self.push(FrontendOp::new(OpRef::NONE, next_val));
                } else {
                    self.pop(); // pop exhausted iterator
                    let target = self.pc + u32::from(delta.get(op_arg)) as usize;
                    self.pc = target;
                }
                Ok(StepAction::Continue)
            }

            // ── Unpack sequence ──
            Instruction::UnpackSequence { count } => {
                let n = count.get(op_arg) as usize;
                let seq = self.pop();
                // Push items in reverse order (Python semantics)
                let obj = seq.concrete.to_pyobj();
                // Concrete unpack: extract items from tuple/list
                let items = pyre_runtime::unpack_sequence_exact(obj, n);
                match items {
                    Ok(items) => {
                        for item in items.into_iter().rev() {
                            self.push(FrontendOp::new(
                                OpRef::NONE,
                                ConcreteValue::from_pyobj(item),
                            ));
                        }
                    }
                    Err(_) => {
                        for _ in 0..n {
                            self.push(FrontendOp::new(OpRef::NONE, ConcreteValue::Null));
                        }
                    }
                }
                Ok(StepAction::Continue)
            }

            // ── String / Format ──
            Instruction::BuildString { count } => {
                let n = count.get(op_arg) as usize;
                let mut parts = Vec::with_capacity(n);
                for _ in 0..n {
                    parts.push(self.pop());
                }
                parts.reverse();
                // Concrete: concatenate string representations
                let mut result = String::new();
                for part in &parts {
                    let obj = part.concrete.to_pyobj();
                    if !obj.is_null() {
                        result.push_str(&pyre_objspace::space::py_str(obj));
                    }
                }
                let str_obj = pyre_object::w_str_new(&result);
                self.push(FrontendOp::new(OpRef::NONE, ConcreteValue::Ref(str_obj)));
                Ok(StepAction::Continue)
            }
            Instruction::FormatSimple => {
                let val = self.pop();
                let s = pyre_objspace::space::py_str(val.concrete.to_pyobj());
                self.push(FrontendOp::new(
                    OpRef::NONE,
                    ConcreteValue::Ref(pyre_object::w_str_new(&s)),
                ));
                Ok(StepAction::Continue)
            }

            // ── Boolean ──
            Instruction::ToBool => {
                let val = self.pop();
                let truth = val.concrete.is_truthy();
                self.push(FrontendOp::new(
                    OpRef::NONE,
                    ConcreteValue::Int(truth as i64),
                ));
                Ok(StepAction::Continue)
            }

            // ── None checks ──
            Instruction::PopJumpIfNone { delta } => {
                let val = self.pop();
                let is_none = matches!(val.concrete, ConcreteValue::Ref(obj) if unsafe { pyre_object::is_none(obj) })
                    || matches!(val.concrete, ConcreteValue::Null);
                if is_none {
                    let target = self.pc + u32::from(delta.get(op_arg)) as usize;
                    self.pc = target;
                }
                Ok(StepAction::Continue)
            }
            Instruction::PopJumpIfNotNone { delta } => {
                let val = self.pop();
                let is_none = matches!(val.concrete, ConcreteValue::Ref(obj) if unsafe { pyre_object::is_none(obj) })
                    || matches!(val.concrete, ConcreteValue::Null);
                if !is_none {
                    let target = self.pc + u32::from(delta.get(op_arg)) as usize;
                    self.pc = target;
                }
                Ok(StepAction::Continue)
            }

            // ── Contains / Is ──
            Instruction::ContainsOp { invert } => {
                let haystack = self.pop();
                let needle = self.pop();
                let result = pyre_objspace::space::py_contains(
                    haystack.concrete.to_pyobj(),
                    needle.concrete.to_pyobj(),
                )
                .unwrap_or(false);
                let inverted = match invert.get(op_arg) {
                    pyre_bytecode::bytecode::Invert::No => result,
                    pyre_bytecode::bytecode::Invert::Yes => !result,
                };
                self.push(FrontendOp::new(
                    OpRef::NONE,
                    ConcreteValue::Int(inverted as i64),
                ));
                Ok(StepAction::Continue)
            }
            Instruction::IsOp { invert } => {
                let b = self.pop();
                let a = self.pop();
                let same = std::ptr::eq(a.concrete.to_pyobj(), b.concrete.to_pyobj());
                let result = match invert.get(op_arg) {
                    pyre_bytecode::bytecode::Invert::No => same,
                    pyre_bytecode::bytecode::Invert::Yes => !same,
                };
                self.push(FrontendOp::new(
                    OpRef::NONE,
                    ConcreteValue::Int(result as i64),
                ));
                Ok(StepAction::Continue)
            }

            // ── LoadFast pairs ──
            Instruction::LoadFastBorrowLoadFastBorrow { var_nums } => {
                let pair = var_nums.get(op_arg);
                let idx1 = u32::from(pair.idx_1()) as usize;
                let idx2 = u32::from(pair.idx_2()) as usize;
                let c1 = self
                    .sym
                    .concrete_locals
                    .get(idx1)
                    .copied()
                    .unwrap_or(ConcreteValue::Null);
                let c2 = self
                    .sym
                    .concrete_locals
                    .get(idx2)
                    .copied()
                    .unwrap_or(ConcreteValue::Null);
                self.push(FrontendOp::new(OpRef::NONE, c1));
                self.push(FrontendOp::new(OpRef::NONE, c2));
                Ok(StepAction::Continue)
            }
            Instruction::StoreFastLoadFast { var_nums } => {
                let pair = var_nums.get(op_arg);
                let store_idx = u32::from(pair.idx_1()) as usize;
                let load_idx = u32::from(pair.idx_2()) as usize;
                let val = self.pop();
                if store_idx < self.sym.concrete_locals.len() {
                    self.sym.concrete_locals[store_idx] = val.concrete;
                }
                let c = self
                    .sym
                    .concrete_locals
                    .get(load_idx)
                    .copied()
                    .unwrap_or(ConcreteValue::Null);
                self.push(FrontendOp::new(OpRef::NONE, c));
                Ok(StepAction::Continue)
            }

            // ── Unsupported (abort trace) ──
            _ => {
                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[metainterp] unsupported instruction at pc={}: {:?}",
                        self.pc - 1,
                        instruction
                    );
                }
                Ok(StepAction::Abort)
            }
        }
    }
}
