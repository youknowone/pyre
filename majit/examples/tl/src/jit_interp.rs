/// JIT-enabled TL interpreter.
///
/// Wraps the base TlInterp with meta-tracing JIT compilation.
/// At backward branches (Br with negative offset or BrCond with negative offset),
/// the warm state triggers tracing/compilation.
use std::collections::HashMap;

use crate::bytecode::ByteCode;
use crate::interp::TlInterp;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmState};

const DEFAULT_THRESHOLD: u32 = 3;

struct CompiledLoop {
    token: LoopToken,
    /// Number of stack slots live at the loop header.
    stack_depth: usize,
}

struct TracingState {
    recorder: TraceRecorder,
    loop_header_pc: usize,
    /// Trace-level stack: maps interpreter stack positions to OpRefs.
    trace_stack: Vec<OpRef>,
    /// Number of stack slots at the loop header (these are the live values).
    header_stack_depth: usize,
    /// The inputarg value (a constant for the trace).
    inputarg: i64,
    /// Constants: OpRef index -> constant value.
    constants: HashMap<u32, i64>,
    next_const_ref: u32,
}

impl TracingState {
    fn new(recorder: TraceRecorder, loop_header_pc: usize) -> Self {
        TracingState {
            recorder,
            loop_header_pc,
            trace_stack: Vec::new(),
            header_stack_depth: 0,
            inputarg: 0,
            constants: HashMap::new(),
            next_const_ref: 10_000,
        }
    }

    fn const_ref(&mut self, value: i64) -> OpRef {
        for (&opref_idx, &v) in &self.constants {
            if v == value {
                return OpRef(opref_idx);
            }
        }
        let opref = OpRef(self.next_const_ref);
        self.next_const_ref += 1;
        self.constants.insert(opref.0, value);
        opref
    }
}

pub struct JitTlInterp {
    interp: TlInterp,
    warm_state: WarmState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    tracing: Option<TracingState>,
}

impl JitTlInterp {
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_THRESHOLD)
    }

    pub fn with_threshold(threshold: u32) -> Self {
        JitTlInterp {
            interp: TlInterp::new(),
            warm_state: WarmState::new(threshold),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    pub fn run(&mut self, bytecode: &[ByteCode], inputarg: i64) -> i64 {
        self.interp.reset();
        self.interp.set_pc(0);

        // We need to drive the interpreter manually for JIT integration.
        // Push inputarg state into the interp.
        let stack = self.interp.stack_mut();
        stack.clear();

        // We'll run using our own loop, delegating execution to the base interp methods.
        let mut pc = 0usize;
        let mut stack: Vec<i64> = Vec::with_capacity(256);

        loop {
            if pc >= bytecode.len() {
                break;
            }
            let instr = &bytecode[pc];
            pc += 1;

            // Record if tracing
            if self.tracing.is_some() {
                let action = self.trace_instruction(instr, pc - 1, &stack, inputarg);
                match action {
                    TraceAction::Continue => {}
                    TraceAction::CloseLoop => {
                        self.close_and_compile_trace(pc - 1, &stack);
                    }
                    TraceAction::Abort => {
                        self.abort_trace();
                    }
                }
            }

            match instr {
                ByteCode::Nop => {}
                ByteCode::Push(v) => stack.push(*v),
                ByteCode::Pop => { stack.pop(); }
                ByteCode::Swap => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(a);
                    stack.push(b);
                }
                ByteCode::Roll(r) => {
                    Self::do_roll(&mut stack, *r as i32);
                }
                ByteCode::Pick(i) => {
                    let n = stack.len() - (*i as usize) - 1;
                    let val = stack[n];
                    stack.push(val);
                }
                ByteCode::Put(i) => {
                    let elem = stack.pop().unwrap();
                    let n = stack.len() - (*i as usize) - 1;
                    stack[n] = elem;
                }
                ByteCode::Add => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(a + b); }
                ByteCode::Sub => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(a - b); }
                ByteCode::Mul => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(a * b); }
                ByteCode::Div => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(a / b); }
                ByteCode::Eq => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(if a == b { 1 } else { 0 }); }
                ByteCode::Ne => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(if a != b { 1 } else { 0 }); }
                ByteCode::Lt => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(if a < b { 1 } else { 0 }); }
                ByteCode::Le => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(if a <= b { 1 } else { 0 }); }
                ByteCode::Gt => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(if a > b { 1 } else { 0 }); }
                ByteCode::Ge => { let b = stack.pop().unwrap(); let a = stack.pop().unwrap(); stack.push(if a >= b { 1 } else { 0 }); }
                ByteCode::BrCond(offset) => {
                    let cond = stack.pop().unwrap();
                    if cond != 0 {
                        let target = (pc as i64 + *offset as i64) as usize;
                        if target < pc && self.tracing.is_none() {
                            let action = self.check_hot(target, &stack, inputarg);
                            match action {
                                BackEdgeAction::Interpret => {}
                                BackEdgeAction::RunCompiled => {
                                    if let Some(result) = self.run_compiled(target, &mut stack) {
                                        return result;
                                    }
                                    pc = self.interp.pc();
                                    continue;
                                }
                            }
                        }
                        pc = target;
                    }
                }
                ByteCode::BrCondStk => {
                    let offset = stack.pop().unwrap();
                    let cond = stack.pop().unwrap();
                    if cond != 0 {
                        pc = (pc as i64 + offset) as usize;
                    }
                }
                ByteCode::Call(offset) => {
                    stack.push(pc as i64);
                    pc = (pc as i64 + *offset as i64) as usize;
                }
                ByteCode::Return => break,
                ByteCode::PushArg => stack.push(inputarg),
                ByteCode::Br(offset) => {
                    let target = (pc as i64 + *offset as i64) as usize;
                    if target < pc && self.tracing.is_none() {
                        let action = self.check_hot(target, &stack, inputarg);
                        match action {
                            BackEdgeAction::Interpret => {}
                            BackEdgeAction::RunCompiled => {
                                if let Some(result) = self.run_compiled(target, &mut stack) {
                                    return result;
                                }
                                pc = self.interp.pc();
                                continue;
                            }
                        }
                    }
                    pc = target;
                }
            }
        }

        stack.pop().unwrap_or(0)
    }

    fn do_roll(stack: &mut Vec<i64>, r: i32) {
        let len = stack.len();
        if r < -1 {
            let i = (len as i32 + r) as usize;
            let n = len - 1;
            let elem = stack[n];
            let mut j = n;
            while j > i {
                stack[j] = stack[j - 1];
                j -= 1;
            }
            stack[i] = elem;
        } else if r > 1 {
            let i = len - r as usize;
            let elem = stack[i];
            for j in i..len - 1 {
                stack[j] = stack[j + 1];
            }
            stack[len - 1] = elem;
        }
    }

    fn check_hot(&mut self, target_pc: usize, stack: &[i64], inputarg: i64) -> BackEdgeAction {
        let green_key = target_pc as u64;
        match self.warm_state.maybe_compile(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing(recorder) => {
                self.start_tracing(recorder, target_pc, stack, inputarg);
                BackEdgeAction::Interpret
            }
            HotResult::AlreadyTracing => BackEdgeAction::Interpret,
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
        }
    }

    fn start_tracing(
        &mut self,
        recorder: TraceRecorder,
        loop_header_pc: usize,
        stack: &[i64],
        inputarg: i64,
    ) {
        let mut state = TracingState::new(recorder, loop_header_pc);
        state.inputarg = inputarg;
        state.header_stack_depth = stack.len();

        // Register input arguments for each stack slot at the loop header.
        for _ in 0..stack.len() {
            let opref = state.recorder.record_input_arg(Type::Int);
            state.trace_stack.push(opref);
        }

        self.tracing = Some(state);
    }

    fn trace_instruction(
        &mut self,
        instr: &ByteCode,
        current_pc: usize,
        runtime_stack: &[i64],
        inputarg: i64,
    ) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();

        match instr {
            ByteCode::Push(v) => {
                let opref = state.const_ref(*v);
                state.trace_stack.push(opref);
            }
            ByteCode::Pop => {
                state.trace_stack.pop();
            }
            ByteCode::PushArg => {
                let opref = state.const_ref(inputarg);
                state.trace_stack.push(opref);
            }
            ByteCode::Swap => {
                let len = state.trace_stack.len();
                if len >= 2 {
                    state.trace_stack.swap(len - 1, len - 2);
                }
            }
            ByteCode::Pick(i) => {
                let n = state.trace_stack.len() - (*i as usize) - 1;
                let opref = state.trace_stack[n];
                state.trace_stack.push(opref);
            }
            ByteCode::Add => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntAdd, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::Sub => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntSub, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::Mul => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntMul, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::Div => {
                return TraceAction::Abort;
            }
            ByteCode::Lt => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntLt, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::Le => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntLe, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::Eq => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntEq, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::Ne => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntNe, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::Gt => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntGt, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::Ge => {
                let b = state.trace_stack.pop().unwrap();
                let a = state.trace_stack.pop().unwrap();
                let result = state.recorder.record_op(OpCode::IntGe, &[a, b]);
                state.trace_stack.push(result);
            }
            ByteCode::BrCond(offset) => {
                let cond_ref = state.trace_stack.pop().unwrap();
                let target = (current_pc as i64 + 1 + *offset as i64) as usize;
                let header = state.loop_header_pc;

                if target == header {
                    // Back-edge to loop header: guard condition is true and close.
                    let fail_descr = make_guard_fail_descr(0, state.header_stack_depth);
                    state.recorder.record_guard(OpCode::GuardTrue, &[cond_ref], fail_descr);
                    return TraceAction::CloseLoop;
                }

                // Forward/other branch: guard the runtime direction.
                let runtime_cond = runtime_stack.last().copied().unwrap_or(0);
                let fail_descr = make_guard_fail_descr(0, state.header_stack_depth);
                if runtime_cond != 0 {
                    state.recorder.record_guard(OpCode::GuardTrue, &[cond_ref], fail_descr);
                } else {
                    state.recorder.record_guard(OpCode::GuardFalse, &[cond_ref], fail_descr);
                }
            }
            ByteCode::Br(offset) => {
                let target = (current_pc as i64 + 1 + *offset as i64) as usize;
                if target == state.loop_header_pc {
                    return TraceAction::CloseLoop;
                }
            }
            ByteCode::Return => return TraceAction::Abort,
            ByteCode::Nop | ByteCode::Roll(_) | ByteCode::Put(_)
            | ByteCode::BrCondStk | ByteCode::Call(_) => {
                return TraceAction::Abort;
            }
        }

        if self.tracing.as_ref().unwrap().recorder.is_too_long() {
            return TraceAction::Abort;
        }

        TraceAction::Continue
    }

    fn close_and_compile_trace(&mut self, _current_pc: usize, _runtime_stack: &[i64]) {
        let state = self.tracing.take().unwrap();
        let green_key = state.loop_header_pc as u64;

        // Jump args = current trace stack (should match header_stack_depth).
        let jump_args: Vec<OpRef> = state.trace_stack.clone();

        let mut recorder = state.recorder;
        recorder.close_loop(&jump_args);
        let trace = recorder.get_trace();

        let mut optimizer = Optimizer::new();
        let optimized_ops = optimizer.optimize(&trace.ops);

        self.backend.set_constants(state.constants);

        let token_num = self.warm_state.alloc_token_number();
        let mut token = LoopToken::new(token_num);

        match self.backend.compile_loop(&trace.inputargs, &optimized_ops, &mut token) {
            Ok(_) => {
                let compiled = CompiledLoop {
                    token,
                    stack_depth: state.header_stack_depth,
                };
                self.compiled_loops.insert(state.loop_header_pc, compiled);

                let install_token_num = self.warm_state.alloc_token_number();
                let install_token = LoopToken::new(install_token_num);
                self.warm_state.install_compiled(green_key, install_token);
            }
            Err(e) => {
                eprintln!("JIT compilation failed: {e}");
                self.warm_state.abort_tracing(green_key, true);
            }
        }
    }

    fn abort_trace(&mut self) {
        if let Some(state) = self.tracing.take() {
            let green_key = state.loop_header_pc as u64;
            state.recorder.abort();
            self.warm_state.abort_tracing(green_key, false);
        }
    }

    fn run_compiled(&mut self, loop_pc: usize, stack: &mut Vec<i64>) -> Option<i64> {
        let compiled = self.compiled_loops.get(&loop_pc)?;

        // Build input args from current stack.
        let depth = compiled.stack_depth;
        let base = stack.len() - depth;
        let args: Vec<Value> = (0..depth)
            .map(|i| Value::Int(stack[base + i]))
            .collect();

        let frame = self.backend.execute_token(&compiled.token, &args);

        // Update stack with output values.
        for i in 0..depth {
            stack[base + i] = self.backend.get_int_value(&frame, i);
        }

        self.interp.set_pc(loop_pc);
        None
    }
}

impl Default for JitTlInterp {
    fn default() -> Self {
        Self::new()
    }
}

enum TraceAction {
    Continue,
    CloseLoop,
    Abort,
}

enum BackEdgeAction {
    Interpret,
    RunCompiled,
}

fn make_guard_fail_descr(fail_index: u32, num_live: usize) -> majit_ir::DescrRef {
    use std::sync::Arc;
    Arc::new(TlFailDescr {
        fail_index,
        fail_arg_types: vec![Type::Int; num_live],
    })
}

#[derive(Debug)]
struct TlFailDescr {
    fail_index: u32,
    fail_arg_types: Vec<Type>,
}

impl majit_ir::Descr for TlFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for TlFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::*;

    #[test]
    fn test_jit_sum_10() {
        let (prog, arg) = sum_program(10);
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&prog, arg), 45);
    }

    #[test]
    fn test_jit_sum_100() {
        let (prog, arg) = sum_program(100);
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&prog, arg), 4950);
    }

    #[test]
    fn test_jit_factorial_10() {
        let (prog, arg) = factorial_program(10);
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&prog, arg), 3_628_800);
    }

    #[test]
    fn test_jit_square_5() {
        let (prog, arg) = square_program(5);
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&prog, arg), 25);
    }

    #[test]
    fn test_jit_square_100() {
        let (prog, arg) = square_program(100);
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&prog, arg), 10_000);
    }

    #[test]
    fn test_jit_matches_interp_sum() {
        let (prog, arg) = sum_program(1_000);
        let mut interp = TlInterp::new();
        let expected = interp.run(&prog, arg);

        let mut jit = JitTlInterp::new();
        let result = jit.run(&prog, arg);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_jit_no_loop() {
        use ByteCode::*;
        let prog = vec![Push(3), Push(7), Add, Return];
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&prog, 0), 10);
    }
}
