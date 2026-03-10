/// JIT-enabled bytecode interpreter for the calc language.
///
/// Wraps the base CalcInterp with meta-tracing JIT compilation.
/// At backward jumps (loop back-edges), the warm state decides whether
/// to start tracing, continue interpreting, or run compiled code.
use std::collections::HashMap;

use crate::bytecode::ByteCode;
use crate::interp::CalcInterp;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmState};

/// Default hot threshold: number of back-edge hits before tracing starts.
const DEFAULT_THRESHOLD: u32 = 3;

/// Per-loop compiled information.
struct CompiledLoop {
    token: LoopToken,
    /// Which variables are live at the loop header (indices into vars[]).
    /// The order matches the input/output argument order.
    live_vars: Vec<u8>,
}

/// State kept while tracing a single loop.
struct TracingState {
    recorder: TraceRecorder,
    /// The PC of the loop header we are tracing.
    loop_header_pc: usize,
    /// Trace-level stack: maps interpreter stack positions to OpRefs.
    trace_stack: Vec<OpRef>,
    /// Trace-level variable bindings: var index -> OpRef.
    trace_vars: HashMap<u8, OpRef>,
    /// Which variables are live at the loop header, in order.
    live_vars: Vec<u8>,
    /// Constants recorded during tracing: OpRef index -> constant value.
    constants: HashMap<u32, i64>,
    /// Next virtual OpRef for constants (allocated from a high range to avoid
    /// collision with the recorder's own OpRef space).
    next_const_ref: u32,
}

impl TracingState {
    fn new(recorder: TraceRecorder, loop_header_pc: usize) -> Self {
        TracingState {
            recorder,
            loop_header_pc,
            trace_stack: Vec::new(),
            trace_vars: HashMap::new(),
            live_vars: Vec::new(),
            constants: HashMap::new(),
            next_const_ref: 10_000,
        }
    }

    /// Get or create a constant OpRef for the given value.
    fn const_ref(&mut self, value: i64) -> OpRef {
        // Check if we already have this constant
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

pub struct JitCalcInterp {
    interp: CalcInterp,
    warm_state: WarmState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    /// Active tracing state (at most one trace at a time).
    tracing: Option<TracingState>,
}

impl JitCalcInterp {
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_THRESHOLD)
    }

    pub fn with_threshold(threshold: u32) -> Self {
        JitCalcInterp {
            interp: CalcInterp::new(),
            warm_state: WarmState::new(threshold),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    pub fn reset(&mut self) {
        self.interp.reset();
        // Keep JIT state (compiled_loops, warm_state) across resets.
    }

    /// Execute the bytecode program and return the final result.
    pub fn run(&mut self, bytecode: &[ByteCode]) -> i64 {
        self.interp.reset();

        loop {
            let pc = self.interp.pc();
            let instr = &bytecode[pc];
            self.interp.set_pc(pc + 1);

            // If we are tracing, record this instruction.
            if self.tracing.is_some() {
                let action = self.trace_instruction(instr, pc, bytecode);
                match action {
                    TraceAction::Continue => {}
                    TraceAction::CloseLoop => {
                        self.close_and_compile_trace(pc);
                    }
                    TraceAction::Abort => {
                        self.abort_trace(pc);
                    }
                }
                // After recording, still execute the instruction normally.
            }

            match instr {
                ByteCode::LoadConst(v) => {
                    self.interp.push_stack(*v);
                }
                ByteCode::LoadVar(idx) => {
                    self.interp.push_stack(self.interp.get_var(*idx));
                }
                ByteCode::StoreVar(idx) => {
                    let val = self.interp.pop_stack();
                    self.interp.set_var(*idx, val);
                }
                ByteCode::Add => self.interp.do_binop(|a, b| a + b),
                ByteCode::Sub => self.interp.do_binop(|a, b| a - b),
                ByteCode::Mul => self.interp.do_binop(|a, b| a * b),
                ByteCode::Div => self.interp.do_binop(|a, b| a / b),
                ByteCode::Mod => self.interp.do_binop(|a, b| a % b),
                ByteCode::Lt => self.interp.do_cmpop(|a, b| a < b),
                ByteCode::Le => self.interp.do_cmpop(|a, b| a <= b),
                ByteCode::Eq => self.interp.do_cmpop(|a, b| a == b),
                ByteCode::Ne => self.interp.do_cmpop(|a, b| a != b),
                ByteCode::Gt => self.interp.do_cmpop(|a, b| a > b),
                ByteCode::Ge => self.interp.do_cmpop(|a, b| a >= b),
                ByteCode::JumpIfFalse(target) => {
                    let val = self.interp.pop_stack();
                    if val == 0 {
                        self.interp.set_pc(*target as usize);
                    }
                }
                ByteCode::Jump(target) => {
                    let target_pc = *target as usize;
                    // Backward jump = back-edge (potential loop header)
                    if target_pc <= pc && self.tracing.is_none() {
                        let action = self.check_hot(target_pc, bytecode);
                        match action {
                            BackEdgeAction::Interpret => {}
                            BackEdgeAction::RunCompiled => {
                                // Run compiled code and update state
                                if let Some(result) = self.run_compiled(target_pc) {
                                    return result;
                                }
                                // If run_compiled didn't halt, continue interpreting
                                continue;
                            }
                        }
                    }
                    self.interp.set_pc(target_pc);
                }
                ByteCode::Print => {
                    let val = self.interp.pop_stack();
                    println!("{val}");
                }
                ByteCode::Halt => {
                    return self.interp.top_or_zero();
                }
            }
        }
    }

    /// Check whether a back-edge target is hot.
    /// Returns what to do next.
    fn check_hot(&mut self, target_pc: usize, bytecode: &[ByteCode]) -> BackEdgeAction {
        let green_key = target_pc as u64;
        match self.warm_state.maybe_compile(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing(recorder) => {
                self.start_tracing(recorder, target_pc, bytecode);
                BackEdgeAction::Interpret
            }
            HotResult::AlreadyTracing => BackEdgeAction::Interpret,
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
        }
    }

    /// Begin tracing at the given loop header.
    fn start_tracing(
        &mut self,
        recorder: TraceRecorder,
        loop_header_pc: usize,
        bytecode: &[ByteCode],
    ) {
        let mut state = TracingState::new(recorder, loop_header_pc);

        // Determine live variables at the loop header by scanning the loop body.
        let live = self.scan_live_vars(loop_header_pc, bytecode);
        state.live_vars = live.clone();

        // Register input arguments for each live variable.
        for &var_idx in &live {
            let opref = state.recorder.record_input_arg(Type::Int);
            state.trace_vars.insert(var_idx, opref);
        }

        self.tracing = Some(state);
    }

    /// Scan forward from loop_header_pc to find which variables are live
    /// (loaded before being stored) within the loop body.
    fn scan_live_vars(&self, header_pc: usize, bytecode: &[ByteCode]) -> Vec<u8> {
        let mut loaded = Vec::new();
        let mut stored = Vec::new();
        let mut pc = header_pc;

        loop {
            if pc >= bytecode.len() {
                break;
            }
            match &bytecode[pc] {
                ByteCode::LoadVar(idx) => {
                    if !stored.contains(idx) && !loaded.contains(idx) {
                        loaded.push(*idx);
                    }
                }
                ByteCode::StoreVar(idx) => {
                    if !stored.contains(idx) {
                        stored.push(*idx);
                    }
                }
                ByteCode::Jump(target) => {
                    if *target as usize == header_pc {
                        break; // reached the back-edge
                    }
                }
                ByteCode::Halt => break,
                _ => {}
            }
            pc += 1;
        }

        // All variables that appear anywhere in the loop (loaded or stored)
        // are needed as input args so we can reconstruct state on guard failure.
        let mut all_vars: Vec<u8> = loaded;
        for v in stored {
            if !all_vars.contains(&v) {
                all_vars.push(v);
            }
        }
        all_vars.sort();
        all_vars
    }

    /// Record one instruction into the active trace.
    fn trace_instruction(
        &mut self,
        instr: &ByteCode,
        _current_pc: usize,
        _bytecode: &[ByteCode],
    ) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();

        match instr {
            ByteCode::LoadConst(v) => {
                let opref = state.const_ref(*v);
                state.trace_stack.push(opref);
            }
            ByteCode::LoadVar(idx) => {
                let opref = state
                    .trace_vars
                    .get(idx)
                    .copied()
                    .unwrap_or_else(|| state.const_ref(self.interp.get_var(*idx)));
                state.trace_stack.push(opref);
            }
            ByteCode::StoreVar(idx) => {
                let opref = state.trace_stack.pop().expect("trace stack underflow");
                state.trace_vars.insert(*idx, opref);
            }
            ByteCode::Add => {
                self.trace_binop(OpCode::IntAdd);
            }
            ByteCode::Sub => {
                self.trace_binop(OpCode::IntSub);
            }
            ByteCode::Mul => {
                self.trace_binop(OpCode::IntMul);
            }
            ByteCode::Div => {
                // No IR div opcode; abort tracing for now.
                return TraceAction::Abort;
            }
            ByteCode::Mod => {
                // No IR mod opcode; abort tracing for now.
                return TraceAction::Abort;
            }
            ByteCode::Lt => {
                self.trace_binop(OpCode::IntLt);
            }
            ByteCode::Le => {
                self.trace_binop(OpCode::IntLe);
            }
            ByteCode::Eq => {
                self.trace_binop(OpCode::IntEq);
            }
            ByteCode::Ne => {
                self.trace_binop(OpCode::IntNe);
            }
            ByteCode::Gt => {
                self.trace_binop(OpCode::IntGt);
            }
            ByteCode::Ge => {
                self.trace_binop(OpCode::IntGe);
            }
            ByteCode::JumpIfFalse(_target) => {
                let cond_ref = state.trace_stack.pop().expect("trace stack underflow");
                // Peek at the actual runtime value to decide which branch to guard.
                // The interpreter hasn't executed this instruction yet, so the value
                // is still on the interpreter stack.
                let runtime_val = *self.interp.stack().last().expect("stack underflow");

                let state = self.tracing.as_mut().unwrap();
                // Create a FailDescr for the guard.
                let fail_descr = make_guard_fail_descr(0, state.live_vars.len());

                if runtime_val != 0 {
                    // Condition is true -> we do NOT jump -> guard that it stays true
                    state
                        .recorder
                        .record_guard(OpCode::GuardTrue, &[cond_ref], fail_descr);
                } else {
                    // Condition is false -> we jump -> guard that it stays false
                    state
                        .recorder
                        .record_guard(OpCode::GuardFalse, &[cond_ref], fail_descr);
                }
            }
            ByteCode::Jump(target) => {
                let target_pc = *target as usize;
                let header = self.tracing.as_ref().unwrap().loop_header_pc;
                if target_pc == header {
                    return TraceAction::CloseLoop;
                }
                // Forward jump inside the loop body: just continue tracing.
            }
            ByteCode::Print => {
                // Print has side effects; pop the trace stack ref.
                let state = self.tracing.as_mut().unwrap();
                state.trace_stack.pop();
            }
            ByteCode::Halt => {
                return TraceAction::Abort;
            }
        }

        // Check trace length limit
        if self.tracing.as_ref().unwrap().recorder.is_too_long() {
            return TraceAction::Abort;
        }

        TraceAction::Continue
    }

    /// Record a binary operation into the trace.
    fn trace_binop(&mut self, opcode: OpCode) {
        let state = self.tracing.as_mut().unwrap();
        let b = state.trace_stack.pop().expect("trace stack underflow");
        let a = state.trace_stack.pop().expect("trace stack underflow");
        let result = state.recorder.record_op(opcode, &[a, b]);
        state.trace_stack.push(result);
    }

    /// Close the loop trace and compile it.
    fn close_and_compile_trace(&mut self, _current_pc: usize) {
        let state = self.tracing.take().unwrap();
        let green_key = state.loop_header_pc as u64;

        // Build the jump args: current OpRef for each live variable.
        let jump_args: Vec<OpRef> = state
            .live_vars
            .iter()
            .map(|var_idx| {
                state
                    .trace_vars
                    .get(var_idx)
                    .copied()
                    .expect("live var has no trace binding")
            })
            .collect();

        let mut recorder = state.recorder;
        recorder.close_loop(&jump_args);

        let trace = recorder.get_trace();

        // Optimize the trace
        let mut optimizer = Optimizer::new();
        let optimized_ops = optimizer.optimize(&trace.ops);

        // Compile the trace
        self.backend.set_constants(state.constants);

        let token_num = self.warm_state.alloc_token_number();
        let mut token = LoopToken::new(token_num);

        match self
            .backend
            .compile_loop(&trace.inputargs, &optimized_ops, &mut token)
        {
            Ok(_info) => {
                let compiled = CompiledLoop {
                    token,
                    live_vars: state.live_vars,
                };
                let loop_pc = state.loop_header_pc;
                self.compiled_loops.insert(loop_pc, compiled);

                // Install into warm state so next time we get RunCompiled
                let install_token_num = self.warm_state.alloc_token_number();
                let install_token = LoopToken::new(install_token_num);
                self.warm_state.install_compiled(green_key, install_token);
            }
            Err(e) => {
                eprintln!("JIT compilation failed: {e}");
                self.warm_state
                    .abort_tracing(green_key, /* dont_trace_here */ true);
            }
        }
    }

    /// Abort the current trace.
    fn abort_trace(&mut self, _current_pc: usize) {
        if let Some(state) = self.tracing.take() {
            let green_key = state.loop_header_pc as u64;
            state.recorder.abort();
            self.warm_state
                .abort_tracing(green_key, /* dont_trace_here */ false);
        }
    }

    /// Run compiled code for a loop at the given PC.
    /// Returns Some(result) if Halt was reached, None to continue interpreting.
    fn run_compiled(&mut self, loop_pc: usize) -> Option<i64> {
        let compiled = self.compiled_loops.get(&loop_pc)?;

        // Build input args from current variable values.
        let args: Vec<Value> = compiled
            .live_vars
            .iter()
            .map(|&var_idx| Value::Int(self.interp.get_var(var_idx)))
            .collect();

        let frame = self.backend.execute_token(&compiled.token, &args);

        // Extract output values and update interpreter state.
        let live_vars = compiled.live_vars.clone();
        for (i, &var_idx) in live_vars.iter().enumerate() {
            let val = self.backend.get_int_value(&frame, i);
            self.interp.set_var(var_idx, val);
        }

        // Guard failed: resume interpreting.
        // The interpreter PC should be set to just after the guard failure point.
        // For now, re-enter the loop header for the next iteration check.
        self.interp.set_pc(loop_pc);
        None
    }
}

impl Default for JitCalcInterp {
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

/// Create a fail descriptor for a guard.
fn make_guard_fail_descr(
    _fail_index: u32,
    num_live_vars: usize,
) -> majit_ir::DescrRef {
    use std::sync::Arc;
    Arc::new(CalcFailDescr {
        fail_index: 0,
        fail_arg_types: vec![Type::Int; num_live_vars],
    })
}

/// Simple fail descriptor for calc guards.
#[derive(Debug)]
struct CalcFailDescr {
    fail_index: u32,
    fail_arg_types: Vec<Type>,
}

impl majit_ir::Descr for CalcFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for CalcFailDescr {
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
    use crate::interp::CalcInterp;

    #[test]
    fn test_jit_sum_matches_interp() {
        let prog = sum_program(1_000_000);

        let mut interp = CalcInterp::new();
        let expected = interp.run(&prog);

        let mut jit = JitCalcInterp::new();
        let result = jit.run(&prog);

        assert_eq!(result, expected);
        assert_eq!(result, 499_999_500_000);
    }

    #[test]
    fn test_jit_sum_small() {
        let prog = sum_program(100);
        let mut jit = JitCalcInterp::new();
        assert_eq!(jit.run(&prog), 4950);
    }

    #[test]
    fn test_jit_sum_zero() {
        let prog = sum_program(0);
        let mut jit = JitCalcInterp::new();
        assert_eq!(jit.run(&prog), 0);
    }

    #[test]
    fn test_jit_factorial() {
        let prog = factorial_program(10);

        let mut interp = CalcInterp::new();
        let expected = interp.run(&prog);

        let mut jit = JitCalcInterp::new();
        let result = jit.run(&prog);

        assert_eq!(result, expected);
        assert_eq!(result, 3_628_800);
    }

    #[test]
    fn test_jit_factorial_20() {
        let prog = factorial_program(20);

        let mut interp = CalcInterp::new();
        let expected = interp.run(&prog);

        let mut jit = JitCalcInterp::new();
        let result = jit.run(&prog);

        assert_eq!(result, expected);
        assert_eq!(result, 2_432_902_008_176_640_000);
    }

    #[test]
    fn test_jit_countdown() {
        // i = 10; while (i > 0) { i = i - 1; } -> result is i = 0
        use ByteCode::*;
        let prog = vec![
            /*  0 */ LoadConst(10),
            /*  1 */ StoreVar(0), // i = 10
            /*  2 */ LoadVar(0),  // loop header
            /*  3 */ LoadConst(0),
            /*  4 */ Gt, // i > 0?
            /*  5 */ JumpIfFalse(11),
            /*  6 */ LoadVar(0),
            /*  7 */ LoadConst(1),
            /*  8 */ Sub, // i - 1
            /*  9 */ StoreVar(0),
            /* 10 */ Jump(2),
            /* 11 */ LoadVar(0),
            /* 12 */ Halt,
        ];

        let mut interp = CalcInterp::new();
        let expected = interp.run(&prog);

        let mut jit = JitCalcInterp::new();
        let result = jit.run(&prog);

        assert_eq!(result, expected);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_jit_no_loop() {
        // Simple program with no loop - should work fine in JIT mode too
        use ByteCode::*;
        let prog = vec![LoadConst(3), LoadConst(7), Add, Halt];
        let mut jit = JitCalcInterp::new();
        assert_eq!(jit.run(&prog), 10);
    }

    #[test]
    fn test_jit_threshold_1() {
        // With threshold=1, the very first back-edge triggers tracing
        let prog = sum_program(100);
        let mut jit = JitCalcInterp::with_threshold(1);
        assert_eq!(jit.run(&prog), 4950);
    }

    #[test]
    fn test_jit_compiled_loop_reuse() {
        // Run the same program twice: second run should hit compiled code
        let prog = sum_program(100);
        let mut jit = JitCalcInterp::with_threshold(2);

        let r1 = jit.run(&prog);
        assert_eq!(r1, 4950);

        jit.reset();
        let r2 = jit.run(&prog);
        assert_eq!(r2, 4950);
    }
}
