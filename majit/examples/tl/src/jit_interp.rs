/// JIT-enabled TL interpreter — structural mirror of tl.py with JitDriver.
///
/// Greens: [pc, code]    (code is constant per trace)
/// Reds:   [inputarg, stack]
///
/// The stack is virtualizable: during tracing, stack slots are mapped to
/// IR operations (OpRef), eliminating memory loads/stores in compiled code.
use std::collections::HashMap;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmEnterState};

const NOP: u8 = 1;
const PUSH: u8 = 2;
const POP: u8 = 3;
const SWAP: u8 = 4;
const PICK: u8 = 6;
const PUT: u8 = 7;
const ADD: u8 = 8;
const SUB: u8 = 9;
const MUL: u8 = 10;
const EQ: u8 = 12;
const NE: u8 = 13;
const LT: u8 = 14;
const LE: u8 = 15;
const GT: u8 = 16;
const GE: u8 = 17;
const BR_COND: u8 = 18;
const RETURN: u8 = 21;
const PUSHARG: u8 = 22;

const DEFAULT_THRESHOLD: u32 = 3;

struct CompiledLoop {
    token: LoopToken,
    num_stack_slots: usize,
}

struct TracingState {
    recorder: TraceRecorder,
    loop_header_pc: usize,
    trace_stack: Vec<OpRef>,
    num_stack_slots: usize,
    constants: HashMap<u32, i64>,
    next_const_ref: u32,
}

impl TracingState {
    fn const_ref(&mut self, value: i64) -> OpRef {
        for (&idx, &v) in &self.constants {
            if v == value {
                return OpRef(idx);
            }
        }
        let opref = OpRef(self.next_const_ref);
        self.next_const_ref += 1;
        self.constants.insert(opref.0, value);
        opref
    }
}

pub struct JitTlInterp {
    warm_state: WarmEnterState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    tracing: Option<TracingState>,
}

impl JitTlInterp {
    pub fn new() -> Self {
        JitTlInterp {
            warm_state: WarmEnterState::new(DEFAULT_THRESHOLD),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    pub fn run(&mut self, code: &[u8], inputarg: i64) -> i64 {
        let mut stack: Vec<i64> = Vec::with_capacity(code.len());
        let mut pc: usize = 0;

        loop {
            // --- tracing: record instruction ---
            if self.tracing.is_some() {
                match self.trace_instruction(code, pc, &stack, inputarg) {
                    TraceAction::Continue => {}
                    TraceAction::CloseLoop => {
                        self.close_and_compile();
                    }
                    TraceAction::Abort => {
                        self.abort_trace();
                    }
                }
            }

            if pc >= code.len() {
                break;
            }

            let opcode = code[pc];
            pc += 1;

            if opcode == NOP {
                // no-op
            } else if opcode == PUSH {
                stack.push(code[pc] as i8 as i64);
                pc += 1;
            } else if opcode == POP {
                stack.pop();
            } else if opcode == SWAP {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a);
                stack.push(b);
            } else if opcode == PICK {
                let i = code[pc] as usize;
                pc += 1;
                let n = stack.len() - i - 1;
                let val = stack[n];
                stack.push(val);
            } else if opcode == PUT {
                let i = code[pc] as usize;
                pc += 1;
                let val = stack.pop().unwrap();
                let n = stack.len() - i - 1;
                stack[n] = val;
            } else if opcode == ADD {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(b + a);
            } else if opcode == SUB {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(b - a);
            } else if opcode == MUL {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(b * a);
            } else if opcode == EQ {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(if b == a { 1 } else { 0 });
            } else if opcode == NE {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(if b != a { 1 } else { 0 });
            } else if opcode == LT {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(if b < a { 1 } else { 0 });
            } else if opcode == LE {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(if b <= a { 1 } else { 0 });
            } else if opcode == GT {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(if b > a { 1 } else { 0 });
            } else if opcode == GE {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(if b >= a { 1 } else { 0 });
            } else if opcode == BR_COND {
                let offset = code[pc] as i8 as i64;
                let target = (pc as i64 + offset + 1) as usize;
                let next_pc = pc + 1;
                let cond = stack.pop().unwrap();
                if cond != 0 {
                    if target < next_pc && self.tracing.is_none() {
                        // Back-edge: check hotness
                        match self.warm_state.maybe_compile(target as u64) {
                            HotResult::NotHot => {}
                            HotResult::StartTracing(recorder) => {
                                self.start_tracing(recorder, target, &stack);
                            }
                            HotResult::AlreadyTracing => {}
                            HotResult::RunCompiled => {
                                if let Some(new_stack) = self.run_compiled(target, &stack) {
                                    stack = new_stack;
                                    pc = target;
                                    continue;
                                }
                            }
                        }
                    }
                    pc = target;
                } else {
                    pc = next_pc;
                }
            } else if opcode == RETURN {
                break;
            } else if opcode == PUSHARG {
                stack.push(inputarg);
            }
        }

        stack.pop().unwrap()
    }

    fn start_tracing(&mut self, recorder: TraceRecorder, loop_header_pc: usize, stack: &[i64]) {
        let num_stack_slots = stack.len();
        let mut state = TracingState {
            recorder,
            loop_header_pc,
            trace_stack: Vec::new(),
            num_stack_slots,
            constants: HashMap::new(),
            next_const_ref: 10_000,
        };

        // Create input args for each stack slot.
        for _ in 0..num_stack_slots {
            let opref = state.recorder.record_input_arg(Type::Int);
            state.trace_stack.push(opref);
        }

        self.tracing = Some(state);
    }

    fn trace_instruction(
        &mut self,
        code: &[u8],
        pc: usize,
        runtime_stack: &[i64],
        inputarg: i64,
    ) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();
        let opcode = code[pc];

        if opcode == NOP {
            // no-op
        } else if opcode == PUSH {
            let val = code[pc + 1] as i8 as i64;
            let opref = state.const_ref(val);
            state.trace_stack.push(opref);
        } else if opcode == POP {
            state.trace_stack.pop();
        } else if opcode == SWAP {
            let len = state.trace_stack.len();
            state.trace_stack.swap(len - 1, len - 2);
        } else if opcode == PICK {
            let i = code[pc + 1] as usize;
            let n = state.trace_stack.len() - i - 1;
            let opref = state.trace_stack[n];
            state.trace_stack.push(opref);
        } else if opcode == PUT {
            let i = code[pc + 1] as usize;
            let opref = state.trace_stack.pop().unwrap();
            let n = state.trace_stack.len() - i - 1;
            state.trace_stack[n] = opref;
        } else if opcode == ADD {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntAdd, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == SUB {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntSub, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == MUL {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntMul, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == EQ {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntEq, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == NE {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntNe, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == LT {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntLt, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == LE {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntLe, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == GT {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntGt, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == GE {
            let a = state.trace_stack.pop().unwrap();
            let b = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntGe, &[b, a]);
            state.trace_stack.push(result);
        } else if opcode == BR_COND {
            let offset = code[pc + 1] as i8 as i64;
            let target = ((pc + 1) as i64 + offset + 1) as usize;
            let cond = state.trace_stack.pop().unwrap();

            // Check runtime condition to determine which path was taken.
            let runtime_cond = *runtime_stack.last().unwrap();

            if runtime_cond != 0 && target == state.loop_header_pc {
                // Back-edge to loop header: guard and close loop.
                let fail_descr = make_fail_descr(state.num_stack_slots);
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[cond], fail_descr);
                return TraceAction::CloseLoop;
            } else if runtime_cond != 0 {
                // Forward branch taken: guard that condition is true.
                let fail_descr = make_fail_descr(state.num_stack_slots);
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[cond], fail_descr);
            } else {
                // Branch not taken: guard that condition is false.
                let fail_descr = make_fail_descr(state.num_stack_slots);
                state
                    .recorder
                    .record_guard(OpCode::GuardFalse, &[cond], fail_descr);
            }
        } else if opcode == PUSHARG {
            let opref = state.const_ref(inputarg);
            state.trace_stack.push(opref);
        } else if opcode == RETURN {
            return TraceAction::Abort;
        } else {
            // Unsupported opcode during tracing (ROLL, CALL, BR_COND_STK, DIV, etc.)
            return TraceAction::Abort;
        }

        if self.tracing.as_ref().unwrap().recorder.is_too_long() {
            return TraceAction::Abort;
        }
        TraceAction::Continue
    }

    fn close_and_compile(&mut self) {
        let state = self.tracing.take().unwrap();
        let green_key = state.loop_header_pc as u64;

        let jump_args: Vec<OpRef> = state.trace_stack.clone();

        let mut recorder = state.recorder;
        recorder.close_loop(&jump_args);
        let trace = recorder.get_trace();

        let mut optimizer = Optimizer::default_pipeline();
        let mut constants = state.constants;

        if std::env::var("MAJIT_LOG").is_ok() {
            eprintln!("--- trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace.ops, &constants));
        }

        let optimized_ops = optimizer.optimize_with_constants(&trace.ops, &mut constants);

        if std::env::var("MAJIT_LOG").is_ok() {
            eprintln!("--- trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        self.backend.set_constants(constants);

        let token_num = self.warm_state.alloc_token_number();
        let mut token = LoopToken::new(token_num);

        match self
            .backend
            .compile_loop(&trace.inputargs, &optimized_ops, &mut token)
        {
            Ok(_) => {
                self.compiled_loops.insert(
                    state.loop_header_pc,
                    CompiledLoop {
                        token,
                        num_stack_slots: state.num_stack_slots,
                    },
                );
                let install_num = self.warm_state.alloc_token_number();
                let install_token = LoopToken::new(install_num);
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
            state.recorder.abort();
            self.warm_state
                .abort_tracing(state.loop_header_pc as u64, false);
        }
    }

    fn run_compiled(&mut self, loop_pc: usize, stack: &[i64]) -> Option<Vec<i64>> {
        let compiled = self.compiled_loops.get(&loop_pc)?;

        let args: Vec<Value> = stack.iter().map(|&v| Value::Int(v)).collect();

        let frame = self.backend.execute_token(&compiled.token, &args);

        let mut new_stack = Vec::new();
        for i in 0..compiled.num_stack_slots {
            new_stack.push(self.backend.get_int_value(&frame, i));
        }

        Some(new_stack)
    }
}

enum TraceAction {
    Continue,
    CloseLoop,
    Abort,
}

fn make_fail_descr(num_live: usize) -> majit_ir::DescrRef {
    use std::sync::Arc;
    Arc::new(TlFailDescr {
        types: vec![Type::Int; num_live],
    })
}

#[derive(Debug)]
struct TlFailDescr {
    types: Vec<Type>,
}

impl majit_ir::Descr for TlFailDescr {
    fn index(&self) -> u32 {
        0
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for TlFailDescr {
    fn fail_index(&self) -> u32 {
        0
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.types
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    /// sum(N) = 1 + 2 + ... + N
    fn sum_bytecode() -> Vec<u8> {
        const PUSH: u8 = 2;
        const PUSHARG: u8 = 22;
        const PICK: u8 = 6;
        const BR_COND: u8 = 18;
        const POP: u8 = 3;
        const RETURN: u8 = 21;
        const SWAP: u8 = 4;
        const ADD: u8 = 8;
        const SUB: u8 = 9;
        vec![
            PUSH, 0,       // acc = 0
            PUSHARG, // counter = N
            // loop (offset 3):
            PICK, 0, // dup counter
            BR_COND, 2,      // if counter != 0, skip to body (offset 9)
            POP,    // pop counter
            RETURN, // body (offset 9):
            SWAP,   // [counter, acc]
            PICK, 1,    // [counter, acc, counter]
            ADD,  // [counter, acc+counter]
            SWAP, // [acc+counter, counter]
            PUSH, 1, SUB, // [acc, counter-1]
            PUSH, 1, BR_COND, 238, // -18: jump to loop (offset 3)
        ]
    }

    #[test]
    fn jit_sum_5() {
        let bc = sum_bytecode();
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&bc, 5), 15);
    }

    #[test]
    fn jit_sum_100() {
        let bc = sum_bytecode();
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&bc, 100), 5050);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = sum_bytecode();
        for a in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![PUSH, 42, RETURN];
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&prog, 0), 42);
    }
}
