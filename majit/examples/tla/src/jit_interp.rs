/// JIT-enabled TLA interpreter — integer-specialized tracing.
///
/// TLA uses wrapped objects (W_IntObject, W_StringObject), but the JIT
/// specializes for integers: the trace_stack holds OpRef values representing
/// raw i64 values, bypassing object allocation and dispatch.
///
/// Greens: [pc, bytecode]
/// Reds:   [self]  (Frame with virtualizable stack)
use std::collections::HashMap;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmState};

use crate::interp::WObject;

const CONST_INT: u8 = 0;
const POP: u8 = 1;
const ADD: u8 = 2;
const RETURN: u8 = 3;
const JUMP_IF: u8 = 4;
const DUP: u8 = 5;
const SUB: u8 = 6;

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

pub struct JitTlaInterp {
    warm_state: WarmState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    tracing: Option<TracingState>,
}

impl JitTlaInterp {
    pub fn new() -> Self {
        JitTlaInterp {
            warm_state: WarmState::new(DEFAULT_THRESHOLD),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    /// Run the TLA interpreter with JIT support.
    /// Internally uses i64 for the stack (integer specialization).
    pub fn run(&mut self, bytecode: &[u8], w_arg: WObject) -> WObject {
        let initial = w_arg.int_value();
        let mut stack: Vec<i64> = Vec::with_capacity(8);
        stack.push(initial);
        let mut pc: usize = 0;

        loop {
            if self.tracing.is_some() {
                match self.trace_instruction(bytecode, pc, &stack) {
                    TraceAction::Continue => {}
                    TraceAction::CloseLoop => {
                        self.close_and_compile();
                    }
                    TraceAction::Abort => {
                        self.abort_trace();
                    }
                }
            }

            if pc >= bytecode.len() {
                break;
            }

            let opcode = bytecode[pc];
            pc += 1;

            if opcode == CONST_INT {
                let value = bytecode[pc] as i64;
                pc += 1;
                stack.push(value);
            } else if opcode == POP {
                stack.pop();
            } else if opcode == DUP {
                let v = *stack.last().unwrap();
                stack.push(v);
            } else if opcode == ADD {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a + b);
            } else if opcode == SUB {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a - b);
            } else if opcode == JUMP_IF {
                let target = bytecode[pc] as usize;
                pc += 1;
                let cond = stack.pop().unwrap();
                if cond != 0 {
                    if target < pc && self.tracing.is_none() {
                        match self.warm_state.maybe_compile(target as u64) {
                            HotResult::NotHot => {}
                            HotResult::StartTracing(recorder) => {
                                self.start_tracing(recorder, target, &stack);
                            }
                            HotResult::AlreadyTracing => {}
                            HotResult::RunCompiled => {
                                if let Some(new_stack) =
                                    self.run_compiled(target, &stack)
                                {
                                    stack = new_stack;
                                    pc = target;
                                    continue;
                                }
                            }
                        }
                    }
                    pc = target;
                }
            } else if opcode == RETURN {
                break;
            }
        }

        WObject::Int(stack.pop().unwrap())
    }

    fn start_tracing(
        &mut self,
        recorder: TraceRecorder,
        loop_header_pc: usize,
        stack: &[i64],
    ) {
        let num_stack_slots = stack.len();
        let mut state = TracingState {
            recorder,
            loop_header_pc,
            trace_stack: Vec::new(),
            num_stack_slots,
            constants: HashMap::new(),
            next_const_ref: 10_000,
        };

        for _ in 0..num_stack_slots {
            let opref = state.recorder.record_input_arg(Type::Int);
            state.trace_stack.push(opref);
        }

        self.tracing = Some(state);
    }

    fn trace_instruction(
        &mut self,
        bytecode: &[u8],
        pc: usize,
        runtime_stack: &[i64],
    ) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();
        let opcode = bytecode[pc];

        if opcode == CONST_INT {
            let value = bytecode[pc + 1] as i64;
            let opref = state.const_ref(value);
            state.trace_stack.push(opref);
        } else if opcode == POP {
            state.trace_stack.pop();
        } else if opcode == DUP {
            let top = *state.trace_stack.last().unwrap();
            state.trace_stack.push(top);
        } else if opcode == ADD {
            let b = state.trace_stack.pop().unwrap();
            let a = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntAdd, &[a, b]);
            state.trace_stack.push(result);
        } else if opcode == SUB {
            let b = state.trace_stack.pop().unwrap();
            let a = state.trace_stack.pop().unwrap();
            let result = state.recorder.record_op(OpCode::IntSub, &[a, b]);
            state.trace_stack.push(result);
        } else if opcode == JUMP_IF {
            let target = bytecode[pc + 1] as usize;
            let cond = state.trace_stack.pop().unwrap();
            let runtime_cond = *runtime_stack.last().unwrap();

            if runtime_cond != 0 && target == state.loop_header_pc {
                let fail_descr = make_fail_descr(state.num_stack_slots);
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[cond], fail_descr);
                return TraceAction::CloseLoop;
            } else if runtime_cond != 0 {
                let fail_descr = make_fail_descr(state.num_stack_slots);
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[cond], fail_descr);
            } else {
                let fail_descr = make_fail_descr(state.num_stack_slots);
                state
                    .recorder
                    .record_guard(OpCode::GuardFalse, &[cond], fail_descr);
            }
        } else if opcode == RETURN {
            return TraceAction::Abort;
        } else {
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
        let optimized_ops = optimizer.optimize(&trace.ops);

        self.backend.set_constants(state.constants);

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

    fn run_compiled(
        &mut self,
        loop_pc: usize,
        stack: &[i64],
    ) -> Option<Vec<i64>> {
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
    Arc::new(TlaFailDescr {
        types: vec![Type::Int; num_live],
    })
}

#[derive(Debug)]
struct TlaFailDescr {
    types: Vec<Type>,
}

impl majit_ir::Descr for TlaFailDescr {
    fn index(&self) -> u32 {
        0
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for TlaFailDescr {
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

    fn countdown_bytecode() -> Vec<u8> {
        vec![
            DUP,              // 0
            CONST_INT, 1,     // 1, 2
            SUB,              // 3
            DUP,              // 4
            JUMP_IF, 1,       // 5, 6 → back to CONST_INT
            POP,              // 7
            RETURN,           // 8
        ]
    }

    #[test]
    fn jit_countdown_5() {
        let bc = countdown_bytecode();
        let mut jit = JitTlaInterp::new();
        let res = jit.run(&bc, WObject::Int(5));
        assert_eq!(res.int_value(), 5);
    }

    #[test]
    fn jit_countdown_100() {
        let bc = countdown_bytecode();
        let mut jit = JitTlaInterp::new();
        let res = jit.run(&bc, WObject::Int(100));
        assert_eq!(res.int_value(), 100);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = countdown_bytecode();
        for n in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::run(&bc, WObject::Int(n)).int_value();
            let mut jit = JitTlaInterp::new();
            let got = jit.run(&bc, WObject::Int(n)).int_value();
            assert_eq!(got, expected, "mismatch for n={n}");
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![CONST_INT, 42, ADD, RETURN];
        let mut jit = JitTlaInterp::new();
        let res = jit.run(&prog, WObject::Int(0));
        assert_eq!(res.int_value(), 42);
    }
}
