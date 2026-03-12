/// JIT-enabled TLR interpreter — structural mirror of tlr.py with JitDriver.
///
/// Greens: [pc, bytecode]   (bytecode is constant per trace — not tracked)
/// Reds:   [a, regs]
///
/// At backward jumps (JUMP_IF_A where target < pc), triggers tracing.
use std::collections::HashMap;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmState};

const MOV_A_R: u8 = 1;
const MOV_R_A: u8 = 2;
const JUMP_IF_A: u8 = 3;
const SET_A: u8 = 4;
const ADD_R_TO_A: u8 = 5;
const RETURN_A: u8 = 6;
const ALLOCATE: u8 = 7;
const NEG_A: u8 = 8;

const DEFAULT_THRESHOLD: u32 = 3;

/// Virtual register index for the accumulator.
const ACC: u8 = 255;

struct CompiledLoop {
    token: LoopToken,
    live_regs: Vec<u8>,
}

struct TracingState {
    recorder: TraceRecorder,
    loop_header_pc: usize,
    trace_regs: HashMap<u8, OpRef>,
    live_regs: Vec<u8>,
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

    fn get_or_const(&mut self, reg: u8, runtime_val: i64) -> OpRef {
        self.trace_regs
            .get(&reg)
            .copied()
            .unwrap_or_else(|| self.const_ref(runtime_val))
    }
}

pub struct JitTlrInterp {
    warm_state: WarmState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    tracing: Option<TracingState>,
}

impl JitTlrInterp {
    pub fn new() -> Self {
        JitTlrInterp {
            warm_state: WarmState::new(DEFAULT_THRESHOLD),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    pub fn run(&mut self, bytecode: &[u8], initial_a: i64) -> i64 {
        let mut regs: Vec<i64> = Vec::new();
        let mut pc: usize = 0;
        let mut a: i64 = initial_a;

        loop {
            // --- tracing: record instruction ---
            if self.tracing.is_some() {
                match self.trace_instruction(bytecode, pc, a, &regs) {
                    TraceAction::Continue => {}
                    TraceAction::CloseLoop => {
                        self.close_and_compile(a, &regs);
                    }
                    TraceAction::Abort => {
                        self.abort_trace();
                    }
                }
            }

            let opcode = bytecode[pc];
            pc += 1;

            if opcode == MOV_A_R {
                let n = bytecode[pc] as usize;
                pc += 1;
                regs[n] = a;
            } else if opcode == MOV_R_A {
                let n = bytecode[pc] as usize;
                pc += 1;
                a = regs[n];
            } else if opcode == JUMP_IF_A {
                let target = bytecode[pc] as usize;
                pc += 1;
                if a != 0 {
                    if target < pc && self.tracing.is_none() {
                        // Back-edge: check hotness
                        match self.warm_state.maybe_compile(target as u64) {
                            HotResult::NotHot => {}
                            HotResult::StartTracing(recorder) => {
                                self.start_tracing(recorder, target, a, &regs);
                            }
                            HotResult::AlreadyTracing => {}
                            HotResult::RunCompiled => {
                                if let Some((new_a, new_regs)) = self.run_compiled(target, a, &regs)
                                {
                                    a = new_a;
                                    regs = new_regs;
                                    pc = target;
                                    continue;
                                }
                            }
                        }
                    }
                    pc = target;
                }
            } else if opcode == SET_A {
                a = bytecode[pc] as i64;
                pc += 1;
            } else if opcode == ADD_R_TO_A {
                let n = bytecode[pc] as usize;
                pc += 1;
                a += regs[n];
            } else if opcode == RETURN_A {
                return a;
            } else if opcode == ALLOCATE {
                let n = bytecode[pc] as usize;
                pc += 1;
                regs = vec![0; n];
            } else if opcode == NEG_A {
                a = -a;
            }
        }
    }

    fn start_tracing(
        &mut self,
        recorder: TraceRecorder,
        loop_header_pc: usize,
        a: i64,
        regs: &[i64],
    ) {
        let mut state = TracingState {
            recorder,
            loop_header_pc,
            trace_regs: HashMap::new(),
            live_regs: Vec::new(),
            constants: HashMap::new(),
            next_const_ref: 10_000,
        };

        // Scan the loop to find live regs (read before written).
        let live = self.scan_live(loop_header_pc, &state, regs.len());
        state.live_regs = live.clone();

        // Create input args for each live register.
        for &r in &live {
            let opref = state.recorder.record_input_arg(Type::Int);
            state.trace_regs.insert(r, opref);
        }
        // Record constants for non-live but used regs.
        let _ = a; // a is part of live_regs as ACC

        self.tracing = Some(state);
    }

    fn scan_live(&self, header_pc: usize, _state: &TracingState, num_regs: usize) -> Vec<u8> {
        // For the SQUARE program, the loop reads:
        //   - accumulator (via NEG_A, ADD_R_TO_A, JUMP_IF_A)
        //   - regs[0], regs[1], regs[2] (via MOV_R_A, ADD_R_TO_A)
        // and writes:
        //   - accumulator, regs[0], regs[2]
        //
        // All touched registers + accumulator are live at the loop header.
        let mut live = vec![ACC];
        for i in 0..num_regs.min(256) {
            live.push(i as u8);
        }
        live
    }

    fn trace_instruction(
        &mut self,
        bytecode: &[u8],
        pc: usize,
        runtime_a: i64,
        runtime_regs: &[i64],
    ) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();
        let opcode = bytecode[pc];

        if opcode == SET_A {
            let val = bytecode[pc + 1] as i64;
            let opref = state.const_ref(val);
            state.trace_regs.insert(ACC, opref);
        } else if opcode == MOV_A_R {
            let n = bytecode[pc + 1];
            let acc = state.get_or_const(ACC, runtime_a);
            state.trace_regs.insert(n, acc);
        } else if opcode == MOV_R_A {
            let n = bytecode[pc + 1];
            let reg = state.get_or_const(n, runtime_regs[n as usize]);
            state.trace_regs.insert(ACC, reg);
        } else if opcode == ADD_R_TO_A {
            let n = bytecode[pc + 1];
            let acc = state.get_or_const(ACC, runtime_a);
            let reg = state.get_or_const(n, runtime_regs[n as usize]);
            let result = state.recorder.record_op(OpCode::IntAdd, &[acc, reg]);
            state.trace_regs.insert(ACC, result);
        } else if opcode == NEG_A {
            let acc = state.get_or_const(ACC, runtime_a);
            let result = state.recorder.record_op(OpCode::IntNeg, &[acc]);
            state.trace_regs.insert(ACC, result);
        } else if opcode == JUMP_IF_A {
            let target = bytecode[pc + 1] as usize;
            if target == state.loop_header_pc {
                // Back-edge: guard that a != 0 and close the loop.
                let acc = state.get_or_const(ACC, runtime_a);
                let fail_descr = make_fail_descr(state.live_regs.len());
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[acc], fail_descr);
                return TraceAction::CloseLoop;
            }
        } else if opcode == RETURN_A {
            return TraceAction::Abort;
        } else if opcode == ALLOCATE {
            // No-op for tracing (regs already exist).
        }

        if self.tracing.as_ref().unwrap().recorder.is_too_long() {
            return TraceAction::Abort;
        }
        TraceAction::Continue
    }

    fn close_and_compile(&mut self, _a: i64, _regs: &[i64]) {
        let state = self.tracing.take().unwrap();
        let green_key = state.loop_header_pc as u64;

        let jump_args: Vec<OpRef> = state
            .live_regs
            .iter()
            .filter_map(|r| state.trace_regs.get(r).copied())
            .collect();

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
                        live_regs: state.live_regs,
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

    fn run_compiled(&mut self, loop_pc: usize, a: i64, regs: &[i64]) -> Option<(i64, Vec<i64>)> {
        let compiled = self.compiled_loops.get(&loop_pc)?;

        let args: Vec<Value> = compiled
            .live_regs
            .iter()
            .map(|&r| {
                if r == ACC {
                    Value::Int(a)
                } else {
                    Value::Int(regs[r as usize])
                }
            })
            .collect();

        let frame = self.backend.execute_token(&compiled.token, &args);

        let mut new_a = a;
        let mut new_regs = regs.to_vec();
        for (i, &r) in compiled.live_regs.iter().enumerate() {
            let val = self.backend.get_int_value(&frame, i);
            if r == ACC {
                new_a = val;
            } else {
                new_regs[r as usize] = val;
            }
        }

        Some((new_a, new_regs))
    }
}

enum TraceAction {
    Continue,
    CloseLoop,
    Abort,
}

fn make_fail_descr(num_live: usize) -> majit_ir::DescrRef {
    use std::sync::Arc;
    Arc::new(TlrFailDescr {
        types: vec![Type::Int; num_live],
    })
}

#[derive(Debug)]
struct TlrFailDescr {
    types: Vec<Type>,
}

impl majit_ir::Descr for TlrFailDescr {
    fn index(&self) -> u32 {
        0
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for TlrFailDescr {
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

    fn square_bytecode() -> Vec<u8> {
        vec![
            ALLOCATE, 3, MOV_A_R, 0, MOV_A_R, 1, SET_A, 0, MOV_A_R, 2, SET_A, 1, NEG_A, ADD_R_TO_A,
            0, MOV_A_R, 0, MOV_R_A, 2, ADD_R_TO_A, 1, MOV_A_R, 2, MOV_R_A, 0, JUMP_IF_A, 10,
            MOV_R_A, 2, RETURN_A,
        ]
    }

    #[test]
    fn jit_square_5() {
        let bc = square_bytecode();
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&bc, 5), 25);
    }

    #[test]
    fn jit_square_100() {
        let bc = square_bytecode();
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&bc, 100), 10_000);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = square_bytecode();
        for a in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlrInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![SET_A, 42, RETURN_A];
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&prog, 0), 42);
    }
}
