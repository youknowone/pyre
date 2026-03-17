/// JIT-enabled tinyframe interpreter — port of rpython/jit/tl/tinyframe/tinyframe.py.
///
/// Greens: [i, code]       (bytecode and position are loop constants)
/// Reds:   [self (frame)]  (registers are virtualizable)
///
/// JIT traces the integer-only path through the register machine.
/// JUMP_IF_ABOVE is the back-edge that triggers tracing.
use std::collections::HashMap;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmEnterState};

use crate::interp::{ADD, Code, JUMP_IF_ABOVE, LOAD, Object, RETURN};

const DEFAULT_THRESHOLD: u32 = 3;

struct CompiledLoop {
    token: LoopToken,
    /// Which registers are live (input/output) for this compiled loop.
    live_regs: Vec<usize>,
}

struct TracingState {
    recorder: TraceRecorder,
    loop_header_pc: usize,
    /// Register → OpRef mapping for live registers.
    trace_regs: HashMap<usize, OpRef>,
    /// Which registers are inputs (in order).
    input_regs: Vec<usize>,
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

pub struct JitTinyFrameInterp {
    warm_state: WarmEnterState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    tracing: Option<TracingState>,
}

impl JitTinyFrameInterp {
    pub fn new() -> Self {
        JitTinyFrameInterp {
            warm_state: WarmEnterState::new(DEFAULT_THRESHOLD),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    /// Run a tinyframe Code with initial integer register values.
    /// `init_regs` maps register index → initial i64 value.
    pub fn run(&mut self, code: &Code, init_regs: &[(usize, i64)]) -> i64 {
        let mut regs: Vec<i64> = vec![0; code.regno];
        for &(r, v) in init_regs {
            regs[r] = v;
        }
        let bytecode = &code.code;
        let mut pc: usize = 0;

        loop {
            if pc >= bytecode.len() {
                break;
            }

            // --- tracing: record instruction ---
            if self.tracing.is_some() {
                match self.trace_instruction(bytecode, pc, &regs) {
                    TraceAction::Continue => {}
                    TraceAction::CloseLoop => {
                        self.close_and_compile();
                    }
                    TraceAction::Abort => {
                        self.abort_trace();
                    }
                }
            }

            let opcode = bytecode[pc];

            if opcode == LOAD {
                let val = bytecode[pc + 1] as i64;
                let reg = bytecode[pc + 2] as usize;
                regs[reg] = val;
                pc += 3;
            } else if opcode == ADD {
                let r1 = bytecode[pc + 1] as usize;
                let r2 = bytecode[pc + 2] as usize;
                let r3 = bytecode[pc + 3] as usize;
                regs[r3] = regs[r1] + regs[r2];
                pc += 4;
            } else if opcode == RETURN {
                let r = bytecode[pc + 1] as usize;
                return regs[r];
            } else if opcode == JUMP_IF_ABOVE {
                let r1 = bytecode[pc + 1] as usize;
                let r2 = bytecode[pc + 2] as usize;
                let tgt = bytecode[pc + 3] as usize;
                if regs[r1] > regs[r2] {
                    // Back-edge: check hotness
                    if self.tracing.is_none() {
                        match self.warm_state.maybe_compile(tgt as u64) {
                            HotResult::NotHot => {}
                            HotResult::StartTracing(recorder) => {
                                self.start_tracing(recorder, tgt, &regs);
                            }
                            HotResult::AlreadyTracing => {}
                            HotResult::RunCompiled => {
                                if let Some(new_regs) = self.run_compiled(tgt, &regs) {
                                    regs = new_regs;
                                    // JIT ran the loop. Continue from after the loop.
                                    // The loop exits at JUMP_IF_ABOVE when condition
                                    // is false, so pc advances past it.
                                    pc += 4;
                                    continue;
                                }
                            }
                        }
                    }
                    pc = tgt;
                } else {
                    pc += 4;
                }
            } else {
                panic!("unsupported opcode in JIT path: {opcode}");
            }
        }

        panic!("fell off end of code");
    }

    fn start_tracing(&mut self, recorder: TraceRecorder, loop_header_pc: usize, regs: &[i64]) {
        let mut state = TracingState {
            recorder,
            loop_header_pc,
            trace_regs: HashMap::new(),
            input_regs: Vec::new(),
            constants: HashMap::new(),
            next_const_ref: 10_000,
        };

        // All registers that have been set become input args.
        // For simplicity, make all registers inputs.
        for i in 0..regs.len() {
            let opref = state.recorder.record_input_arg(Type::Int);
            state.trace_regs.insert(i, opref);
            state.input_regs.push(i);
        }

        self.tracing = Some(state);
    }

    fn trace_instruction(&mut self, bytecode: &[u8], pc: usize, _regs: &[i64]) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();
        let opcode = bytecode[pc];

        if opcode == LOAD {
            let val = bytecode[pc + 1] as i64;
            let reg = bytecode[pc + 2] as usize;
            let opref = state.const_ref(val);
            state.trace_regs.insert(reg, opref);
        } else if opcode == ADD {
            let r1 = bytecode[pc + 1] as usize;
            let r2 = bytecode[pc + 2] as usize;
            let r3 = bytecode[pc + 3] as usize;
            let a = *state.trace_regs.get(&r1).unwrap();
            let b = *state.trace_regs.get(&r2).unwrap();
            let result = state.recorder.record_op(OpCode::IntAdd, &[a, b]);
            state.trace_regs.insert(r3, result);
        } else if opcode == JUMP_IF_ABOVE {
            let r1 = bytecode[pc + 1] as usize;
            let r2 = bytecode[pc + 2] as usize;
            let tgt = bytecode[pc + 3] as usize;

            if tgt == state.loop_header_pc {
                // Back-edge to our loop header: guard and close.
                let a = *state.trace_regs.get(&r1).unwrap();
                let b = *state.trace_regs.get(&r2).unwrap();
                let cond = state.recorder.record_op(OpCode::IntGt, &[a, b]);
                let fail_descr = make_fail_descr(state.input_regs.len());

                // fail_args: current register values (after loop body)
                let fail_args: Vec<OpRef> = state
                    .input_regs
                    .iter()
                    .map(|r| *state.trace_regs.get(r).unwrap())
                    .collect();
                state.recorder.record_guard_with_fail_args(
                    OpCode::GuardTrue,
                    &[cond],
                    fail_descr,
                    &fail_args,
                );

                return TraceAction::CloseLoop;
            } else {
                return TraceAction::Abort;
            }
        } else if opcode == RETURN {
            return TraceAction::Abort;
        } else {
            return TraceAction::Abort;
        }

        if state.recorder.is_too_long() {
            return TraceAction::Abort;
        }
        TraceAction::Continue
    }

    fn close_and_compile(&mut self) {
        let state = self.tracing.take().unwrap();
        let green_key = state.loop_header_pc as u64;

        let jump_args: Vec<OpRef> = state
            .input_regs
            .iter()
            .map(|r| *state.trace_regs.get(r).unwrap())
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
                        live_regs: state.input_regs,
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

    fn run_compiled(&mut self, loop_pc: usize, regs: &[i64]) -> Option<Vec<i64>> {
        let compiled = self.compiled_loops.get(&loop_pc)?;

        let values: Vec<Value> = regs.iter().map(|&v| Value::Int(v)).collect();
        let frame = self.backend.execute_token(&compiled.token, &values);

        let mut new_regs = regs.to_vec();
        for (i, &reg_idx) in compiled.live_regs.iter().enumerate() {
            new_regs[reg_idx] = self.backend.get_int_value(&frame, i);
        }

        Some(new_regs)
    }
}

enum TraceAction {
    Continue,
    CloseLoop,
    Abort,
}

fn make_fail_descr(num_live: usize) -> majit_ir::DescrRef {
    use std::sync::Arc;
    Arc::new(TinyFrameFailDescr {
        types: vec![Type::Int; num_live],
    })
}

#[derive(Debug)]
struct TinyFrameFailDescr {
    types: Vec<Type>,
}

impl majit_ir::Descr for TinyFrameFailDescr {
    fn index(&self) -> u32 {
        0
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for TinyFrameFailDescr {
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

    #[test]
    fn jit_loop_count_to_100() {
        let code = interp::compile(
            "
        main:
        LOAD 1 => r1
        LOAD 100 => r2
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        RETURN r0
        ",
        );
        let mut jit = JitTinyFrameInterp::new();
        let result = jit.run(&code, &[]);
        assert_eq!(result, 100);
    }

    #[test]
    fn jit_loop_sum() {
        // loop.tf: sum from 1 to N
        let code = interp::compile(
            "
        main:
        LOAD 0 => r1
        LOAD 1 => r2
        @add
        ADD r2 r1 => r1
        JUMP_IF_ABOVE r0 r1 @add
        RETURN r1
        ",
        );
        let mut jit = JitTinyFrameInterp::new();
        let result = jit.run(&code, &[(0, 100)]);
        assert_eq!(result, 100);
    }

    #[test]
    fn jit_matches_interp() {
        let code = interp::compile(
            "
        main:
        LOAD 1 => r1
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        RETURN r0
        ",
        );

        for n in [10, 50, 100, 255] {
            // Interpreter
            let mut frame = interp::Frame::new(&code);
            frame.registers[2] = Some(Object::Int(n));
            let interp_result = frame.interpret(&code).as_int();

            // JIT
            let mut jit = JitTinyFrameInterp::new();
            let jit_result = jit.run(&code, &[(2, n)]);

            assert_eq!(jit_result, interp_result, "count_to({n}) mismatch");
        }
    }
}
