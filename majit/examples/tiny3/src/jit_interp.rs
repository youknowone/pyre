/// JIT-enabled tiny3 interpreter — port of rpython/jit/tl/tiny3_hotpath.py.
///
/// Greens: [pos, bytecode]   (bytecode and position are loop constants)
/// Reds:   [args]
///
/// Identical JIT strategy to tiny2: trace the integer-only path.
/// When all args are integers, the loop body compiles to pure i64 arithmetic.
/// The difference from tiny2 is the interpreter: non-integer values are FloatBox
/// (not StrBox), and mixed int/float arithmetic casts to float.
use std::collections::HashMap;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmState};

const DEFAULT_THRESHOLD: u32 = 3;

struct CompiledLoop {
    token: LoopToken,
    num_args: usize,
}

struct TracingState {
    recorder: TraceRecorder,
    loop_header_pos: usize,
    /// args[i] → OpRef mapping (the live state).
    trace_args: Vec<OpRef>,
    /// Intermediate computation stack (maps to OpRef during tracing).
    trace_stack: Vec<OpRef>,
    num_args: usize,
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

pub struct JitTiny3Interp {
    warm_state: WarmState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    tracing: Option<TracingState>,
}

impl JitTiny3Interp {
    pub fn new() -> Self {
        JitTiny3Interp {
            warm_state: WarmState::new(DEFAULT_THRESHOLD),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    /// Run a word-based program with integer args.
    /// Returns the result: stack top if non-empty, else args[0].
    pub fn run(&mut self, bytecode: &[&str], args: &mut Vec<i64>) -> i64 {
        let mut stack: Vec<i64> = Vec::new();
        let mut loops: Vec<usize> = Vec::new();
        let mut pos: usize = 0;

        while pos < bytecode.len() {
            // --- tracing: record instruction ---
            if self.tracing.is_some() {
                match self.trace_instruction(bytecode, pos) {
                    TraceAction::Continue => {}
                    TraceAction::CloseLoop => {
                        self.close_and_compile();
                    }
                    TraceAction::Abort => {
                        self.abort_trace();
                    }
                }
            }

            let opcode = bytecode[pos];
            pos += 1;

            if opcode == "ADD" || opcode == "SUB" || opcode == "MUL" {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let result = match opcode {
                    "ADD" => a + b,
                    "SUB" => a - b,
                    "MUL" => a * b,
                    _ => unreachable!(),
                };
                stack.push(result);
            } else if opcode.starts_with('#') {
                let n = parse_int(opcode, 1) as usize;
                stack.push(args[n - 1]);
            } else if opcode.starts_with("->#") {
                let n = parse_int(opcode, 3) as usize;
                let val = stack.pop().unwrap();
                args[n - 1] = val;
            } else if opcode == "{" {
                loops.push(pos);
            } else if opcode == "}" {
                let flag = stack.pop().unwrap();
                if flag == 0 {
                    loops.pop();
                } else {
                    let target = *loops.last().unwrap();

                    if self.tracing.is_none() {
                        match self.warm_state.maybe_compile(target as u64) {
                            HotResult::NotHot => {}
                            HotResult::StartTracing(recorder) => {
                                self.start_tracing(recorder, target, args);
                            }
                            HotResult::AlreadyTracing => {}
                            HotResult::RunCompiled => {
                                if let Some(new_args) = self.run_compiled(target, args) {
                                    *args = new_args;
                                    loops.pop();
                                    continue;
                                }
                            }
                        }
                    }

                    pos = target;
                }
            } else {
                // Integer literal
                stack.push(parse_int(opcode, 0));
            }
        }

        if !stack.is_empty() {
            stack.pop().unwrap()
        } else {
            args[0]
        }
    }

    fn start_tracing(&mut self, recorder: TraceRecorder, loop_header_pos: usize, args: &[i64]) {
        let num_args = args.len();
        let mut state = TracingState {
            recorder,
            loop_header_pos,
            trace_args: Vec::new(),
            trace_stack: Vec::new(),
            num_args,
            constants: HashMap::new(),
            next_const_ref: 10_000,
        };

        for _ in 0..num_args {
            let opref = state.recorder.record_input_arg(Type::Int);
            state.trace_args.push(opref);
        }

        self.tracing = Some(state);
    }

    fn trace_instruction(&mut self, bytecode: &[&str], pos: usize) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();
        let opcode = bytecode[pos];

        if opcode == "ADD" || opcode == "SUB" || opcode == "MUL" {
            let b = state.trace_stack.pop().unwrap();
            let a = state.trace_stack.pop().unwrap();
            let ir_op = match opcode {
                "ADD" => OpCode::IntAdd,
                "SUB" => OpCode::IntSub,
                "MUL" => OpCode::IntMul,
                _ => unreachable!(),
            };
            let result = state.recorder.record_op(ir_op, &[a, b]);
            state.trace_stack.push(result);
        } else if opcode.starts_with('#') {
            let n = parse_int(opcode, 1) as usize;
            let opref = state.trace_args[n - 1];
            state.trace_stack.push(opref);
        } else if opcode.starts_with("->#") {
            let n = parse_int(opcode, 3) as usize;
            let opref = state.trace_stack.pop().unwrap();
            state.trace_args[n - 1] = opref;
        } else if opcode == "{" {
            return TraceAction::Abort;
        } else if opcode == "}" {
            let flag_ref = state.trace_stack.pop().unwrap();

            let zero = state.const_ref(0);
            let cond = state.recorder.record_op(OpCode::IntNe, &[flag_ref, zero]);
            let fail_descr = make_fail_descr(state.num_args);
            let fail_args: Vec<OpRef> = state.trace_args.clone();
            state.recorder.record_guard_with_fail_args(
                OpCode::GuardTrue,
                &[cond],
                fail_descr,
                &fail_args,
            );

            return TraceAction::CloseLoop;
        } else {
            let val = parse_int(opcode, 0);
            let opref = state.const_ref(val);
            state.trace_stack.push(opref);
        }

        if state.recorder.is_too_long() {
            return TraceAction::Abort;
        }
        TraceAction::Continue
    }

    fn close_and_compile(&mut self) {
        let state = self.tracing.take().unwrap();
        let green_key = state.loop_header_pos as u64;

        let jump_args: Vec<OpRef> = state.trace_args.clone();

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
                    state.loop_header_pos,
                    CompiledLoop {
                        token,
                        num_args: state.num_args,
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
                .abort_tracing(state.loop_header_pos as u64, false);
        }
    }

    fn run_compiled(&mut self, loop_pos: usize, args: &[i64]) -> Option<Vec<i64>> {
        let compiled = self.compiled_loops.get(&loop_pos)?;

        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        let frame = self.backend.execute_token(&compiled.token, &values);

        let mut new_args = Vec::new();
        for i in 0..compiled.num_args {
            new_args.push(self.backend.get_int_value(&frame, i));
        }

        Some(new_args)
    }
}

fn parse_int(s: &str, start: usize) -> i64 {
    let s = &s[start..];
    let mut res: i64 = 0;
    for c in s.chars() {
        let d = c as i64 - '0' as i64;
        res = res * 10 + d;
    }
    res
}

enum TraceAction {
    Continue,
    CloseLoop,
    Abort,
}

fn make_fail_descr(num_live: usize) -> majit_ir::DescrRef {
    use std::sync::Arc;
    Arc::new(Tiny3FailDescr {
        types: vec![Type::Int; num_live],
    })
}

#[derive(Debug)]
struct Tiny3FailDescr {
    types: Vec<Type>,
}

impl majit_ir::Descr for Tiny3FailDescr {
    fn index(&self) -> u32 {
        0
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for Tiny3FailDescr {
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
    fn jit_fibonacci_single() {
        let prog: Vec<&str> = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![1i64, 1, 11];
        let result = jit.run(&prog, &mut args);
        assert_eq!(result, 89);
    }

    #[test]
    fn jit_fibonacci_matches_interp() {
        let prog_str = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1";
        let prog: Vec<&str> = prog_str.split_whitespace().collect();

        for n in [5, 10, 11, 15, 20] {
            let mut interp_args = vec![
                interp::Box::Int(1),
                interp::Box::Int(1),
                interp::Box::Int(n),
            ];
            let interp_result = interp::interpret(&prog, &mut interp_args);
            let expected = interp::repr_stack(&interp_result);

            let mut jit = JitTiny3Interp::new();
            let mut jit_args = vec![1i64, 1, n];
            let jit_result = jit.run(&prog, &mut jit_args);

            assert_eq!(jit_result.to_string(), expected, "fib({n}) mismatch");
        }
    }

    #[test]
    fn jit_countdown() {
        let prog: Vec<&str> = "{ #1 #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![5i64];
        jit.run(&prog, &mut args);
        assert_eq!(args[0], 0);
    }
}
