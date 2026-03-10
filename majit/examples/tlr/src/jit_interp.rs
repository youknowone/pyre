/// JIT-enabled TLR interpreter.
///
/// Wraps the base TlrInterp with meta-tracing JIT compilation.
/// At backward jumps (JumpIfA with target < pc), the warm state decides
/// whether to start tracing, continue interpreting, or run compiled code.
use std::collections::HashMap;

use crate::bytecode::ByteCode;
use crate::interp::TlrInterp;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmState};

const DEFAULT_THRESHOLD: u32 = 3;

struct CompiledLoop {
    token: LoopToken,
    /// Which registers are live at the loop header, in order.
    /// Index 0 in live_regs corresponds to the accumulator (represented as reg 255).
    live_regs: Vec<u8>,
}

/// Accumulator is represented as a virtual register index.
const ACC_REG: u8 = 255;

struct TracingState {
    recorder: TraceRecorder,
    loop_header_pc: usize,
    /// Trace-level register bindings: reg index -> OpRef.
    /// ACC_REG (255) represents the accumulator.
    trace_regs: HashMap<u8, OpRef>,
    /// Which registers are live at the loop header, in order.
    live_regs: Vec<u8>,
    /// Constants recorded during tracing: OpRef index -> constant value.
    constants: HashMap<u32, i64>,
    next_const_ref: u32,
}

impl TracingState {
    fn new(recorder: TraceRecorder, loop_header_pc: usize) -> Self {
        TracingState {
            recorder,
            loop_header_pc,
            trace_regs: HashMap::new(),
            live_regs: Vec::new(),
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

pub struct JitTlrInterp {
    interp: TlrInterp,
    warm_state: WarmState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    tracing: Option<TracingState>,
}

impl JitTlrInterp {
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_THRESHOLD)
    }

    pub fn with_threshold(threshold: u32) -> Self {
        JitTlrInterp {
            interp: TlrInterp::new(),
            warm_state: WarmState::new(threshold),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    pub fn run(&mut self, bytecode: &[ByteCode], initial_a: i64) -> i64 {
        self.interp.reset();
        self.interp.set_accumulator(initial_a);

        loop {
            let pc = self.interp.pc();
            let instr = &bytecode[pc];
            self.interp.set_pc(pc + 1);

            // If tracing, record this instruction.
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
            }

            match instr {
                ByteCode::MovAR(n) => {
                    let a = self.interp.accumulator();
                    self.interp.set_reg(*n, a);
                }
                ByteCode::MovRA(n) => {
                    let v = self.interp.get_reg(*n);
                    self.interp.set_accumulator(v);
                }
                ByteCode::JumpIfA(target) => {
                    let a = self.interp.accumulator();
                    if a != 0 {
                        let target_pc = *target as usize;
                        // Backward jump = potential loop
                        if target_pc <= pc && self.tracing.is_none() {
                            let action = self.check_hot(target_pc, bytecode);
                            match action {
                                BackEdgeAction::Interpret => {}
                                BackEdgeAction::RunCompiled => {
                                    if let Some(result) = self.run_compiled(target_pc) {
                                        return result;
                                    }
                                    continue;
                                }
                            }
                        }
                        self.interp.set_pc(target_pc);
                    }
                }
                ByteCode::SetA(val) => {
                    self.interp.set_accumulator(*val);
                }
                ByteCode::AddRToA(n) => {
                    let a = self.interp.accumulator();
                    let r = self.interp.get_reg(*n);
                    self.interp.set_accumulator(a + r);
                }
                ByteCode::ReturnA => {
                    return self.interp.accumulator();
                }
                ByteCode::Allocate(n) => {
                    self.interp.ensure_regs(*n as usize);
                }
                ByteCode::NegA => {
                    let a = self.interp.accumulator();
                    self.interp.set_accumulator(-a);
                }
            }
        }
    }

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

    fn start_tracing(
        &mut self,
        recorder: TraceRecorder,
        loop_header_pc: usize,
        bytecode: &[ByteCode],
    ) {
        let mut state = TracingState::new(recorder, loop_header_pc);

        // Scan the loop to find live registers (including accumulator).
        let live = self.scan_live_regs(loop_header_pc, bytecode);
        state.live_regs = live.clone();

        // Register input arguments for each live register.
        for &reg_idx in &live {
            let opref = state.recorder.record_input_arg(Type::Int);
            state.trace_regs.insert(reg_idx, opref);
        }

        self.tracing = Some(state);
    }

    /// Scan the loop body to find which registers (and accumulator) are live.
    fn scan_live_regs(&self, header_pc: usize, bytecode: &[ByteCode]) -> Vec<u8> {
        let mut loaded = Vec::new();
        let mut stored = Vec::new();
        let mut pc = header_pc;

        loop {
            if pc >= bytecode.len() {
                break;
            }
            match &bytecode[pc] {
                ByteCode::MovRA(n) => {
                    if !stored.contains(n) && !loaded.contains(n) {
                        loaded.push(*n);
                    }
                }
                ByteCode::AddRToA(n) => {
                    // Reads accumulator (implicitly) and register n.
                    if !stored.contains(&ACC_REG) && !loaded.contains(&ACC_REG) {
                        loaded.push(ACC_REG);
                    }
                    if !stored.contains(n) && !loaded.contains(n) {
                        loaded.push(*n);
                    }
                }
                ByteCode::MovAR(n) => {
                    // Reads accumulator, writes register n.
                    if !stored.contains(&ACC_REG) && !loaded.contains(&ACC_REG) {
                        loaded.push(ACC_REG);
                    }
                    if !stored.contains(n) {
                        stored.push(*n);
                    }
                }
                ByteCode::NegA | ByteCode::ReturnA => {
                    if !stored.contains(&ACC_REG) && !loaded.contains(&ACC_REG) {
                        loaded.push(ACC_REG);
                    }
                }
                ByteCode::SetA(_) => {
                    if !stored.contains(&ACC_REG) {
                        stored.push(ACC_REG);
                    }
                }
                ByteCode::JumpIfA(target) => {
                    // Reads accumulator.
                    if !stored.contains(&ACC_REG) && !loaded.contains(&ACC_REG) {
                        loaded.push(ACC_REG);
                    }
                    if *target as usize == header_pc {
                        break;
                    }
                }
                ByteCode::Allocate(_) => {}
            }
            pc += 1;
        }

        let mut all: Vec<u8> = loaded;
        for v in stored {
            if !all.contains(&v) {
                all.push(v);
            }
        }
        all.sort();
        all
    }

    fn trace_instruction(
        &mut self,
        instr: &ByteCode,
        _current_pc: usize,
        _bytecode: &[ByteCode],
    ) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();

        match instr {
            ByteCode::SetA(val) => {
                let opref = state.const_ref(*val);
                state.trace_regs.insert(ACC_REG, opref);
            }
            ByteCode::MovAR(n) => {
                let acc = state
                    .trace_regs
                    .get(&ACC_REG)
                    .copied()
                    .unwrap_or_else(|| state.const_ref(self.interp.accumulator()));
                state.trace_regs.insert(*n, acc);
            }
            ByteCode::MovRA(n) => {
                let reg = state
                    .trace_regs
                    .get(n)
                    .copied()
                    .unwrap_or_else(|| state.const_ref(self.interp.get_reg(*n)));
                state.trace_regs.insert(ACC_REG, reg);
            }
            ByteCode::AddRToA(n) => {
                let acc = state
                    .trace_regs
                    .get(&ACC_REG)
                    .copied()
                    .unwrap_or_else(|| state.const_ref(self.interp.accumulator()));
                let reg = state
                    .trace_regs
                    .get(n)
                    .copied()
                    .unwrap_or_else(|| state.const_ref(self.interp.get_reg(*n)));
                let result = state.recorder.record_op(OpCode::IntAdd, &[acc, reg]);
                state.trace_regs.insert(ACC_REG, result);
            }
            ByteCode::NegA => {
                let acc = state
                    .trace_regs
                    .get(&ACC_REG)
                    .copied()
                    .unwrap_or_else(|| state.const_ref(self.interp.accumulator()));
                let result = state.recorder.record_op(OpCode::IntNeg, &[acc]);
                state.trace_regs.insert(ACC_REG, result);
            }
            ByteCode::JumpIfA(target) => {
                let target_pc = *target as usize;
                let header = state.loop_header_pc;

                if target_pc == header {
                    // Back-edge: the accumulator must be nonzero (we only jump if a != 0).
                    let acc = state
                        .trace_regs
                        .get(&ACC_REG)
                        .copied()
                        .unwrap_or_else(|| state.const_ref(self.interp.accumulator()));
                    let fail_descr = make_guard_fail_descr(0, state.live_regs.len());
                    state
                        .recorder
                        .record_guard(OpCode::GuardTrue, &[acc], fail_descr);
                    return TraceAction::CloseLoop;
                }
                // Non-back-edge conditional jump inside loop: guard the condition.
                let acc = state
                    .trace_regs
                    .get(&ACC_REG)
                    .copied()
                    .unwrap_or_else(|| state.const_ref(self.interp.accumulator()));
                let fail_descr = make_guard_fail_descr(0, state.live_regs.len());

                let runtime_a = self.interp.accumulator();
                let state = self.tracing.as_mut().unwrap();
                if runtime_a != 0 {
                    state
                        .recorder
                        .record_guard(OpCode::GuardTrue, &[acc], fail_descr);
                } else {
                    state
                        .recorder
                        .record_guard(OpCode::GuardFalse, &[acc], fail_descr);
                }
            }
            ByteCode::Allocate(_) => {
                // Allocate doesn't affect trace (registers already exist).
            }
            ByteCode::ReturnA => {
                return TraceAction::Abort;
            }
        }

        if self.tracing.as_ref().unwrap().recorder.is_too_long() {
            return TraceAction::Abort;
        }

        TraceAction::Continue
    }

    fn close_and_compile_trace(&mut self, _current_pc: usize) {
        let state = self.tracing.take().unwrap();
        let green_key = state.loop_header_pc as u64;

        // Build jump args: current OpRef for each live register.
        let jump_args: Vec<OpRef> = state
            .live_regs
            .iter()
            .map(|reg_idx| {
                state
                    .trace_regs
                    .get(reg_idx)
                    .copied()
                    .expect("live reg has no trace binding")
            })
            .collect();

        let mut recorder = state.recorder;
        recorder.close_loop(&jump_args);
        let trace = recorder.get_trace();

        let mut optimizer = Optimizer::new();
        let optimized_ops = optimizer.optimize(&trace.ops);

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
                    live_regs: state.live_regs,
                };
                self.compiled_loops.insert(state.loop_header_pc, compiled);

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

    fn abort_trace(&mut self, _current_pc: usize) {
        if let Some(state) = self.tracing.take() {
            let green_key = state.loop_header_pc as u64;
            state.recorder.abort();
            self.warm_state
                .abort_tracing(green_key, /* dont_trace_here */ false);
        }
    }

    fn run_compiled(&mut self, loop_pc: usize) -> Option<i64> {
        let compiled = self.compiled_loops.get(&loop_pc)?;

        let args: Vec<Value> = compiled
            .live_regs
            .iter()
            .map(|&reg_idx| {
                let val = if reg_idx == ACC_REG {
                    self.interp.accumulator()
                } else {
                    self.interp.get_reg(reg_idx)
                };
                Value::Int(val)
            })
            .collect();

        let frame = self.backend.execute_token(&compiled.token, &args);

        let live_regs = compiled.live_regs.clone();
        for (i, &reg_idx) in live_regs.iter().enumerate() {
            let val = self.backend.get_int_value(&frame, i);
            if reg_idx == ACC_REG {
                self.interp.set_accumulator(val);
            } else {
                self.interp.set_reg(reg_idx, val);
            }
        }

        self.interp.set_pc(loop_pc);
        None
    }
}

impl Default for JitTlrInterp {
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
    Arc::new(TlrFailDescr {
        fail_index,
        fail_arg_types: vec![Type::Int; num_live],
    })
}

#[derive(Debug)]
struct TlrFailDescr {
    fail_index: u32,
    fail_arg_types: Vec<Type>,
}

impl majit_ir::Descr for TlrFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for TlrFailDescr {
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
    fn test_jit_square_5() {
        let (prog, a) = square_program(5);
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&prog, a), 25);
    }

    #[test]
    fn test_jit_square_100() {
        let (prog, a) = square_program(100);
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&prog, a), 10_000);
    }

    #[test]
    fn test_jit_sum_10() {
        let (prog, a) = sum_program(10);
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&prog, a), 45);
    }

    #[test]
    fn test_jit_sum_100() {
        let (prog, a) = sum_program(100);
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&prog, a), 4950);
    }

    #[test]
    fn test_jit_sum_1m() {
        let (prog, a) = sum_program(1_000_000);
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&prog, a), 499_999_500_000);
    }

    #[test]
    fn test_jit_matches_interp_square() {
        let (prog, a) = square_program(1000);
        let mut interp = TlrInterp::new();
        let expected = interp.run(&prog, a);

        let mut jit = JitTlrInterp::new();
        let result = jit.run(&prog, a);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_jit_matches_interp_sum() {
        let (prog, a) = sum_program(10_000);
        let mut interp = TlrInterp::new();
        let expected = interp.run(&prog, a);

        let mut jit = JitTlrInterp::new();
        let result = jit.run(&prog, a);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_jit_no_loop() {
        let prog = vec![ByteCode::SetA(42), ByteCode::ReturnA];
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&prog, 0), 42);
    }

    #[test]
    fn test_jit_threshold_1() {
        let (prog, a) = sum_program(100);
        let mut jit = JitTlrInterp::with_threshold(1);
        assert_eq!(jit.run(&prog, a), 4950);
    }
}
