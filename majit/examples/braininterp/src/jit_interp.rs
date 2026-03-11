/// JIT-enabled Brainfuck interpreter — structural mirror of braininterp.py with JitDriver.
///
/// Greens: [pc, code]    (code is constant per trace)
/// Reds:   [pointer, tape]
///
/// The tape is virtualizable: during tracing, tape cells accessed at known
/// positions are mapped to IR operations (OpRef), eliminating memory
/// loads/stores in compiled code.
///
/// Back-edge detection: `]` that jumps backward to matching `[` is the
/// loop back-edge. When it becomes hot, tracing starts at the `[` header.
///
/// Before tracing begins, the loop body is pre-scanned to determine which
/// tape offsets are accessed. Input args are registered upfront for all
/// accessed cells, as required by the trace recorder.
use std::collections::HashMap;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::recorder::TraceRecorder;
use majit_trace::warmstate::{HotResult, WarmState};

const TAPE_SIZE: usize = 30000;
const DEFAULT_THRESHOLD: u32 = 3;

struct CompiledLoop {
    token: LoopToken,
    /// Sorted list of tape positions that are the input/output args.
    tape_positions: Vec<usize>,
}

struct TracingState {
    recorder: TraceRecorder,
    loop_header_pc: usize,
    /// Maps absolute tape position -> OpRef for the current value.
    trace_tape: HashMap<usize, OpRef>,
    /// Sorted tape positions at loop entry (determines input arg order).
    entry_positions: Vec<usize>,
    /// Current tape pointer during tracing.
    trace_pointer: usize,
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

    fn get_tape_cell(&self, pos: usize) -> OpRef {
        *self
            .trace_tape
            .get(&pos)
            .expect("tape cell should be pre-registered")
    }

    fn set_tape_cell(&mut self, pos: usize, opref: OpRef) {
        self.trace_tape.insert(pos, opref);
    }
}

pub struct JitBrainInterp {
    warm_state: WarmState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<usize, CompiledLoop>,
    tracing: Option<TracingState>,
}

impl JitBrainInterp {
    pub fn new() -> Self {
        JitBrainInterp {
            warm_state: WarmState::new(DEFAULT_THRESHOLD),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
        }
    }

    pub fn run(&mut self, code: &[u8]) -> String {
        let mut tape = vec![0u8; TAPE_SIZE];
        let mut pointer: usize = 0;
        let mut output = String::new();
        let mut pc: usize = 0;

        while pc < code.len() {
            // --- tracing: record instruction ---
            if self.tracing.is_some() {
                match self.trace_instruction(code, pc, &tape, pointer) {
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

            let ch = code[pc];
            if ch == b'>' {
                pointer += 1;
                pc += 1;
            } else if ch == b'<' {
                pointer -= 1;
                pc += 1;
            } else if ch == b'+' {
                tape[pointer] = tape[pointer].wrapping_add(1);
                pc += 1;
            } else if ch == b'-' {
                tape[pointer] = tape[pointer].wrapping_sub(1);
                pc += 1;
            } else if ch == b'.' {
                output.push(tape[pointer] as char);
                pc += 1;
            } else if ch == b',' {
                tape[pointer] = 0;
                pc += 1;
            } else if ch == b'[' {
                if tape[pointer] == 0 {
                    let mut need: i32 = 1;
                    let mut p = pc + 1;
                    while need > 0 {
                        if code[p] == b']' {
                            need -= 1;
                        } else if code[p] == b'[' {
                            need += 1;
                        }
                        p += 1;
                    }
                    pc = p;
                } else {
                    pc += 1;
                }
            } else if ch == b']' {
                if tape[pointer] != 0 {
                    // Find matching '[' — this is the back-edge.
                    let target = find_matching_open(code, pc);

                    if self.tracing.is_none() {
                        // Back-edge: check hotness.
                        match self.warm_state.maybe_compile(target as u64) {
                            HotResult::NotHot => {}
                            HotResult::StartTracing(recorder) => {
                                self.start_tracing(
                                    recorder, code, target, &tape, pointer,
                                );
                            }
                            HotResult::AlreadyTracing => {}
                            HotResult::RunCompiled => {
                                if let Some(new_cells) =
                                    self.run_compiled(target, &tape)
                                {
                                    // Write results back to tape.
                                    let compiled =
                                        self.compiled_loops.get(&target).unwrap();
                                    for (i, &pos) in
                                        compiled.tape_positions.iter().enumerate()
                                    {
                                        tape[pos] = new_cells[i] as u8;
                                    }
                                    // After compiled loop, the `[` condition is
                                    // false (guard failed), so skip past `]`.
                                    pc += 1;
                                    continue;
                                }
                            }
                        }
                    }

                    pc = target;
                } else {
                    pc += 1;
                }
            } else {
                // Unknown character: skip.
                pc += 1;
            }
        }

        output
    }

    fn start_tracing(
        &mut self,
        mut recorder: TraceRecorder,
        code: &[u8],
        loop_header_pc: usize,
        tape: &[u8],
        pointer: usize,
    ) {
        // Pre-scan the loop body to find all tape offsets accessed.
        let offsets = scan_loop_offsets(code, loop_header_pc);

        // Convert relative offsets to absolute tape positions.
        let mut entry_positions: Vec<usize> = offsets
            .iter()
            .map(|&off| (pointer as isize + off) as usize)
            .collect();
        entry_positions.sort();
        entry_positions.dedup();

        // Register input args for all accessed tape cells upfront.
        let mut trace_tape = HashMap::new();
        for &pos in &entry_positions {
            let opref = recorder.record_input_arg(Type::Int);
            trace_tape.insert(pos, opref);
        }

        let _ = tape;
        let state = TracingState {
            recorder,
            loop_header_pc,
            trace_tape,
            entry_positions,
            trace_pointer: pointer,
            constants: HashMap::new(),
            next_const_ref: 10_000,
        };

        self.tracing = Some(state);
    }

    fn trace_instruction(
        &mut self,
        code: &[u8],
        pc: usize,
        tape: &[u8],
        pointer: usize,
    ) -> TraceAction {
        let state = self.tracing.as_mut().unwrap();
        let ch = code[pc];

        if ch == b'>' {
            state.trace_pointer += 1;
        } else if ch == b'<' {
            state.trace_pointer -= 1;
        } else if ch == b'+' {
            let pos = state.trace_pointer;
            let cell = state.get_tape_cell(pos);
            let one = state.const_ref(1);
            let result = state.recorder.record_op(OpCode::IntAdd, &[cell, one]);
            state.set_tape_cell(pos, result);
        } else if ch == b'-' {
            let pos = state.trace_pointer;
            let cell = state.get_tape_cell(pos);
            let one = state.const_ref(1);
            let result = state.recorder.record_op(OpCode::IntSub, &[cell, one]);
            state.set_tape_cell(pos, result);
        } else if ch == b'.' {
            // Output during tracing — we cannot trace I/O, abort.
            return TraceAction::Abort;
        } else if ch == b',' {
            return TraceAction::Abort;
        } else if ch == b'[' {
            // Nested loop entry: guard the condition.
            let pos = state.trace_pointer;
            let cell = state.get_tape_cell(pos);
            let runtime_val = tape[pointer];

            let zero = state.const_ref(0);
            let fail_descr = make_fail_descr(state.entry_positions.len());
            if runtime_val == 0 {
                // Loop skipped: guard that cell is zero.
                let cmp = state
                    .recorder
                    .record_op(OpCode::IntEq, &[cell, zero]);
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[cmp], fail_descr);
            } else {
                // Loop entered: guard that cell is non-zero.
                let cmp = state
                    .recorder
                    .record_op(OpCode::IntNe, &[cell, zero]);
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[cmp], fail_descr);
            }
        } else if ch == b']' {
            let pos = state.trace_pointer;
            let cell = state.get_tape_cell(pos);
            let runtime_val = tape[pointer];

            let zero = state.const_ref(0);

            if runtime_val != 0 {
                let target = find_matching_open(code, pc);

                let fail_descr = make_fail_descr(state.entry_positions.len());
                let cmp = state
                    .recorder
                    .record_op(OpCode::IntNe, &[cell, zero]);
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[cmp], fail_descr);

                if target == state.loop_header_pc {
                    // Back-edge to loop header: close loop.
                    return TraceAction::CloseLoop;
                }
                // Inner loop back-edge: guard emitted, continue tracing.
            } else {
                // Loop exit: guard that cell is zero.
                let fail_descr = make_fail_descr(state.entry_positions.len());
                let cmp = state
                    .recorder
                    .record_op(OpCode::IntEq, &[cell, zero]);
                state
                    .recorder
                    .record_guard(OpCode::GuardTrue, &[cmp], fail_descr);
            }
        }

        if self.tracing.as_ref().unwrap().recorder.is_too_long() {
            return TraceAction::Abort;
        }
        TraceAction::Continue
    }

    fn close_and_compile(&mut self) {
        let state = self.tracing.take().unwrap();
        let green_key = state.loop_header_pc as u64;

        // Build jump args in the same order as entry_positions.
        let jump_args: Vec<OpRef> = state
            .entry_positions
            .iter()
            .map(|pos| state.trace_tape[pos])
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
                        tape_positions: state.entry_positions.clone(),
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
        tape: &[u8],
    ) -> Option<Vec<i64>> {
        let compiled = self.compiled_loops.get(&loop_pc)?;

        let args: Vec<Value> = compiled
            .tape_positions
            .iter()
            .map(|&pos| Value::Int(tape[pos] as i64))
            .collect();

        let frame = self.backend.execute_token(&compiled.token, &args);

        let mut results = Vec::new();
        for i in 0..compiled.tape_positions.len() {
            results.push(self.backend.get_int_value(&frame, i));
        }

        Some(results)
    }
}

enum TraceAction {
    Continue,
    CloseLoop,
    Abort,
}

/// Find the matching '[' for a ']' at the given position.
fn find_matching_open(code: &[u8], close_pos: usize) -> usize {
    let mut need: i32 = 1;
    let mut p = close_pos - 1;
    while need > 0 {
        if code[p] == b']' {
            need += 1;
        } else if code[p] == b'[' {
            need -= 1;
        }
        if need > 0 {
            p -= 1;
        }
    }
    p
}

/// Pre-scan a loop body (from `[` to matching `]`) to find all relative
/// tape offsets that are read or written. Returns sorted, deduplicated
/// offsets relative to the pointer at loop entry.
fn scan_loop_offsets(code: &[u8], open_bracket: usize) -> Vec<isize> {
    let mut offsets = Vec::new();
    let mut rel: isize = 0;
    let mut pc = open_bracket + 1;
    let mut depth: i32 = 1;

    while depth > 0 && pc < code.len() {
        let ch = code[pc];
        if ch == b'>' {
            rel += 1;
        } else if ch == b'<' {
            rel -= 1;
        } else if ch == b'+' || ch == b'-' || ch == b'.' || ch == b',' {
            offsets.push(rel);
        } else if ch == b'[' {
            offsets.push(rel); // condition check reads current cell
            depth += 1;
        } else if ch == b']' {
            offsets.push(rel); // condition check reads current cell
            depth -= 1;
        }
        pc += 1;
    }

    // Also include offset 0 for the outer loop condition at '['.
    offsets.push(0);

    offsets.sort();
    offsets.dedup();
    offsets
}

fn make_fail_descr(num_live: usize) -> majit_ir::DescrRef {
    use std::sync::Arc;
    Arc::new(BfFailDescr {
        types: vec![Type::Int; num_live],
    })
}

#[derive(Debug)]
struct BfFailDescr {
    types: Vec<Type>,
}

impl majit_ir::Descr for BfFailDescr {
    fn index(&self) -> u32 {
        0
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for BfFailDescr {
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
    fn jit_simple_loop() {
        // Move cell0 to cell1: +++++[->+<]
        let mut jit = JitBrainInterp::new();
        let output = jit.run(b"+++++[->+<]");
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn jit_multiply() {
        // cell1 = 9 * 9 = 81
        let mut jit = JitBrainInterp::new();
        let output = jit.run(b"+++++++++[>+++++++++<-]");
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn jit_matches_interp_move() {
        let code = b"+++++[->+<]";
        let expected = interp::interpret(code);
        let mut jit = JitBrainInterp::new();
        let got = jit.run(code);
        assert_eq!(got, expected);
    }

    #[test]
    fn jit_matches_interp_multiply() {
        let code = b"+++++++++[>+++++++++<-]";
        let expected = interp::interpret(code);
        let mut jit = JitBrainInterp::new();
        let got = jit.run(code);
        assert_eq!(got, expected);
    }

    #[test]
    fn jit_no_loop() {
        let mut jit = JitBrainInterp::new();
        let output = jit.run(b"+++");
        assert_eq!(output.len(), 0);
    }
}
