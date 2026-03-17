/// JIT-enabled Brainfuck interpreter using JitDriver + JitState.
///
/// Greens: [pc, code]    (code is constant per trace)
/// Reds:   [pointer, tape]
///
/// The tape is virtualizable: during tracing, tape cells accessed at known
/// positions are mapped to IR operations (OpRef), eliminating memory
/// loads/stores in compiled code.
///
/// This example hand-writes `trace_instruction` for educational purposes.
/// In production, the `#[jit_interp]` proc macro auto-generates tracing
/// code from the interpreter's match dispatch — see aheuijit for an example.
///
/// Back-edge detection: `]` that jumps backward to matching `[` is the
/// loop back-edge. When it becomes hot, tracing starts at the `[` header.
///
/// Before tracing begins, the loop body is pre-scanned to determine which
/// tape offsets are accessed. Input args are registered upfront for all
/// accessed cells, as required by the trace recorder.
use std::collections::HashMap;

use majit_ir::{OpCode, OpRef};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

const TAPE_SIZE: usize = 30000;
const DEFAULT_THRESHOLD: u32 = 3;

// ── JitState types ──

/// Red variables: tape pointer + tape contents.
pub struct BfState {
    pointer: usize,
    tape: Vec<u8>,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct BfMeta {
    /// Sorted absolute tape positions that are live at the loop header.
    tape_positions: Vec<usize>,
    /// Tape pointer at trace start (initial trace_pointer for BfSym).
    initial_pointer: usize,
}

/// Symbolic state during tracing — OpRef for each live tape cell.
pub struct BfSym {
    /// Maps absolute tape position → current OpRef.
    tape: HashMap<usize, OpRef>,
    /// Ordered list of tape positions (same order as inputargs).
    tape_positions: Vec<usize>,
    /// Current tape pointer during tracing (tracked symbolically as pointer moves).
    trace_pointer: usize,
}

impl BfSym {
    fn get_tape_cell(&self, pos: usize) -> OpRef {
        *self
            .tape
            .get(&pos)
            .expect("tape cell should be pre-registered")
    }

    fn set_tape_cell(&mut self, pos: usize, opref: OpRef) {
        self.tape.insert(pos, opref);
    }
}

impl JitState for BfState {
    type Meta = BfMeta;
    type Sym = BfSym;
    type Env = [u8];

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> BfMeta {
        let offsets = scan_loop_offsets(env, header_pc);
        let mut tape_positions: Vec<usize> = offsets
            .iter()
            .map(|&off| (self.pointer as isize + off) as usize)
            .collect();
        tape_positions.sort();
        tape_positions.dedup();
        BfMeta {
            tape_positions,
            initial_pointer: self.pointer,
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        meta.tape_positions
            .iter()
            .map(|&pos| self.tape[pos] as i64)
            .collect()
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> BfSym {
        let mut tape = HashMap::new();
        for (i, &pos) in meta.tape_positions.iter().enumerate() {
            tape.insert(pos, OpRef(i as u32));
        }
        BfSym {
            tape,
            tape_positions: meta.tape_positions.clone(),
            trace_pointer: meta.initial_pointer,
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        meta.tape_positions.iter().all(|&pos| pos < self.tape.len())
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        for (i, &pos) in meta.tape_positions.iter().enumerate() {
            self.tape[pos] = values[i] as u8;
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.tape_positions
            .iter()
            .filter_map(|pos| sym.tape.get(pos).copied())
            .collect()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

/// Trace one instruction, recording IR into ctx.
fn trace_instruction(
    ctx: &mut TraceCtx,
    sym: &mut BfSym,
    code: &[u8],
    pc: usize,
    tape: &[u8],
    pointer: usize,
    header_pc: usize,
) -> TraceAction {
    let ch = code[pc];

    if ch == b'>' {
        sym.trace_pointer += 1;
    } else if ch == b'<' {
        sym.trace_pointer -= 1;
    } else if ch == b'+' {
        let pos = sym.trace_pointer;
        let cell = sym.get_tape_cell(pos);
        let one = ctx.const_int(1);
        let result = ctx.record_op(OpCode::IntAdd, &[cell, one]);
        sym.set_tape_cell(pos, result);
    } else if ch == b'-' {
        let pos = sym.trace_pointer;
        let cell = sym.get_tape_cell(pos);
        let one = ctx.const_int(1);
        let result = ctx.record_op(OpCode::IntSub, &[cell, one]);
        sym.set_tape_cell(pos, result);
    } else if ch == b'.' || ch == b',' {
        // I/O during tracing — abort.
        return TraceAction::Abort;
    } else if ch == b'[' {
        // Nested loop entry: guard the condition.
        let pos = sym.trace_pointer;
        let cell = sym.get_tape_cell(pos);
        let runtime_val = tape[pointer];

        let zero = ctx.const_int(0);
        let num_live = sym.tape_positions.len();
        if runtime_val == 0 {
            let cmp = ctx.record_op(OpCode::IntEq, &[cell, zero]);
            ctx.record_guard(OpCode::GuardTrue, &[cmp], num_live);
        } else {
            let cmp = ctx.record_op(OpCode::IntNe, &[cell, zero]);
            ctx.record_guard(OpCode::GuardTrue, &[cmp], num_live);
        }
    } else if ch == b']' {
        let pos = sym.trace_pointer;
        let cell = sym.get_tape_cell(pos);
        let runtime_val = tape[pointer];

        let zero = ctx.const_int(0);
        let num_live = sym.tape_positions.len();

        if runtime_val != 0 {
            let target = find_matching_open(code, pc);

            let cmp = ctx.record_op(OpCode::IntNe, &[cell, zero]);
            ctx.record_guard(OpCode::GuardTrue, &[cmp], num_live);

            if target == header_pc {
                return TraceAction::CloseLoop;
            }
            // Inner loop back-edge: guard emitted, continue tracing.
        } else {
            // Loop exit: guard that cell is zero.
            let cmp = ctx.record_op(OpCode::IntEq, &[cell, zero]);
            ctx.record_guard(OpCode::GuardTrue, &[cmp], num_live);
        }
    }

    TraceAction::Continue
}

pub struct JitBrainInterp {
    driver: JitDriver<BfState>,
}

impl JitBrainInterp {
    pub fn new() -> Self {
        JitBrainInterp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    pub fn run(&mut self, code: &[u8]) -> String {
        let mut state = BfState {
            pointer: 0,
            tape: vec![0u8; TAPE_SIZE],
        };
        let mut output = String::new();
        let mut pc: usize = 0;

        while pc < code.len() {
            // jit_merge_point: trace instruction if tracing is active.
            let header_pc = self
                .driver
                .current_trace_green_key()
                .map(|k| k as usize)
                .unwrap_or(0);
            let pointer = state.pointer;
            let tape_snapshot: *const [u8] = &*state.tape;
            self.driver.merge_point(|ctx, sym| {
                // Safety: tape is not modified during tracing — only symbolic ops are recorded.
                let tape = unsafe { &*tape_snapshot };
                trace_instruction(ctx, sym, code, pc, tape, pointer, header_pc)
            });

            if pc >= code.len() {
                break;
            }

            let ch = code[pc];
            if ch == b'>' {
                state.pointer += 1;
                pc += 1;
            } else if ch == b'<' {
                state.pointer -= 1;
                pc += 1;
            } else if ch == b'+' {
                state.tape[state.pointer] = state.tape[state.pointer].wrapping_add(1);
                pc += 1;
            } else if ch == b'-' {
                state.tape[state.pointer] = state.tape[state.pointer].wrapping_sub(1);
                pc += 1;
            } else if ch == b'.' {
                output.push(state.tape[state.pointer] as char);
                pc += 1;
            } else if ch == b',' {
                state.tape[state.pointer] = 0;
                pc += 1;
            } else if ch == b'[' {
                if state.tape[state.pointer] == 0 {
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
                if state.tape[state.pointer] != 0 {
                    let target = find_matching_open(code, pc);

                    // can_enter_jit: back-edge detection.
                    if !self.driver.is_tracing() && target < pc {
                        if self.driver.back_edge(target, &mut state, code, || {}) {
                            // Compiled loop ran to completion; guard failure means
                            // the `[` condition is now false — skip past `]`.
                            pc += 1;
                            continue;
                        }
                    }

                    pc = target;
                } else {
                    pc += 1;
                }
            } else {
                pc += 1;
            }
        }

        output
    }
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
