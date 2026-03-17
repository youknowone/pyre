/// JIT-enabled TL interpreter using JitDriver + JitState.
///
/// Greens: [pc, code]    (code is constant per trace — not tracked)
/// Reds:   [inputarg, stack]
///
/// The stack is virtualizable: during tracing, stack slots are mapped to
/// IR operations (OpRef), eliminating memory loads/stores in compiled code.
/// In RPython's tl.py, this is declared as `_virtualizable_ = ['stackpos', 'stack[*]']`.
/// Here, stack slots are passed directly as inputargs/jump_args to achieve
/// the same optimization — stack contents stay in JIT registers across iterations.
///
/// This example hand-writes `trace_instruction` for educational purposes.
/// In production, the `#[jit_interp]` proc macro auto-generates tracing
/// code from the interpreter's match dispatch — see aheuijit for an example.
use majit_ir::{OpCode, OpRef};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

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

// ── JitState types ──

/// Red variables: inputarg + stack slots.
pub struct TlState {
    inputarg: i64,
    stack: Vec<i64>,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct TlMeta {
    header_pc: usize,
    /// Number of stack slots live at the loop header.
    num_stack_slots: usize,
}

/// Symbolic state during tracing — OpRef for each stack slot.
pub struct TlSym {
    /// Symbolic stack: each entry is an OpRef tracking the IR value.
    trace_stack: Vec<OpRef>,
    /// Number of stack slots at trace start (= number of inputargs).
    num_stack_slots: usize,
}

impl JitState for TlState {
    type Meta = TlMeta;
    type Sym = TlSym;
    type Env = [u8];

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> TlMeta {
        TlMeta {
            header_pc,
            num_stack_slots: self.stack.len(),
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        self.stack[..meta.num_stack_slots].to_vec()
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> TlSym {
        let trace_stack: Vec<OpRef> = (0..meta.num_stack_slots)
            .map(|i| OpRef(i as u32))
            .collect();
        TlSym {
            trace_stack,
            num_stack_slots: meta.num_stack_slots,
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        self.stack.len() >= meta.num_stack_slots
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        self.stack.truncate(meta.num_stack_slots);
        for (i, &v) in values.iter().enumerate() {
            if i < self.stack.len() {
                self.stack[i] = v;
            }
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.trace_stack.clone()
    }

    fn validate_close(sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        sym.trace_stack.len() == sym.num_stack_slots
    }
}

/// Trace one instruction, recording IR into ctx.
fn trace_instruction(
    ctx: &mut TraceCtx,
    sym: &mut TlSym,
    code: &[u8],
    pc: usize,
    state: &TlState,
    header_pc: usize,
) -> TraceAction {
    let opcode = code[pc];

    if opcode == NOP {
        // no-op
    } else if opcode == PUSH {
        let val = code[pc + 1] as i8 as i64;
        let opref = ctx.const_int(val);
        sym.trace_stack.push(opref);
    } else if opcode == POP {
        sym.trace_stack.pop();
    } else if opcode == SWAP {
        let len = sym.trace_stack.len();
        sym.trace_stack.swap(len - 1, len - 2);
    } else if opcode == PICK {
        let i = code[pc + 1] as usize;
        let n = sym.trace_stack.len() - i - 1;
        let opref = sym.trace_stack[n];
        sym.trace_stack.push(opref);
    } else if opcode == PUT {
        let i = code[pc + 1] as usize;
        let opref = sym.trace_stack.pop().unwrap();
        let n = sym.trace_stack.len() - i - 1;
        sym.trace_stack[n] = opref;
    } else if opcode == ADD {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntAdd, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == SUB {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntSub, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == MUL {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntMul, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == EQ {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntEq, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == NE {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntNe, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == LT {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntLt, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == LE {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntLe, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == GT {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntGt, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == GE {
        let a = sym.trace_stack.pop().unwrap();
        let b = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntGe, &[b, a]);
        sym.trace_stack.push(result);
    } else if opcode == BR_COND {
        let offset = code[pc + 1] as i8 as i64;
        let target = ((pc + 1) as i64 + offset + 1) as usize;
        let cond = sym.trace_stack.pop().unwrap();

        // Check runtime condition to determine which path was taken.
        let runtime_cond = *state.stack.last().unwrap();

        if runtime_cond != 0 && target == header_pc {
            // Back-edge to loop header: guard and close loop.
            let num_live = sym.trace_stack.len();
            ctx.record_guard(OpCode::GuardTrue, &[cond], num_live);
            return TraceAction::CloseLoop;
        } else if runtime_cond != 0 {
            // Forward branch taken: guard that condition is true.
            let num_live = sym.trace_stack.len();
            ctx.record_guard(OpCode::GuardTrue, &[cond], num_live);
        } else {
            // Branch not taken: guard that condition is false.
            let num_live = sym.trace_stack.len();
            ctx.record_guard(OpCode::GuardFalse, &[cond], num_live);
        }
    } else if opcode == PUSHARG {
        let opref = ctx.const_int(state.inputarg);
        sym.trace_stack.push(opref);
    } else if opcode == RETURN {
        return TraceAction::Abort;
    } else {
        // Unsupported opcode during tracing (ROLL, CALL, BR_COND_STK, DIV, etc.)
        return TraceAction::Abort;
    }

    TraceAction::Continue
}

pub struct JitTlInterp {
    driver: JitDriver<TlState>,
}

impl JitTlInterp {
    pub fn new() -> Self {
        JitTlInterp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    pub fn run(&mut self, code: &[u8], inputarg: i64) -> i64 {
        let mut state = TlState {
            inputarg,
            stack: Vec::with_capacity(code.len()),
        };
        let mut pc: usize = 0;

        loop {
            // jit_merge_point(code=code, pc=pc, inputarg=inputarg, stack=stack)
            let header_pc = self.driver.current_trace_green_key()
                .map(|k| k as usize)
                .unwrap_or(0);
            self.driver.merge_point(|ctx, sym| {
                trace_instruction(ctx, sym, code, pc, &state, header_pc)
            });

            if pc >= code.len() {
                break;
            }

            let opcode = code[pc];
            pc += 1;

            if opcode == NOP {
                // no-op
            } else if opcode == PUSH {
                state.stack.push(code[pc] as i8 as i64);
                pc += 1;
            } else if opcode == POP {
                state.stack.pop();
            } else if opcode == SWAP {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(a);
                state.stack.push(b);
            } else if opcode == PICK {
                let i = code[pc] as usize;
                pc += 1;
                let n = state.stack.len() - i - 1;
                let val = state.stack[n];
                state.stack.push(val);
            } else if opcode == PUT {
                let i = code[pc] as usize;
                pc += 1;
                let val = state.stack.pop().unwrap();
                let n = state.stack.len() - i - 1;
                state.stack[n] = val;
            } else if opcode == ADD {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(b + a);
            } else if opcode == SUB {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(b - a);
            } else if opcode == MUL {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(b * a);
            } else if opcode == EQ {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(if b == a { 1 } else { 0 });
            } else if opcode == NE {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(if b != a { 1 } else { 0 });
            } else if opcode == LT {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(if b < a { 1 } else { 0 });
            } else if opcode == LE {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(if b <= a { 1 } else { 0 });
            } else if opcode == GT {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(if b > a { 1 } else { 0 });
            } else if opcode == GE {
                let a = state.stack.pop().unwrap();
                let b = state.stack.pop().unwrap();
                state.stack.push(if b >= a { 1 } else { 0 });
            } else if opcode == BR_COND {
                let offset = code[pc] as i8 as i64;
                let target = (pc as i64 + offset + 1) as usize;
                let next_pc = pc + 1;
                let cond = state.stack.pop().unwrap();
                if cond != 0 {
                    // can_enter_jit(code=code, pc=target, inputarg=inputarg, stack=stack)
                    if target < next_pc {
                        if self.driver.back_edge(target, &mut state, code, || {}) {
                            pc = target;
                            continue;
                        }
                    }
                    pc = target;
                } else {
                    pc = next_pc;
                }
            } else if opcode == RETURN {
                break;
            } else if opcode == PUSHARG {
                state.stack.push(inputarg);
            }
        }

        state.stack.pop().unwrap()
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

    /// Exercises the JIT with many input sizes: small values stay interpreted,
    /// larger values trigger trace compilation and run compiled code.
    /// The guard exit path (counter == 0) is exercised on every input,
    /// verifying that fallback from compiled code produces correct results.
    ///
    /// Bridge compilation is handled automatically by MetaInterp: when a guard
    /// fails repeatedly, MetaInterp begins tracing a bridge from the guard's
    /// fail point. No explicit test setup is needed — the mechanism activates
    /// whenever guard failure count exceeds the bridge threshold.
    #[test]
    fn jit_various_sizes() {
        let bc = sum_bytecode();
        for a in [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    /// Uses the sum program with a shared JitTlInterp instance across multiple
    /// calls. The first few small inputs are interpreted; once the back-edge
    /// counter reaches the threshold the loop compiles. Subsequent inputs run
    /// compiled code and exit via the loop-exit guard (counter == 0).
    /// This exercises the guard exit path with a persistent compiled trace,
    /// which is the prerequisite for bridge compilation in MetaInterp.
    #[test]
    fn jit_bridge_exercise() {
        let bc = sum_bytecode();
        let mut jit = JitTlInterp::new();
        for a in [3, 5, 10, 20, 50, 100] {
            let expected = interp::interpret(&bc, a);
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }
}
