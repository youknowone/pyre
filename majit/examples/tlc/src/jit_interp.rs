/// JIT-enabled TLC interpreter using JitDriver + JitState.
///
/// Greens: [pc, code]   (code is constant per trace — not tracked)
/// Reds:   [frame, pool]
///
/// Traces integer-only loops. Back-edges on BR, BR_COND.
/// Aborts on object operations (NIL, CONS, CAR, CDR, NEW, GETATTR, SETATTR,
/// SEND, CALL, DIV on cons, BR_COND_STK, PUSHARGN).
use majit_ir::{OpCode, OpRef};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

use crate::interp::{self, ConstantPool, Obj};

const NOP: u8 = interp::NOP;
const PUSH: u8 = interp::PUSH;
const POP: u8 = interp::POP;
const SWAP: u8 = interp::SWAP;
const ROLL: u8 = interp::ROLL;
const PICK: u8 = interp::PICK;
const PUT: u8 = interp::PUT;
const ADD: u8 = interp::ADD;
const SUB: u8 = interp::SUB;
const MUL: u8 = interp::MUL;
const EQ: u8 = interp::EQ;
const NE: u8 = interp::NE;
const LT: u8 = interp::LT;
const LE: u8 = interp::LE;
const GT: u8 = interp::GT;
const GE: u8 = interp::GE;
const BR: u8 = interp::BR;
const BR_COND: u8 = interp::BR_COND;
const RETURN: u8 = interp::RETURN;
const PUSHARG: u8 = interp::PUSHARG;

const DEFAULT_THRESHOLD: u32 = 3;

// ── JitState types ──

/// Red variables: stack slots + inputarg.
pub struct TlcState {
    stack: Vec<i64>,
    inputarg: i64,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct TlcMeta {
    header_pc: usize,
    /// Number of stack slots live at the loop header.
    num_stack_slots: usize,
}

/// Symbolic state during tracing — OpRef for each stack slot.
pub struct TlcSym {
    /// Symbolic trace stack: OpRef per slot.
    trace_stack: Vec<OpRef>,
    /// Number of inputargs (= stack depth at trace start).
    num_stack_slots: usize,
}

impl JitState for TlcState {
    type Meta = TlcMeta;
    type Sym = TlcSym;
    type Env = [u8];

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> TlcMeta {
        TlcMeta {
            header_pc,
            num_stack_slots: self.stack.len(),
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        self.stack[..meta.num_stack_slots].to_vec()
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> TlcSym {
        let mut trace_stack = Vec::with_capacity(meta.num_stack_slots);
        for i in 0..meta.num_stack_slots {
            trace_stack.push(OpRef(i as u32));
        }
        TlcSym {
            trace_stack,
            num_stack_slots: meta.num_stack_slots,
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        self.stack.len() >= meta.num_stack_slots
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        self.stack.clear();
        self.stack.extend_from_slice(&values[..meta.num_stack_slots]);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.trace_stack.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

/// Trace one instruction, recording IR into ctx.
fn trace_instruction(
    ctx: &mut TraceCtx,
    sym: &mut TlcSym,
    code: &[u8],
    pc: usize,
    runtime_stack: &[i64],
    inputarg: i64,
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
    } else if opcode == ROLL {
        let r = code[pc + 1] as i8 as i64;
        if r < -1 {
            let i = sym.trace_stack.len() as i64 + r;
            assert!(i >= 0, "IndexError in trace ROLL");
            let val = sym.trace_stack.pop().unwrap();
            sym.trace_stack.insert(i as usize, val);
        } else if r > 1 {
            let i = sym.trace_stack.len() as i64 - r;
            assert!(i >= 0, "IndexError in trace ROLL");
            let val = sym.trace_stack.remove(i as usize);
            sym.trace_stack.push(val);
        }
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
    } else if opcode == BR {
        let offset = code[pc + 1] as i8 as i64;
        let target = ((pc + 1) as i64 + offset + 1) as usize;
        if target == header_pc {
            return TraceAction::CloseLoop;
        }
        // Forward BR: no guard needed, just continue.
    } else if opcode == BR_COND {
        let offset = code[pc + 1] as i8 as i64;
        let target = ((pc + 1) as i64 + offset + 1) as usize;
        let cond = sym.trace_stack.pop().unwrap();

        let runtime_cond = *runtime_stack.last().unwrap();

        if runtime_cond != 0 && target == header_pc {
            // Back-edge to loop header: guard and close loop.
            let num_live = sym.num_stack_slots;
            ctx.record_guard(OpCode::GuardTrue, &[cond], num_live);
            return TraceAction::CloseLoop;
        } else if runtime_cond != 0 {
            // Forward branch taken: guard true.
            let num_live = sym.num_stack_slots;
            ctx.record_guard(OpCode::GuardTrue, &[cond], num_live);
        } else {
            // Branch not taken: guard false.
            let num_live = sym.num_stack_slots;
            ctx.record_guard(OpCode::GuardFalse, &[cond], num_live);
        }
    } else if opcode == PUSHARG {
        let opref = ctx.const_int(inputarg);
        sym.trace_stack.push(opref);
    } else if opcode == RETURN {
        return TraceAction::Abort;
    } else {
        // Unsupported opcode during tracing (object operations)
        return TraceAction::Abort;
    }

    TraceAction::Continue
}

pub struct JitTlcInterp {
    driver: JitDriver<TlcState>,
}

impl JitTlcInterp {
    pub fn new() -> Self {
        JitTlcInterp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    /// Run the TLC interpreter with JIT support.
    /// Only traces integer-only loops; falls back to plain interpreter for
    /// object operations.
    pub fn run(&mut self, code: &[u8], inputarg: i64, pool: &ConstantPool) -> i64 {
        let mut state = TlcState {
            stack: Vec::with_capacity(32),
            inputarg,
        };
        let mut pc: usize = 0;

        loop {
            // jit_merge_point(code=code, pc=pc, stack=stack, inputarg=inputarg)
            let header_pc = self
                .driver
                .current_trace_green_key()
                .map(|k| k as usize)
                .unwrap_or(0);
            self.driver.merge_point(|ctx, sym| {
                trace_instruction(ctx, sym, code, pc, &state.stack, state.inputarg, header_pc)
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
            } else if opcode == ROLL {
                let r = code[pc] as i8 as i64;
                if r < -1 {
                    let i = state.stack.len() as i64 + r;
                    assert!(i >= 0, "IndexError");
                    let val = state.stack.pop().unwrap();
                    state.stack.insert(i as usize, val);
                } else if r > 1 {
                    let i = state.stack.len() as i64 - r;
                    assert!(i >= 0, "IndexError");
                    let val = state.stack.remove(i as usize);
                    state.stack.push(val);
                }
                pc += 1;
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
            } else if opcode == BR {
                let old_pc = pc;
                let offset = code[pc] as i8 as i64;
                pc = ((pc as i64) + offset + 1) as usize;

                // can_enter_jit at backward jumps
                if old_pc > pc {
                    if self.driver.back_edge(pc, &mut state, code, || {}) {
                        continue;
                    }
                }
            } else if opcode == BR_COND {
                let offset = code[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                let next_pc = pc + 1;
                let cond = state.stack.pop().unwrap();
                if cond != 0 {
                    // can_enter_jit at backward jumps
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
            } else {
                // For object operations (NIL, CONS, CAR, CDR, NEW, etc.),
                // fall back to the full interpreter.
                pc -= 1; // back up to re-read the opcode
                let args = vec![Obj::Int(inputarg)];
                let full_result = interp::interp_eval(code, pc, args, pool);
                return match full_result {
                    Some(obj) => obj.int_o(),
                    None => {
                        if let Some(&top) = state.stack.last() {
                            top
                        } else {
                            0
                        }
                    }
                };
            }
        }

        state.stack.pop().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    /// Fibonacci using ROLL -- pure integer loop, good JIT candidate.
    fn fibo_bytecode(pool: &mut ConstantPool) -> Vec<u8> {
        interp::compile(
            include_str!("../../../../rpython/jit/tl/fibo.tlc.src"),
            pool,
        )
    }

    #[test]
    fn jit_fibo_7() {
        let mut pool = ConstantPool::new();
        let bc = fibo_bytecode(&mut pool);
        let mut jit = JitTlcInterp::new();
        assert_eq!(jit.run(&bc, 7, &pool), 13);
    }

    #[test]
    fn jit_fibo_matches_interp() {
        let mut pool = ConstantPool::new();
        let bc = fibo_bytecode(&mut pool);
        for n in [1, 2, 3, 5, 7, 10, 15] {
            let expected = interp::interp(&bc, 0, n, &pool);
            let mut jit = JitTlcInterp::new();
            let got = jit.run(&bc, n, &pool);
            assert_eq!(got, expected, "fibo mismatch for n={n}");
        }
    }

    /// Simple integer countdown loop (no object ops).
    #[test]
    fn jit_countdown() {
        let mut pool = ConstantPool::new();
        let bc = interp::compile(
            "
            PUSHARG         # [n]
        loop:
            PUSH 1
            SUB             # [n-1]
            PICK 0          # [n-1, n-1]
            BR_COND loop    # [n-1] if n-1 != 0
            RETURN
        ",
            &mut pool,
        );
        let mut jit = JitTlcInterp::new();
        assert_eq!(jit.run(&bc, 100, &pool), 0);
    }

    #[test]
    fn jit_sum() {
        let mut pool = ConstantPool::new();
        let bc = interp::compile(
            "
            PUSH 0          # [acc=0]
            PUSHARG         # [acc, n]
        loop:
            PICK 0          # [acc, n, n]
            BR_COND body
            POP
            RETURN
        body:
            SWAP            # [n, acc]
            PICK 1          # [n, acc, n]
            ADD             # [n, acc+n]
            SWAP            # [acc+n, n]
            PUSH 1
            SUB             # [acc, n-1]
            PUSH 1
            BR_COND loop
        ",
            &mut pool,
        );
        let mut jit = JitTlcInterp::new();
        assert_eq!(jit.run(&bc, 10, &pool), 55);
        let mut jit2 = JitTlcInterp::new();
        assert_eq!(jit2.run(&bc, 100, &pool), 5050);
    }
}
