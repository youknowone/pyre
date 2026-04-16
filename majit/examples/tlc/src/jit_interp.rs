/// JIT-enabled TLC interpreter via `#[jit_interp]` with `state_fields`.
///
/// RPython parity: tlc.py JitDriver(greens=['pc','code'], reds=['frame','pool']).
/// TLC's Frame has a plain list stack (no `_virtualizable_`); we use
/// `state_fields = { stackpos: int, stack: [int; virt] }` to mirror the
/// virtualizable-stack shape used by tl.py/tla.py for the integer-only trace.
///
/// Greens: [pc]
/// Reds:   [stackpos, stack]
///
/// Only integer-stack opcodes are traced. Object opcodes (NIL, CONS, CAR, CDR,
/// NEW, GETATTR, SETATTR, SEND) cause guard failure in RPython and are absent
/// from this function, matching that behavior.
use crate::interp::{self, ConstantPool};

// ── State ──

pub type Bytecode = [u8];

trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}

impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

const STACK_CAP: usize = 1024;

struct TlcState {
    stackpos: i64,
    stack: Vec<i64>,
}

/// Stack rotation — residual CALL in the trace.
///
/// RPython parity: tlc.py:284 ROLL (inline). Writing it as a raw-pointer
/// function with `#[dont_look_inside]` lets the JIT emit a single residual
/// CALL instead of tracing the inner shuffle.
#[majit_macros::dont_look_inside]
extern "C" fn tlc_roll(stack_ptr: usize, stackpos: i64, r: i64) {
    let stack = unsafe { std::slice::from_raw_parts_mut(stack_ptr as *mut i64, stackpos as usize) };
    let len = stack.len();
    if r < -1 {
        // Move top element to position len+r (counted from bottom).
        let i = len as i64 + r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let elem = stack[len - 1];
        for j in (i..len - 1).rev() {
            stack[j + 1] = stack[j];
        }
        stack[i] = elem;
    } else if r > 1 {
        // Move element at position len-r to top.
        let i = len as i64 - r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let elem = stack[i];
        for j in i..len - 1 {
            stack[j] = stack[j + 1];
        }
        stack[len - 1] = elem;
    }
}

// ── Opcodes ──

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
// DIV not traced: IntObj.div() in tlc.py:144 uses Python 2 floor division (//),
// which differs from Rust's truncating division for negative operands.
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

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = TlcState,
    env = Bytecode,
    auto_calls = true,
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlcState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = TlcState {
        stackpos: 0,
        stack: vec![0i64; STACK_CAP],
    };

    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;

        match opcode {
            NOP => {}
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos = state.stackpos + 1;
            }
            POP => {
                state.stackpos = state.stackpos - 1;
            }
            SWAP => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 1) as usize] = b;
                state.stack[(state.stackpos - 2) as usize] = a;
            }
            ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos = state.stackpos - 1;
            }
            SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos = state.stackpos - 1;
            }
            MUL => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b * a;
                state.stackpos = state.stackpos - 1;
            }
            EQ => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b == a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            NE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b != a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            LT => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b < a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            LE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b <= a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            GT => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b > a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            GE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b >= a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            ROLL => {
                let r = program[pc] as i8 as i64;
                pc += 1;
                tlc_roll(state.stack.as_mut_ptr() as usize, state.stackpos, r);
            }
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let v = state.stack[(state.stackpos as usize) - i - 1];
                state.stack[state.stackpos as usize] = v;
                state.stackpos = state.stackpos + 1;
            }
            PUT => {
                let i = program[pc] as usize;
                pc += 1;
                state.stackpos = state.stackpos - 1;
                let v = state.stack[state.stackpos as usize];
                state.stack[(state.stackpos as usize) - i] = v;
            }
            PUSHARG => {
                state.stack[state.stackpos as usize] = inputarg;
                state.stackpos = state.stackpos + 1;
            }
            BR_COND => {
                let target = ((pc as i64) + program[pc] as i8 as i64 + 1) as usize;
                let next_pc = pc + 1;
                state.stackpos = state.stackpos - 1;
                let jump = state.stack[state.stackpos as usize] != 0;
                if jump {
                    if target < next_pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
                pc = next_pc;
            }
            BR => {
                let target = ((pc as i64) + program[pc] as i8 as i64 + 1) as usize;
                let next_pc = pc + 1;
                if target < next_pc {
                    can_enter_jit!(driver, target, &mut state, program, || {});
                }
                pc = target;
                continue;
            }
            RETURN => break,
            _ => break,
        }
    }

    if state.stackpos == 0 {
        0
    } else {
        state.stackpos = state.stackpos - 1;
        state.stack[state.stackpos as usize]
    }
}

// ── Public wrapper matching the old API ──

pub struct JitTlcInterp {
    threshold: u32,
}

impl JitTlcInterp {
    pub fn new() -> Self {
        JitTlcInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// Run the TLC interpreter with JIT support.
    /// Only traces integer-only loops; unknown opcodes cause loop exit.
    pub fn run(&mut self, code: &[u8], inputarg: i64, _pool: &ConstantPool) -> i64 {
        mainloop(code, inputarg, self.threshold)
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
