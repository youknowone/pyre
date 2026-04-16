/// JIT-enabled TL interpreter via `#[jit_interp]` with `state_fields`.
///
/// RPython parity: tl.py JitDriver(greens=['pc','code'], reds=['inputarg','stack'],
/// virtualizables=['stack']). Stack._virtualizable_ = ['stackpos', 'stack[*]']
/// at tl.py:14 maps directly to `state_fields = { stackpos: int, stack: [int; virt] }`.
///
/// Greens: [pc, code]
/// Reds:   [inputarg, stackpos, stack]  (inputarg is a function parameter — red by nature)
use majit_metainterp::jit::promote;

/// Stack rotation — @dont_look_inside in RPython (tl.py:43).
///
/// Operates on the live portion of the stack `stack[0..stackpos]`.
/// The JIT does not trace into this function; it emits a residual CALL.
#[majit_macros::dont_look_inside]
extern "C" fn storage_roll(stack_ptr: usize, stackpos: i64, r: i64) {
    let stack = unsafe { std::slice::from_raw_parts_mut(stack_ptr as *mut i64, stackpos as usize) };
    let len = stack.len();
    if r < -1 {
        // tl.py:45-55
        let i = len as i64 + r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let n = len - 1;
        let elem = stack[n];
        for j in (i..n).rev() {
            stack[j + 1] = stack[j];
        }
        stack[i] = elem;
    } else if r > 1 {
        // tl.py:56-65
        let i = len as i64 - r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let elem = stack[i];
        for j in i..len - 1 {
            stack[j] = stack[j + 1];
        }
        let n = len - 1;
        stack[n] = elem;
    }
}

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

/// tl.py:13-14 Stack object. `_virtualizable_ = ['stackpos', 'stack[*]']`.
/// tl.py:17 `Stack(size)` — `size` is the bytecode length; the caller
/// (`interp_eval`) passes `len(code)`. See tl.py:120.
struct TlState {
    stackpos: i64,
    stack: Vec<i64>,
}

// ── Opcodes ──

const NOP: u8 = 1;
const PUSH: u8 = 2;
const POP: u8 = 3;
const SWAP: u8 = 4;
const ROLL: u8 = 5;
const PICK: u8 = 6;
const PUT: u8 = 7;
const ADD: u8 = 8;
const SUB: u8 = 9;
const MUL: u8 = 10;
const DIV: u8 = 11;
const EQ: u8 = 12;
const NE: u8 = 13;
const LT: u8 = 14;
const LE: u8 = 15;
const GT: u8 = 16;
const GE: u8 = 17;
const BR_COND: u8 = 18;
const BR_COND_STK: u8 = 19;
const CALL: u8 = 20;
const RETURN: u8 = 21;
const PUSHARG: u8 = 22;

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = TlState,
    env = Bytecode,
    auto_calls = true,
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = TlState {
        stackpos: 0,
        stack: vec![0i64; program.len()],
    };

    while pc < program.len() {
        jit_merge_point!();
        // tl.py:88  stack.stackpos = promote(stack.stackpos)
        state.stackpos = promote(state.stackpos);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            NOP => {}
            // tl.py:94-96
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos = state.stackpos + 1;
            }
            // tl.py:98-99
            POP => {
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:101-104
            SWAP => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 1) as usize] = b;
                state.stack[(state.stackpos - 2) as usize] = a;
            }
            // tl.py:106-109  Stack.roll() is @dont_look_inside
            ROLL => {
                let r = program[pc] as i8 as i64;
                pc += 1;
                storage_roll(state.stack.as_mut_ptr() as usize, state.stackpos, r);
            }
            // tl.py:111-113  Stack.pick(i): duplicate stack[stackpos - i - 1]
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let v = state.stack[(state.stackpos as usize) - i - 1];
                state.stack[state.stackpos as usize] = v;
                state.stackpos = state.stackpos + 1;
            }
            // tl.py:115-117  Stack.put(i): pop and store at stackpos - i - 1
            PUT => {
                let i = program[pc] as usize;
                pc += 1;
                state.stackpos = state.stackpos - 1;
                let v = state.stack[state.stackpos as usize];
                state.stack[(state.stackpos as usize) - i] = v;
            }
            // tl.py:119-121
            ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:123-125
            SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:127-129
            MUL => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b * a;
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:131-133
            DIV => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b / a;
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:135-157 — inline comparisons (no helper functions)
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
            // tl.py:159-165
            BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                state.stackpos = state.stackpos - 1;
                let jump = state.stack[state.stackpos as usize] != 0;
                if jump {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            // tl.py:167-172
            BR_COND_STK => {
                state.stackpos = state.stackpos - 1;
                let offset = state.stack[state.stackpos as usize];
                state.stackpos = state.stackpos - 1;
                let cond = state.stack[state.stackpos as usize];
                if cond != 0 {
                    let target = (pc as i64 + offset) as usize;
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            // tl.py:174-178
            CALL => {
                let offset = program[pc] as i8 as i64;
                pc += 1;
                let res = crate::interp::interpret_at(program, (pc as i64 + offset) as usize, 0);
                state.stack[state.stackpos as usize] = res;
                state.stackpos = state.stackpos + 1;
            }
            // tl.py:180-181
            RETURN => break,
            // tl.py:183-184
            PUSHARG => {
                state.stack[state.stackpos as usize] = inputarg;
                state.stackpos = state.stackpos + 1;
            }
            _ => {}
        }
    }

    state.stackpos = state.stackpos - 1;
    state.stack[state.stackpos as usize]
}

// ── Public wrapper matching the old API ──

pub struct JitTlInterp {
    threshold: u32,
}

impl JitTlInterp {
    pub fn new() -> Self {
        JitTlInterp { threshold: 3 }
    }

    pub fn run(&mut self, bytecode: &[u8], inputarg: i64) -> i64 {
        mainloop(bytecode, inputarg, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    /// sum(N) = 1 + 2 + ... + N
    fn sum_bytecode() -> Vec<u8> {
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
