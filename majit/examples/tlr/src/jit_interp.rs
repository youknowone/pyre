/// JIT-enabled TLR interpreter — auto-generated tracing via `#[jit_interp]` + `state_fields`.
///
/// Matches RPython's tlr.py line-by-line: write the interpreter, get JIT for free.
///
/// Greens: [pc, bytecode]
/// Reds:   [a, regs]  (tracked via state_fields)

pub type Bytecode = [u8];

trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}
impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

struct TlrState {
    a: i64,
    regs: Vec<i64>,
}

const MOV_A_R: u8 = 1;
const MOV_R_A: u8 = 2;
const JUMP_IF_A: u8 = 3;
const SET_A: u8 = 4;
const ADD_R_TO_A: u8 = 5;
const RETURN_A: u8 = 6;
const ALLOCATE: u8 = 7;
const NEG_A: u8 = 8;

const DEFAULT_THRESHOLD: u32 = 3;

#[majit_macros::jit_interp(
    state = TlrState,
    env = Bytecode,
    state_fields = {
        a: int,
        regs: [int],
    },
)]
fn mainloop(program: &Bytecode, initial_a: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlrState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = TlrState {
        a: initial_a,
        regs: Vec::new(),
    };

    // while True: — RPython tlr.py:22
    loop {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;

        match opcode {
            MOV_A_R => {
                let n = program[pc] as usize;
                pc += 1;
                state.regs[n] = state.a;
            }
            MOV_R_A => {
                let n = program[pc] as usize;
                pc += 1;
                state.a = state.regs[n];
            }
            JUMP_IF_A => {
                let target = program[pc] as usize;
                pc += 1;
                let jump = state.a != 0;
                if jump {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            SET_A => {
                state.a = program[pc] as i64;
                pc += 1;
            }
            ADD_R_TO_A => {
                let n = program[pc] as usize;
                pc += 1;
                state.a = state.a + state.regs[n];
            }
            RETURN_A => {
                return state.a;
            }
            ALLOCATE => {
                let n = program[pc] as usize;
                pc += 1;
                state.regs = vec![0; n];
            }
            NEG_A => {
                state.a = 0 - state.a;
            }
            _ => {}
        }
    }
}

// ── Public wrapper matching the old API ──

pub struct JitTlrInterp {
    threshold: u32,
}

impl JitTlrInterp {
    pub fn new() -> Self {
        JitTlrInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn run(&mut self, bytecode: &[u8], initial_a: i64) -> i64 {
        mainloop(bytecode, initial_a, self.threshold)
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

    /// Exercises the JIT with many input sizes: small values stay interpreted,
    /// larger values trigger trace compilation and run compiled code.
    /// The guard exit path (a == 0 at loop end) is exercised on every input,
    /// verifying that fallback from compiled code produces correct results.
    #[test]
    fn jit_various_sizes() {
        let bc = square_bytecode();
        for a in [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlrInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }
}
