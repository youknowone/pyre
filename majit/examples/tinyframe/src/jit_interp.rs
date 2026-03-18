/// JIT-enabled tinyframe interpreter — auto-generated tracing via `#[jit_interp]` + `state_fields`.
///
/// Greens: [pc, bytecode]
/// Reds:   [regs]  (tracked via state_fields)
use crate::interp::{ADD, JUMP_IF_ABOVE, LOAD, RETURN};

pub type Bytecode = [u8];

trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}
impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

struct TinyFrameState {
    regs: Vec<i64>,
}

const DEFAULT_THRESHOLD: u32 = 3;

#[majit_macros::jit_interp(
    state = TinyFrameState,
    env = Bytecode,
    state_fields = {
        regs: [int],
    },
)]
fn mainloop(
    program: &Bytecode,
    num_regs: usize,
    init_regs: &[(usize, i64)],
    threshold: u32,
) -> i64 {
    let mut driver: majit_meta::JitDriver<TinyFrameState> = majit_meta::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = TinyFrameState {
        regs: vec![0; num_regs],
    };
    for &(r, v) in init_regs {
        state.regs[r] = v;
    }

    loop {
        jit_merge_point!();
        let opcode = program[pc];

        match opcode {
            LOAD => {
                let val = program[pc + 1] as i64;
                let reg = program[pc + 2] as usize;
                state.regs[reg] = val;
                pc += 3;
            }
            ADD => {
                let r1 = program[pc + 1] as usize;
                let r2 = program[pc + 2] as usize;
                let r3 = program[pc + 3] as usize;
                state.regs[r3] = state.regs[r1] + state.regs[r2];
                pc += 4;
            }
            JUMP_IF_ABOVE => {
                let r1 = program[pc + 1] as usize;
                let r2 = program[pc + 2] as usize;
                let tgt = program[pc + 3] as usize;
                let v1 = state.regs[r1];
                let v2 = state.regs[r2];
                let jump = v1 > v2;
                if jump {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            RETURN => {
                let r = program[pc + 1] as usize;
                return state.regs[r];
            }
            _ => {
                break;
            }
        }
    }
    panic!("fell off end of code");
}

// -- Public wrapper matching the old API --

pub struct JitTinyFrameInterp {
    threshold: u32,
}

impl JitTinyFrameInterp {
    pub fn new() -> Self {
        JitTinyFrameInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// Run a tinyframe Code with initial integer register values.
    pub fn run(&mut self, code: &crate::interp::Code, init_regs: &[(usize, i64)]) -> i64 {
        mainloop(&code.code, code.regno, init_regs, self.threshold)
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
            frame.registers[2] = Some(interp::Object::Int(n));
            let interp_result = frame.interpret(&code).as_int();

            // JIT
            let mut jit = JitTinyFrameInterp::new();
            let jit_result = jit.run(&code, &[(2, n)]);

            assert_eq!(jit_result, interp_result, "count_to({n}) mismatch");
        }
    }
}
