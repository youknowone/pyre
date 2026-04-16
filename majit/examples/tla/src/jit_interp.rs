/// JIT-enabled TLA interpreter via `#[jit_interp]` with `state_fields`.
///
/// RPython parity: tla.py Frame `_virtualizable_ = ['stackpos', 'stack[*]']`
/// (tla.py:98). Integer-only trace — strings cause trace abort.
///
/// Greens: [pc, bytecode]
/// Reds:   [stackpos, stack]  (tracked via state_fields)

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

struct TlaState {
    stackpos: i64,
    stack: Vec<i64>,
}

// ── Opcodes ──

const CONST_INT: u8 = 0;
const POP: u8 = 1;
const ADD: u8 = 2;
const RETURN: u8 = 3;
const JUMP_IF: u8 = 4;
const DUP: u8 = 5;
const SUB: u8 = 6;
const NEWSTR: u8 = 7;

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = TlaState,
    env = Bytecode,
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, initial_value: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlaState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = TlaState {
        stackpos: 1,
        stack: {
            let mut s = vec![0i64; STACK_CAP];
            s[0] = initial_value;
            s
        },
    };

    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;

        match opcode {
            CONST_INT => {
                let value = program[pc] as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos = state.stackpos + 1;
            }
            POP => {
                state.stackpos = state.stackpos - 1;
            }
            DUP => {
                let v = state.stack[(state.stackpos - 1) as usize];
                state.stack[state.stackpos as usize] = v;
                state.stackpos = state.stackpos + 1;
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
            JUMP_IF => {
                let target = program[pc] as usize;
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
            NEWSTR => {
                // String operations cause trace abort — RPython would
                // guard-fail on non-int type (W_StringObject vs W_IntObject).
                pc += 1;
                break;
            }
            RETURN => break,
            _ => {}
        }
    }

    state.stackpos = state.stackpos - 1;
    state.stack[state.stackpos as usize]
}

// ── Public wrapper matching the old API ──

pub struct JitTlaInterp {
    threshold: u32,
}

impl JitTlaInterp {
    pub fn new() -> Self {
        JitTlaInterp { threshold: 3 }
    }

    pub fn run(
        &mut self,
        bytecode: &[u8],
        w_arg: crate::interp::WObject,
    ) -> crate::interp::WObject {
        let val = match &w_arg {
            crate::interp::WObject::Int(v) => *v,
            _ => panic!("JIT only supports integer args"),
        };
        let result = mainloop(bytecode, val, self.threshold);
        crate::interp::WObject::Int(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    fn countdown_bytecode() -> Vec<u8> {
        vec![DUP, CONST_INT, 1, SUB, DUP, JUMP_IF, 1, POP, RETURN]
    }

    #[test]
    fn jit_countdown_5() {
        let bc = countdown_bytecode();
        let mut jit = JitTlaInterp::new();
        let result = jit.run(&bc, interp::WObject::Int(5));
        assert_eq!(result.int_value(), 5);
    }

    #[test]
    fn jit_countdown_30() {
        let bc = countdown_bytecode();
        let mut jit = JitTlaInterp::new();
        let result = jit.run(&bc, interp::WObject::Int(30));
        assert_eq!(result.int_value(), 30);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = countdown_bytecode();
        for n in [1, 2, 5, 10, 20, 30, 40] {
            let expected = interp::run(&bc, interp::WObject::Int(n));
            let mut jit = JitTlaInterp::new();
            let got = jit.run(&bc, interp::WObject::Int(n));
            assert_eq!(got.int_value(), expected.int_value(), "mismatch for n={n}");
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![RETURN];
        let mut jit = JitTlaInterp::new();
        let result = jit.run(&prog, interp::WObject::Int(42));
        assert_eq!(result.int_value(), 42);
    }
}
