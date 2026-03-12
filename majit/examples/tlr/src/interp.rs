/// Interpreter for TLR — direct translation of rpython/jit/tl/tlr.py:interpret().
///
/// Bytecode format is identical: &[u8] with single-byte opcodes and args.

const MOV_A_R: u8 = 1;
const MOV_R_A: u8 = 2;
const JUMP_IF_A: u8 = 3;
const SET_A: u8 = 4;
const ADD_R_TO_A: u8 = 5;
const RETURN_A: u8 = 6;
const ALLOCATE: u8 = 7;
const NEG_A: u8 = 8;

pub fn interpret(bytecode: &[u8], a: i64) -> i64 {
    let mut regs: Vec<i64> = Vec::new();
    let mut pc: usize = 0;
    let mut a = a;

    loop {
        // jitdriver.jit_merge_point(...)
        let opcode = bytecode[pc];
        pc += 1;

        if opcode == MOV_A_R {
            let n = bytecode[pc] as usize;
            pc += 1;
            regs[n] = a;
        } else if opcode == MOV_R_A {
            let n = bytecode[pc] as usize;
            pc += 1;
            a = regs[n];
        } else if opcode == JUMP_IF_A {
            let target = bytecode[pc] as usize;
            pc += 1;
            if a != 0 {
                // if target < pc: can_enter_jit(...)
                pc = target;
            }
        } else if opcode == SET_A {
            a = bytecode[pc] as i64;
            pc += 1;
        } else if opcode == ADD_R_TO_A {
            let n = bytecode[pc] as usize;
            pc += 1;
            a += regs[n];
        } else if opcode == RETURN_A {
            return a;
        } else if opcode == ALLOCATE {
            let n = bytecode[pc] as usize;
            pc += 1;
            regs = vec![0; n];
        } else if opcode == NEG_A {
            a = -a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_bytecode() -> Vec<u8> {
        vec![
            ALLOCATE, 3, MOV_A_R, 0, MOV_A_R, 1, SET_A, 0, MOV_A_R, 2, SET_A, 1, NEG_A, ADD_R_TO_A,
            0, MOV_A_R, 0, MOV_R_A, 2, ADD_R_TO_A, 1, MOV_A_R, 2, MOV_R_A, 0, JUMP_IF_A, 10,
            MOV_R_A, 2, RETURN_A,
        ]
    }

    #[test]
    fn test_square_1() {
        assert_eq!(interpret(&square_bytecode(), 1), 1);
    }

    #[test]
    fn test_square_5() {
        assert_eq!(interpret(&square_bytecode(), 5), 25);
    }

    #[test]
    fn test_square_10() {
        assert_eq!(interpret(&square_bytecode(), 10), 100);
    }

    #[test]
    fn test_square_100() {
        assert_eq!(interpret(&square_bytecode(), 100), 10_000);
    }

    #[test]
    fn test_simple_return() {
        let prog = vec![RETURN_A];
        assert_eq!(interpret(&prog, 42), 42);
    }

    #[test]
    fn test_set_and_return() {
        let prog = vec![SET_A, 7, RETURN_A];
        assert_eq!(interpret(&prog, 0), 7);
    }
}
