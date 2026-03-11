/// Interpreter for TL — direct translation of rpython/jit/tl/tl.py.
///
/// Stack-based interpreter with integer values and virtualizable stack.

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

pub fn interpret(code: &[u8], inputarg: i64) -> i64 {
    interpret_at(code, 0, inputarg)
}

fn interpret_at(code: &[u8], mut pc: usize, inputarg: i64) -> i64 {
    let mut stack: Vec<i64> = Vec::with_capacity(code.len());

    while pc < code.len() {
        let opcode = code[pc];
        pc += 1;

        if opcode == NOP {
            // no-op
        } else if opcode == PUSH {
            stack.push(code[pc] as i8 as i64);
            pc += 1;
        } else if opcode == POP {
            stack.pop();
        } else if opcode == SWAP {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(a);
            stack.push(b);
        } else if opcode == ROLL {
            let r = code[pc] as i8 as i64;
            pc += 1;
            roll(&mut stack, r);
        } else if opcode == PICK {
            let i = code[pc] as i8 as usize;
            pc += 1;
            let n = stack.len() - i - 1;
            let val = stack[n];
            stack.push(val);
        } else if opcode == PUT {
            let i = code[pc] as i8 as usize;
            pc += 1;
            let val = stack.pop().unwrap();
            let n = stack.len() - i - 1;
            stack[n] = val;
        } else if opcode == ADD {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(b + a);
        } else if opcode == SUB {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(b - a);
        } else if opcode == MUL {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(b * a);
        } else if opcode == DIV {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(b / a);
        } else if opcode == EQ {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(if b == a { 1 } else { 0 });
        } else if opcode == NE {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(if b != a { 1 } else { 0 });
        } else if opcode == LT {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(if b < a { 1 } else { 0 });
        } else if opcode == LE {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(if b <= a { 1 } else { 0 });
        } else if opcode == GT {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(if b > a { 1 } else { 0 });
        } else if opcode == GE {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(if b >= a { 1 } else { 0 });
        } else if opcode == BR_COND {
            let offset = code[pc] as i8 as i64;
            let cond = stack.pop().unwrap();
            if cond != 0 {
                pc = (pc as i64 + offset + 1) as usize;
            } else {
                pc += 1;
            }
        } else if opcode == BR_COND_STK {
            let offset = stack.pop().unwrap();
            let cond = stack.pop().unwrap();
            if cond != 0 {
                pc = (pc as i64 + offset) as usize;
            }
        } else if opcode == CALL {
            let offset = code[pc] as i8 as i64;
            pc += 1;
            let res = interpret_at(code, (pc as i64 + offset) as usize, 0);
            stack.push(res);
        } else if opcode == RETURN {
            break;
        } else if opcode == PUSHARG {
            stack.push(inputarg);
        }
    }

    stack.pop().unwrap()
}

fn roll(stack: &mut Vec<i64>, r: i64) {
    let len = stack.len();
    if r < -1 {
        let i = (len as i64 + r) as usize;
        let elem = stack[len - 1];
        for j in (i..len - 1).rev() {
            stack[j + 1] = stack[j];
        }
        stack[i] = elem;
    } else if r > 1 {
        let i = len - r as usize;
        let elem = stack[i];
        for j in i..len - 1 {
            stack[j] = stack[j + 1];
        }
        stack[len - 1] = elem;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push() {
        assert_eq!(interpret(&[PUSH, 16], 0), 16);
    }

    #[test]
    fn test_push_add() {
        assert_eq!(interpret(&[PUSH, 42, PUSH, 100, ADD], 0), 142);
    }

    #[test]
    fn test_push_pop() {
        assert_eq!(interpret(&[PUSH, 16, PUSH, 42, PUSH, 100, POP], 0), 42);
    }

    #[test]
    fn test_swap() {
        assert_eq!(interpret(&[PUSH, 42, PUSH, 84, SWAP], 0), 42);
    }

    #[test]
    fn test_pick() {
        assert_eq!(interpret(&[PUSH, 7, PUSH, 8, PUSH, 9, PICK, 0], 0), 9);
    }

    #[test]
    fn test_roll() {
        // ROLL 3: rotate top 3 elements up → bottom element comes to top
        assert_eq!(
            interpret(&[PUSH, 1, PUSH, 2, PUSH, 3, ROLL, 3], 0),
            1
        );
    }

    #[test]
    fn test_pusharg() {
        assert_eq!(interpret(&[PUSHARG, RETURN], 42), 42);
    }

    #[test]
    fn test_sub() {
        assert_eq!(interpret(&[PUSH, 10, PUSH, 3, SUB], 0), 7);
    }

    #[test]
    fn test_mul() {
        assert_eq!(interpret(&[PUSH, 6, PUSH, 7, MUL], 0), 42);
    }

    #[test]
    fn test_comparisons() {
        assert_eq!(interpret(&[PUSH, 3, PUSH, 5, LT], 0), 1);
        assert_eq!(interpret(&[PUSH, 5, PUSH, 3, LT], 0), 0);
        assert_eq!(interpret(&[PUSH, 3, PUSH, 3, EQ], 0), 1);
        assert_eq!(interpret(&[PUSH, 3, PUSH, 4, EQ], 0), 0);
    }

    /// sum(N) = 1 + 2 + ... + N
    fn sum_bytecode() -> Vec<u8> {
        vec![
            PUSH, 0,      // acc = 0
            PUSHARG,       // counter = N
            // loop (offset 3):
            PICK, 0,       // dup counter
            BR_COND, 2,    // if counter != 0, skip to body (offset 9)
            POP,           // pop counter
            RETURN,
            // body (offset 9):
            SWAP,          // [counter, acc]
            PICK, 1,       // [counter, acc, counter]
            ADD,           // [counter, acc+counter]
            SWAP,          // [acc+counter, counter]
            PUSH, 1,
            SUB,           // [acc, counter-1]
            PUSH, 1,
            BR_COND, 238,  // -18: jump to loop (offset 3)
        ]
    }

    /// factorial(N) = N!
    fn factorial_bytecode() -> Vec<u8> {
        vec![
            PUSH, 1,       // acc = 1
            PUSHARG,        // counter = N
            // loop (offset 3):
            PICK, 0,        // dup counter
            PUSH, 1,
            LE,             // counter <= 1?
            BR_COND, 12,    // if true, goto exit (offset 22)
            SWAP,           // [counter, acc]
            PICK, 1,        // [counter, acc, counter]
            MUL,            // [counter, acc*counter]
            SWAP,           // [acc*counter, counter]
            PUSH, 1,
            SUB,            // counter -= 1
            PUSH, 1,
            BR_COND, 237,   // -19: jump to loop (offset 3)
            // exit (offset 22):
            POP,            // pop counter
            RETURN,
        ]
    }

    #[test]
    fn test_sum_3() {
        assert_eq!(interpret(&sum_bytecode(), 3), 6);
    }

    #[test]
    fn test_sum_10() {
        assert_eq!(interpret(&sum_bytecode(), 10), 55);
    }

    #[test]
    fn test_sum_100() {
        assert_eq!(interpret(&sum_bytecode(), 100), 5050);
    }

    #[test]
    fn test_factorial_5() {
        assert_eq!(interpret(&factorial_bytecode(), 5), 120);
    }

    #[test]
    fn test_factorial_7() {
        assert_eq!(interpret(&factorial_bytecode(), 7), 5040);
    }

    #[test]
    fn test_call_return() {
        // CALL jumps to offset pc+1+offset, executes there, returns result
        // Code: CALL 1, RETURN, PUSH 42
        // CALL at offset 0: operand at 1 = 1, pc becomes 2.
        // Call interpret_at(code, 2 + 1 = 3, 0) → executes PUSH 42 → returns 42
        // Stack: [42], then RETURN → 42
        let code = vec![CALL, 1, RETURN, PUSH, 42];
        assert_eq!(interpret(&code, 0), 42);
    }
}
