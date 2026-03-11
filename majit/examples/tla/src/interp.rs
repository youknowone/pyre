/// Interpreter for TLA — direct translation of rpython/jit/tl/tla/tla.py.
///
/// Object-oriented stack machine with wrapped values (W_IntObject, W_StringObject).
/// The frame is virtualizable in RPython: `_virtualizable_ = ['stackpos', 'stack[*]']`.

const CONST_INT: u8 = 0;
const POP: u8 = 1;
const ADD: u8 = 2;
const RETURN: u8 = 3;
const JUMP_IF: u8 = 4;
const DUP: u8 = 5;
const SUB: u8 = 6;
const NEWSTR: u8 = 7;

#[derive(Clone, Debug)]
pub enum WObject {
    Int(i64),
    Str(String),
}

impl WObject {
    fn is_true(&self) -> bool {
        match self {
            WObject::Int(v) => *v != 0,
            WObject::Str(s) => !s.is_empty(),
        }
    }

    fn add(&self, other: &WObject) -> WObject {
        match (self, other) {
            (WObject::Int(a), WObject::Int(b)) => WObject::Int(a + b),
            _ => panic!("unsupported add"),
        }
    }

    fn sub(&self, other: &WObject) -> WObject {
        match (self, other) {
            (WObject::Int(a), WObject::Int(b)) => WObject::Int(a - b),
            _ => panic!("unsupported sub"),
        }
    }

    pub fn int_value(&self) -> i64 {
        match self {
            WObject::Int(v) => *v,
            _ => panic!("not an int"),
        }
    }
}

pub fn run(bytecode: &[u8], w_arg: WObject) -> WObject {
    let mut stack: Vec<WObject> = Vec::with_capacity(8);
    stack.push(w_arg);
    let mut pc: usize = 0;

    while pc < bytecode.len() {
        let opcode = bytecode[pc];
        pc += 1;

        if opcode == CONST_INT {
            let value = bytecode[pc] as i64;
            pc += 1;
            stack.push(WObject::Int(value));
        } else if opcode == POP {
            stack.pop();
        } else if opcode == DUP {
            let w_x = stack.last().unwrap().clone();
            stack.push(w_x);
        } else if opcode == ADD {
            let w_y = stack.pop().unwrap();
            let w_x = stack.pop().unwrap();
            stack.push(w_x.add(&w_y));
        } else if opcode == SUB {
            let w_y = stack.pop().unwrap();
            let w_x = stack.pop().unwrap();
            stack.push(w_x.sub(&w_y));
        } else if opcode == JUMP_IF {
            let target = bytecode[pc] as usize;
            pc += 1;
            let w_x = stack.pop().unwrap();
            if w_x.is_true() {
                pc = target;
            }
        } else if opcode == NEWSTR {
            let ch = bytecode[pc] as char;
            pc += 1;
            stack.push(WObject::Str(ch.to_string()));
        } else if opcode == RETURN {
            let w_x = stack.pop().unwrap();
            assert!(stack.is_empty(), "stack not empty at RETURN");
            return w_x;
        }
    }

    stack.pop().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_return() {
        let res = run(&[RETURN], WObject::Int(42));
        assert_eq!(res.int_value(), 42);
    }

    #[test]
    fn test_pop() {
        let res = run(&[CONST_INT, 99, POP, RETURN], WObject::Int(42));
        assert_eq!(res.int_value(), 42);
    }

    #[test]
    fn test_dup_add() {
        let res = run(&[DUP, ADD, RETURN], WObject::Int(41));
        assert_eq!(res.int_value(), 82);
    }

    #[test]
    fn test_add() {
        let res = run(&[CONST_INT, 20, ADD, RETURN], WObject::Int(22));
        assert_eq!(res.int_value(), 42);
    }

    #[test]
    fn test_sub() {
        let res = run(&[CONST_INT, 20, SUB, RETURN], WObject::Int(22));
        assert_eq!(res.int_value(), 2);
    }

    #[test]
    fn test_jump_if() {
        let code = [JUMP_IF, 5, CONST_INT, 123, RETURN, CONST_INT, 234, RETURN];
        let res = run(&code, WObject::Int(0));
        assert_eq!(res.int_value(), 123);

        let res = run(&code, WObject::Int(1));
        assert_eq!(res.int_value(), 234);
    }

    #[test]
    fn test_newstr() {
        let res = run(
            &[POP, NEWSTR, b'x', RETURN],
            WObject::Int(0),
        );
        match res {
            WObject::Str(s) => assert_eq!(s, "x"),
            _ => panic!("expected string"),
        }
    }

    /// Countdown: decrement N to 0.
    /// Bytecodes: DUP, CONST_INT 1, SUB, DUP, JUMP_IF 1, POP, RETURN
    fn countdown_bytecode() -> Vec<u8> {
        vec![
            DUP,              // 0
            CONST_INT, 1,     // 1, 2
            SUB,              // 3
            DUP,              // 4
            JUMP_IF, 1,       // 5, 6 → back to CONST_INT
            POP,              // 7
            RETURN,           // 8
        ]
    }

    #[test]
    fn test_countdown_3() {
        let bc = countdown_bytecode();
        let res = run(&bc, WObject::Int(3));
        assert_eq!(res.int_value(), 3);
    }

    #[test]
    fn test_countdown_10() {
        let bc = countdown_bytecode();
        let res = run(&bc, WObject::Int(10));
        assert_eq!(res.int_value(), 10);
    }

    #[test]
    fn test_countdown_100() {
        let bc = countdown_bytecode();
        let res = run(&bc, WObject::Int(100));
        assert_eq!(res.int_value(), 100);
    }
}
