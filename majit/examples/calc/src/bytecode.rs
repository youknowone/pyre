/// Bytecode definitions for the calc interpreter.

#[derive(Debug, Clone, PartialEq)]
pub enum ByteCode {
    /// Push a constant onto the stack.
    LoadConst(i64),
    /// Push the value of a variable (0-25 = a-z) onto the stack.
    LoadVar(u8),
    /// Pop the top of the stack and store it in a variable (0-25 = a-z).
    StoreVar(u8),
    /// Pop two values, push their sum.
    Add,
    /// Pop two values, push their difference (a - b where b is top).
    Sub,
    /// Pop two values, push their product.
    Mul,
    /// Pop two values, push their quotient (a / b where b is top).
    Div,
    /// Pop two values, push their remainder (a % b where b is top).
    Mod,
    /// Pop two values, push 1 if a < b, else 0.
    Lt,
    /// Pop two values, push 1 if a <= b, else 0.
    Le,
    /// Pop two values, push 1 if a == b, else 0.
    Eq,
    /// Pop two values, push 1 if a != b, else 0.
    Ne,
    /// Pop two values, push 1 if a > b, else 0.
    Gt,
    /// Pop two values, push 1 if a >= b, else 0.
    Ge,
    /// Pop the top of the stack; jump to the target if it is zero.
    JumpIfFalse(u16),
    /// Unconditional jump to the target.
    Jump(u16),
    /// Pop the top of the stack and print it.
    Print,
    /// Stop execution.
    Halt,
}

/// Build the "sum 0..n" program:
///
/// ```text
/// sum = 0; i = 0;
/// while (i < n) { sum = sum + i; i = i + 1; }
/// result = sum
/// ```
///
/// Variables: sum = var 0 (a), i = var 1 (b).
pub fn sum_program(n: i64) -> Vec<ByteCode> {
    use ByteCode::*;
    vec![
        /*  0 */ LoadConst(0),    // sum = 0
        /*  1 */ StoreVar(0),
        /*  2 */ LoadConst(0),    // i = 0
        /*  3 */ StoreVar(1),
        /*  4 */ LoadVar(1),      // loop: push i
        /*  5 */ LoadConst(n),    //        push n
        /*  6 */ Lt,              //        i < n?
        /*  7 */ JumpIfFalse(17), //        if false -> end
        /*  8 */ LoadVar(0),      // sum = sum + i
        /*  9 */ LoadVar(1),
        /* 10 */ Add,
        /* 11 */ StoreVar(0),
        /* 12 */ LoadVar(1),      // i = i + 1
        /* 13 */ LoadConst(1),
        /* 14 */ Add,
        /* 15 */ StoreVar(1),
        /* 16 */ Jump(4),         // goto loop
        /* 17 */ LoadVar(0),      // push sum as result
        /* 18 */ Halt,
    ]
}

/// Build a simple program that computes factorial(n) iteratively.
///
/// ```text
/// result = 1; i = 1;
/// while (i <= n) { result = result * i; i = i + 1; }
/// ```
///
/// Variables: result = var 0 (a), i = var 1 (b).
pub fn factorial_program(n: i64) -> Vec<ByteCode> {
    use ByteCode::*;
    vec![
        /*  0 */ LoadConst(1),    // result = 1
        /*  1 */ StoreVar(0),
        /*  2 */ LoadConst(1),    // i = 1
        /*  3 */ StoreVar(1),
        /*  4 */ LoadVar(1),      // loop: push i
        /*  5 */ LoadConst(n),    //        push n
        /*  6 */ Le,              //        i <= n?
        /*  7 */ JumpIfFalse(17), //        if false -> end
        /*  8 */ LoadVar(0),      // result = result * i
        /*  9 */ LoadVar(1),
        /* 10 */ Mul,
        /* 11 */ StoreVar(0),
        /* 12 */ LoadVar(1),      // i = i + 1
        /* 13 */ LoadConst(1),
        /* 14 */ Add,
        /* 15 */ StoreVar(1),
        /* 16 */ Jump(4),         // goto loop
        /* 17 */ LoadVar(0),      // push result
        /* 18 */ Halt,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_program_structure() {
        let prog = sum_program(100);
        assert!(!prog.is_empty());
        assert_eq!(prog.last(), Some(&ByteCode::Halt));
    }

    #[test]
    fn factorial_program_structure() {
        let prog = factorial_program(10);
        assert!(!prog.is_empty());
        assert_eq!(prog.last(), Some(&ByteCode::Halt));
    }
}
