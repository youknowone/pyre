/// Bytecode definitions for TL (Toy Language) — stack-based interpreter.
///
/// Port of rpython/jit/tl/tl.py + rpython/jit/tl/tlopcode.py.
/// Supports stack operations, arithmetic, comparisons, branches, and PUSHARG.

#[derive(Debug, Clone, PartialEq)]
pub enum ByteCode {
    Nop,
    /// Push an immediate value onto the stack.
    Push(i64),
    /// Pop and discard top of stack.
    Pop,
    /// Swap the top two stack elements.
    Swap,
    /// Roll: rotate stack elements. Positive = rotate up, negative = rotate down.
    Roll(i8),
    /// Pick: duplicate the element at depth `i` from the top.
    Pick(u8),
    /// Put: pop top and store it at depth `i` from the new top.
    Put(u8),
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    // Comparisons (push 1 or 0)
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Conditional branch: pop top; if nonzero, jump by relative offset from next PC.
    BrCond(i16),
    /// Conditional branch with offset from stack: pop [cond, offset]; if cond, jump by offset.
    BrCondStk,
    /// Call: push return address, jump forward by offset.
    Call(i16),
    /// Return: stop execution of current function/program.
    Return,
    /// Push the input argument onto the stack.
    PushArg,
    /// Unconditional branch by relative offset from next PC.
    Br(i16),
}

/// Build a "sum 0..n" program in TL bytecode.
///
/// Uses PUSHARG for n. BrCond/Br offsets are relative to the *next* PC.
pub fn sum_program(n: i64) -> (Vec<ByteCode>, i64) {
    use ByteCode::*;
    // Stack at loop header: [sum, i]
    let prog = vec![
        /*  0 */ Push(0),       // sum = 0            stack: [0]
        /*  1 */ Push(0),       // i = 0              stack: [sum, i]
        // loop header at pc=2:                       stack: [sum, i]
        /*  2 */ Pick(0),       // dup i              stack: [sum, i, i]
        /*  3 */ PushArg,       // push n             stack: [sum, i, i, n]
        /*  4 */ Lt,            // i < n?             stack: [sum, i, cond]
        /*  5 */ BrCond(1),     // if cond: goto 7    stack: [sum, i]
        /*  6 */ Br(7),         // goto 14 (exit)     stack: [sum, i]
        // body (pc=7):                               stack: [sum, i]
        /*  7 */ Swap,          //                    stack: [i, sum]
        /*  8 */ Pick(1),       //                    stack: [i, sum, i]
        /*  9 */ Add,           //                    stack: [i, sum+i]
        /* 10 */ Swap,          //                    stack: [sum', i]
        /* 11 */ Push(1),       //                    stack: [sum', i, 1]
        /* 12 */ Add,           //                    stack: [sum', i+1]
        /* 13 */ Br(-12),       // goto 2 (14-12=2)   stack: [sum', i']
        // exit (pc=14):                              stack: [sum, i]
        /* 14 */ Pop,           // discard i          stack: [sum]
        /* 15 */ Return,        //                    returns sum
    ];
    (prog, n)
}

/// Build a "factorial(n)" program.
pub fn factorial_program(n: i64) -> (Vec<ByteCode>, i64) {
    use ByteCode::*;
    let prog = vec![
        /*  0 */ Push(1),       // result = 1         stack: [1]
        /*  1 */ Push(1),       // i = 1              stack: [result, i]
        // loop header at pc=2:                       stack: [result, i]
        /*  2 */ Pick(0),       // dup i              stack: [result, i, i]
        /*  3 */ PushArg,       // push n             stack: [result, i, i, n]
        /*  4 */ Le,            // i <= n?            stack: [result, i, cond]
        /*  5 */ BrCond(1),     // if cond: goto 7    stack: [result, i]
        /*  6 */ Br(7),         // goto 14 (exit)     stack: [result, i]
        // body (pc=7):                               stack: [result, i]
        /*  7 */ Swap,          //                    stack: [i, result]
        /*  8 */ Pick(1),       //                    stack: [i, result, i]
        /*  9 */ Mul,           //                    stack: [i, result*i]
        /* 10 */ Swap,          //                    stack: [result', i]
        /* 11 */ Push(1),       //                    stack: [result', i, 1]
        /* 12 */ Add,           //                    stack: [result', i+1]
        /* 13 */ Br(-12),       // goto 2             stack: [result', i']
        // exit (pc=14):                              stack: [result, i]
        /* 14 */ Pop,           // discard i          stack: [result]
        /* 15 */ Return,
    ];
    (prog, n)
}

/// Build a "square(a)" program — compute a*a by repeated addition.
pub fn square_program(a: i64) -> (Vec<ByteCode>, i64) {
    use ByteCode::*;
    // counter = a, res = 0. loop: res += a; counter--; if counter!=0 goto loop
    let prog = vec![
        /*  0 */ Push(0),       // res = 0            stack: [0]
        /*  1 */ PushArg,       // counter = a        stack: [res, counter]
        // loop header at pc=2:                       stack: [res, counter]
        /*  2 */ Swap,          //                    stack: [counter, res]
        /*  3 */ PushArg,       // push a             stack: [counter, res, a]
        /*  4 */ Add,           //                    stack: [counter, res+a]
        /*  5 */ Swap,          //                    stack: [res', counter]
        /*  6 */ Push(1),       //                    stack: [res', counter, 1]
        /*  7 */ Sub,           //                    stack: [res', counter-1]
        /*  8 */ Pick(0),       // dup counter        stack: [res', counter', counter']
        /*  9 */ BrCond(-8),    // if counter'!=0: goto 2 (10-8=2). Pops counter'.
        //                                            stack: [res', counter']
        /* 10 */ Pop,           // discard counter    stack: [res]
        /* 11 */ Return,
    ];
    (prog, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_program_structure() {
        let (prog, _) = sum_program(100);
        assert!(!prog.is_empty());
        assert_eq!(prog.last(), Some(&ByteCode::Return));
    }

    #[test]
    fn factorial_program_structure() {
        let (prog, _) = factorial_program(10);
        assert!(!prog.is_empty());
        assert_eq!(prog.last(), Some(&ByteCode::Return));
    }

    #[test]
    fn square_program_structure() {
        let (prog, _) = square_program(5);
        assert!(!prog.is_empty());
        assert_eq!(prog.last(), Some(&ByteCode::Return));
    }
}
