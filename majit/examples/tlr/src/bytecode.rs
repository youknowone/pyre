/// Bytecode definitions for the TLR (Toy Language - Register) interpreter.
///
/// Port of rpython/jit/tl/tlr.py — a register-based interpreter with
/// an accumulator and dynamically-allocated register file.

#[derive(Debug, Clone, PartialEq)]
pub enum ByteCode {
    /// Move accumulator value into register N.
    MovAR(u8),
    /// Move register N value into accumulator.
    MovRA(u8),
    /// If accumulator is nonzero, jump to target PC.
    JumpIfA(u16),
    /// Set accumulator to an immediate value.
    SetA(i64),
    /// Add register N to accumulator.
    AddRToA(u8),
    /// Return the accumulator value (halt).
    ReturnA,
    /// Allocate N registers (initialized to 0).
    Allocate(u8),
    /// Negate the accumulator.
    NegA,
}

/// Build the "square" program from tlr.py: compute a*a by repeated addition.
///
/// ```text
/// regs = [0] * 3
/// regs[0] = a       // i = a (loop counter)
/// regs[1] = a       // copy of a
/// regs[2] = 0       // res = 0
/// loop:
///   i = i - 1        (SET_A 1, NEG_A, ADD_R_TO_A 0, MOV_A_R 0)
///   res = res + a    (MOV_R_A 2, ADD_R_TO_A 1, MOV_A_R 2)
///   if i != 0: goto loop  (MOV_R_A 0, JUMP_IF_A loop)
/// return res
/// ```
pub fn square_program(a: i64) -> (Vec<ByteCode>, i64) {
    use ByteCode::*;
    let prog = vec![
        /*  0 */ Allocate(3),
        /*  1 */ SetA(a),
        /*  2 */ MovAR(0),    // i = a
        /*  3 */ MovAR(1),    // regs[1] = a (copy)
        /*  4 */ SetA(0),
        /*  5 */ MovAR(2),    // res = 0
        // loop header at 6:
        /*  6 */ SetA(1),
        /*  7 */ NegA,
        /*  8 */ AddRToA(0),
        /*  9 */ MovAR(0),    // i--
        /* 10 */ MovRA(2),
        /* 11 */ AddRToA(1),
        /* 12 */ MovAR(2),    // res += a
        /* 13 */ MovRA(0),
        /* 14 */ JumpIfA(6),  // if i != 0: goto loop
        /* 15 */ MovRA(2),
        /* 16 */ ReturnA,
    ];
    (prog, a)
}

/// Build a "sum 0..n" program: compute 0 + 1 + 2 + ... + (n-1).
///
/// ```text
/// regs = [0] * 3
/// regs[0] = n       // limit
/// regs[1] = 0       // i (counter)
/// regs[2] = 0       // sum
/// loop:
///   sum = sum + i    (MOV_R_A 2, ADD_R_TO_A 1, MOV_A_R 2)
///   i = i + 1        (MOV_R_A 1, ADD immediate 1 via SET_A trick)
///   if i < n: goto loop
/// return sum
/// ```
///
/// Since TLR has no comparison ops, we implement "loop n times" by
/// counting down a separate counter from n to 0, while incrementing i.
pub fn sum_program(n: i64) -> (Vec<ByteCode>, i64) {
    use ByteCode::*;
    // Strategy: counter = n, i = 0, sum = 0
    // loop: sum += i; i += 1; counter -= 1; if counter != 0 goto loop
    let prog = vec![
        /*  0 */ Allocate(4),
        /*  1 */ SetA(n),
        /*  2 */ MovAR(0),    // regs[0] = counter = n
        /*  3 */ SetA(0),
        /*  4 */ MovAR(1),    // regs[1] = i = 0
        /*  5 */ MovAR(2),    // regs[2] = sum = 0
        /*  6 */ SetA(1),
        /*  7 */ MovAR(3),    // regs[3] = constant 1
        // loop header at 8:
        /*  8 */ MovRA(2),
        /*  9 */ AddRToA(1),
        /* 10 */ MovAR(2),    // sum += i
        /* 11 */ MovRA(1),
        /* 12 */ AddRToA(3),
        /* 13 */ MovAR(1),    // i += 1
        /* 14 */ SetA(-1),
        /* 15 */ AddRToA(0),
        /* 16 */ MovAR(0),    // counter -= 1
        /* 17 */ JumpIfA(8),  // if counter != 0: goto loop
        /* 18 */ MovRA(2),
        /* 19 */ ReturnA,     // return sum
    ];
    (prog, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_program_structure() {
        let (prog, _) = square_program(5);
        assert!(!prog.is_empty());
        assert_eq!(prog.last(), Some(&ByteCode::ReturnA));
    }

    #[test]
    fn sum_program_structure() {
        let (prog, _) = sum_program(100);
        assert!(!prog.is_empty());
        assert_eq!(prog.last(), Some(&ByteCode::ReturnA));
    }
}
