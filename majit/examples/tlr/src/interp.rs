/// The bytecode interpreter for TLR (Toy Language - Register).
///
/// Port of rpython/jit/tl/tlr.py:interpret().
/// Register-based with a single accumulator.
use crate::bytecode::ByteCode;

pub struct TlrInterp {
    /// Accumulator register.
    a: i64,
    /// General-purpose register file.
    regs: Vec<i64>,
    /// Program counter.
    pc: usize,
}

impl TlrInterp {
    pub fn new() -> Self {
        Self {
            a: 0,
            regs: Vec::new(),
            pc: 0,
        }
    }

    pub fn reset(&mut self) {
        self.a = 0;
        self.regs.clear();
        self.pc = 0;
    }

    /// Execute the bytecode program with initial accumulator value.
    pub fn run(&mut self, bytecode: &[ByteCode], initial_a: i64) -> i64 {
        self.a = initial_a;
        self.pc = 0;
        self.regs.clear();

        loop {
            let instr = &bytecode[self.pc];
            self.pc += 1;
            match instr {
                ByteCode::MovAR(n) => {
                    self.regs[*n as usize] = self.a;
                }
                ByteCode::MovRA(n) => {
                    self.a = self.regs[*n as usize];
                }
                ByteCode::JumpIfA(target) => {
                    if self.a != 0 {
                        self.pc = *target as usize;
                    }
                }
                ByteCode::SetA(val) => {
                    self.a = *val;
                }
                ByteCode::AddRToA(n) => {
                    self.a += self.regs[*n as usize];
                }
                ByteCode::ReturnA => {
                    return self.a;
                }
                ByteCode::Allocate(n) => {
                    self.regs = vec![0; *n as usize];
                }
                ByteCode::NegA => {
                    self.a = -self.a;
                }
            }
        }
    }

    // -- Public accessors for JIT integration --

    pub fn pc(&self) -> usize {
        self.pc
    }

    pub fn set_pc(&mut self, pc: usize) {
        self.pc = pc;
    }

    pub fn accumulator(&self) -> i64 {
        self.a
    }

    pub fn set_accumulator(&mut self, a: i64) {
        self.a = a;
    }

    pub fn get_reg(&self, n: u8) -> i64 {
        self.regs[n as usize]
    }

    pub fn set_reg(&mut self, n: u8, val: i64) {
        self.regs[n as usize] = val;
    }

    pub fn num_regs(&self) -> usize {
        self.regs.len()
    }

    pub fn ensure_regs(&mut self, n: usize) {
        if self.regs.len() < n {
            self.regs.resize(n, 0);
        }
    }
}

impl Default for TlrInterp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::*;

    #[test]
    fn test_set_and_return() {
        let prog = vec![ByteCode::SetA(42), ByteCode::ReturnA];
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, 0), 42);
    }

    #[test]
    fn test_accumulator_passthrough() {
        let prog = vec![ByteCode::ReturnA];
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, 99), 99);
    }

    #[test]
    fn test_allocate_and_mov() {
        let prog = vec![
            ByteCode::Allocate(3),
            ByteCode::SetA(7),
            ByteCode::MovAR(1),
            ByteCode::SetA(0),
            ByteCode::AddRToA(1),
            ByteCode::ReturnA,
        ];
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, 0), 7);
    }

    #[test]
    fn test_neg() {
        let prog = vec![ByteCode::SetA(5), ByteCode::NegA, ByteCode::ReturnA];
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, 0), -5);
    }

    #[test]
    fn test_square_5() {
        let (prog, a) = square_program(5);
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, a), 25);
    }

    #[test]
    fn test_square_100() {
        let (prog, a) = square_program(100);
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, a), 10_000);
    }

    #[test]
    fn test_sum_10() {
        let (prog, a) = sum_program(10);
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, a), 45);
    }

    #[test]
    fn test_sum_100() {
        let (prog, a) = sum_program(100);
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, a), 4950);
    }

    #[test]
    fn test_sum_1000() {
        let (prog, a) = sum_program(1000);
        let mut interp = TlrInterp::new();
        assert_eq!(interp.run(&prog, a), 499_500);
    }
}
