/// Stack-based bytecode interpreter for TL (Toy Language).
///
/// Port of rpython/jit/tl/tl.py:interp().
use crate::bytecode::ByteCode;

pub struct TlInterp {
    stack: Vec<i64>,
    pc: usize,
    inputarg: i64,
}

impl TlInterp {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(256),
            pc: 0,
            inputarg: 0,
        }
    }

    pub fn reset(&mut self) {
        self.stack.clear();
        self.pc = 0;
        self.inputarg = 0;
    }

    /// Execute bytecode with the given input argument.
    pub fn run(&mut self, bytecode: &[ByteCode], inputarg: i64) -> i64 {
        self.stack.clear();
        self.pc = 0;
        self.inputarg = inputarg;

        loop {
            if self.pc >= bytecode.len() {
                break;
            }
            let instr = bytecode[self.pc].clone();
            self.pc += 1;

            match instr {
                ByteCode::Nop => {}
                ByteCode::Push(v) => {
                    self.stack.push(v);
                }
                ByteCode::Pop => {
                    self.stack.pop().expect("stack underflow on Pop");
                }
                ByteCode::Swap => {
                    let a = self.stack.pop().expect("stack underflow");
                    let b = self.stack.pop().expect("stack underflow");
                    self.stack.push(a);
                    self.stack.push(b);
                }
                ByteCode::Roll(r) => {
                    self.roll(r as i32);
                }
                ByteCode::Pick(i) => {
                    let n = self.stack.len() - (i as usize) - 1;
                    let val = self.stack[n];
                    self.stack.push(val);
                }
                ByteCode::Put(i) => {
                    let elem = self.stack.pop().expect("stack underflow");
                    let n = self.stack.len() - (i as usize) - 1;
                    self.stack[n] = elem;
                }
                ByteCode::Add => self.binop(|a, b| a + b),
                ByteCode::Sub => self.binop(|a, b| a - b),
                ByteCode::Mul => self.binop(|a, b| a * b),
                ByteCode::Div => self.binop(|a, b| a / b),
                ByteCode::Eq => self.cmpop(|a, b| a == b),
                ByteCode::Ne => self.cmpop(|a, b| a != b),
                ByteCode::Lt => self.cmpop(|a, b| a < b),
                ByteCode::Le => self.cmpop(|a, b| a <= b),
                ByteCode::Gt => self.cmpop(|a, b| a > b),
                ByteCode::Ge => self.cmpop(|a, b| a >= b),
                ByteCode::BrCond(offset) => {
                    let cond = self.stack.pop().expect("stack underflow");
                    if cond != 0 {
                        self.pc = (self.pc as i64 + offset as i64) as usize;
                    }
                }
                ByteCode::BrCondStk => {
                    let offset = self.stack.pop().expect("stack underflow");
                    let cond = self.stack.pop().expect("stack underflow");
                    if cond != 0 {
                        self.pc = (self.pc as i64 + offset) as usize;
                    }
                }
                ByteCode::Call(offset) => {
                    // Push return address and jump.
                    self.stack.push(self.pc as i64);
                    self.pc = (self.pc as i64 + offset as i64) as usize;
                }
                ByteCode::Return => {
                    break;
                }
                ByteCode::PushArg => {
                    self.stack.push(self.inputarg);
                }
                ByteCode::Br(offset) => {
                    self.pc = (self.pc as i64 + offset as i64) as usize;
                }
            }
        }

        self.stack.pop().unwrap_or(0)
    }

    fn roll(&mut self, r: i32) {
        let len = self.stack.len();
        if r < -1 {
            let i = (len as i32 + r) as usize;
            let n = len - 1;
            let elem = self.stack[n];
            let mut j = n;
            while j > i {
                self.stack[j] = self.stack[j - 1];
                j -= 1;
            }
            self.stack[i] = elem;
        } else if r > 1 {
            let i = len - r as usize;
            let elem = self.stack[i];
            for j in i..len - 1 {
                self.stack[j] = self.stack[j + 1];
            }
            self.stack[len - 1] = elem;
        }
    }

    // -- Public accessors for JIT integration --

    pub fn pc(&self) -> usize {
        self.pc
    }

    pub fn set_pc(&mut self, pc: usize) {
        self.pc = pc;
    }

    pub fn stack(&self) -> &[i64] {
        &self.stack
    }

    pub fn stack_mut(&mut self) -> &mut Vec<i64> {
        &mut self.stack
    }

    pub fn inputarg(&self) -> i64 {
        self.inputarg
    }

    pub fn push(&mut self, val: i64) {
        self.stack.push(val);
    }

    pub fn pop(&mut self) -> i64 {
        self.stack.pop().expect("stack underflow")
    }

    pub fn top(&self) -> i64 {
        *self.stack.last().expect("stack empty")
    }

    #[inline(always)]
    fn binop(&mut self, op: impl FnOnce(i64, i64) -> i64) {
        let b = self.stack.pop().expect("stack underflow");
        let a = self.stack.pop().expect("stack underflow");
        self.stack.push(op(a, b));
    }

    #[inline(always)]
    fn cmpop(&mut self, op: impl FnOnce(i64, i64) -> bool) {
        let b = self.stack.pop().expect("stack underflow");
        let a = self.stack.pop().expect("stack underflow");
        self.stack.push(if op(a, b) { 1 } else { 0 });
    }
}

impl Default for TlInterp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::*;

    #[test]
    fn test_push_return() {
        let prog = vec![ByteCode::Push(42), ByteCode::Return];
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, 0), 42);
    }

    #[test]
    fn test_pusharg() {
        let prog = vec![ByteCode::PushArg, ByteCode::Return];
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, 99), 99);
    }

    #[test]
    fn test_add() {
        let prog = vec![
            ByteCode::Push(3),
            ByteCode::Push(7),
            ByteCode::Add,
            ByteCode::Return,
        ];
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, 0), 10);
    }

    #[test]
    fn test_sub() {
        let prog = vec![
            ByteCode::Push(10),
            ByteCode::Push(3),
            ByteCode::Sub,
            ByteCode::Return,
        ];
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, 0), 7);
    }

    #[test]
    fn test_swap() {
        let prog = vec![
            ByteCode::Push(1),
            ByteCode::Push(2),
            ByteCode::Swap,
            ByteCode::Return,
        ];
        let mut interp = TlInterp::new();
        // After swap: [2, 1], return pops 1
        assert_eq!(interp.run(&prog, 0), 1);
    }

    #[test]
    fn test_pick() {
        let prog = vec![
            ByteCode::Push(10),
            ByteCode::Push(20),
            ByteCode::Push(30),
            ByteCode::Pick(2), // dup element at depth 2 (= 10)
            ByteCode::Return,
        ];
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, 0), 10);
    }

    #[test]
    fn test_br_cond() {
        // if 1: skip Push(99), return 42
        let prog = vec![
            ByteCode::Push(42),
            ByteCode::Push(1),
            ByteCode::BrCond(1), // goto 4
            ByteCode::Push(99),
            ByteCode::Return,
        ];
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, 0), 42);
    }

    #[test]
    fn test_br_cond_false() {
        // if 0: don't skip Push(99), return 99
        let prog = vec![
            ByteCode::Push(42),
            ByteCode::Push(0),
            ByteCode::BrCond(1), // don't jump
            ByteCode::Push(99),
            ByteCode::Return,
        ];
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, 0), 99);
    }

    #[test]
    fn test_sum_10() {
        let (prog, arg) = sum_program(10);
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, arg), 45);
    }

    #[test]
    fn test_sum_100() {
        let (prog, arg) = sum_program(100);
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, arg), 4950);
    }

    #[test]
    fn test_sum_1000() {
        let (prog, arg) = sum_program(1000);
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, arg), 499_500);
    }

    #[test]
    fn test_factorial_5() {
        let (prog, arg) = factorial_program(5);
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, arg), 120);
    }

    #[test]
    fn test_factorial_10() {
        let (prog, arg) = factorial_program(10);
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, arg), 3_628_800);
    }

    #[test]
    fn test_factorial_20() {
        let (prog, arg) = factorial_program(20);
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, arg), 2_432_902_008_176_640_000);
    }

    #[test]
    fn test_square_5() {
        let (prog, arg) = square_program(5);
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, arg), 25);
    }

    #[test]
    fn test_square_100() {
        let (prog, arg) = square_program(100);
        let mut interp = TlInterp::new();
        assert_eq!(interp.run(&prog, arg), 10_000);
    }
}
