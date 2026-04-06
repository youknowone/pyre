/// The bytecode interpreter for the calc language.

use crate::bytecode::ByteCode;
use majit_metainterp::{conditional_call, record_known_result};

/// Pure comparison — @elidable_promote.
/// rlib/jit.py:180 — promotes both arguments, then calls the elidable body.
/// The JIT emits guard_value on both operands and constant-folds the result.
#[majit_macros::elidable_promote]
fn calc_lt(a: i64, b: i64) -> i64 { if a < b { 1 } else { 0 } }
#[majit_macros::elidable_promote]
fn calc_le(a: i64, b: i64) -> i64 { if a <= b { 1 } else { 0 } }
#[majit_macros::elidable_promote]
fn calc_eq(a: i64, b: i64) -> i64 { if a == b { 1 } else { 0 } }
#[majit_macros::elidable_promote]
fn calc_ne(a: i64, b: i64) -> i64 { if a != b { 1 } else { 0 } }
#[majit_macros::elidable_promote]
fn calc_gt(a: i64, b: i64) -> i64 { if a > b { 1 } else { 0 } }
#[majit_macros::elidable_promote]
fn calc_ge(a: i64, b: i64) -> i64 { if a >= b { 1 } else { 0 } }

/// Debug logging — @not_in_trace.
/// rlib/jit.py:260 — disappears from JIT traces, only called in interpreter mode.
#[majit_macros::not_in_trace]
fn debug_trace_opcode(_pc: usize, _name: &str) {
    // In debug builds, this could log the executing opcode.
    // The JIT skips this call entirely in compiled traces.
}

pub struct CalcInterp {
    stack: Vec<i64>,
    vars: [i64; 26],
    pc: usize,
}

impl CalcInterp {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(64),
            vars: [0; 26],
            pc: 0,
        }
    }

    /// Reset interpreter state so it can run a new program.
    pub fn reset(&mut self) {
        self.stack.clear();
        self.vars = [0; 26];
        self.pc = 0;
    }

    /// Execute the bytecode program and return the final result.
    ///
    /// The result is the top-of-stack value when `Halt` is reached,
    /// or 0 if the stack is empty.
    pub fn run(&mut self, bytecode: &[ByteCode]) -> i64 {
        loop {
            debug_trace_opcode(self.pc, "calc_step");
            let instr = &bytecode[self.pc];
            self.pc += 1;
            match instr {
                ByteCode::LoadConst(v) => {
                    self.stack.push(*v);
                }
                ByteCode::LoadVar(idx) => {
                    self.stack.push(self.vars[*idx as usize]);
                }
                ByteCode::StoreVar(idx) => {
                    let val = self.stack.pop().expect("stack underflow on StoreVar");
                    self.vars[*idx as usize] = val;
                }
                ByteCode::Add => self.binop(|a, b| a + b),
                ByteCode::Sub => self.binop(|a, b| a - b),
                ByteCode::Mul => self.binop(|a, b| a * b),
                ByteCode::Div => self.binop(|a, b| a / b),
                ByteCode::Mod => self.binop(|a, b| a % b),
                ByteCode::Lt => self.cmpop_elidable(calc_lt),
                ByteCode::Le => self.cmpop_elidable(calc_le),
                ByteCode::Eq => self.cmpop_elidable(calc_eq),
                ByteCode::Ne => self.cmpop_elidable(calc_ne),
                ByteCode::Gt => self.cmpop_elidable(calc_gt),
                ByteCode::Ge => self.cmpop_elidable(calc_ge),
                ByteCode::JumpIfFalse(target) => {
                    let val = self.stack.pop().expect("stack underflow on JumpIfFalse");
                    // rlib/jit.py:1301 — conditional_call: bridge-free code
                    conditional_call!(val != 0, debug_trace_opcode, self.pc, "branch_not_taken");
                    if val == 0 {
                        self.pc = *target as usize;
                    }
                }
                ByteCode::Jump(target) => {
                    self.pc = *target as usize;
                }
                ByteCode::Print => {
                    let val = self.stack.pop().expect("stack underflow on Print");
                    println!("{val}");
                }
                ByteCode::Halt => {
                    return self.stack.last().copied().unwrap_or(0);
                }
            }
        }
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

    /// Comparison using @elidable helper + record_known_result.
    #[inline(always)]
    fn cmpop_elidable(&mut self, op: fn(i64, i64) -> i64) {
        let b = self.stack.pop().expect("stack underflow");
        let a = self.stack.pop().expect("stack underflow");
        let result = op(a, b);
        record_known_result!(result, op, a, b);
        self.stack.push(result);
    }
}

impl Default for CalcInterp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::*;

    #[test]
    fn test_load_const_halt() {
        let prog = vec![ByteCode::LoadConst(42), ByteCode::Halt];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 42);
    }

    #[test]
    fn test_add() {
        let prog = vec![
            ByteCode::LoadConst(3),
            ByteCode::LoadConst(7),
            ByteCode::Add,
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 10);
    }

    #[test]
    fn test_sub() {
        let prog = vec![
            ByteCode::LoadConst(10),
            ByteCode::LoadConst(3),
            ByteCode::Sub,
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 7);
    }

    #[test]
    fn test_mul() {
        let prog = vec![
            ByteCode::LoadConst(6),
            ByteCode::LoadConst(7),
            ByteCode::Mul,
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 42);
    }

    #[test]
    fn test_div() {
        let prog = vec![
            ByteCode::LoadConst(20),
            ByteCode::LoadConst(4),
            ByteCode::Div,
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 5);
    }

    #[test]
    fn test_mod() {
        let prog = vec![
            ByteCode::LoadConst(17),
            ByteCode::LoadConst(5),
            ByteCode::Mod,
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 2);
    }

    #[test]
    fn test_variables() {
        let prog = vec![
            ByteCode::LoadConst(99),
            ByteCode::StoreVar(0),
            ByteCode::LoadVar(0),
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 99);
    }

    #[test]
    fn test_comparison_ops() {
        // 3 < 5 => 1
        let prog = vec![
            ByteCode::LoadConst(3),
            ByteCode::LoadConst(5),
            ByteCode::Lt,
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 1);

        // 5 < 3 => 0
        interp.reset();
        let prog = vec![
            ByteCode::LoadConst(5),
            ByteCode::LoadConst(3),
            ByteCode::Lt,
            ByteCode::Halt,
        ];
        assert_eq!(interp.run(&prog), 0);

        // 3 <= 3 => 1
        interp.reset();
        let prog = vec![
            ByteCode::LoadConst(3),
            ByteCode::LoadConst(3),
            ByteCode::Le,
            ByteCode::Halt,
        ];
        assert_eq!(interp.run(&prog), 1);

        // 3 == 3 => 1
        interp.reset();
        let prog = vec![
            ByteCode::LoadConst(3),
            ByteCode::LoadConst(3),
            ByteCode::Eq,
            ByteCode::Halt,
        ];
        assert_eq!(interp.run(&prog), 1);

        // 3 != 4 => 1
        interp.reset();
        let prog = vec![
            ByteCode::LoadConst(3),
            ByteCode::LoadConst(4),
            ByteCode::Ne,
            ByteCode::Halt,
        ];
        assert_eq!(interp.run(&prog), 1);

        // 5 > 3 => 1
        interp.reset();
        let prog = vec![
            ByteCode::LoadConst(5),
            ByteCode::LoadConst(3),
            ByteCode::Gt,
            ByteCode::Halt,
        ];
        assert_eq!(interp.run(&prog), 1);

        // 5 >= 5 => 1
        interp.reset();
        let prog = vec![
            ByteCode::LoadConst(5),
            ByteCode::LoadConst(5),
            ByteCode::Ge,
            ByteCode::Halt,
        ];
        assert_eq!(interp.run(&prog), 1);
    }

    #[test]
    fn test_jump() {
        // Jump over a LoadConst(999), should get 42
        let prog = vec![
            /*  0 */ ByteCode::LoadConst(42),
            /*  1 */ ByteCode::Jump(3),
            /*  2 */ ByteCode::LoadConst(999), // skipped
            /*  3 */ ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 42);
    }

    #[test]
    fn test_jump_if_false() {
        // Condition true: should NOT jump
        let prog = vec![
            ByteCode::LoadConst(1),
            ByteCode::JumpIfFalse(4),
            ByteCode::LoadConst(42),
            ByteCode::Halt,
            ByteCode::LoadConst(99),
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 42);

        // Condition false: should jump
        interp.reset();
        let prog = vec![
            ByteCode::LoadConst(0),
            ByteCode::JumpIfFalse(4),
            ByteCode::LoadConst(42),
            ByteCode::Halt,
            ByteCode::LoadConst(99),
            ByteCode::Halt,
        ];
        assert_eq!(interp.run(&prog), 99);
    }

    #[test]
    fn test_sum_program() {
        let prog = sum_program(100);
        let mut interp = CalcInterp::new();
        let result = interp.run(&prog);
        // sum(0..100) = 4950
        assert_eq!(result, 4950);
    }

    #[test]
    fn test_sum_program_zero() {
        let prog = sum_program(0);
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 0);
    }

    #[test]
    fn test_sum_program_one() {
        let prog = sum_program(1);
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 0);
    }

    #[test]
    fn test_sum_program_large() {
        let prog = sum_program(1_000_000);
        let mut interp = CalcInterp::new();
        let result = interp.run(&prog);
        // sum(0..1_000_000) = 999_999 * 1_000_000 / 2 = 499_999_500_000
        assert_eq!(result, 499_999_500_000);
    }

    #[test]
    fn test_factorial_program() {
        let prog = factorial_program(10);
        let mut interp = CalcInterp::new();
        // 10! = 3628800
        assert_eq!(interp.run(&prog), 3_628_800);
    }

    #[test]
    fn test_factorial_zero() {
        let prog = factorial_program(0);
        let mut interp = CalcInterp::new();
        // 0! = 1 (loop body never executes, result stays 1)
        assert_eq!(interp.run(&prog), 1);
    }

    #[test]
    fn test_halt_empty_stack() {
        let prog = vec![ByteCode::Halt];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 0);
    }

    #[test]
    fn test_reset() {
        let mut interp = CalcInterp::new();
        let prog = sum_program(10);
        assert_eq!(interp.run(&prog), 45);

        interp.reset();
        let prog = sum_program(5);
        assert_eq!(interp.run(&prog), 10);
    }
}
