/// The bytecode interpreter for the calc language.
use crate::bytecode::ByteCode;

/// Debug logging — @not_in_trace: disappears from the final assembler.
#[majit_macros::not_in_trace]
fn trace_halt(result: i64) {
    if cfg!(debug_assertions) {
        eprintln!("[calc] halt with result={result}");
    }
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
                ByteCode::Lt => self.cmpop(|a, b| a < b),
                ByteCode::Le => self.cmpop(|a, b| a <= b),
                ByteCode::Eq => self.cmpop(|a, b| a == b),
                ByteCode::Ne => self.cmpop(|a, b| a != b),
                ByteCode::Gt => self.cmpop(|a, b| a > b),
                ByteCode::Ge => self.cmpop(|a, b| a >= b),
                ByteCode::JumpIfFalse(target) => {
                    let val = self.stack.pop().expect("stack underflow on JumpIfFalse");
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
                    let result = self.stack.last().copied().unwrap_or(0);
                    trace_halt(result);
                    return result;
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
}

impl Default for CalcInterp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::ByteCode;

    #[test]
    fn test_simple_add() {
        let prog = vec![
            ByteCode::LoadConst(3),
            ByteCode::LoadConst(4),
            ByteCode::Add,
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 7);
    }

    #[test]
    fn test_comparison() {
        let prog = vec![
            ByteCode::LoadConst(5),
            ByteCode::LoadConst(3),
            ByteCode::Lt,
            ByteCode::Halt,
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 0); // 5 < 3 == false
    }

    #[test]
    fn test_loop_sum() {
        // sum = 0; i = 10; while i > 0: sum += i; i -= 1
        // Exercises: Jump, JumpIfFalse, StoreVar, LoadVar, Sub, Add, Gt
        let prog = vec![
            ByteCode::LoadConst(0),  //  0: push 0
            ByteCode::StoreVar(0),   //  1: sum = 0     (var a)
            ByteCode::LoadConst(10), //  2: push 10
            ByteCode::StoreVar(1),   //  3: i = 10      (var b)
            // loop header (pc=4)
            ByteCode::LoadVar(1),      //  4: push i
            ByteCode::LoadConst(0),    //  5: push 0
            ByteCode::Gt,              //  6: i > 0?
            ByteCode::JumpIfFalse(15), //  7: if false → pc 15 (load result)
            ByteCode::LoadVar(0),      //  8: push sum
            ByteCode::LoadVar(1),      //  9: push i
            ByteCode::Add,             // 10: sum + i
            ByteCode::StoreVar(0),     // 11: sum = sum + i
            ByteCode::LoadVar(1),      // 12: push i
            ByteCode::LoadConst(1),    // 13: push 1
            ByteCode::Sub,             // 14: i - 1
            ByteCode::StoreVar(1),     // 15: i = i - 1  — wait, JumpIfFalse(15) lands here
                                       // Fix: JumpIfFalse target must skip the loop body entirely.
                                       // Re-layout with correct targets:
        ];
        // Correct layout:
        let prog = vec![
            ByteCode::LoadConst(0),  //  0
            ByteCode::StoreVar(0),   //  1: sum = 0
            ByteCode::LoadConst(10), //  2
            ByteCode::StoreVar(1),   //  3: i = 10
            // loop header (pc=4)
            ByteCode::LoadVar(1),      //  4: push i
            ByteCode::LoadConst(0),    //  5: push 0
            ByteCode::Gt,              //  6: i > 0?
            ByteCode::JumpIfFalse(16), //  7: if false → pc 16 (result)
            ByteCode::LoadVar(0),      //  8: push sum
            ByteCode::LoadVar(1),      //  9: push i
            ByteCode::Add,             // 10: sum + i
            ByteCode::StoreVar(0),     // 11: sum = sum + i
            ByteCode::LoadVar(1),      // 12: push i
            ByteCode::LoadConst(1),    // 13: push 1
            ByteCode::Sub,             // 14: i - 1
            ByteCode::StoreVar(1),     // 15: i = i - 1
            ByteCode::Jump(4),         // 16 — but this IS pc 16, conflict
        ];
        // The issue: Jump(4) at pc=16 means JumpIfFalse(16) lands on Jump(4).
        // That's wrong. Fix: JumpIfFalse should go to pc=17.
        let prog = vec![
            ByteCode::LoadConst(0),    //  0
            ByteCode::StoreVar(0),     //  1: sum = 0
            ByteCode::LoadConst(10),   //  2
            ByteCode::StoreVar(1),     //  3: i = 10
            ByteCode::LoadVar(1),      //  4: push i
            ByteCode::LoadConst(0),    //  5: push 0
            ByteCode::Gt,              //  6: i > 0?
            ByteCode::JumpIfFalse(17), //  7: if false → pc 17 (LoadVar sum result)
            ByteCode::LoadVar(0),      //  8: push sum
            ByteCode::LoadVar(1),      //  9: push i
            ByteCode::Add,             // 10: sum + i
            ByteCode::StoreVar(0),     // 11: sum = sum + i
            ByteCode::LoadVar(1),      // 12: push i
            ByteCode::LoadConst(1),    // 13: push 1
            ByteCode::Sub,             // 14: i - 1
            ByteCode::StoreVar(1),     // 15: i = i - 1
            ByteCode::Jump(4),         // 16: back to loop header
            ByteCode::LoadVar(0),      // 17: push sum (result)
            ByteCode::Halt,            // 18
        ];
        let mut interp = CalcInterp::new();
        assert_eq!(interp.run(&prog), 55); // 1+2+...+10 = 55
    }
}
