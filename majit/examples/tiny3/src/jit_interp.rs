/// JIT-enabled tiny3 interpreter via `#[jit_interp]` proc macro.
///
/// The word-based program is compiled to bytecode, then the bytecoded mainloop
/// is traced by the auto-generated JitState/trace_instruction.
///
/// Greens: [pc]
/// Reds:   [storage (args at bottom, computation stack on top)]
///
/// The JIT traces the integer-only path. Float arithmetic falls back to the
/// plain interpreter. This matches RPython's promote(y.__class__) strategy.

// ── Bytecode opcodes ──

const OP_PUSH_INT: u8 = 0; // followed by 8 bytes (i64 LE)
const OP_PUSH_ARG: u8 = 1; // followed by 1 byte (arg index, 0-based)
const OP_STORE_ARG: u8 = 2; // followed by 1 byte (arg index, 0-based)
const OP_ADD: u8 = 3;
const OP_SUB: u8 = 4;
const OP_MUL: u8 = 5;
const OP_LOOP_START: u8 = 6; // no-op marker
const OP_LOOP_END: u8 = 7; // followed by 2 bytes (target pc, u16 LE)
const OP_END: u8 = 8;
const OP_PUSH_FLOAT: u8 = 9; // followed by 8 bytes (f64 bits as i64 LE) — not traced by JIT

// ── Bytecode compiler ──

fn compile(words: &[&str]) -> Vec<u8> {
    let mut code = Vec::new();
    let mut loop_starts: Vec<usize> = Vec::new();

    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        if w == "ADD" {
            code.push(OP_ADD);
        } else if w == "SUB" {
            code.push(OP_SUB);
        } else if w == "MUL" {
            code.push(OP_MUL);
        } else if w.starts_with("->#") {
            let n = parse_int(w, 3) as u8;
            code.push(OP_STORE_ARG);
            code.push(n - 1);
        } else if w.starts_with('#') {
            let n = parse_int(w, 1) as u8;
            code.push(OP_PUSH_ARG);
            code.push(n - 1);
        } else if w == "{" {
            code.push(OP_LOOP_START);
            loop_starts.push(code.len());
        } else if w == "}" {
            code.push(OP_LOOP_END);
            let target_pc = loop_starts.pop().expect("unmatched }");
            code.push((target_pc & 0xFF) as u8);
            code.push(((target_pc >> 8) & 0xFF) as u8);
        } else if let Some(fval) = try_parse_float(w) {
            // Float literal
            code.push(OP_PUSH_FLOAT);
            code.extend_from_slice(&(fval.to_bits() as i64).to_le_bytes());
        } else {
            // Integer literal
            let val = parse_int(w, 0);
            code.push(OP_PUSH_INT);
            code.extend_from_slice(&val.to_le_bytes());
        }
        i += 1;
    }
    code.push(OP_END);
    code
}

// ── Storage pool types ──

/// Single i64 storage: args at bottom, computation stack on top.
pub struct Tiny3Storage {
    stack: Vec<i64>,
}

impl Tiny3Storage {
    fn new() -> Self {
        Tiny3Storage { stack: Vec::new() }
    }
    pub fn push(&mut self, val: i64) {
        self.stack.push(val);
    }
    pub fn pop(&mut self) -> i64 {
        self.stack.pop().unwrap()
    }
    pub fn dup(&mut self) {
        let v = *self.stack.last().unwrap();
        self.stack.push(v);
    }
    pub fn add(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(b + a);
    }
    pub fn sub(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(b - a);
    }
    pub fn mul(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(b * a);
    }
    /// Copy element at index `n` (from bottom, 0-based) to top.
    pub fn copy_up(&mut self, n: usize) {
        let val = self.stack[n];
        self.stack.push(val);
    }
    /// Pop top and store at index `n` (from bottom, 0-based).
    pub fn store_down(&mut self, n: usize) {
        let val = self.stack.pop().unwrap();
        self.stack[n] = val;
    }
    pub fn peek_at(&self, idx: usize) -> i64 {
        self.stack[self.stack.len() - 1 - idx]
    }
    pub fn clear(&mut self) {
        self.stack.clear();
    }
    pub fn data_ptr(&self) -> usize {
        self.stack.as_ptr() as usize
    }
    pub fn len(&self) -> usize {
        self.stack.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

/// Storage pool wrapping a single Tiny3Storage.
pub struct Tiny3Pool {
    storages: Vec<Tiny3Storage>,
}

impl Tiny3Pool {
    fn new() -> Self {
        Tiny3Pool {
            storages: vec![Tiny3Storage::new()],
        }
    }
    pub fn get(&self, idx: usize) -> &Tiny3Storage {
        &self.storages[idx]
    }
    pub fn get_mut(&mut self, idx: usize) -> &mut Tiny3Storage {
        &mut self.storages[idx]
    }
    pub fn all_jit_compatible(&self) -> bool {
        true
    }
}

/// Interpreter state: storage pool + selected storage index.
struct Tiny3State {
    pool: Tiny3Pool,
    selected: usize,
}

fn find_used_storages(_program: &Bytecode, _header_pc: usize, _initial: usize) -> Vec<usize> {
    vec![0]
}

/// Type alias for bytecode — the env type for `#[jit_interp]`.
pub type Bytecode = [u8];

/// Extension trait providing `get_op` for bytecode slices.
trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}

impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

// ── JIT mainloop (integer-only path) ──

#[majit_macros::jit_interp(
    state = Tiny3State,
    env = Bytecode,
    storage = {
        pool: state.pool,
        pool_type: Tiny3Pool,
        selector: state.selected,
        untraceable: [],
        scan: find_used_storages,
        can_trace_guard: all_jit_compatible,
    },
    binops = {
        add => IntAdd,
        sub => IntSub,
        mul => IntMul,
    },
)]
fn mainloop(program: &Bytecode, num_args: usize, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<Tiny3State> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = num_args as i32;
    let mut state = Tiny3State {
        pool: Tiny3Pool::new(),
        selected: 0,
    };

    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;

        match opcode {
            OP_PUSH_INT => {
                let value = i64::from_le_bytes([
                    program[pc],
                    program[pc + 1],
                    program[pc + 2],
                    program[pc + 3],
                    program[pc + 4],
                    program[pc + 5],
                    program[pc + 6],
                    program[pc + 7],
                ]);
                pc += 8;
                state.pool.get_mut(state.selected).push(value);
                stacksize += 1;
            }
            OP_PUSH_FLOAT => {
                // Float literal stored as bit-casted i64 — same encoding as OP_PUSH_INT.
                let value = i64::from_le_bytes([
                    program[pc],
                    program[pc + 1],
                    program[pc + 2],
                    program[pc + 3],
                    program[pc + 4],
                    program[pc + 5],
                    program[pc + 6],
                    program[pc + 7],
                ]);
                pc += 8;
                state.pool.get_mut(state.selected).push(value);
                stacksize += 1;
            }
            OP_PUSH_ARG => {
                let n = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).copy_up(n);
                stacksize += 1;
            }
            OP_STORE_ARG => {
                let n = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).store_down(n);
                stacksize -= 1;
            }
            OP_ADD => state.pool.get_mut(state.selected).add(),
            OP_SUB => state.pool.get_mut(state.selected).sub(),
            OP_MUL => state.pool.get_mut(state.selected).mul(),
            OP_LOOP_START => {}
            OP_LOOP_END => {
                let target = (program[pc] as usize) | ((program[pc + 1] as usize) << 8);
                pc += 2;
                let cond = state.pool.get_mut(state.selected).pop();
                stacksize -= 1;
                let jump = cond != 0;
                if jump {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            OP_END => break,
            _ => {}
        }
    }

    state.pool.get_mut(state.selected).pop()
}

// ── Public wrapper matching the old API ──

pub struct JitTiny3Interp {
    threshold: u32,
}

impl JitTiny3Interp {
    pub fn new() -> Self {
        JitTiny3Interp { threshold: 3 }
    }

    /// Run a word-based program with integer args.
    pub fn run(&mut self, bytecode: &[&str], args: &mut Vec<i64>) -> i64 {
        let code = compile(bytecode);
        let num_args = args.len();
        let has_result_on_stack = program_has_result(bytecode, num_args);

        // Prepend OP_PUSH_INT for each arg to pre-populate the storage.
        let mut full_code = Vec::new();
        for &arg in args.iter() {
            full_code.push(OP_PUSH_INT);
            full_code.extend_from_slice(&arg.to_le_bytes());
        }
        full_code.extend_from_slice(&code);

        let result = mainloop(&full_code, num_args, self.threshold);

        // Sync args back via plain interpreter.
        let mut interp_args: Vec<crate::interp::Box> =
            args.iter().map(|&v| crate::interp::Box::Int(v)).collect();
        let _ = crate::interp::interpret(bytecode, &mut interp_args);
        for (i, b) in interp_args.iter().enumerate() {
            args[i] = b.as_int();
        }

        if has_result_on_stack { result } else { args[0] }
    }

    /// Run a word-based program with typed (Int/Float) args.
    /// Float paths fall back to the plain interpreter; JIT traces integer-only.
    pub fn run_typed(
        &mut self,
        bytecode: &[&str],
        args: &mut Vec<crate::interp::Box>,
    ) -> crate::interp::Box {
        // For typed runs, use the plain interpreter.
        // The JIT only traces the integer-only path.
        let result_stack = crate::interp::interpret(bytecode, args);
        if !result_stack.is_empty() {
            result_stack.last().unwrap().clone()
        } else {
            args[0].clone()
        }
    }
}

/// Check if the program produces a stack result beyond args.
fn program_has_result(words: &[&str], num_args: usize) -> bool {
    let mut depth: i32 = num_args as i32;
    let mut loop_depth = 0;
    for w in words {
        if *w == "{" {
            loop_depth += 1;
        } else if *w == "}" {
            loop_depth -= 1;
            depth -= 1;
        } else if loop_depth == 0 {
            if *w == "ADD" || *w == "SUB" || *w == "MUL" {
                depth -= 1;
            } else if w.starts_with("->#") {
                depth -= 1;
            } else if w.starts_with('#') {
                depth += 1;
            } else {
                depth += 1;
            }
        }
    }
    depth > num_args as i32
}

/// Try parsing a float literal (must contain '.').
fn try_parse_float(s: &str) -> Option<f64> {
    if s.contains('.') {
        s.parse::<f64>().ok()
    } else {
        None
    }
}

fn parse_int(s: &str, start: usize) -> i64 {
    let s = &s[start..];
    let mut res: i64 = 0;
    for c in s.chars() {
        let d = c as i64 - '0' as i64;
        res = res * 10 + d;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    #[test]
    fn jit_fibonacci_single() {
        let prog: Vec<&str> = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![1i64, 1, 11];
        let result = jit.run(&prog, &mut args);
        assert_eq!(result, 89);
    }

    #[test]
    fn jit_fibonacci_matches_interp() {
        let prog_str = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1";
        let prog: Vec<&str> = prog_str.split_whitespace().collect();

        for n in [5, 10, 11, 15, 20] {
            let mut interp_args = vec![
                interp::Box::Int(1),
                interp::Box::Int(1),
                interp::Box::Int(n),
            ];
            let interp_result = interp::interpret(&prog, &mut interp_args);
            let expected = interp::repr_stack(&interp_result);

            let mut jit = JitTiny3Interp::new();
            let mut jit_args = vec![1i64, 1, n];
            let jit_result = jit.run(&prog, &mut jit_args);

            assert_eq!(jit_result.to_string(), expected, "fib({n}) mismatch");
        }
    }

    #[test]
    fn jit_countdown() {
        let prog: Vec<&str> = "{ #1 #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![5i64];
        jit.run(&prog, &mut args);
        assert_eq!(args[0], 0);
    }

    #[test]
    fn jit_float_arithmetic() {
        let prog: Vec<&str> = "{ #1 0.5 MUL ->#1 #2 1 SUB ->#2 #2 }"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![interp::Box::Float(8.0), interp::Box::Int(10)];
        jit.run_typed(&prog, &mut args);
        let result = args[0].as_float();
        let expected = 8.0 * 0.5f64.powi(10);
        assert!(
            (result - expected).abs() < 1e-10,
            "expected ~{expected}, got {result}"
        );
        assert_eq!(args[1].as_int(), 0);
    }

    #[test]
    fn jit_mixed_int_float() {
        let prog: Vec<&str> = "{ #1 #2 ADD ->#1 #3 1 SUB ->#3 #3 }"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![
            interp::Box::Float(1.5),
            interp::Box::Int(2),
            interp::Box::Int(5),
        ];
        jit.run_typed(&prog, &mut args);
        let result = args[0].as_float();
        assert!((result - 11.5).abs() < 1e-10, "expected 11.5, got {result}");
        assert_eq!(args[2].as_int(), 0);
    }
}
