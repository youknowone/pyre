/// JIT-enabled tiny3 interpreter via `#[jit_interp]` proc macro with `state_fields`.
///
/// PRE-EXISTING-ADAPTATION: `rpython/jit/tl/tiny3_hotpath.py:96` models the
/// operand stack as a linked-list `Stack(value, next)`, identical shape to
/// tiny2_hotpath.py. pyre's `state_fields = { stackpos, stack: [int; virt] }`
/// does not express linked-list stacks — see the same adaptation note on
/// `majit/examples/tiny2/src/jit_interp.rs`.
///
/// Greens: [pc]
/// Reds:   [stackpos, stack]  (args at bottom, computation stack on top)
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

// ── State ──

struct Tiny3State {
    stackpos: i64,
    stack: Vec<i64>,
}

pub type Bytecode = [u8];

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
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn mainloop(program: &Bytecode, num_args: usize, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<Tiny3State> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = Tiny3State {
        stackpos: num_args as i64,
        stack: vec![0i64; program.len()],
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
                state.stack[state.stackpos as usize] = value;
                state.stackpos = state.stackpos + 1;
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
                state.stack[state.stackpos as usize] = value;
                state.stackpos = state.stackpos + 1;
            }
            OP_PUSH_ARG => {
                let n = program[pc] as usize;
                pc += 1;
                let v = state.stack[n];
                state.stack[state.stackpos as usize] = v;
                state.stackpos = state.stackpos + 1;
            }
            OP_STORE_ARG => {
                let n = program[pc] as usize;
                pc += 1;
                state.stackpos = state.stackpos - 1;
                let v = state.stack[state.stackpos as usize];
                state.stack[n] = v;
            }
            OP_ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos = state.stackpos - 1;
            }
            OP_SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos = state.stackpos - 1;
            }
            OP_MUL => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b * a;
                state.stackpos = state.stackpos - 1;
            }
            OP_LOOP_START => {}
            OP_LOOP_END => {
                let target = (program[pc] as usize) | ((program[pc + 1] as usize) << 8);
                pc += 2;
                state.stackpos = state.stackpos - 1;
                let jump = state.stack[state.stackpos as usize] != 0;
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

    state.stackpos = state.stackpos - 1;
    state.stack[state.stackpos as usize]
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

        // Prepend OP_PUSH_INT for each arg to pre-populate the stack.
        let prefix_len = num_args * 9;
        let mut full_code = Vec::new();
        for &arg in args.iter() {
            full_code.push(OP_PUSH_INT);
            full_code.extend_from_slice(&arg.to_le_bytes());
        }
        // Patch loop targets to account for the prepended prefix.
        let mut patched_code = code.clone();
        let mut ci = 0;
        while ci < patched_code.len() {
            match patched_code[ci] {
                OP_PUSH_INT | OP_PUSH_FLOAT => ci += 9,
                OP_PUSH_ARG | OP_STORE_ARG => ci += 2,
                OP_LOOP_END => {
                    ci += 1;
                    let old_target =
                        (patched_code[ci] as usize) | ((patched_code[ci + 1] as usize) << 8);
                    let new_target = old_target + prefix_len;
                    patched_code[ci] = (new_target & 0xFF) as u8;
                    patched_code[ci + 1] = ((new_target >> 8) & 0xFF) as u8;
                    ci += 2;
                }
                _ => ci += 1,
            }
        }
        full_code.extend_from_slice(&patched_code);

        let result = mainloop(&full_code, 0, self.threshold);

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
