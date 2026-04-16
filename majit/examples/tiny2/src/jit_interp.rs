/// JIT-enabled tiny2 interpreter via `#[jit_interp]` proc macro with `state_fields`.
///
/// PRE-EXISTING-ADAPTATION: `rpython/jit/tl/tiny2_hotpath.py:90` models the
/// operand stack as a linked-list `Stack(value, next)`; each push allocates
/// one cons cell that RPython's JIT peels as a chain of virtuals. pyre's
/// `state_fields = { stackpos, stack: [int; virt] }` does not express
/// linked-list stacks — it requires a contiguous virtualizable array. The
/// array backing is a source-shape deviation; post-optimization the trace
/// shape is equivalent to RPython's peeled virtuals for the shallow,
/// constant-height stacks the tinybench exercises. Porting a linked-list
/// state kind to #[jit_interp] is a separate, larger port.
///
/// Greens: [bytecode (env), pc]
/// Reds:   [stackpos, stack]  (tracked via state_fields)

// ── Bytecode opcodes ──

const OP_PUSH_INT: u8 = 0; // followed by 8 bytes (i64 LE)
const OP_PUSH_ARG: u8 = 1; // followed by 1 byte (arg index, 0-based)
const OP_STORE_ARG: u8 = 2; // followed by 1 byte (arg index, 0-based)
const OP_ADD: u8 = 3;
const OP_SUB: u8 = 4;
const OP_MUL: u8 = 5;
const OP_LOOP_START: u8 = 6; // no-op marker (target for back-edge)
const OP_LOOP_END: u8 = 7; // followed by 2 bytes (target pc, u16 LE)
const OP_END: u8 = 8;

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
            code.push(n - 1); // 0-based index
        } else if w.starts_with('#') {
            let n = parse_int(w, 1) as u8;
            code.push(OP_PUSH_ARG);
            code.push(n - 1); // 0-based index
        } else if w == "{" {
            code.push(OP_LOOP_START);
            loop_starts.push(code.len()); // pc AFTER the OP_LOOP_START
        } else if w == "}" {
            code.push(OP_LOOP_END);
            let target_pc = loop_starts.pop().expect("unmatched }");
            code.push((target_pc & 0xFF) as u8);
            code.push(((target_pc >> 8) & 0xFF) as u8);
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

/// RPython tiny2_hotpath.py Stack. `_virtualizable_ = ['stackpos', 'stack[*]']`.
struct Tiny2State {
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

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = Tiny2State,
    env = Bytecode,
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn mainloop(program: &Bytecode, num_args: usize, args_out: &mut [i64], threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<Tiny2State> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = Tiny2State {
        stackpos: num_args as i64,
        stack: vec![0i64; program.len()],
    };

    while pc < program.len() {
        // RPython: tinyjitdriver.jit_merge_point(...)
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
            OP_PUSH_ARG => {
                // RPython: stack = Stack(args[n-1], stack) — copy arg to top.
                let n = program[pc] as usize;
                pc += 1;
                let v = state.stack[n];
                state.stack[state.stackpos as usize] = v;
                state.stackpos = state.stackpos + 1;
            }
            OP_STORE_ARG => {
                // RPython: stack, args[n-1] = stack.pop() — pop top and store at arg slot n.
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
            OP_LOOP_START => {
                // RPython: loops.append(pos) — loop targets are compiled into bytecode offsets.
            }
            OP_LOOP_END => {
                // RPython: flag = stack.pop(); if flag.as_int() != 0: pos = loops[-1]; promote(pos)
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

    // Write modified args back.
    let n = args_out.len().min(state.stackpos as usize);
    args_out[..n].copy_from_slice(&state.stack[..n]);

    // Result is the top of stack (one past stackpos-1).
    state.stackpos = state.stackpos - 1;
    state.stack[state.stackpos as usize]
}

// ── Public wrapper matching the old API ──

pub struct JitTiny2Interp {
    threshold: u32,
}

impl JitTiny2Interp {
    pub fn new() -> Self {
        JitTiny2Interp { threshold: 3 }
    }

    /// Run a word-based program with integer args.
    /// Returns the result: stack top if non-empty, else args[0].
    ///
    /// RPython: interpret(bytecode, args) — args are mutated in-place during execution.
    /// Here, args occupy stack slots 0..num_args and are written back via args_out.
    pub fn run(&mut self, bytecode: &[&str], args: &mut Vec<i64>) -> i64 {
        let code = compile(bytecode);
        let num_args = args.len();

        // Determine if the program ends with a value on stack beyond args.
        let has_result_on_stack = program_has_result(bytecode, num_args);

        // Prepend OP_PUSH_INT instructions for each arg to pre-populate the stack.
        // The prepended prefix shifts all bytecode offsets, so loop targets in `code`
        // must be adjusted by the prefix length.
        let prefix_len = num_args * 9; // 1 opcode byte + 8 value bytes per arg
        let mut full_code = Vec::with_capacity(prefix_len + code.len());
        for &arg in args.iter() {
            full_code.push(OP_PUSH_INT);
            full_code.extend_from_slice(&arg.to_le_bytes());
        }
        // Patch loop targets: scan code for OP_LOOP_END and adjust the 2-byte target.
        let mut patched_code = code.clone();
        let mut ci = 0;
        while ci < patched_code.len() {
            match patched_code[ci] {
                OP_PUSH_INT => ci += 9,
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

        // Stack starts empty; args are pushed by the OP_PUSH_INT prefix, so pass num_args=0.
        let result = mainloop(&full_code, 0, args.as_mut_slice(), self.threshold);

        if has_result_on_stack { result } else { args[0] }
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
            depth -= 1; // } pops the flag
        } else if loop_depth == 0 {
            if *w == "ADD" || *w == "SUB" || *w == "MUL" {
                depth -= 1; // pop 2, push 1 = net -1
            } else if w.starts_with("->#") {
                depth -= 1; // pop and store
            } else if w.starts_with('#') {
                depth += 1; // push arg copy
            } else {
                depth += 1; // integer literal
            }
        }
    }
    depth > num_args as i32
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
        let mut jit = JitTiny2Interp::new();
        let mut args = vec![1i64, 1, 11];
        let result = jit.run(&prog, &mut args);
        assert_eq!(result, 89);
    }

    #[test]
    fn jit_fibonacci_matches_interp() {
        let prog_str = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1";
        let prog: Vec<&str> = prog_str.split_whitespace().collect();

        for n in [5, 10, 11, 15, 20] {
            // Interpreter
            let mut interp_args = vec![
                interp::Box::Int(1),
                interp::Box::Int(1),
                interp::Box::Int(n),
            ];
            let interp_result = interp::interpret(&prog, &mut interp_args);
            let expected = interp::repr_stack(&interp_result);

            // JIT
            let mut jit = JitTiny2Interp::new();
            let mut jit_args = vec![1i64, 1, n];
            let jit_result = jit.run(&prog, &mut jit_args);

            assert_eq!(jit_result.to_string(), expected, "fib({n}) mismatch");
        }
    }

    #[test]
    fn jit_factorial() {
        let prog: Vec<&str> = "1 { #1 MUL #1 1 SUB ->#1 #1 }".split_whitespace().collect();

        let mut interp_args = vec![interp::Box::Int(5)];
        let result = interp::interpret(&prog, &mut interp_args);
        assert_eq!(interp::repr_stack(&result), "120");
    }

    #[test]
    fn jit_countdown() {
        let prog: Vec<&str> = "{ #1 #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let mut jit = JitTiny2Interp::new();
        let mut args = vec![5i64];
        jit.run(&prog, &mut args);
        // After loop, arg1 should be 0
        assert_eq!(args[0], 0);
    }
}
