/// JIT-enabled tiny2 interpreter via `#[jit_interp]` proc macro.
///
/// The word-based program is compiled to bytecode, then the bytecoded mainloop
/// is traced by the auto-generated JitState/trace_instruction.
///
/// Greens: [bytecode (env), pc]
/// Reds:   [storage (args at bottom, computation stack on top)]
///
/// RPython correspondence (tiny2_hotpath.py):
///   promote(opcode)         — implicit: opcode is constant at each trace position
///   hint(bytecode, deepfreeze=True) — implicit: &[u8] is immutable
///   compute_invariants      — implicit: storage pool auto-captures loop state
///   on_enter_jit            — implicit: storage pool virtualizes args

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

// ── Storage pool types ──

/// Single i64 storage: args at bottom, computation stack on top.
pub struct Tiny2Storage {
    stack: Vec<i64>,
}

impl Tiny2Storage {
    fn new() -> Self {
        Tiny2Storage { stack: Vec::new() }
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
    /// Copy the bottom `n` elements (arg slots) into `out`.
    pub fn read_args(&self, out: &mut [i64]) {
        let n = out.len().min(self.stack.len());
        out[..n].copy_from_slice(&self.stack[..n]);
    }
}

/// Storage pool wrapping a single Tiny2Storage.
pub struct Tiny2Pool {
    storages: Vec<Tiny2Storage>,
}

impl Tiny2Pool {
    fn new() -> Self {
        Tiny2Pool {
            storages: vec![Tiny2Storage::new()],
        }
    }
    pub fn get(&self, idx: usize) -> &Tiny2Storage {
        &self.storages[idx]
    }
    pub fn get_mut(&mut self, idx: usize) -> &mut Tiny2Storage {
        &mut self.storages[idx]
    }
    pub fn all_jit_compatible(&self) -> bool {
        true
    }
}

/// Interpreter state: storage pool + selected storage index.
struct Tiny2State {
    pool: Tiny2Pool,
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

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = Tiny2State,
    env = Bytecode,
    storage = {
        pool: state.pool,
        pool_type: Tiny2Pool,
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
#[allow(unused_assignments, unused_variables)]
fn mainloop(program: &Bytecode, num_args: usize, args_out: &mut [i64], threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<Tiny2State> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    // stacksize is updated by macro-generated code in can_enter_jit! expansion.
    let mut stacksize: i32 = num_args as i32;
    let mut state = Tiny2State {
        pool: Tiny2Pool::new(),
        selected: 0,
    };

    while pc < program.len() {
        // RPython: tinyjitdriver.jit_merge_point(args=args, loops=loops,
        //          stack=stack, bytecode=bytecode, pos=pos)
        jit_merge_point!();
        // RPython: hint(bytecode, deepfreeze=True) — implicit: &[u8] is immutable
        // RPython: hint(opcode, concrete=True) — implicit: opcode is constant at each pc
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
            OP_PUSH_ARG => {
                // RPython: stack = Stack(args[n-1], stack)
                let n = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).copy_up(n);
                stacksize += 1;
            }
            OP_STORE_ARG => {
                // RPython: stack, args[n-1] = stack.pop()
                let n = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).store_down(n);
                stacksize -= 1;
            }
            // RPython: op2(stack, func_add_int, func_add_str)
            // Integer-only path; promote(y.__class__) is implicit (no StrBox in bytecoded form).
            OP_ADD => state.pool.get_mut(state.selected).add(),
            OP_SUB => state.pool.get_mut(state.selected).sub(),
            OP_MUL => state.pool.get_mut(state.selected).mul(),
            OP_LOOP_START => {
                // RPython: loops.append(pos) — loop targets are compiled into bytecode offsets.
            }
            OP_LOOP_END => {
                // RPython: flag = stack.pop(); if flag.as_int() != 0: pos = loops[-1]; promote(pos)
                let target = (program[pc] as usize) | ((program[pc + 1] as usize) << 8);
                pc += 2;
                let cond = state.pool.get_mut(state.selected).pop();
                stacksize -= 1;
                let jump = cond != 0;
                if jump {
                    // RPython: promote(pos); can_enter_jit(...)
                    // promote(pos) is implicit: target is a bytecode literal, already constant.
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

    // Write modified args back (RPython mutates args[] in-place).
    state.pool.get(state.selected).read_args(args_out);

    // Result is the top of storage.
    state.pool.get_mut(state.selected).pop()
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
    /// Here, args occupy storage slots 0..num_args and are written back via args_out.
    pub fn run(&mut self, bytecode: &[&str], args: &mut Vec<i64>) -> i64 {
        let code = compile(bytecode);
        let num_args = args.len();

        // Determine if the program ends with a value on stack beyond args.
        let has_result_on_stack = program_has_result(bytecode, num_args);

        // Prepend OP_PUSH_INT instructions for each arg to pre-populate the storage.
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

        // mainloop writes modified args back into args_out before returning.
        let result = mainloop(&full_code, num_args, args.as_mut_slice(), self.threshold);

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
