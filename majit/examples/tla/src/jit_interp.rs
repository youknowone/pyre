/// JIT-enabled TLA interpreter — auto-generated tracing via `#[jit_interp]`.
///
/// The `#[jit_interp]` proc macro transforms this interpreter function into
/// a meta-tracing JIT: it auto-generates `trace_instruction` and `JitState`
/// impl from the match dispatch arms. No manual IR recording needed.
///
/// This matches RPython's abstraction: write the interpreter, get JIT for free.
///
/// Greens: [pc]
/// Reds:   [stack (via storage pool)]

// ── Storage pool types ──

/// Single i64 stack storage.
pub struct TlaStorage {
    stack: Vec<i64>,
}

impl TlaStorage {
    fn new() -> Self {
        TlaStorage { stack: Vec::new() }
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
    pub fn len(&self) -> usize {
        self.stack.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
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
    pub fn get_op(&self, _pc: usize) -> u8 {
        0 // unused — opcode is read from env/program directly
    }
}

/// Storage pool wrapping a single TlaStorage.
pub struct TlaPool {
    storages: Vec<TlaStorage>,
}

impl TlaPool {
    fn new() -> Self {
        TlaPool {
            storages: vec![TlaStorage::new()],
        }
    }
    pub fn get(&self, idx: usize) -> &TlaStorage {
        &self.storages[idx]
    }
    pub fn get_mut(&mut self, idx: usize) -> &mut TlaStorage {
        &mut self.storages[idx]
    }
    pub fn all_jit_compatible(&self) -> bool {
        true
    }
}

/// Interpreter state: storage pool + selected storage index.
struct TlaState {
    pool: TlaPool,
    selected: usize,
}

fn find_used_storages(_program: &Bytecode, _header_pc: usize, _initial: usize) -> Vec<usize> {
    vec![0]
}

/// Type alias for bytecode — the env type for `#[jit_interp]`.
pub type Bytecode = [u8];

/// Extension trait providing `get_op` for bytecode slices.
/// Required by `#[jit_interp]` generated code.
trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}

impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

// ── Opcodes ──

const CONST_INT: u8 = 0;
const POP: u8 = 1;
const ADD: u8 = 2;
const RETURN: u8 = 3;
const JUMP_IF: u8 = 4;
const DUP: u8 = 5;
const SUB: u8 = 6;
#[allow(dead_code)]
const NEWSTR: u8 = 7;

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = TlaState,
    env = Bytecode,
    storage = {
        pool: state.pool,
        pool_type: TlaPool,
        selector: state.selected,
        untraceable: [],
        scan: find_used_storages,
        can_trace_guard: all_jit_compatible,
    },
    binops = {
        add => IntAdd,
        sub => IntSub,
    },
)]
pub fn mainloop(program: &Bytecode, initial_value: i64, threshold: u32) -> i64 {
    let mut driver: majit_meta::JitDriver<TlaState> = majit_meta::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = TlaState {
        pool: TlaPool::new(),
        selected: 0,
    };

    // Push initial value onto the stack.
    state.pool.get_mut(state.selected).push(initial_value);
    stacksize = 1;

    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;

        match opcode {
            CONST_INT => {
                let value = program[pc] as i64;
                pc += 1;
                state.pool.get_mut(state.selected).push(value);
                stacksize += 1;
            }
            POP => {
                state.pool.get_mut(state.selected).pop();
                stacksize -= 1;
            }
            DUP => {
                state.pool.get_mut(state.selected).dup();
                stacksize += 1;
            }
            ADD => state.pool.get_mut(state.selected).add(),
            SUB => state.pool.get_mut(state.selected).sub(),
            JUMP_IF => {
                let target = program[pc] as usize;
                pc += 1;
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
            RETURN => break,
            _ => {}
        }
    }

    state.pool.get_mut(state.selected).pop()
}

// ── Public wrapper matching the old API ──

pub struct JitTlaInterp {
    threshold: u32,
}

impl JitTlaInterp {
    pub fn new() -> Self {
        JitTlaInterp { threshold: 3 }
    }

    pub fn run(&mut self, bytecode: &[u8], w_arg: crate::interp::WObject) -> crate::interp::WObject {
        let val = match &w_arg {
            crate::interp::WObject::Int(v) => *v,
            _ => panic!("JIT only supports integer args"),
        };
        let result = mainloop(bytecode, val, self.threshold);
        crate::interp::WObject::Int(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    fn countdown_bytecode() -> Vec<u8> {
        vec![
            DUP, CONST_INT, 1, SUB, DUP, JUMP_IF, 1, POP, RETURN,
        ]
    }

    #[test]
    fn jit_countdown_5() {
        let bc = countdown_bytecode();
        let mut jit = JitTlaInterp::new();
        let result = jit.run(&bc, interp::WObject::Int(5));
        assert_eq!(result.int_value(), 5);
    }

    #[test]
    fn jit_countdown_100() {
        let bc = countdown_bytecode();
        let mut jit = JitTlaInterp::new();
        let result = jit.run(&bc, interp::WObject::Int(100));
        assert_eq!(result.int_value(), 100);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = countdown_bytecode();
        for n in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::run(&bc, interp::WObject::Int(n));
            let mut jit = JitTlaInterp::new();
            let got = jit.run(&bc, interp::WObject::Int(n));
            assert_eq!(
                got.int_value(),
                expected.int_value(),
                "mismatch for n={n}"
            );
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![RETURN];
        let mut jit = JitTlaInterp::new();
        let result = jit.run(&prog, interp::WObject::Int(42));
        assert_eq!(result.int_value(), 42);
    }
}
