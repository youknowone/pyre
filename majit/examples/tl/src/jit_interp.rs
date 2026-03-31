/// JIT-enabled TL interpreter — auto-generated tracing via `#[jit_interp]`.
///
/// The `#[jit_interp]` proc macro transforms this interpreter function into
/// a meta-tracing JIT: it auto-generates `trace_instruction` and `JitState`
/// impl from the match dispatch arms. No manual IR recording needed.
///
/// Greens: [pc]
/// Reds:   [stack (via storage pool)]
/// Hint to the JIT that this value should be treated as a compile-time constant.
/// During tracing, the tracer records a GUARD_VALUE. Non-tracing mode: identity.
#[inline(always)]
fn hint_promote<T: Copy>(val: T) -> T {
    val
}

// ── Storage pool types ──

/// Single i64 stack storage.
/// _virtualizable_ = ['stackpos', 'stack[*]'] — tl.py:14 (implicit via state_fields)
pub struct TlStorage {
    stack: Vec<i64>,
}

impl TlStorage {
    fn new() -> Self {
        TlStorage { stack: Vec::new() }
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
    pub fn swap(&mut self) {
        let len = self.stack.len();
        self.stack.swap(len - 1, len - 2);
    }
    pub fn pick(&mut self, i: usize) {
        let n = self.stack.len() - i - 1;
        let val = self.stack[n];
        self.stack.push(val);
    }
    pub fn put(&mut self, i: usize) {
        let val = self.stack.pop().unwrap();
        let n = self.stack.len() - i - 1;
        self.stack[n] = val;
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
    pub fn div(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(b / a);
    }
    /// Stack rotation — @dont_look_inside in RPython (tl.py:43).
    /// The JIT does not trace into it; in tracing mode, break exits trace.
    pub fn roll(&mut self, r: i64) {
        let len = self.stack.len();
        if r < -1 {
            let i = (len as i64 + r) as usize;
            let elem = self.stack[len - 1];
            for j in (i..len - 1).rev() {
                self.stack[j + 1] = self.stack[j];
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

/// Storage pool wrapping a single TlStorage.
pub struct TlPool {
    storages: Vec<TlStorage>,
}

impl TlPool {
    fn new() -> Self {
        TlPool {
            storages: vec![TlStorage::new()],
        }
    }
    pub fn get(&self, idx: usize) -> &TlStorage {
        &self.storages[idx]
    }
    pub fn get_mut(&mut self, idx: usize) -> &mut TlStorage {
        &mut self.storages[idx]
    }
    pub fn all_jit_compatible(&self) -> bool {
        true
    }
}

/// Interpreter state: storage pool + selected storage index.
/// hint(stack, access_directly=True) — tl.py:78 (implicit via state_fields)
struct TlState {
    pool: TlPool,
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

const NOP: u8 = 1;
const PUSH: u8 = 2;
const POP: u8 = 3;
const SWAP: u8 = 4;
const ROLL: u8 = 5;
const PICK: u8 = 6;
const PUT: u8 = 7;
const ADD: u8 = 8;
const SUB: u8 = 9;
const MUL: u8 = 10;
const DIV: u8 = 11;
const EQ: u8 = 12;
const NE: u8 = 13;
const LT: u8 = 14;
const LE: u8 = 15;
const GT: u8 = 16;
const GE: u8 = 17;
const BR_COND: u8 = 18;
const BR_COND_STK: u8 = 19;
const CALL: u8 = 20;
const RETURN: u8 = 21;
const PUSHARG: u8 = 22;

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = TlState,
    env = Bytecode,
    storage = {
        pool: state.pool,
        pool_type: TlPool,
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
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = TlState {
        pool: TlPool::new(),
        selected: 0,
    };

    while pc < program.len() {
        jit_merge_point!();
        // promote(stack.stackpos) — tl.py:88
        // Makes stackpos a compile-time constant via GUARD_VALUE.
        stacksize = hint_promote(stacksize);
        let opcode = program[pc];
        pc += 1;

        match opcode {
            NOP => {}
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.pool.get_mut(state.selected).push(value);
                stacksize += 1;
            }
            POP => {
                state.pool.get_mut(state.selected).pop();
                stacksize -= 1;
            }
            SWAP => {
                state.pool.get_mut(state.selected).swap();
            }
            ROLL => {
                // Stack.roll() is @dont_look_inside in RPython — the JIT does
                // not trace into it. We delegate to TlStorage::roll().
                let r = program[pc] as i8 as i64;
                pc += 1;
                state.pool.get_mut(state.selected).roll(r);
            }
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).pick(i);
                stacksize += 1;
            }
            PUT => {
                let i = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).put(i);
                stacksize -= 1;
            }
            ADD => state.pool.get_mut(state.selected).add(),
            SUB => state.pool.get_mut(state.selected).sub(),
            MUL => state.pool.get_mut(state.selected).mul(),
            DIV => state.pool.get_mut(state.selected).div(),
            EQ => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b == a { 1 } else { 0 });
                stacksize -= 1;
            }
            NE => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b != a { 1 } else { 0 });
                stacksize -= 1;
            }
            LT => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b < a { 1 } else { 0 });
                stacksize -= 1;
            }
            LE => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b <= a { 1 } else { 0 });
                stacksize -= 1;
            }
            GT => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b > a { 1 } else { 0 });
                stacksize -= 1;
            }
            GE => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b >= a { 1 } else { 0 });
                stacksize -= 1;
            }
            BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
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
            BR_COND_STK => {
                // Pops TWO values: offset (top), then condition.
                let stk = state.pool.get_mut(state.selected);
                let offset = stk.pop();
                let cond = stk.pop();
                stacksize -= 2;
                if cond != 0 {
                    let target = (pc as i64 + offset) as usize;
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            CALL => {
                // Recursive function call. Reads offset from bytecode, calls
                // the pure interpreter at (pc + offset). The JIT does not
                // trace into recursive calls.
                let offset = program[pc] as i8 as i64;
                pc += 1;
                let res = crate::interp::interpret_at(program, (pc as i64 + offset) as usize, 0);
                state.pool.get_mut(state.selected).push(res);
                stacksize += 1;
            }
            RETURN => break,
            PUSHARG => {
                state.pool.get_mut(state.selected).push(inputarg);
                stacksize += 1;
            }
            _ => {}
        }
    }

    state.pool.get_mut(state.selected).pop()
}

// ── Public wrapper matching the old API ──

pub struct JitTlInterp {
    threshold: u32,
}

impl JitTlInterp {
    pub fn new() -> Self {
        JitTlInterp { threshold: 3 }
    }

    pub fn run(&mut self, bytecode: &[u8], inputarg: i64) -> i64 {
        mainloop(bytecode, inputarg, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    /// sum(N) = 1 + 2 + ... + N
    fn sum_bytecode() -> Vec<u8> {
        vec![
            PUSH, 0,       // acc = 0
            PUSHARG, // counter = N
            // loop (offset 3):
            PICK, 0, // dup counter
            BR_COND, 2,      // if counter != 0, skip to body (offset 9)
            POP,    // pop counter
            RETURN, // body (offset 9):
            SWAP,   // [counter, acc]
            PICK, 1,    // [counter, acc, counter]
            ADD,  // [counter, acc+counter]
            SWAP, // [acc+counter, counter]
            PUSH, 1, SUB, // [acc, counter-1]
            PUSH, 1, BR_COND, 238, // -18: jump to loop (offset 3)
        ]
    }

    #[test]
    fn jit_sum_5() {
        let bc = sum_bytecode();
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&bc, 5), 15);
    }

    #[test]
    fn jit_sum_100() {
        let bc = sum_bytecode();
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&bc, 100), 5050);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = sum_bytecode();
        for a in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![PUSH, 42, RETURN];
        let mut jit = JitTlInterp::new();
        assert_eq!(jit.run(&prog, 0), 42);
    }

    #[test]
    fn jit_various_sizes() {
        let bc = sum_bytecode();
        for a in [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    #[test]
    fn jit_bridge_exercise() {
        let bc = sum_bytecode();
        let mut jit = JitTlInterp::new();
        for a in [3, 5, 10, 20, 50, 100] {
            let expected = interp::interpret(&bc, a);
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }
}
