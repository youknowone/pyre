/// JIT-enabled TL interpreter — auto-generated tracing via `#[jit_interp]`.
///
/// The `#[jit_interp]` proc macro transforms this interpreter function into
/// a meta-tracing JIT: it auto-generates `trace_instruction` and `JitState`
/// impl from the match dispatch arms. No manual IR recording needed.
///
/// Greens: [pc, code]
/// Reds:   [inputarg, stack]
/// Virtualizables: [stack]  — tl.py:14, tl.py:71

/// Hint to the JIT that this value should be treated as a compile-time constant.
/// During tracing, the tracer records a GUARD_VALUE. Non-tracing mode: identity.
/// RPython: promote(x) → hint(x, promote=True) — rlib/jit.py:125.
#[inline(always)]
fn hint_promote<T: Copy>(val: T) -> T {
    val
}

/// Stack rotation — @dont_look_inside in RPython (tl.py:43).
///
/// Operates on the live portion of the stack `stack[0..stackpos]`.
/// The JIT does not trace into this function; it emits a residual CALL.
#[majit_macros::dont_look_inside]
extern "C" fn storage_roll(stack_ptr: usize, stackpos: i64, r: i64) {
    let stack = unsafe { std::slice::from_raw_parts_mut(stack_ptr as *mut i64, stackpos as usize) };
    let len = stack.len();
    if r < -1 {
        // tl.py:45-55
        let i = len as i64 + r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let n = len - 1;
        let elem = stack[n];
        for j in (i..n).rev() {
            stack[j + 1] = stack[j];
        }
        stack[i] = elem;
    } else if r > 1 {
        // tl.py:56-65
        let i = len as i64 - r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let elem = stack[i];
        for j in i..len - 1 {
            stack[j] = stack[j + 1];
        }
        let n = len - 1;
        stack[n] = elem;
    }
}

// ── Storage pool types ──

/// tl.py:13-14 Stack object.
/// _virtualizable_ = ['stackpos', 'stack[*]']
pub struct TlStorage {
    /// tl.py:19  self.stackpos — always store a known-nonneg integer here.
    pub stackpos: i32,
    /// tl.py:18  self.stack = [0] * size
    stack: Vec<i64>,
}

impl TlStorage {
    fn new() -> Self {
        TlStorage {
            stackpos: 0,
            stack: Vec::new(),
        }
    }
    fn new_with_size(size: usize) -> Self {
        TlStorage {
            stackpos: 0,
            stack: vec![0; size],
        }
    }
    /// tl.py:21-23  Stack.append
    pub fn push(&mut self, val: i64) {
        let pos = self.stackpos as usize;
        if pos >= self.stack.len() {
            self.stack.resize(pos + 1, 0);
        }
        self.stack[pos] = val;
        self.stackpos += 1;
    }
    /// tl.py:25-30  Stack.pop
    pub fn pop(&mut self) -> i64 {
        let stackpos = self.stackpos - 1;
        assert!(stackpos >= 0, "IndexError in pop");
        self.stackpos = stackpos; // always store a known-nonneg integer here
        self.stack[stackpos as usize]
    }
    pub fn dup(&mut self) {
        let v = self.stack[(self.stackpos - 1) as usize];
        self.push(v);
    }
    /// tl.py:101-104  SWAP
    pub fn swap(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(a);
        self.push(b);
    }
    /// tl.py:32-35  Stack.pick
    pub fn pick(&mut self, i: usize) {
        let n = self.stackpos as usize - i - 1;
        self.push(self.stack[n]);
    }
    /// tl.py:37-41  Stack.put
    pub fn put(&mut self, i: usize) {
        let elem = self.pop();
        let n = self.stackpos as usize - i - 1;
        self.stack[n] = elem;
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
    /// tl.py:43  @dont_look_inside Stack.roll
    pub fn roll(&mut self, r: i64) {
        storage_roll(self.stack.as_mut_ptr() as usize, self.stackpos as i64, r);
    }
    pub fn len(&self) -> usize {
        self.stackpos as usize
    }
    pub fn is_empty(&self) -> bool {
        self.stackpos == 0
    }
    pub fn peek_at(&self, idx: usize) -> i64 {
        self.stack[self.stackpos as usize - 1 - idx]
    }
    pub fn clear(&mut self) {
        self.stackpos = 0;
    }
    pub fn data_ptr(&self) -> usize {
        self.stack.as_ptr() as usize
    }
    pub fn data_mut_ptr(&mut self) -> usize {
        self.stack.as_mut_ptr() as usize
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
/// tl.py:78  stack = hint(stack, access_directly=True)
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
    auto_calls = true,
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
    // stacksize mirrors stack.stackpos for the macro's guard recovery.
    let mut stacksize: i32 = 0;
    let mut state = TlState {
        pool: TlPool::new(),
        selected: 0,
    };

    while pc < program.len() {
        jit_merge_point!();
        // tl.py:88  stack.stackpos = promote(stack.stackpos)
        stacksize = hint_promote(state.pool.get(state.selected).stackpos);
        state.pool.get_mut(state.selected).stackpos = stacksize;

        let opcode = program[pc];
        pc += 1;

        match opcode {
            NOP => {}
            // tl.py:94-96
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.pool.get_mut(state.selected).push(value);
            }
            // tl.py:98-99
            POP => {
                state.pool.get_mut(state.selected).pop();
            }
            // tl.py:101-104
            SWAP => {
                state.pool.get_mut(state.selected).swap();
            }
            // tl.py:106-109  Stack.roll() is @dont_look_inside
            ROLL => {
                let r = program[pc] as i8 as i64;
                pc += 1;
                let stk = state.pool.get_mut(state.selected);
                storage_roll(stk.data_mut_ptr(), stk.stackpos as i64, r);
            }
            // tl.py:111-113
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).pick(i);
            }
            // tl.py:115-117
            PUT => {
                let i = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).put(i);
            }
            // tl.py:119-121
            ADD => state.pool.get_mut(state.selected).add(),
            // tl.py:123-125
            SUB => state.pool.get_mut(state.selected).sub(),
            // tl.py:127-129
            MUL => state.pool.get_mut(state.selected).mul(),
            // tl.py:131-133
            DIV => state.pool.get_mut(state.selected).div(),
            // tl.py:135-137
            EQ => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b == a { 1 } else { 0 });
            }
            NE => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b != a { 1 } else { 0 });
            }
            LT => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b < a { 1 } else { 0 });
            }
            LE => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b <= a { 1 } else { 0 });
            }
            GT => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b > a { 1 } else { 0 });
            }
            GE => {
                let a = state.pool.get_mut(state.selected).pop();
                let b = state.pool.get_mut(state.selected).pop();
                state
                    .pool
                    .get_mut(state.selected)
                    .push(if b >= a { 1 } else { 0 });
            }
            // tl.py:159-165
            BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                let cond = state.pool.get_mut(state.selected).pop();
                let jump = cond != 0;
                if jump {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            // tl.py:167-172
            BR_COND_STK => {
                let stk = state.pool.get_mut(state.selected);
                let offset = stk.pop();
                let cond = stk.pop();
                if cond != 0 {
                    let target = (pc as i64 + offset) as usize;
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            // tl.py:174-178
            CALL => {
                let offset = program[pc] as i8 as i64;
                pc += 1;
                let res = crate::interp::interpret_at(program, (pc as i64 + offset) as usize, 0);
                state.pool.get_mut(state.selected).push(res);
            }
            // tl.py:180-181
            RETURN => break,
            // tl.py:183-184
            PUSHARG => {
                state.pool.get_mut(state.selected).push(inputarg);
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
