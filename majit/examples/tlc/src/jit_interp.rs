/// JIT-enabled TLC interpreter — auto-generated tracing via `#[jit_interp]`.
///
/// The `#[jit_interp]` proc macro transforms this interpreter function into
/// a meta-tracing JIT: it auto-generates `trace_instruction` and `JitState`
/// impl from the match dispatch arms. No manual IR recording needed.
///
/// Greens: [pc]
/// Reds:   [stack (via storage pool)]
///
/// RPython correspondence (tlc.py):
///   - JitDriver(greens=['pc', 'code'], reds=['frame', 'pool']) — tlc.py:231
///   - @elidable Class.get() — tlc.py:76 (class descriptor lookup cached as pure)
///     Not yet implemented: requires extracting a helper and jitcode lowerer support.
///   - Object opcodes (NIL, CONS, CAR, CDR, NEW, etc.) are not handled here —
///     they cause guard failure in RPython, effectively breaking out of the trace.
///   - Integer-only loop opcodes map directly to IR via #[jit_interp] binops.
use crate::interp::{self, ConstantPool};

// ── Storage pool types ──

/// Single i64 stack storage.
pub struct TlcStorage {
    stack: Vec<i64>,
}

impl TlcStorage {
    fn new() -> Self {
        TlcStorage { stack: Vec::new() }
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
    pub fn eq(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(if b == a { 1 } else { 0 });
    }
    pub fn ne(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(if b != a { 1 } else { 0 });
    }
    pub fn lt(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(if b < a { 1 } else { 0 });
    }
    pub fn le(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(if b <= a { 1 } else { 0 });
    }
    pub fn gt(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(if b > a { 1 } else { 0 });
    }
    pub fn ge(&mut self) {
        let a = self.pop();
        let b = self.pop();
        self.push(if b >= a { 1 } else { 0 });
    }
    /// Stack rotation — inline in tlc.py:284 (no @dont_look_inside).
    pub fn roll(&mut self, r: i64) {
        if r < -1 {
            let i = self.stack.len() as i64 + r;
            assert!(i >= 0, "IndexError in ROLL");
            let val = self.stack.pop().unwrap();
            self.stack.insert(i as usize, val);
        } else if r > 1 {
            let i = self.stack.len() as i64 - r;
            assert!(i >= 0, "IndexError in ROLL");
            let val = self.stack.remove(i as usize);
            self.stack.push(val);
        }
    }
    pub fn pick_at(&mut self, depth: usize) {
        let n = self.stack.len() - depth - 1;
        let val = self.stack[n];
        self.stack.push(val);
    }
    pub fn put_at(&mut self, depth: usize) {
        let val = self.stack.pop().unwrap();
        let n = self.stack.len() - depth - 1;
        self.stack[n] = val;
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
}

/// Storage pool wrapping a single TlcStorage.
pub struct TlcPool {
    storages: Vec<TlcStorage>,
}

impl TlcPool {
    fn new() -> Self {
        TlcPool {
            storages: vec![TlcStorage::new()],
        }
    }
    pub fn get(&self, idx: usize) -> &TlcStorage {
        &self.storages[idx]
    }
    pub fn get_mut(&mut self, idx: usize) -> &mut TlcStorage {
        &mut self.storages[idx]
    }
    pub fn all_jit_compatible(&self) -> bool {
        true
    }
}

/// Interpreter state: storage pool + selected storage index.
struct TlcState {
    pool: TlcPool,
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

const NOP: u8 = interp::NOP;
const PUSH: u8 = interp::PUSH;
const POP: u8 = interp::POP;
const SWAP: u8 = interp::SWAP;
const ROLL: u8 = interp::ROLL;
const PICK: u8 = interp::PICK;
const PUT: u8 = interp::PUT;
const ADD: u8 = interp::ADD;
const SUB: u8 = interp::SUB;
const MUL: u8 = interp::MUL;
// DIV not traced: IntObj.div() in tlc.py:144 uses Python 2 floor division (//),
// which differs from Rust's truncating division for negative operands.
// Object opcodes (NIL, CONS, CAR, CDR, NEW, GETATTR, SETATTR, SEND) are also
// not traced — they cause guard failure, breaking out of the compiled trace.
const EQ: u8 = interp::EQ;
const NE: u8 = interp::NE;
const LT: u8 = interp::LT;
const LE: u8 = interp::LE;
const GT: u8 = interp::GT;
const GE: u8 = interp::GE;
const BR: u8 = interp::BR;
const BR_COND: u8 = interp::BR_COND;
const RETURN: u8 = interp::RETURN;
const PUSHARG: u8 = interp::PUSHARG;

const DEFAULT_THRESHOLD: u32 = 3;

// ── JIT mainloop ──

// #[jit_interp] generates trace_instruction + JitState from the match arms below.
// Only integer-stack opcodes are traced; object opcodes (NIL, CONS, NEW, SEND, etc.)
// are absent from this function, so reaching them would abort tracing — matching
// RPython's behavior where polymorphic dispatch on Obj causes guard failure.
#[majit_macros::jit_interp(
    state = TlcState,
    env = Bytecode,
    storage = {
        pool: state.pool,
        pool_type: TlcPool,
        selector: state.selected,
        untraceable: [],
        scan: find_used_storages,
        can_trace_guard: all_jit_compatible,
    },
    binops = {
        add => IntAdd,
        sub => IntSub,
        mul => IntMul,
        eq => IntEq,
        ne => IntNe,
        lt => IntLt,
        le => IntLe,
        gt => IntGt,
        ge => IntGe,
    },
)]
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlcState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = TlcState {
        pool: TlcPool::new(),
        selected: 0,
    };

    while pc < program.len() {
        jit_merge_point!();
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
            SWAP => state.pool.get_mut(state.selected).swap(),
            ADD => state.pool.get_mut(state.selected).add(),
            SUB => state.pool.get_mut(state.selected).sub(),
            MUL => state.pool.get_mut(state.selected).mul(),
            EQ => state.pool.get_mut(state.selected).eq(),
            NE => state.pool.get_mut(state.selected).ne(),
            LT => state.pool.get_mut(state.selected).lt(),
            LE => state.pool.get_mut(state.selected).le(),
            GT => state.pool.get_mut(state.selected).gt(),
            GE => state.pool.get_mut(state.selected).ge(),
            ROLL => {
                let r = program[pc] as i8 as i64;
                pc += 1;
                state.pool.get_mut(state.selected).roll(r);
            }
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).pick_at(i);
                stacksize += 1;
            }
            PUT => {
                let i = program[pc] as usize;
                pc += 1;
                state.pool.get_mut(state.selected).put_at(i);
                stacksize -= 1;
            }
            PUSHARG => {
                state.pool.get_mut(state.selected).push(inputarg);
                stacksize += 1;
            }
            BR_COND => {
                let target = ((pc as i64) + program[pc] as i8 as i64 + 1) as usize;
                let next_pc = pc + 1;
                let cond = state.pool.get_mut(state.selected).pop();
                stacksize -= 1;
                let jump = cond != 0;
                if jump {
                    if target < next_pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
                pc = next_pc;
            }
            BR => {
                let target = ((pc as i64) + program[pc] as i8 as i64 + 1) as usize;
                let next_pc = pc + 1;
                if target < next_pc {
                    can_enter_jit!(driver, target, &mut state, program, || {});
                }
                pc = target;
                continue;
            }
            RETURN => break,
            _ => break,
        }
    }

    if state.pool.get(state.selected).is_empty() {
        0
    } else {
        state.pool.get_mut(state.selected).pop()
    }
}

// ── Public wrapper matching the old API ──

pub struct JitTlcInterp {
    threshold: u32,
}

impl JitTlcInterp {
    pub fn new() -> Self {
        JitTlcInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// Run the TLC interpreter with JIT support.
    /// Only traces integer-only loops; unknown opcodes cause loop exit.
    pub fn run(&mut self, code: &[u8], inputarg: i64, _pool: &ConstantPool) -> i64 {
        mainloop(code, inputarg, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    /// Fibonacci using ROLL -- pure integer loop, good JIT candidate.
    fn fibo_bytecode(pool: &mut ConstantPool) -> Vec<u8> {
        interp::compile(
            include_str!("../../../../rpython/jit/tl/fibo.tlc.src"),
            pool,
        )
    }

    #[test]
    fn jit_fibo_7() {
        let mut pool = ConstantPool::new();
        let bc = fibo_bytecode(&mut pool);
        let mut jit = JitTlcInterp::new();
        assert_eq!(jit.run(&bc, 7, &pool), 13);
    }

    #[test]
    fn jit_fibo_matches_interp() {
        let mut pool = ConstantPool::new();
        let bc = fibo_bytecode(&mut pool);
        for n in [1, 2, 3, 5, 7, 10, 15] {
            let expected = interp::interp(&bc, 0, n, &pool);
            let mut jit = JitTlcInterp::new();
            let got = jit.run(&bc, n, &pool);
            assert_eq!(got, expected, "fibo mismatch for n={n}");
        }
    }

    /// Simple integer countdown loop (no object ops).
    #[test]
    fn jit_countdown() {
        let mut pool = ConstantPool::new();
        let bc = interp::compile(
            "
            PUSHARG         # [n]
        loop:
            PUSH 1
            SUB             # [n-1]
            PICK 0          # [n-1, n-1]
            BR_COND loop    # [n-1] if n-1 != 0
            RETURN
        ",
            &mut pool,
        );
        let mut jit = JitTlcInterp::new();
        assert_eq!(jit.run(&bc, 100, &pool), 0);
    }

    #[test]
    fn jit_sum() {
        let mut pool = ConstantPool::new();
        let bc = interp::compile(
            "
            PUSH 0          # [acc=0]
            PUSHARG         # [acc, n]
        loop:
            PICK 0          # [acc, n, n]
            BR_COND body
            POP
            RETURN
        body:
            SWAP            # [n, acc]
            PICK 1          # [n, acc, n]
            ADD             # [n, acc+n]
            SWAP            # [acc+n, n]
            PUSH 1
            SUB             # [acc, n-1]
            PUSH 1
            BR_COND loop
        ",
            &mut pool,
        );
        let mut jit = JitTlcInterp::new();
        assert_eq!(jit.run(&bc, 10, &pool), 55);
        let mut jit2 = JitTlcInterp::new();
        assert_eq!(jit2.run(&bc, 100, &pool), 5050);
    }
}
