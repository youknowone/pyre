/// JIT-enabled TL interpreter via `#[jit_interp]` with `state_fields`.
///
/// RPython parity: tl.py JitDriver(greens=['pc','code'], reds=['inputarg','stack'],
/// virtualizables=['stack']). Stack._virtualizable_ = ['stackpos', 'stack[*]']
/// at tl.py:14 maps directly to `state_fields = { stackpos: int, stack: [int; virt] }`.
///
/// Greens: [pc, code]
/// Reds:   [inputarg, stackpos, stack]  (inputarg is a function parameter — red by nature)
use majit_metainterp::jit::promote;

// [SPIKE-S0/FR] throwaway compile counter to measure portal-call compile-through.
pub static SPIKE_COMPILES: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

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

// ── State ──

pub type Bytecode = [u8];

#[expect(
    dead_code,
    reason = "the jit_interp macro resolves bytecode reads through this trait surface"
)]
trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}

impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

/// tl.py:13-14 Stack object. `_virtualizable_ = ['stackpos', 'stack[*]']`.
/// tl.py:17 `Stack(size)` — `size` is the bytecode length; the caller
/// (`interp_eval`) passes `len(code)`. See tl.py:120.
struct TlState {
    stackpos: i64,
    stack: Vec<i64>,
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
    greens = [pc, program],
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
    recursive_entry = crate::interp::interpret_recursive,
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlState> =
        majit_metainterp::JitDriver::new(threshold);
    // [SPIKE-S0/FR] count compiled loops.
    driver.set_on_compile_loop(|_gk, _b, _a| {
        SPIKE_COMPILES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let stacksize: i32 = 0;
    let mut state = TlState {
        stackpos: 0,
        stack: vec![0i64; program.len()],
    };

    // RPython warmspot.py:281-289 canonical-liveness install hook.
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!();
        // tl.py:88  stack.stackpos = promote(stack.stackpos)
        state.stackpos = promote(state.stackpos);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            NOP => {}
            // tl.py:94-96
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos = state.stackpos + 1;
            }
            // tl.py:98-99
            POP => {
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:101-104
            SWAP => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 1) as usize] = b;
                state.stack[(state.stackpos - 2) as usize] = a;
            }
            // tl.py:106-109  Stack.roll() is @dont_look_inside
            ROLL => {
                let r = program[pc] as i8 as i64;
                pc += 1;
                storage_roll(state.stack.as_mut_ptr() as usize, state.stackpos, r);
            }
            // tl.py:111-113  Stack.pick(i): duplicate stack[stackpos - i - 1]
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let v = state.stack[(state.stackpos as usize) - i - 1];
                state.stack[state.stackpos as usize] = v;
                state.stackpos = state.stackpos + 1;
            }
            // tl.py:115-117  Stack.put(i): pop and store at stackpos - i - 1
            PUT => {
                let i = program[pc] as usize;
                pc += 1;
                state.stackpos = state.stackpos - 1;
                let v = state.stack[state.stackpos as usize];
                state.stack[(state.stackpos as usize) - i] = v;
            }
            // tl.py:119-121
            ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:123-125
            SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:127-129
            MUL => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b * a;
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:131-133
            DIV => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b / a;
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:135-157 — inline comparisons (no helper functions)
            EQ => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b == a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            NE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b != a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            LT => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b < a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            LE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b <= a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            GT => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b > a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            GE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b >= a { 1 } else { 0 };
                state.stackpos = state.stackpos - 1;
            }
            // tl.py:159-165
            BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
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
            // tl.py:167-172
            BR_COND_STK => {
                state.stackpos = state.stackpos - 1;
                let offset = state.stack[state.stackpos as usize];
                state.stackpos = state.stackpos - 1;
                let cond = state.stack[state.stackpos as usize];
                if cond != 0 {
                    let target = (pc as i64 + offset) as usize;
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            // tl.py:174-178 — `res = interp(code, pc + offset)`, a recursive
            // portal re-entry.  Greens in declaration order [pc, program];
            // the concrete fallback is `interpret_recursive` (declared via
            // `recursive_entry`), the JIT path emits BC_RECURSIVE_CALL_INT.
            CALL => {
                let offset = program[pc] as i8 as i64;
                pc += 1;
                let target = (pc as i64 + offset) as usize;
                let res = recursive_portal_call!(driver, target, program);
                state.stack[state.stackpos as usize] = res;
                state.stackpos = state.stackpos + 1;
            }
            // tl.py:180-181
            RETURN => break,
            // tl.py:183-184
            PUSHARG => {
                state.stack[state.stackpos as usize] = inputarg;
                state.stackpos = state.stackpos + 1;
            }
            _ => {}
        }
    }

    state.stackpos = state.stackpos - 1;
    state.stack[state.stackpos as usize]
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

    /// A hot loop whose body issues a recursive `CALL` to a constant-returning
    /// subroutine — exercises `recursive_portal_call!` → BC_RECURSIVE_CALL_INT
    /// end to end.  The loop runs `N` times, each iteration adds the
    /// subroutine's result (3) to the accumulator, so `interpret(prog, N) ==
    /// 3 * N`.  The JIT must match the interpreter on every input.
    fn call_loop_bytecode() -> Vec<u8> {
        vec![
            PUSHARG, // counter = N                       [counter]
            PUSH, 0, // acc = 0                           [counter, acc]
            // loop (offset 3):
            SWAP, // [acc, counter]
            PICK, 0, // dup counter   [acc, counter, counter]
            BR_COND, 2, // pop top; if counter != 0 → body (offset 10)
            // exit (counter == 0):  [acc, counter]
            POP,    // [acc]
            RETURN, // return acc
            // body (offset 10):     [acc, counter]
            SWAP, // [counter, acc]
            CALL, 10,   // call subroutine (offset 23) → [counter, acc, 3]
            ADD,  // [counter, acc+3]
            SWAP, // [acc+3, counter]
            PUSH, 1, SUB,  // counter -= 1   [acc+3, counter-1]
            SWAP, // [counter-1, acc+3]   (loop-top stack shape)
            PUSH, 1, BR_COND, 236, // -20: jump back to loop (offset 3)
            // subroutine (offset 23): fresh stack, returns 3
            PUSH, 3, RETURN,
        ]
    }

    /// [SPIKE-S0/FR] Measure whether a portal call inside a hot loop compiles
    /// through. Control = pure loop (no call). Probe = leaf call in loop body.
    #[test]
    fn spike_portal_call_compile_through() {
        use core::sync::atomic::Ordering;

        SPIKE_COMPILES.store(0, Ordering::Relaxed);
        let bc = sum_bytecode();
        let mut jit = JitTlInterp::new();
        let got = jit.run(&bc, 500);
        let control = SPIKE_COMPILES.load(Ordering::Relaxed);
        eprintln!("[SPIKE] control sum(500)={got} compiles={control}");

        SPIKE_COMPILES.store(0, Ordering::Relaxed);
        let bc = call_loop_bytecode();
        let mut jit = JitTlInterp::new();
        let got = jit.run(&bc, 500);
        let probe = SPIKE_COMPILES.load(Ordering::Relaxed);
        eprintln!("[SPIKE] probe  call_loop(500)={got} compiles={probe}");

        assert_eq!(got, 1500, "leaf-call loop still computes 3*N");
    }

    #[test]
    fn jit_recursive_call_matches_interp() {
        let bc = call_loop_bytecode();
        // Sanity: the interpreter computes 3 * N.
        assert_eq!(interp::interpret(&bc, 4), 12);
        for a in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "recursive-call mismatch for a={a}");
        }
    }

    /// #184 recursive CALL_ASSEMBLER portal entry: the macro-generated
    /// `JitCodeSym::recursive_fresh_entry_reds` yields fresh-frame reds in
    /// `extract_live` order — stackpos zeroed, a fresh vable identity Ref
    /// distinct from the caller, and the stack re-allocated at the caller's
    /// captured capacity (read from the sym's `stack_len_value` cache).
    #[test]
    fn recursive_fresh_entry_reds_layout() {
        use majit_metainterp::{JitCodeSym as _, JitState as _};
        let program = sum_bytecode();
        let caller = TlState {
            stackpos: 7,
            stack: vec![1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0],
        };
        let meta = caller.build_meta(0, program.as_slice());
        let mut sym = <TlState as majit_metainterp::JitState>::create_sym(&meta, 0);
        // Seeds `sym.stack_len_value` from the caller's live capacity.
        caller.initialize_sym(&mut sym, &meta);
        let (values, _owner) = sym
            .recursive_fresh_entry_reds()
            .expect("ref-scalar-free state-field interp must support portal entry");
        // tl extract_live order: [stackpos (Int), &state (Ref), stack.len() (Int)].
        assert_eq!(values.len(), 3, "stackpos + vable ptr + stack len");
        assert_eq!(values[0], majit_ir::Value::Int(0), "fresh stackpos zeroed");
        match values[1] {
            majit_ir::Value::Ref(majit_ir::GcRef(p)) => {
                assert_ne!(p, 0, "fresh vable base must be non-null");
                assert_ne!(
                    p, &caller as *const TlState as usize,
                    "fresh base must differ from the caller's state",
                );
            }
            ref other => panic!("slot 1 must be the vable identity Ref, got {other:?}"),
        }
        assert_eq!(
            values[2],
            majit_ir::Value::Int(caller.stack.len() as i64),
            "fresh stack re-allocated at the caller's captured capacity",
        );
    }

    /// #184 S3f-1: the host alloc/free targets the recursive dispatcher records
    /// as residual `CallR`/`CallN` for the compiled caller loop.  Exercises the
    /// macro-generated `extern "C"` pair directly through the `JitCodeSym` seam:
    /// `alloc(cap)` returns a fresh `Box::into_raw`-ed `TlState` (stackpos 0,
    /// `stack` of length `cap`, all zero), `free` drops it without crashing.
    #[test]
    fn recursive_fresh_alloc_free_roundtrip() {
        use majit_metainterp::{JitCodeSym as _, JitState as _};
        let program = sum_bytecode();
        let caller = TlState {
            stackpos: 3,
            stack: vec![9, 8, 7, 0, 0, 0, 0, 0],
        };
        let meta = caller.build_meta(0, program.as_slice());
        let mut sym = <TlState as majit_metainterp::JitState>::create_sym(&meta, 0);
        caller.initialize_sym(&mut sym, &meta);
        let (alloc_fp, free_fp) = sym
            .recursive_fresh_alloc_free_targets()
            .expect("single-virt-array state-field interp must support portal alloc/free");
        let alloc: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(alloc_fp) };
        let free: extern "C" fn(i64) = unsafe { core::mem::transmute(free_fp) };

        let cap: i64 = 12;
        let raw = alloc(cap);
        assert_ne!(raw, 0, "fresh alloc must return a non-null pointer");
        assert_ne!(
            raw as usize, &caller as *const TlState as usize,
            "fresh state must differ from the caller's state",
        );
        unsafe {
            let fresh = &*(raw as *const TlState);
            assert_eq!(fresh.stackpos, 0, "fresh stackpos zeroed");
            assert_eq!(
                fresh.stack.len(),
                cap as usize,
                "fresh stack sized at the requested capacity",
            );
            assert!(
                fresh.stack.iter().all(|&x| x == 0),
                "fresh stack zero-initialised",
            );
        }
        // Must reclaim the Box::into_raw allocation without double-free / crash.
        free(raw);
        // A null free is a no-op (the dispatcher never frees a null, but the
        // compiled guard-fail path must tolerate it).
        free(0);
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

    /// A loop whose body branches on `counter > 50` (a *forward* BR_COND, so
    /// it is an in-trace guard, not a back-edge).  Traced while `counter > 50`,
    /// it guard-fails on every iteration once `counter <= 50` — driving the
    /// state-field blackhole forward-resume path (the `!should_bridge` arm of
    /// `back_edge_internal`) rather than the clean `is_finish` loop-exit.
    /// The computed value is irrelevant; the assertion is that the JIT result
    /// equals the plain interpreter result across the guard-failure divergence.
    fn divergent_branch_bytecode() -> Vec<u8> {
        vec![
            PUSH, 0,       // [0] acc = 0
            PUSHARG, // [2] counter = N            stack = [acc, counter]
            // loop header (offset 3):
            PICK, 0, //       [3] dup counter
            BR_COND, 2,      //    [5] if counter != 0 -> body(9); else fall through
            POP,    //      [7] pop counter
            RETURN, //      [8] return acc
            // body (offset 9):
            PICK, 0, //       [9]  dup counter            [acc, ctr, ctr]
            PUSH, 50, //      [11] push 50                [acc, ctr, ctr, 50]
            GT, //      [13] ctr > 50 ?             [acc, ctr, (ctr>50)]
            BR_COND, 5, //    [14] if ctr>50 -> skip_extra(21)
            // not-taken (ctr <= 50): acc += 1 (the divergent path)
            SWAP, //          [16] [ctr, acc]
            PUSH, 1,    //       [17] [ctr, acc, 1]
            ADD,  //          [19] [ctr, acc+1]
            SWAP, //          [20] [acc+1, ctr]
            // skip_extra (offset 21): common tail — acc += counter; counter -= 1
            SWAP, //          [21] [ctr, acc]
            PICK, 1,    //       [22] [ctr, acc, ctr]
            ADD,  //          [24] [ctr, acc+ctr]
            SWAP, //          [25] [acc+ctr, ctr]
            PUSH, 1,   //       [26] [acc+ctr, ctr, 1]
            SUB, //          [28] [acc+ctr, ctr-1]
            PUSH, 1, //       [29] push 1 (unconditional back-jump cond)
            BR_COND, 226, //  [31] -30: jump to loop header(3)
        ]
    }

    #[test]
    fn jit_divergent_branch_matches_interp() {
        let bc = divergent_branch_bytecode();
        // N spans both sides of the `counter > 50` split so the traced
        // (`> 50`) path guard-fails for the lower half of every run.
        for a in [3, 5, 49, 50, 51, 60, 100, 200] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }
}
