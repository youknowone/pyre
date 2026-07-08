//! Single-pass whole-circuit-close regression example.
//!
//! Every other `majit/examples` interpreter uses the plain
//! `jit_merge_point!()` marker. This one uses the `; state` selector form
//! (`jit_merge_point!(driver, program, pc; state)`) — the single-pass
//! whole-circuit-close path. When the walk closes a loop, the merge-point hook
//! writes the walk-final scalar state field(s) back into native `state`,
//! re-derives storage-backed caches via `recover` (a no-op here — no
//! storage-derived caches), then direct-enters the compiled loop rather than
//! replaying the walked body. That path is the sole/default trace-close path
//! after issue #344 Phase B, yet inside this repository only aheui — a separate
//! git repo, outside CI — exercised it. This crate closes that coverage gap.
//!
//! The interpreter is a minimal stack machine, structurally the tl example's
//! virtualizable-stack shape (`state_fields = { stackpos: int, stack: [int;
//! virt] }`): a scalar `stackpos` the loop mutates plus a loop-carried virt
//! array. That is the shape whose single-pass close is known to converge (a
//! scalar-only interpreter with a residual adjacent to a tight back-edge
//! collapses its per-opcode merge points and never reaches the loop header).
//!
//! ## Reds regime
//!
//! The loop-carried reds are recovered through the state-field side channel:
//! `stackpos` is written back by `writeback_scalar_state_fields_from_sym` and
//! the virt `stack` lives on the heap the compiled code mutates in place. This
//! is the EMPTY-reds case the driver publishes at CloseLoop
//! (`single_pass_outcome = Some((pc, Vec::new()))` in `jitdriver.rs`, where the
//! empty reds vector is documented as INTENTIONALLY empty). The macro's
//! non-empty-reds transfer branch (`if !__sp_reds.is_empty()` — a
//! `restore_values` into native state) therefore stays UNEXERCISED: the driver
//! hard-codes empty reds at the close, so no example crate can drive that
//! branch without changing the driver/macro. A genuinely register-resident,
//! non-storage-recoverable loop-carried red is not expressible through this
//! macro surface — every loop-carried value must land in a declared state
//! field, which the state-field write-back (not `__sp_reds`) transfers. Adding
//! coverage for the non-empty-reds branch is a driver/macro change, out of
//! scope for an example crate.

/// Bytecode stream. Byte-wide opcodes/operands, same shape as the tl env.
pub type Bytecode = [u8];

// ── Opcodes ──
const PUSH: u8 = 2; // [PUSH, imm]: push a signed-byte immediate
const POP: u8 = 3; // pop top
const SWAP: u8 = 4; // swap the top two
const PICK: u8 = 6; // [PICK, i]: duplicate stack[stackpos - i - 1]
const ADD: u8 = 8; // pop a, b; push b + a
const SUB: u8 = 9; // pop a, b; push b - a
const BR_COND: u8 = 18; // [BR_COND, off]: pop cond; if cond != 0 jump
const RETURN: u8 = 21; // return top
const PUSHARG: u8 = 22; // push the input argument
const TOUCH: u8 = 30; // residual: side-effecting, result-neutral stack touch

// ── Countable side-effecting residual ──

/// Number of `touch` invocations, observed by the tests. A walk-vs-native
/// double-execution of the residual during single-pass tracing would inflate
/// this beyond the interpreter's count.
#[cfg(test)]
static TOUCH_CALLS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Side-effecting residual, `@dont_look_inside` — the JIT does not trace into
/// it; it emits a residual CALL. `#[dont_look_inside]` is non-elidable and may
/// raise, so the optimizer keeps the call. It reads and rewrites the live
/// top-of-stack through the raw stack pointer: a genuine heap side effect that
/// forces the virtualizable stack, yet leaves every value unchanged. The
/// computed result is therefore independent of how many times `touch` runs, so
/// the call count alone is the double-execution detector. Modelled on tl's
/// `storage_roll`.
#[majit_macros::dont_look_inside]
extern "C" fn touch(stack_ptr: usize, stackpos: i64) {
    #[cfg(test)]
    TOUCH_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    if stackpos > 0 {
        let stack =
            unsafe { std::slice::from_raw_parts_mut(stack_ptr as *mut i64, stackpos as usize) };
        let top = stack[(stackpos - 1) as usize];
        stack[(stackpos - 1) as usize] = top;
    }
}

// ── State ──

/// Virtualizable stack: a scalar `stackpos` plus a loop-carried virt array.
struct StackState {
    stackpos: i64,
    stack: Vec<i64>,
}

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = StackState,
    env = Bytecode,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<StackState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let stacksize: i32 = 0;
    let mut state = StackState {
        stackpos: 0,
        stack: vec![0i64; program.len()],
    };

    // warmspot.py:281-289 canonical-liveness install hook.
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        // `; state` selects the single-pass close: the walk's final state is
        // transferred into `state` here (write-back + recover) instead of being
        // replayed. Byte-identical to `jit_merge_point!()` until the walk closes
        // a loop.
        jit_merge_point!(driver, program, pc; state);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos = state.stackpos + 1;
            }
            POP => {
                state.stackpos = state.stackpos - 1;
            }
            SWAP => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 1) as usize] = b;
                state.stack[(state.stackpos - 2) as usize] = a;
            }
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let v = state.stack[(state.stackpos as usize) - i - 1];
                state.stack[state.stackpos as usize] = v;
                state.stackpos = state.stackpos + 1;
            }
            ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos = state.stackpos - 1;
            }
            SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos = state.stackpos - 1;
            }
            // Residual @dont_look_inside touch of the live stack.
            TOUCH => {
                touch(state.stack.as_mut_ptr() as usize, state.stackpos);
            }
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
            RETURN => break,
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

// ── Plain reference interpreter ──

/// The same bytecode executed with no JIT. `TOUCH` is a result-neutral no-op
/// here (it does not call the counted residual), so the plain result equals the
/// JIT result and the residual count stays a pure JIT-side signal — mirroring
/// how tl's `interp` uses an uncounted `roll`.
pub fn interp(program: &Bytecode, inputarg: i64) -> i64 {
    let mut pc: usize = 0;
    let mut stack: Vec<i64> = Vec::with_capacity(program.len());

    while pc < program.len() {
        let opcode = program[pc];
        pc += 1;
        match opcode {
            PUSH => {
                stack.push(program[pc] as i8 as i64);
                pc += 1;
            }
            POP => {
                stack.pop();
            }
            SWAP => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a);
                stack.push(b);
            }
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let n = stack.len() - i - 1;
                let v = stack[n];
                stack.push(v);
            }
            ADD => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(b + a);
            }
            SUB => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(b - a);
            }
            TOUCH => {}
            BR_COND => {
                let offset = program[pc] as i8 as i64;
                let cond = stack.pop().unwrap();
                if cond != 0 {
                    pc = (pc as i64 + offset + 1) as usize;
                } else {
                    pc += 1;
                }
            }
            RETURN => break,
            PUSHARG => {
                stack.push(inputarg);
            }
            _ => {}
        }
    }
    stack.pop().unwrap()
}

/// `sum(N) = N + (N-1) + ... + 1`, a hot loop with no residual — used by the
/// output-match / smoke tests and `main`. The result is `N*(N+1)/2`. Identical
/// to tl's sum loop; it carries no `TOUCH` so it never perturbs the residual
/// counter (the tests run in parallel and share `TOUCH_CALLS`).
///
///   0: PUSH 0            [0]           acc = 0
///   2: PUSHARG           [0, N]        counter = N
///   loop header @ 3:
///   3: PICK 0            [acc, c, c]   dup counter
///   5: BR_COND 2         [acc, c]      if c != 0 -> body @9 (pops the dup)
///   7: POP               [acc]         c == 0: drop counter
///   8: RETURN                          return acc
///   body @ 9:
///   9: SWAP              [c, acc]
///   10: PICK 1           [c, acc, c]
///   12: ADD              [c, acc+c]
///   13: SWAP             [acc+c, c]
///   14: PUSH 1 SUB       [acc, c-1]    decrement counter
///   17: PUSH 1           [acc, c-1, 1] unconditional back-jump cond
///   19: BR_COND 238      -> jump to loop header @3
fn sum_program() -> Vec<u8> {
    vec![
        PUSH, 0,       // 0
        PUSHARG, // 2
        PICK, 0, // 3  (loop header)
        BR_COND, 2,      // 5  -> body @9
        POP,    // 7
        RETURN, // 8
        SWAP,   // 9
        PICK, 1,    // 10
        ADD,  // 12
        SWAP, // 13
        PUSH, 1, SUB, // 14
        PUSH, 1, // 17
        BR_COND, 238, // 19 offset byte @20 -> target = 20 + (-18) + 1 = 3
    ]
}

/// The sum loop with a leading `TOUCH` residual in the body — used only by the
/// residual-count test. `touch` fires exactly once per iteration, so the loop
/// runs it exactly N times. The result is still `N*(N+1)/2` because `touch` is
/// result-neutral.
///
///   body @ 9:
///   9: TOUCH             [acc, c]      residual (result-neutral)
///   10: SWAP             [c, acc]
///   11: PICK 1           [c, acc, c]
///   13: ADD              [c, acc+c]
///   14: SWAP             [acc+c, c]
///   15: PUSH 1 SUB       [acc, c-1]    decrement counter
///   18: PUSH 1           [acc, c-1, 1] unconditional back-jump cond
///   20: BR_COND 237      -> jump to loop header @3
#[cfg(test)]
fn touch_loop_program() -> Vec<u8> {
    vec![
        PUSH, 0,       // 0
        PUSHARG, // 2
        PICK, 0, // 3  (loop header)
        BR_COND, 2,      // 5  -> body @9
        POP,    // 7
        RETURN, // 8
        TOUCH,  // 9  residual
        SWAP,   // 10
        PICK, 1,    // 11
        ADD,  // 13
        SWAP, // 14
        PUSH, 1, SUB, // 15
        PUSH, 1, // 18
        BR_COND, 237, // 20 offset byte @21 -> target = 21 + (-19) + 1 = 3
    ]
}

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let program = sum_program();
    let result = mainloop(&program, n, 3);
    println!("sum({n}) [single-pass JIT] = {result}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::Ordering;

    /// The plain interpreter and the single-pass JIT mainloop must compute the
    /// identical result across a range of inputs (each exercises the CloseLoop
    /// single-pass close).
    #[test]
    fn jit_output_matches_interp() {
        let program = sum_program();
        for n in [1_i64, 2, 3, 5, 10, 20, 50, 100, 200] {
            let expected = interp(&program, n);
            let got = mainloop(&program, n, 3);
            assert_eq!(got, n * (n + 1) / 2, "sum({n}) closed form");
            assert_eq!(got, expected, "JIT diverged from interp for n={n}");
        }
    }

    /// THE regression this crate exists to catch. The residual `touch` fires
    /// exactly once per loop iteration, and the loop runs a known number of
    /// times. Under single-pass tracing (the walk is the sole executor) the
    /// residual must run exactly the interpreter's count — a walk-vs-native
    /// double-execution during the trace-then-close would inflate it.
    #[test]
    fn jit_residual_not_double_executed() {
        let program = touch_loop_program();
        let n: i64 = 50;

        let expected = interp(&program, n);
        TOUCH_CALLS.store(0, Ordering::Relaxed);
        let got = mainloop(&program, n, 3);
        let jit_touches = TOUCH_CALLS.load(Ordering::Relaxed);

        assert_eq!(got, expected, "JIT result diverged from interp");
        // One TOUCH per iteration; N iterations before the counter hits 0.
        let expected_touches = n as u32;
        assert_eq!(
            jit_touches, expected_touches,
            "residual touch executed {jit_touches}× but the loop runs exactly \
             {expected_touches} iterations — a walk-vs-native double-execution \
             during single-pass tracing would inflate this count"
        );
    }

    /// Smoke test: a program with no back-edge never enters the JIT.
    #[test]
    fn jit_no_loop() {
        let program = vec![PUSH, 42, RETURN];
        assert_eq!(mainloop(&program, 0, 3), 42);
    }
}
