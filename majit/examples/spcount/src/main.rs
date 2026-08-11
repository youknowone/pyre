//! Single-pass whole-circuit-close regression example.
//!
//! This stack machine uses the `jit_merge_point!(...; state)` form. On loop
//! close, the walk-final scalar state is written back and the compiled loop is
//! entered directly. The `TOUCH` residual lets tests detect accidental
//! execution by both the trace walk and the native interpreter.

/// Bytecode stream. Byte-wide opcodes/operands, same shape as the tl env.
pub type Bytecode = [u8];

// Opcodes
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

// Countable side-effecting residual

/// Number of `touch` invocations, observed by the tests. A walk-vs-native
/// double-execution of the residual during single-pass tracing would inflate
/// this beyond the interpreter's count.
#[cfg(test)]
static TOUCH_CALLS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Number of loops the driver compiled/closed, observed by the tests. The
/// residual-count canary only exercises the single-pass close if a trace
/// actually compiled; this counter lets the test assert that it did, so a run
/// that never starts tracing (or aborts before the `; state` close) fails
/// loudly instead of passing vacuously. Mirrors tl's `SPIKE_COMPILES`.
static SPCOUNT_COMPILES: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Optimized op count of the most recently compiled loop body.
///
/// A compile *count* says a trace closed; it does not say the trace did any
/// work. An entirely empty dispatch still compiles one trace whose whole
/// optimized body is `Finish()` — `ops_after == 1` — and that degenerate body
/// satisfies every inequality a real loop satisfies. `SPCOUNT_COMPILES` alone
/// therefore cannot tell a live tier from a hollow one, which is why the third
/// callback parameter is captured here instead of discarded.
static SPCOUNT_LAST_OPS_AFTER: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// Loop-shape flags recorded with [`SPCOUNT_LAST_OPS_AFTER`]. They distinguish
/// a body that reaches its back edge from an empty or always-failing body
/// without relying only on a measured operation count.
static SPCOUNT_LAST_HAS_JUMP: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);
static SPCOUNT_LAST_ALWAYS_FAILS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Side-effecting residual, `@dont_look_inside` — the JIT does not trace into
/// it; it emits a residual CALL. `#[dont_look_inside]` is non-elidable and may
/// raise, so the optimizer keeps the call. It is result-neutral (its only
/// observable effect is the counted call), so the computed sum is independent
/// of how many times `touch` runs and the call count alone is the
/// double-execution detector. The argument is the scalar `stackpos` — a
/// jit-lowerable value, so the TOUCH arm compiles into the trace (a virt-array
/// base-pointer argument would degrade to a `BC_ABORT` stub and never compile).
/// Modelled on tl's `storage_roll`.
#[majit_macros::dont_look_inside]
extern "C" fn touch(stackpos: i64) {
    #[cfg(test)]
    TOUCH_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let _ = stackpos;
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
    // Count compiled loops so the residual-count test can assert the
    // single-pass close actually ran. Mirrors tl's SPIKE_COMPILES hook.
    //
    // The third parameter is the optimized op count of the closed body. It is
    // captured rather than discarded because the count alone cannot separate a
    // real compiled loop from a bare `Finish()` — see SPCOUNT_LAST_OPS_AFTER.
    driver.set_on_compile_loop(|_gk, _before, after, opcodes| {
        SPCOUNT_COMPILES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        SPCOUNT_LAST_OPS_AFTER.store(after, core::sync::atomic::Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape::of(opcodes);
        SPCOUNT_LAST_HAS_JUMP.store(shape.has_jump, core::sync::atomic::Ordering::Relaxed);
        SPCOUNT_LAST_ALWAYS_FAILS.store(
            shape.has_always_fails,
            core::sync::atomic::Ordering::Relaxed,
        );
    });
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
                state.stackpos += 1;
            }
            POP => {
                state.stackpos -= 1;
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
                state.stackpos += 1;
            }
            ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos -= 1;
            }
            SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos -= 1;
            }
            // Residual @dont_look_inside touch of the live stack.
            TOUCH => {
                touch(state.stackpos);
            }
            BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                state.stackpos -= 1;
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
                state.stackpos += 1;
            }
            _ => {}
        }
    }

    state.stackpos -= 1;
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

/// Two loops in ONE program, at two different headers, so one run holds two
/// distinct green keys. `greens = [pc, program]` keys a merge point by the
/// header it was reached at, so the inner header @11 and the outer header @3
/// are different keys with the same `program` — which is the shape no other
/// example crate has. spcount already compiled two keys, but they came from two
/// different *programs* in two different tests; nothing could ever be already
/// compiled while another loop was being traced.
///
/// The inner loop runs `K` times per outer iteration, so at `threshold = 3` it
/// reaches its trip count first and compiles while the outer loop is still
/// cold. When the outer back edge later arms and its walk reaches the inner
/// header, `has_compiled_targets_fn` is true there — the `already_compiled_here`
/// branch (`dispatch.rs:5798`) publishes `close_jump_into_key`, which is the
/// cross-loop close.
///
/// The inner loop is a pure spin (it only decrements its own counter), so the
/// result is the same `N*(N+1)/2` the sum fixture computes and can be checked
/// against [`interp`] rather than against a hand-computed constant.
///
///    0: PUSH 0            [acc]
///    2: PUSHARG           [acc, i]
///    3: PICK 0            OUTER header  [acc, i, i]
///    5: BR_COND +2        -> outer body @9      [acc, i]
///    7: POP               [acc]
///    8: RETURN
///    9: PUSH K            outer body: j = K     [acc, i, j]
///   11: PICK 0            INNER header  [acc, i, j, j]
///   13: BR_COND +5        -> inner body @20     [acc, i, j]
///   15: POP               inner exit: drop j==0 [acc, i]
///   16: PUSH 1            [acc, i, 1]
///   18: BR_COND +7        -> outer tail @27     [acc, i]
///   20: PUSH 1 SUB        inner body: j -= 1    [acc, i, j-1]
///   23: PUSH 1
///   25: BR_COND -16       -> INNER header @11
///   27: SWAP              outer tail            [i, acc]
///   28: PICK 1            [i, acc, i]
///   30: ADD               [i, acc+i]
///   31: SWAP              [acc+i, i]
///   32: PUSH 1 SUB        i -= 1                [acc, i-1]
///   35: PUSH 1
///   37: BR_COND -36       -> OUTER header @3
#[cfg(test)]
fn nested_loop_program(k: i8) -> Vec<u8> {
    vec![
        PUSH, 0,       // 0
        PUSHARG, // 2
        PICK, 0, // 3   outer header
        BR_COND, 2,      // 5   offset @6  -> 6 + 2 + 1 = 9
        POP,    // 7
        RETURN, // 8
        PUSH, k as u8, // 9   j = K
        PICK, 0, // 11  inner header
        BR_COND, 5,   // 13  offset @14 -> 14 + 5 + 1 = 20
        POP, // 15  inner exit
        PUSH, 1, // 16
        BR_COND, 7, // 18  offset @19 -> 19 + 7 + 1 = 27
        PUSH, 1, SUB, // 20  inner body
        PUSH, 1, // 23
        BR_COND, 240,  // 25  offset @26 -> 26 + (-16) + 1 = 11
        SWAP, // 27  outer tail
        PICK, 1,    // 28
        ADD,  // 30
        SWAP, // 31
        PUSH, 1, SUB, // 32
        PUSH, 1, // 35
        BR_COND, 220, // 37  offset @38 -> 38 + (-36) + 1 = 3
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
    use majit_metainterp::{RefusalKind, refusal_kind};

    /// Serializes the tier probe against every other test that runs the JIT.
    ///
    /// `SPCOUNT_COMPILES` and `SPCOUNT_LAST_OPS_AFTER` are process-wide, and
    /// `libtest` runs these tests on parallel threads. Both counters must be
    /// zeroed and read inside one window or a concurrent test's compile lands
    /// between the reset and the load. `SPCOUNT_LAST_OPS_AFTER` needs this even
    /// more than the count does: it is last-writer-wins, so without the lock it
    /// can report another fixture's body size, which is a *plausible* number and
    /// therefore will not look wrong.
    ///
    /// The lock only works if EVERY test that enters the JIT takes it, not
    /// just the probe — a one-sided lock serializes nothing. This was not
    /// hypothetical: with `jit_output_matches_interp` still calling `mainloop`
    /// directly, the probe read `2 compile(s)` for a fixture that compiles
    /// exactly one, and could have pinned that test's 13-op body instead of this
    /// one's 17. Hence [`run_jit`]. Neither helper may call the other — a plain
    /// mutex re-entered on one thread deadlocks.
    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// For tests that assert only on the result. They still compile, so they
    /// must not run inside the probe's window. See [`PROBE_LOCK`].
    fn run_jit(program: &[u8], inputarg: i64) -> i64 {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        mainloop(program, inputarg, 3)
    }

    fn compile_probe(
        program: &[u8],
        inputarg: i64,
    ) -> (i64, u32, u32, usize, majit_metainterp::LoopBodyShape) {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        TOUCH_CALLS.store(0, Ordering::Relaxed);
        SPCOUNT_COMPILES.store(0, Ordering::Relaxed);
        SPCOUNT_LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        // Reset to the values `closes_a_loop()` rejects, so a hook that never
        // fires fails the shape assertion instead of passing it untouched.
        SPCOUNT_LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        SPCOUNT_LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        let got = mainloop(program, inputarg, 3);
        (
            got,
            TOUCH_CALLS.load(Ordering::Relaxed),
            SPCOUNT_COMPILES.load(Ordering::Relaxed),
            SPCOUNT_LAST_OPS_AFTER.load(Ordering::Relaxed),
            majit_metainterp::LoopBodyShape {
                has_jump: SPCOUNT_LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: SPCOUNT_LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        )
    }

    /// The plain interpreter and the single-pass JIT mainloop must compute the
    /// identical result across a range of inputs (each exercises the CloseLoop
    /// single-pass close).
    #[test]
    fn jit_output_matches_interp() {
        let program = sum_program();
        for n in [1_i64, 2, 3, 5, 10, 20, 50, 100, 200] {
            let expected = interp(&program, n);
            let got = run_jit(&program, n);
            assert_eq!(got, n * (n + 1) / 2, "sum({n}) closed form");
            assert_eq!(got, expected, "JIT diverged from interp for n={n}");
        }
    }

    #[test]
    fn jit_residual_not_double_executed() {
        let program = touch_loop_program();
        let n: i64 = 50;

        let expected = interp(&program, n);
        let (got, jit_touches, compiles, ops_after, shape) = compile_probe(&program, n);

        // The residual-count canary is only meaningful if a trace actually
        // compiled and closed via the `; state` single-pass path. Without this
        // guard the loop could merely interpret (no tracing, or an abort before
        // the close), run `touch` exactly n× anyway, and leave the count green
        // vacuously.
        assert!(
            compiles >= 1,
            "single-pass trace never compiled — canary would pass vacuously"
        );

        assert!(
            shape.closes_a_loop(),
            "compiled {compiles} trace(s) but the body {} ({shape:?}) — the \
             canary would count n residual calls the interpreter made, not the \
             JIT",
            shape.why_not().unwrap_or("closes a loop")
        );

        assert_eq!(
            ops_after, 17,
            "compiled loop body is {ops_after} ops across {compiles} compile(s), \
             not the pinned 17 — a value of 1 means the body is a bare \
             `Finish()`, i.e. a dispatch that lowered nothing at all"
        );

        // Equality over a NAMED set, never `is_empty()`: PUSHARG is a known
        // abort stub here (`state.stack[state.stackpos as usize] = inputarg;` —
        // the lowerer cannot express the store of a loop-external input), so an
        // emptiness check would be red on day one and the natural response would
        // be to weaken it. Pinning the set instead means a SECOND arm degrading
        // is a failure rather than a silent addition, and PUSHARG lowering again
        // is also a failure — the prompt to re-measure `ops_after` above.
        let mut sp_arms: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "StackState")
            .collect();
        sp_arms.sort_unstable_by_key(|a| a.arm);
        let degraded: Vec<&str> = sp_arms.iter().map(|a| a.arm).collect();
        assert_eq!(
            degraded,
            ["PUSHARG"],
            "the degraded-arm set moved; every trace reaching an abort stub aborts"
        );

        // The CAUSE, which the name set above cannot see: the comment on the
        // name pin says PUSHARG degrades because the lowerer cannot express the
        // store of a loop-external input. That was prose; this asserts it.
        let causes: Vec<(&str, RefusalKind)> = sp_arms
            .iter()
            .map(|a| (a.arm, refusal_kind(a.reason)))
            .collect();
        assert_eq!(
            causes,
            [("PUSHARG", RefusalKind::UnlowerableStmt)],
            "PUSHARG still degrades but a different mechanism is refusing it. \
             `RefusalKind::Unclassified` means majit grew a refusal family the \
             classifier does not know — add it in `majit-metainterp`, do not \
             re-record this pin"
        );
        assert!(
            sp_arms[0].reason.contains("inputarg"),
            "PUSHARG's refusal no longer names the loop-external input it \
             stores: {}",
            sp_arms[0].reason
        );

        assert_eq!(got, expected, "JIT result diverged from interp");
        // One TOUCH per iteration; N iterations before the counter hits 0.
        let expected_touches = n as u32;
        assert_eq!(
            jit_touches, expected_touches,
            "residual touch executed {jit_touches}× but the loop runs exactly \
             {expected_touches} iterations — a walk-vs-native double-execution \
             during single-pass tracing would inflate this count"
        );
        println!(
            "[tier-alive] touch_loop({n}) = {got}, compiled {compiles} loop(s) of \
             {ops_after} ops, {jit_touches} residual calls, degraded {degraded:?}"
        );
    }

    /// Smoke test: a program with no back-edge never enters the JIT.
    #[test]
    fn jit_no_loop() {
        let program = vec![PUSH, 42, RETURN];
        assert_eq!(run_jit(&program, 0), 42);
    }

    /// The nested program must compute what the plain interpreter computes.
    ///
    /// This is the precondition for reading anything else off
    /// [`nested_loop_program`]: a bytecode with two loops is easy to get subtly
    /// wrong (every `BR_COND` offset is relative to its own operand byte), and
    /// a wrong program that happens to compile two loops would look exactly
    /// like a right one to the tier assertions.
    #[test]
    fn nested_jit_output_matches_interp() {
        let program = nested_loop_program(3);
        for n in [1_i64, 2, 3, 5, 10, 20] {
            let expected = interp(&program, n);
            assert_eq!(
                run_jit(&program, n),
                expected,
                "nested_loop_program({n}) diverged from the plain interpreter",
            );
            assert_eq!(
                expected,
                n * (n + 1) / 2,
                "the inner loop must be a pure spin, leaving the outer sum unchanged",
            );
        }
    }

    #[test]
    #[ignore = "end-state gate: the outer loop does not trace once the inner loop is compiled"]
    fn nested_loops_compile_two_keys_in_one_run() {
        let program = nested_loop_program(3);
        let (got, _touches, compiles, _ops_after, _shape) = compile_probe(&program, 8);
        assert_eq!(got, 36, "sum(8) = 36");
        assert!(
            compiles >= 2,
            "both the inner header @11 and the outer header @3 must compile in \
             one run — got {compiles} compile(s); with only one, no loop is ever \
             already-compiled while another is traced and the cross-loop close \
             is unreachable by construction",
        );
    }
}
