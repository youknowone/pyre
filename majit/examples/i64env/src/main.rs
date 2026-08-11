//! `[i64]`-env regression example.
//!
//! Proves the `#[jit_interp]` macro reads an `env` whose element type is wider
//! than a byte (`pub type Code = [i64];`) at the correct stride. The macro
//! lowers every `program[pc + N]` read with a descr whose `item_size` matches
//! the env element (`size_of::<<Code as Index<usize>>::Output>()` = 8), so the
//! load scales the index by 8 instead of reading a stray byte. A byte-wide
//! descr (the previous hardcoding) would read the wrong word and miscompile.
//!
//! The register file is `[int; virt]` because it is loop-carried: a plain
//! `[int]` array element is not restored on a CloseLoop guard deopt. (See the
//! macro's loop-carried-plain-array diagnostic.)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The env: an i64-word bytecode stream. The whole point of this example is
/// that the element is 8 bytes wide, not 1.
pub type Code = [i64];

// Opcodes and operands are full i64 words. Values can exceed a byte to make a
// byte-stride miscompile observable.
const OP_LOAD: i64 = 0; // [LOAD, imm, dst]
const OP_ADD: i64 = 1; // [ADD, a, b, dst]
const OP_JUMP_IF_ABOVE: i64 = 2; // [JIA, a, b, target_pc]
const OP_RETURN: i64 = 3; // [RETURN, reg]

/// Hot loops majit compiled — evidence the JIT tier traced + compiled.
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Ops in the last compiled loop body after optimization. `COMPILES` counts
/// TRACES, not WORK: an entirely empty dispatch still compiles one, whose whole
/// optimized body is `Finish()` — `ops_after == 1`. Pinning this value is what
/// separates a real body from that degenerate one.
pub static LAST_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);

/// Shape of the last compiled loop body — see [`majit_metainterp::LoopBodyShape`].
///
/// Held as two flags rather than the struct itself so the recording stays
/// lock-free on the compile path; the probe rebuilds the struct inside the same
/// lock window it reads the counters in, because this is as process-global as
/// they are.
pub static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
pub static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

struct VmState {
    regs: Vec<i64>,
    /// What `OP_RETURN` hands back. The `; state` merge point leaves the
    /// dispatch loop through `break` before the in-arm `return` can run, so the
    /// result has to arrive in `state` and be returned by the post-loop tail.
    ret: i64,
}

#[majit_macros::jit_interp(
    state = VmState,
    env = Code,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
        ret: int,
    },
)]
fn mainloop(program: &Code, num_regs: usize, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<VmState> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _ops_before, ops_after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
        LAST_OPS_AFTER.store(ops_after, Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape::of(opcodes);
        LAST_HAS_JUMP.store(shape.has_jump, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(shape.has_always_fails, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let _stacksize: i32 = 0;
    let mut state = VmState {
        regs: vec![0; num_regs],
        ret: 0,
    };

    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    loop {
        jit_merge_point!(driver, program, pc; state);
        if pc == 0 {
            can_enter_jit!(driver, pc, &mut state, program, || {});
        }
        let opcode = program[pc];
        match opcode {
            OP_LOAD => {
                let val = program[pc + 1];
                let reg = program[pc + 2] as usize;
                state.regs[reg] = val;
                pc += 3;
            }
            OP_ADD => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] + state.regs[b];
                pc += 4;
            }
            OP_JUMP_IF_ABOVE => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let tgt = program[pc + 3] as usize;
                if state.regs[a] > state.regs[b] {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            OP_RETURN => {
                let r = program[pc + 1] as usize;
                state.ret = state.regs[r];
                return state.ret;
            }
            _ => panic!("fell off end of code"),
        }
    }
    state.ret
}

/// `r0` counts up by `step` while `n` > `r0`, exiting at `r0 >= n`. `step` and
/// `n` are full i64 words (intentionally > 255 in tests).
fn count_program(n: i64, step: i64) -> Vec<i64> {
    vec![
        OP_LOAD,
        step,
        1, // r1 = step
        OP_LOAD,
        n,
        2, // r2 = n
        OP_LOAD,
        0,
        0, // r0 = 0
        // @l1 = pc 9
        OP_ADD,
        0,
        1,
        0, // r0 = r0 + r1
        OP_JUMP_IF_ABOVE,
        2,
        0,
        9, // if r2 > r0 goto @l1
        OP_RETURN,
        0, // return r0
    ]
}

fn run(program: &Code, num_regs: usize, threshold: u32) -> i64 {
    mainloop(program, num_regs, threshold)
}

fn main() {
    let result = run(&count_program(1000, 1), 3, 3);
    println!("count to 1000 (step 1) = {result}");
}

/// Same computation as `count_program`, but the loop header is at **pc 0** —
/// the portal entry pc. `r0` is never reset, so the two `OP_LOAD`s re-running
/// each iteration are idempotent and the result is still `n`.
///
///   pc  0: LOAD 1 -> r1
///   pc  3: LOAD n -> r2
///   pc  6: ADD  r0 = r0 + r1
///   pc 10: JIA  if r2 > r0 goto 0      <- back edge to pc 0
///   pc 14: RETURN r0
/// Straight-line, no jump at all: control never returns to pc 0.
///
///   pc  0: LOAD 111 -> r1
///   pc  3: LOAD 222 -> r2
///   pc  6: ADD  r0 = r1 + r2
///   pc 10: RETURN r0            (= 333)
#[cfg(test)]
fn straight_line_program() -> Vec<i64> {
    vec![
        OP_LOAD, 111, 1, // r1 = 111
        OP_LOAD, 222, 2, // r2 = 222
        OP_ADD, 1, 2, 0, // r0 = r1 + r2
        OP_RETURN, 0,
    ]
}

/// The traced program's terminal is its FIRST instruction: the walk is armed
/// at pc 0 and the very next thing it executes is the interpreter's `return`.
/// Isolates the portal-return termination path from everything else.
#[cfg(test)]
fn return_immediately() -> Vec<i64> {
    vec![OP_RETURN, 0]
}

// One instruction per source line, operands beside their opcode. rustfmt would
// put each element on its own line, which detaches every `// r1 = 1` comment
// from the row it annotates.
#[rustfmt::skip]
#[cfg(test)]
fn loop_header_at_zero(n: i64) -> Vec<i64> {
    vec![
        OP_LOAD, 1, 1, // r1 = 1
        OP_LOAD, n, 2, // r2 = n
        OP_ADD, 0, 1, 0, // r0 = r0 + r1
        OP_JUMP_IF_ABOVE, 2, 0, 0, // if r2 > r0 goto 0
        OP_RETURN, 0,
    ]
}

/// Byte-identical to `loop_header_at_zero` except the back edge targets **pc 3**
/// instead of pc 0. The discriminant for "is pc 0 special, or just early?".
#[rustfmt::skip]
#[cfg(test)]
fn loop_header_at_three(n: i64) -> Vec<i64> {
    vec![
        OP_LOAD, 1, 1, // r1 = 1
        OP_LOAD, n, 2, // r2 = n            <- header
        OP_ADD, 0, 1, 0, // r0 = r0 + r1
        OP_JUMP_IF_ABOVE, 2, 0, 3, // if r2 > r0 goto 3
        OP_RETURN, 0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `program` with the counter reset, returning `(result, compiles)`
    /// observed under [`PROBE_LOCK`].
    fn probe(label: &str, program: &Code, num_regs: usize, threshold: u32) -> (i64, usize) {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        let r = run(program, num_regs, threshold);
        let compiles = COMPILES.load(Ordering::Relaxed);
        eprintln!("[probe] {label} result={r} COMPILES={compiles}");
        (r, compiles)
    }

    /// How many unroll-free fallback compiles `jit_tier_is_alive` currently
    /// sees — see the block at its assertion. `MC_DIAG` slot 73.
    const EXPECT_UNPEELED: u64 = 1;

    /// Like [`probe`], but also reports `LAST_OPS_AFTER`, read *inside* the
    /// lock.
    ///
    /// Reading it after the guard drops would be a race for exactly the reason
    /// [`PROBE_LOCK`] exists: `LAST_OPS_AFTER` is process-global, so another
    /// test's compile can land between `run` returning and the load, and the
    /// gate would then pin a body it never ran. Both counters are reset inside
    /// the lock too, so a zero here means *this* run compiled nothing rather
    /// than inheriting a previous run's value.
    ///
    /// Must not be called from [`probe`] or [`run_locked`], nor they from it:
    /// [`PROBE_LOCK`] is a plain mutex and re-entering it on one thread
    /// deadlocks.
    fn probe_with_ops(
        label: &str,
        program: &Code,
        num_regs: usize,
        threshold: u32,
    ) -> (i64, usize, usize, u64, majit_metainterp::LoopBodyShape) {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        // Slot 72 is process-global and CUMULATIVE, so it is read as a delta
        // across this run and inside `PROBE_LOCK` — the same window the other
        // counters are reset in. An absolute read would carry every other
        // test's compiles, which is the defect `PROBE_LOCK` exists to prevent.
        let unpeeled_before = majit_metainterp::mc_diag(73);
        let r = run(program, num_regs, threshold);
        let unpeeled = majit_metainterp::mc_diag(73) - unpeeled_before;
        let compiles = COMPILES.load(Ordering::Relaxed);
        let ops_after = LAST_OPS_AFTER.load(Ordering::Relaxed);
        eprintln!(
            "[probe] {label} result={r} COMPILES={compiles} OPS_AFTER={ops_after} \
             UNPEELED={unpeeled}"
        );
        (
            r,
            compiles,
            ops_after,
            unpeeled,
            majit_metainterp::LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        )
    }

    /// For tests that assert only on the result: they still compile, so they
    /// must not run inside a probe's window. See [`PROBE_LOCK`].
    fn run_locked(program: &Code, num_regs: usize, threshold: u32) -> i64 {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        run(program, num_regs, threshold)
    }

    /// Loop header at pc 0 (the portal entry pc). Same arithmetic as the
    /// pc-3 and pc-9 variants; only the back-edge target differs.
    #[test]
    fn header_at_zero() {
        let (r, _) = probe("header_at_zero", &loop_header_at_zero(1000), 3, 3);
        assert_eq!(r, 1000);
    }

    /// Loop header at pc 3.
    #[test]
    fn header_at_three() {
        let (r, _) = probe("header_at_three", &loop_header_at_three(1000), 3, 3);
        assert_eq!(r, 1000);
    }

    /// Portal-entry door, straight-line program: the walk is armed at pc 0
    /// (threshold 1 trips on the single entry-door hit) and control never
    /// returns to pc 0.
    #[test]
    fn entry_door_straight_line() {
        let (r, compiles) = probe("entry_door_straight_line", &straight_line_program(), 3, 1);
        assert_eq!(r, 333);
        assert!(
            compiles >= 1,
            "the straight-line entry walk must reach its return terminator and mint a procedure"
        );
    }

    /// Same entry door, but the program loops back to pc 0 — control does
    /// return to the armed pc. Isolates "never returns" from "pc 0".
    #[test]
    fn entry_door_loops_back_to_zero() {
        let (r, compiles) = probe(
            "entry_door_loops_back_to_zero",
            &loop_header_at_zero(1000),
            3,
            1,
        );
        assert_eq!(r, 1000);
        // The paired control for the test above: same door, same armed pc,
        // but control does return there, so this one closed a loop even
        // before in-arm returns lowered. It must keep doing so.
        assert!(
            compiles >= 1,
            "the looping entry walk must still close its loop and compile"
        );
    }

    /// Walk armed at pc 0; the first instruction executed is `OP_RETURN`.
    #[test]
    fn entry_door_return_immediately() {
        let (r, _) = probe("entry_door_return_immediately", &return_immediately(), 3, 1);
        assert_eq!(r, 0);
    }

    #[test]
    fn jit_tier_is_alive() {
        let (got, compiles, ops_after, unpeeled, shape) =
            probe_with_ops("jit_tier_is_alive", &count_program(1000, 1), 3, 3);
        assert_eq!(
            unpeeled, EXPECT_UNPEELED,
            "expected {EXPECT_UNPEELED} unroll-free fallback compile(s) out of \
             {compiles}, saw {unpeeled}"
        );
        // The body actually closes a loop — see `LoopBodyShape`. A compile
        // count and an op count together still accept a body that bails out on
        // its first pass; this is the term that does not. Sound HERE because
        // this fixture loops: on a straight-line subject a `Jump`-less body is
        // the right answer, not a defect.
        assert!(
            shape.closes_a_loop(),
            "compiled {ops_after} ops but the body {} ({shape:?})",
            shape.why_not().unwrap_or("closes a loop")
        );
        assert_eq!(
            got, 1000,
            "count_program(1000, 1) counts r0 up by 1 per pass while r2 > r0, so \
             the answer IS the trip count"
        );

        let degraded: Vec<&str> = majit_metainterp::degraded_dispatch_arms()
            .iter()
            .filter(|a| a.interp == "VmState")
            .map(|a| a.arm)
            .collect();
        assert_eq!(
            degraded,
            Vec::<&str>::new(),
            "dispatch arms degraded to abort stubs: {degraded:?} — an equality \
             rather than is_empty() so this also catches a name disappearing, \
             which is what emptying the dispatch looks like"
        );

        assert!(
            compiles >= 1,
            "compiled {compiles} loops — the JIT tier is inert and the \
             interpreter is answering alone, which every other assertion in this \
             file would still pass"
        );

        assert_eq!(
            ops_after, 4,
            "compiled body is {ops_after} ops, not the pinned 4 — 1 is a bare \
             `Finish()`, i.e. a dispatch that lowered nothing at all, and 9 is \
             the peeled shape the legacy bare merge point produced"
        );
        println!(
            "[tier-alive] count_program(1000, 1) = {got}, compiled {compiles} loop(s) of {ops_after} ops, 0 degraded arms"
        );
    }

    #[test]
    fn straight_line_program_folds_to_one_finish() {
        let (got, compiles, ops_after, _unpeeled, shape) =
            probe_with_ops("straight_line_body", &straight_line_program(), 3, 1);
        // The counterpart to `jit_tier_is_alive`'s shape assertion, and the
        // reason that one has to be conditional: same crate, same probe,
        // opposite shape. This subject has no loop, so its body correctly
        // carries no back edge — `closes_a_loop()` is FALSE here and that is
        // health. The pair is what shows the predicate discriminating rather
        // than merely passing everywhere.
        assert_eq!(
            shape,
            majit_metainterp::LoopBodyShape::default(),
            "a straight-line body should carry neither a back edge nor an \
             always-failing guard"
        );
        // Load-bearing: this is what separates `Finish(333)` from a `Finish(0)`
        // that lowered nothing. See the doc comment.
        assert_eq!(got, 333, "straight_line_program must answer 111 + 222");
        assert!(
            compiles >= 1,
            "the straight-line walk must mint a procedure"
        );
        assert_eq!(
            ops_after, 1,
            "straight-line body is {ops_after} ops, not the pinned 1 — 2 was \
             the pre-fix shape, where the add could not fold because the \
             walk headed past its arming pc"
        );
        println!("[straight-line] 333 from a {ops_after}-op body with no back edge");
    }

    /// Loop header at pc 9 — the pre-existing shape, as a control.
    #[test]
    fn header_at_nine() {
        let (r, _) = probe("header_at_nine", &count_program(1000, 1), 3, 3);
        assert_eq!(r, 1000);
    }

    /// The env element is 8 bytes wide and the immediate `n = 1000` does not
    /// fit a byte. A byte-stride descr would read the wrong word and never
    /// reach 1000.
    #[test]
    fn i64_env_reads_wide_immediates() {
        let (result, compiles) = probe(
            "i64_env_reads_wide_immediates",
            &count_program(1000, 1),
            3,
            3,
        );
        assert_eq!(result, 1000, "i64-env loop must compute 1000");
        assert!(
            compiles >= 1,
            "majit should have compiled the hot loop at least once"
        );
    }

    /// Correctness across inputs (each exercises the CloseLoop guard-exit
    /// deopt), all with byte-overflowing immediates.
    #[test]
    fn i64_env_varies_n() {
        for n in [300_i64, 500, 1000, 4096, 100_000] {
            let r = run_locked(&count_program(n, 1), 3, 3);
            assert_eq!(r, n, "count to {n} mismatch");
        }
    }

    /// A step > 255 proves the ADD operand word is read at the right stride
    /// too: counting by 7 up to a multiple of 7 lands exactly on `n`.
    #[test]
    fn i64_env_wide_step() {
        let r = run_locked(&count_program(7 * 300, 7), 3, 3);
        assert_eq!(r, 7 * 300);
    }
}
