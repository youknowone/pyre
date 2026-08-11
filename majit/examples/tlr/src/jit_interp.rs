/// JIT-enabled TLR interpreter — auto-generated tracing via `#[jit_interp]` + `state_fields`.
///
/// Matches RPython's tlr.py line-by-line: write the interpreter, get JIT for free.
///
/// Greens: [pc, bytecode]
/// Reds:   [a, regs]  (tracked via state_fields)
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub type Bytecode = [u8];

/// Hot loops majit compiled. The only positive evidence the JIT tier is alive:
/// a green suite, agreement with `interp::interpret` and an exact result are
/// all satisfied by an interpreter answering alone.
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Ops in the last compiled loop body after optimization.
///
/// `COMPILES > 0` is necessary but NOT sufficient: an entirely empty dispatch
/// still compiles a trace — one whose whole optimized body is `Finish()`, i.e.
/// `ops_after == 1`. A compile counter counts TRACES, not WORK. This is the
/// term that separates a compiled loop from a compiled nothing.
pub static LAST_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);

/// Shape of the last compiled loop body — see [`majit_metainterp::LoopBodyShape`].
///
/// Held as two flags rather than the struct itself so the recording stays
/// lock-free on the compile path; the probe rebuilds the struct inside the same
/// lock window it reads the counters in, because this is as process-global as
/// they are.
pub static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
pub static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

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

struct TlrState {
    a: i64,
    regs: Vec<i64>,
}

const MOV_A_R: u8 = 1;
const MOV_R_A: u8 = 2;
const JUMP_IF_A: u8 = 3;
const SET_A: u8 = 4;
const ADD_R_TO_A: u8 = 5;
const RETURN_A: u8 = 6;
const ALLOCATE: u8 = 7;
const NEG_A: u8 = 8;
/// Two-byte little-endian immediate, the shift-or form
/// `(program[pc] | program[pc + 1] << 8)`.
///
/// It exists to make that form observable. The corpus's other shift-or sites
/// are `tiny2/src/jit_interp.rs:198` and `tiny3:209`, and both sit inside
/// `OP_LOOP_END`, which is a degraded abort stub for an unrelated reason
/// (`break`/`continue`), in crates that compile **zero** traces. So the form
/// has never been asked to lower anywhere, and "it already works" was neither
/// established nor refutable. `SET_A_WIDE` puts it in an arm inside a loop that
/// does close a trace, in a crate that compiles.
const SET_A_WIDE: u8 = 9;

const DEFAULT_THRESHOLD: u32 = 3;

// `greens = [pc, program]` lets the operand reads (`program[pc + N]`)
// constant-fold so the loop traces and compiles. `regs` is `[int; virt]`
// (virtualizable), not plain `[int]`: a loop-carried plain `[int]` element is
// kept in a trace register and is *not* restored to the array on a CloseLoop
// guard deopt, so the post-loop value reads back as the pre-loop one. A virt
// array writes through to the heap-backing Vec, which the deopt path reads
// directly — the same mechanism braininterp relies on. (`a` is a plain scalar
// red, which is restored correctly.)
#[majit_macros::jit_interp(
    state = TlrState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        a: int,
        regs: [int; virt],
    },
)]
fn mainloop(program: &Bytecode, initial_a: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlrState> =
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
    let mut state = TlrState {
        a: initial_a,
        regs: Vec::new(),
    };

    // RPython warmspot.py:281-289 canonical-liveness install hook.
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    // while True: — RPython tlr.py:22
    loop {
        // `; state` selects the single-executor close: the walk's final state is
        // transferred into `state` here and the native loop resumes at the close
        // pc, instead of discarding the walk outcome and re-running the circuit
        // the walk already executed.
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        pc += 1;

        match opcode {
            MOV_A_R => {
                let n = program[pc] as usize;
                pc += 1;
                state.regs[n] = state.a;
            }
            MOV_R_A => {
                let n = program[pc] as usize;
                pc += 1;
                state.a = state.regs[n];
            }
            JUMP_IF_A => {
                let target = program[pc] as usize;
                pc += 1;
                let jump = state.a != 0;
                if jump {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            SET_A => {
                state.a = program[pc] as i64;
                pc += 1;
            }
            ADD_R_TO_A => {
                let n = program[pc] as usize;
                pc += 1;
                state.a += state.regs[n];
            }
            RETURN_A => {
                return state.a;
            }
            ALLOCATE => {
                let n = program[pc] as usize;
                pc += 1;
                state.regs = vec![0; n];
            }
            NEG_A => {
                state.a = 0 - state.a;
            }
            SET_A_WIDE => {
                state.a = (program[pc] as i64) | ((program[pc + 1] as i64) << 8);
                pc += 2;
            }
            _ => {}
        }
    }
    // Reached only when the merge point broke out on a walk that already ran
    // `RETURN_A`, so the result is whatever that opcode left in the accumulator.
    // `a` is a plain scalar state field, so `writeback_scalar_state_fields`
    // has already pushed the walk-final value into native `state` by here.
    state.a
}

// ── Public wrapper matching the old API ──

pub struct JitTlrInterp {
    threshold: u32,
}

impl Default for JitTlrInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl JitTlrInterp {
    pub fn new() -> Self {
        JitTlrInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn run(&mut self, bytecode: &[u8], initial_a: i64) -> i64 {
        mainloop(bytecode, initial_a, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;
    use majit_metainterp::{RefusalKind, refusal_kind};

    fn square_bytecode() -> Vec<u8> {
        vec![
            ALLOCATE, 3, MOV_A_R, 0, MOV_A_R, 1, SET_A, 0, MOV_A_R, 2, SET_A, 1, NEG_A, ADD_R_TO_A,
            0, MOV_A_R, 0, MOV_R_A, 2, ADD_R_TO_A, 1, MOV_A_R, 2, MOV_R_A, 0, JUMP_IF_A, 10,
            MOV_R_A, 2, RETURN_A,
        ]
    }

    /// The wide immediate this fixture decodes: 258 = `0x0102`, chosen so that
    /// BOTH operand bytes are load-bearing. A value under 256 would decode
    /// correctly even if the high byte were dropped entirely.
    const WIDE: i64 = 258;
    /// The narrow control's immediate. Any value < 256; only its width matters.
    const NARROW: i64 = 200;

    fn imm_loop_bytecode(wide: bool) -> Vec<u8> {
        let mut prog = vec![
            ALLOCATE, 2, // regs = [counter, acc]
            MOV_A_R, 0, // regs[0] = initial_a
            SET_A, 0, //
            MOV_A_R, 1, // regs[1] = 0
        ];
        let loop_target = prog.len() as u8; // 8
        if wide {
            prog.extend_from_slice(&[SET_A_WIDE, (WIDE & 0xff) as u8, (WIDE >> 8) as u8]);
        } else {
            prog.extend_from_slice(&[SET_A, NARROW as u8]);
        }
        // acc += a; then counter -= 1 via `SET_A 1; NEG_A; ADD_R_TO_A 0`
        // (tlr has no subtract); loop back while the counter is non-zero;
        // finally return the accumulator.
        prog.extend_from_slice(&[
            ADD_R_TO_A,
            1,
            MOV_A_R,
            1,
            SET_A,
            1,
            NEG_A,
            ADD_R_TO_A,
            0,
            MOV_A_R,
            0,
            JUMP_IF_A,
            loop_target,
            MOV_R_A,
            1,
            RETURN_A,
        ]);
        prog
    }

    #[test]
    fn jit_wide_immediate_folds() {
        const PASSES: i64 = 20;

        let narrow_bc = imm_loop_bytecode(false);
        let (narrow_got, narrow_compiles, narrow_ops, ..) = compile_probe(&narrow_bc, PASSES);
        assert_eq!(
            narrow_got,
            NARROW * PASSES,
            "narrow control computed {narrow_got}, not {NARROW}*{PASSES}"
        );
        assert_eq!(narrow_got, interp::interpret(&narrow_bc, PASSES));
        assert!(
            narrow_compiles >= 1,
            "the narrow control compiled nothing, so the wide arm's result below \
             is not attributable to the immediate width (compiles={narrow_compiles})"
        );

        let wide_bc = imm_loop_bytecode(true);
        let (wide_got, wide_compiles, wide_ops, ..) = compile_probe(&wide_bc, PASSES);
        // Decoding, not just compiling: a dropped high byte yields 2*PASSES
        // instead of 258*PASSES, and a dropped statement yields 0.
        assert_eq!(
            wide_got,
            WIDE * PASSES,
            "wide immediate decoded to {} per pass, not {WIDE} — a value of {} \
             means the high byte was dropped",
            wide_got / PASSES,
            WIDE & 0xff
        );
        assert_eq!(wide_got, interp::interpret(&wide_bc, PASSES));
        assert!(
            wide_compiles >= 1,
            "the wide-immediate loop compiled nothing while the narrow control \
             compiled {narrow_compiles} — the shift-or form blocks tracing"
        );
        // THE RESULT: 11 ops, EQUAL to the narrow control's 11.
        //
        // The two-byte read folded to a constant exactly as the one-byte read
        // did. Had the wide immediate survived as a residual read of the green
        // bytecode, this body would carry the extra loads and the shift/or and
        // exceed the control. It does not — so the shift-or wide-immediate form
        // already folds, and it is NOT a lowering gap.
        //
        // The equality with `narrow_ops` is asserted rather than the literal
        // alone, because the pair is the finding: a future change that inflates
        // both equally would keep an absolute pin green while destroying the
        // property this test exists to state.
        assert_eq!(
            wide_ops, narrow_ops,
            "wide immediate no longer folds to the same body as the narrow \
             control ({wide_ops} vs {narrow_ops}) — it has become a residual read"
        );
        assert_eq!(
            wide_ops, 11,
            "wide-immediate loop body is {wide_ops} ops, not the pinned 11"
        );
        println!(
            "[wide-imm] narrow={narrow_got} ({narrow_compiles} loop(s), {narrow_ops} ops), \
             wide={wide_got} ({wide_compiles} loop(s), {wide_ops} ops)"
        );
    }

    fn realloc_loop_bytecode(realloc: bool) -> Vec<u8> {
        let head = if realloc {
            [ALLOCATE, 2]
        } else {
            [ADD_R_TO_A, 1]
        };
        vec![
            ALLOCATE, 2, // prologue: regs is non-empty before the loop
            // loop target = 2
            head[0], head[1], // the varied instruction
            MOV_A_R, 0, // regs[0] = a
            SET_A, 1,     // a = 1
            NEG_A, // a = -1
            ADD_R_TO_A, 0, // a = regs[0] - 1
            JUMP_IF_A, 2, // back edge while a != 0
            RETURN_A,
        ]
    }

    #[test]
    fn jit_realloc_in_traced_loop() {
        const PASSES: i64 = 20;

        let control_bc = realloc_loop_bytecode(false);
        let (control_got, control_compiles, control_ops, ..) = compile_probe(&control_bc, PASSES);
        assert_eq!(control_got, interp::interpret(&control_bc, PASSES));
        assert!(
            control_compiles >= 1,
            "the control loop compiled nothing, so the probe's zero below is \
             not attributable to ALLOCATE (compiles={control_compiles})"
        );

        let bc = realloc_loop_bytecode(true);
        // Correctness first and unconditionally: this is the assertion that
        // catches a stale-mirror miscompile, and it must not be guarded by a
        // compile count that a miscompiling build would still satisfy.
        let expected = interp::interpret(&bc, PASSES);
        let (got, compiles, ops_after, ..) = compile_probe(&bc, PASSES);
        assert_eq!(
            got, expected,
            "realloc-in-loop disagrees with the interpreter ({got} vs \
             {expected}) — a virt-array mirror cached across the reallocation \
             would produce exactly this"
        );
        assert_eq!(
            got, control_got,
            "the two arms must compute the same value; if they do not, the \
             control is not a control"
        );

        assert_eq!(
            compiles, 0,
            "the reallocating loop compiled {compiles} trace(s) of {ops_after} \
             ops — ALLOCATE has started lowering, so re-check the result \
             assertion above against a stale virt-array mirror before pinning \
             a new number"
        );
        println!(
            "[realloc] countdown({PASSES}) = {got}; control compiled \
             {control_compiles} loop(s) of {control_ops} ops, realloc arm compiled {compiles}"
        );
    }

    /// [`COMPILES`] is process-global, so under the default parallel libtest
    /// runner a concurrent `run` lands inside [`compile_probe`]'s
    /// store/run/load window and the probe reads someone else's compile. The
    /// lock therefore covers *every* call that can compile, not just the
    /// probe's own — [`run_jit`] and [`compile_probe`] are the only two ways a
    /// test may enter the JIT, and neither may call the other (a plain mutex
    /// re-entered on one thread deadlocks).
    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// For tests that assert only on the result. They still compile, so they
    /// must not run inside the probe's window. See [`PROBE_LOCK`].
    fn run_jit(bc: &[u8], initial_a: i64) -> i64 {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        JitTlrInterp::new().run(bc, initial_a)
    }

    /// Run with both counters reset, returning `(result, compiles, ops_after)`.
    ///
    /// [`LAST_OPS_AFTER`] is read here rather than at the call site, and reset
    /// here rather than nowhere. Both counters are process-global, so both need
    /// the same treatment [`PROBE_LOCK`] exists to give [`COMPILES`]: a load
    /// taken after the guard drops can observe a concurrent test's compile, and
    /// a counter that is never stored to zero retains whatever the last compile
    /// anywhere in the process left behind. Unreset, a zero from this probe is
    /// indistinguishable from an inherited value.
    fn compile_probe(
        bc: &[u8],
        initial_a: i64,
    ) -> (i64, usize, usize, majit_metainterp::LoopBodyShape) {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        let got = JitTlrInterp::new().run(bc, initial_a);
        (
            got,
            COMPILES.load(Ordering::Relaxed),
            LAST_OPS_AFTER.load(Ordering::Relaxed),
            majit_metainterp::LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        )
    }

    /// A real loop body was compiled, and exactly one known arm is degraded.
    ///
    /// This certifies that the JIT tier produced a non-empty compiled body. It
    /// is silent on whether that body's resume data is well-formed: a trace can
    /// carry a malformed promote snapshot and still be counted here.
    ///
    /// All three parts are needed and none implies another:
    ///
    /// 1. `COMPILES` 0 → non-zero. A green suite, agreement with
    ///    `interp::interpret` and an exact result are all satisfied by the
    ///    interpreter answering alone.
    /// 2. `ops_after` pinned by equality. `compiles >= 1` is necessary but NOT
    ///    sufficient: an entirely empty dispatch still compiles a trace — one
    ///    whose whole optimized body is `Finish()`, i.e. `ops_after == 1`. A
    ///    compile counter counts TRACES, not WORK, and every inequality a real
    ///    loop satisfies that degenerate body satisfies too.
    /// 3. The degraded-arm set, pinned as an **equality over a named set**
    ///    rather than an emptiness check. `ALLOCATE` does not lower today, so
    ///    asserting the list is empty would just fail; asserting it equals
    ///    exactly this one catches a *new* arm silently degrading, and fails
    ///    loudly on the day `ALLOCATE` starts lowering. The suite is green with
    ///    `ALLOCATE` degraded only because `square_bytecode` issues it once at
    ///    pc 0, outside the hot loop — a program whose traced loop reached it
    ///    would abort every trace.
    ///
    /// The registry is process-wide, so it is filtered to `state = TlrState`,
    /// and read *after* a run because nothing installs the dispatch JitCode
    /// until the interpreter is entered.
    #[test]
    fn jit_tier_is_alive() {
        let (got, compiles, ops_after, shape) = compile_probe(&square_bytecode(), 100);
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
        assert_eq!(got, 10_000, "square(100) must still answer 10000");

        let tlr_arms: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "TlrState")
            .collect();
        let degraded: Vec<&str> = tlr_arms.iter().map(|a| a.arm).collect();
        assert_eq!(
            degraded,
            ["ALLOCATE"],
            "the degraded-arm set moved. A NEW name means an arm silently \
             stopped lowering and every trace reaching it now aborts; a MISSING \
             name means that arm lowers again"
        );

        let causes: Vec<(&str, RefusalKind)> = tlr_arms
            .iter()
            .map(|a| (a.arm, refusal_kind(a.reason)))
            .collect();
        assert_eq!(
            causes,
            [("ALLOCATE", RefusalKind::GreenWriteback)],
            "ALLOCATE still degrades but a different mechanism is refusing it. \
             `RefusalKind::Unclassified` means majit grew a refusal family the \
             classifier does not know — add it in `majit-metainterp`, do not \
             re-record this pin"
        );
        assert!(
            tlr_arms[0].reason.contains("pc += 1"),
            "ALLOCATE's refusal no longer names the green write that stops \
             lowering before the reallocation: {}",
            tlr_arms[0].reason
        );

        // Zero-vs-nonzero is the property; a later change that legitimately
        // mints more than one artifact is not this regression.
        assert!(
            compiles >= 1,
            "square(100) compiled {compiles} loops — the JIT tier is inert and \
             the interpreter is answering alone, which every other assertion in \
             this file would still pass"
        );
        assert_eq!(
            ops_after, 11,
            "compiled loop body is {ops_after} ops, not the pinned 11 — a value \
             of 1 means the body is a bare `Finish()`, i.e. a dispatch that \
             lowered nothing at all"
        );
        println!(
            "[tier-alive] square(100) = {got}, compiled {compiles} loop(s) of {ops_after} ops, degraded {degraded:?}"
        );
    }

    #[test]
    fn jit_square_5() {
        let bc = square_bytecode();
        assert_eq!(run_jit(&bc, 5), 25);
    }

    #[test]
    fn jit_square_100() {
        let bc = square_bytecode();
        assert_eq!(run_jit(&bc, 100), 10_000);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = square_bytecode();
        for a in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::interpret(&bc, a);
            let got = run_jit(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![SET_A, 42, RETURN_A];
        assert_eq!(run_jit(&prog, 0), 42);
    }

    /// Exercises the JIT with many input sizes: small values stay interpreted,
    /// larger values trigger trace compilation and run compiled code.
    /// The guard exit path (a == 0 at loop end) is exercised on every input,
    /// verifying that fallback from compiled code produces correct results.
    #[test]
    fn jit_various_sizes() {
        let bc = square_bytecode();
        for a in [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000] {
            let expected = interp::interpret(&bc, a);
            let got = run_jit(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }
}
