/// JIT-enabled Brainfuck interpreter using `#[jit_interp]` and virtualizable
/// state fields.
///
/// The program counter and bytecode are green inputs. The pointer and tape are
/// red state, with tape cells represented as symbolic values while tracing.
/// A backward `]` branch identifies the loop header.
pub type Bytecode = [u8];

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Hot loops majit compiled.
///
/// Until this counter existed, this crate had **no** observable for the JIT
/// tier at all — no compile count, no `ops_after`, no degraded-arm read. Its
/// two tests compare JIT output against the interpreter, which an interpreter
/// answering alone satisfies perfectly. That is not hypothetical here: this
/// example ran its whole suite green while `Total ops recorded` went 9610 -> 0
/// under an experiment that declared `greens` (see the header).
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Ops in the last compiled loop body after optimization.
///
/// `COMPILES > 0` is necessary but NOT sufficient: an entirely empty dispatch
/// still compiles a trace — one whose whole optimized body is `Finish()`, i.e.
/// `ops_after == 1`. A compile counter counts TRACES, not WORK.
///
/// On THIS crate it does not separate a compiled loop from a compiled
/// nothing either, and it never could: see [`LAST_OPS_BEFORE`].
pub static LAST_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);

/// Shape of the last compiled loop body — see [`majit_metainterp::LoopBodyShape`].
///
/// Held as two flags rather than the struct itself so the recording stays
/// lock-free on the compile path; the probe rebuilds the struct inside the same
/// lock window it reads the counters in, because this is as process-global as
/// they are.
pub static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
pub static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

/// Operations recorded before optimization for the most recently compiled
/// loop. Tests compare this with [`LAST_OPS_AFTER`] to distinguish a useful
/// loop body from a segmented trace-limit terminator.
pub static LAST_OPS_BEFORE: AtomicUsize = AtomicUsize::new(0);

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

#[cfg(not(test))]
const TAPE_SIZE: usize = 30000;
#[cfg(test)]
const TAPE_SIZE: usize = 128;
const DEFAULT_THRESHOLD: u32 = 3;

struct BfState {
    pointer: i64,
    tape: Vec<i64>,
}

#[majit_macros::jit_interp(
    state = BfState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        pointer: int,
        tape: [int; virt],
    },
)]
fn mainloop(program: &Bytecode, threshold: u32) -> String {
    let mut driver: majit_metainterp::JitDriver<BfState> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, ops_before, ops_after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
        LAST_OPS_BEFORE.store(ops_before, Ordering::Relaxed);
        LAST_OPS_AFTER.store(ops_after, Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape::of(opcodes);
        LAST_HAS_JUMP.store(shape.has_jump, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(shape.has_always_fails, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let _stacksize: i32 = 0;
    let mut state = BfState {
        pointer: 0,
        tape: vec![0i64; TAPE_SIZE],
    };
    let mut output = String::new();

    // RPython warmspot.py: make_jitcodes() before pyjitpl.py finish_setup().
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    loop {
        if pc >= program.len() {
            break;
        }
        // Still the legacy bare form, but no longer for the reason it was
        // reverted: that blocker is fixed and the conversion is still not
        // exercisable.
        //
        // The arms below match byte-character literals (`b'>'`, `b'+'`, …), and
        // those lower now. "majit macros: accept byte and char literals in
        // dispatch arm patterns" routes `Pat::Lit` through `pat_lit_int_value`,
        // which takes `Lit::Byte` and `Lit::Char`; "majit macros: register a
        // dispatch arm the lowerer drops for an unsupported pattern" makes the
        // remaining skip path register instead of vanishing. This interpreter
        // now names one: `BfState::b']' lowered to an abort stub`.
        //
        // The pre-fix reading is kept because it is why the bare form is here.
        // Before those two, every arm was dropped through a bare `continue` that
        // ran before any stub was built, so nothing registered and the dispatch
        // was empty. On the converted form, `MAJIT_STATS=1 --nocapture`, exact
        // binary: `Traces compiled: 1` but `Total ops recorded: 1` and
        // `Total ops after opt: 1` — the compiled body was the bare `Finish()` of
        // an empty dispatch, against the tla control's 21 recorded / 10 after opt
        // / 1 guard failure on the same instrument. That is the measurement
        // behind "`Traces compiled > 0` proves nothing on its own", which still
        // holds. Its companion no longer does: an empty `degraded_dispatch_arms()`
        // was compatible with total loss only while the skip path registered
        // nothing, and it now registers.
        //
        // What blocks the conversion today:
        // The tier is live and every trace runs away to `trace_limit` and is
        // segmented. Tracing starts at `key=113`, the `[`, and records 1602
        // repetitions of that one arm's guard — `GuardValue(v0, 0);
        // IntEq(v2, 0); GuardFalse` — on unchanged operands, with no pc advance
        // and no tape write. Convert only once that stops.
        //
        // STOP: The `pc = pc + 1` suspicion that used to be recorded here is
        // REFUTED as the account of `main()`'s non-termination, and the
        // refutation is one measurement: the identical 1602 repetitions appear
        // for a SEVEN-pass program that finishes in 0.03s. The runaway is a
        // fixed-size, per-trace-attempt event, not a trip-count-dependent one —
        // ops recorded is 1205/2405/4805/9605 for `trace_limit`
        // 1500/3000/6000/12000, i.e. a function of the limit and of nothing
        // else. A quantity that does not move with the workload cannot explain
        // a symptom that only the large workload shows. See #47.
        jit_merge_point!();
        let ch = program[pc];

        match ch {
            b'>' => {
                state.pointer += 1;
                pc += 1;
            }
            b'<' => {
                state.pointer -= 1;
                pc += 1;
            }
            b'+' => {
                state.tape[state.pointer as usize] += 1;
                pc += 1;
            }
            b'-' => {
                state.tape[state.pointer as usize] -= 1;
                pc += 1;
            }
            b'.' => break,
            b',' => break,
            b'[' => {
                if state.tape[state.pointer as usize] == 0 {
                    let mut need: i32 = 1;
                    let mut p = pc + 1;
                    while need > 0 {
                        if program[p] == b']' {
                            need -= 1;
                        } else if program[p] == b'[' {
                            need += 1;
                        }
                        p += 1;
                    }
                    pc = p;
                } else {
                    pc += 1;
                }
            }
            b']' if state.tape[state.pointer as usize] != 0 => {
                let target = find_matching_open(program, pc);
                if target < pc {
                    can_enter_jit!(driver, target, &mut state, program, || {});
                }
                pc = target;
                continue;
            }
            _ => {
                pc += 1;
            }
        }
    }

    // Handle I/O after breaking out of the traced loop.
    // The trace aborts on '.' and ','; the interpreter handles them here
    // and re-enters the loop.
    while pc < program.len() {
        let ch = program[pc];
        if ch == b'.' {
            output.push(state.tape[state.pointer as usize] as u8 as char);
            pc += 1;
        } else if ch == b',' {
            state.tape[state.pointer as usize] = 0;
            pc += 1;
        } else {
            // Re-enter the traced mainloop for the remaining program.
            let remaining_output = mainloop_resume(program, &mut state, &mut driver, pc, threshold);
            output.push_str(&remaining_output);
            break;
        }
    }

    output
}

/// Resume mainloop execution from a given pc, reusing existing state and driver.
fn mainloop_resume(
    program: &Bytecode,
    state: &mut BfState,
    _driver: &mut majit_metainterp::JitDriver<BfState>,
    mut pc: usize,
    _threshold: u32,
) -> String {
    let mut output = String::new();

    while pc < program.len() {
        let ch = program[pc];
        if ch == b'>' {
            state.pointer += 1;
            pc += 1;
        } else if ch == b'<' {
            state.pointer -= 1;
            pc += 1;
        } else if ch == b'+' {
            state.tape[state.pointer as usize] += 1;
            pc += 1;
        } else if ch == b'-' {
            state.tape[state.pointer as usize] -= 1;
            pc += 1;
        } else if ch == b'.' {
            output.push(state.tape[state.pointer as usize] as u8 as char);
            pc += 1;
        } else if ch == b',' {
            state.tape[state.pointer as usize] = 0;
            pc += 1;
        } else if ch == b'[' {
            if state.tape[state.pointer as usize] == 0 {
                let mut need: i32 = 1;
                let mut p = pc + 1;
                while need > 0 {
                    if program[p] == b']' {
                        need -= 1;
                    } else if program[p] == b'[' {
                        need += 1;
                    }
                    p += 1;
                }
                pc = p;
            } else {
                pc += 1;
            }
        } else if ch == b']' {
            if state.tape[state.pointer as usize] != 0 {
                let target = find_matching_open(program, pc);
                pc = target;
            } else {
                pc += 1;
            }
        } else {
            pc += 1;
        }
    }

    output
}

/// Find the matching '[' for a ']' at the given position.
fn find_matching_open(code: &[u8], close_pos: usize) -> usize {
    let mut need: i32 = 1;
    let mut p = close_pos - 1;
    while need > 0 {
        if code[p] == b']' {
            need += 1;
        } else if code[p] == b'[' {
            need -= 1;
        }
        if need > 0 {
            p -= 1;
        }
    }
    p
}

pub struct JitBrainInterp {
    threshold: u32,
}

impl Default for JitBrainInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl JitBrainInterp {
    pub fn new() -> Self {
        JitBrainInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn run(&mut self, code: &[u8]) -> String {
        mainloop(code, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::interp;
    use majit_metainterp::{RefusalKind, refusal_kind};

    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `prog` and return only its output.
    ///
    /// Not an alias for `JitBrainInterp::new().run()`: the lock is the point.
    /// Neither this nor [`compile_probe`] may call the other — the mutex is
    /// plain, and re-entering it on one thread deadlocks.
    fn run_jit(prog: &[u8]) -> String {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        JitBrainInterp::new().run(prog)
    }

    /// Run `prog` with the tier counters reset, returning
    /// `(output, compiles, ops_before, ops_after)`.
    fn compile_probe(
        prog: &[u8],
    ) -> (String, usize, usize, usize, majit_metainterp::LoopBodyShape) {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_BEFORE.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        let got = JitBrainInterp::new().run(prog);
        (
            got,
            COMPILES.load(Ordering::Relaxed),
            LAST_OPS_BEFORE.load(Ordering::Relaxed),
            LAST_OPS_AFTER.load(Ordering::Relaxed),
            majit_metainterp::LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        )
    }

    fn multiply_gate_bf(a: u8, b: u8) -> Vec<u8> {
        let mut prog = Vec::new();
        for _ in 0..a {
            prog.push(b'+');
        }
        prog.push(b'[');
        prog.push(b'>');
        for _ in 0..b {
            prog.push(b'+');
        }
        prog.extend_from_slice(b"<-]");
        prog.extend_from_slice(b">.");
        prog
    }

    #[test]
    fn jit_tier_aborts_in_the_loop_arm_stub() {
        const A: u8 = 7;
        const B: u8 = 3;
        let prog = multiply_gate_bf(A, B);
        let (got, compiles, ops_before, ops_after, _shape) = compile_probe(&prog);

        // The interpreter must still answer correctly whatever the tier does.
        assert_eq!(
            got.chars().count(),
            1,
            "multiply({A}*{B}) printed {got:?}; the gate expects exactly one cell"
        );
        let cell = got.chars().next().unwrap() as u32;
        assert_eq!(
            cell,
            u32::from(A) * u32::from(B),
            "multiply({A}*{B}) left {cell} in cell 1, so the loop ran {} passes \
             rather than {A}",
            cell / u32::from(B)
        );

        let bf_arms: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "BfState")
            .collect();
        let degraded: Vec<&str> = bf_arms.iter().map(|a| a.arm).collect();
        assert_eq!(
            degraded,
            vec!["b'['", "b']'"],
            "degraded dispatch arms moved — every trace reaching one aborts"
        );

        // The CAUSE, which the name set above cannot see: an arm can keep
        // degrading for an entirely different reason. See tl's `jit_tier_is_alive`
        // for the measurement that motivated this — three A/B arms whose name
        // sets and pass counts were identical while the mechanism changed.
        let causes: Vec<(&str, RefusalKind)> = bf_arms
            .iter()
            .map(|a| (a.arm, refusal_kind(a.reason)))
            .collect();
        assert_eq!(
            causes,
            [
                ("b'['", RefusalKind::GreenWriteback),
                ("b']'", RefusalKind::UnlowerableStmt)
            ],
            "an arm still degrades, but a different mechanism is refusing it now"
        );
        let bracket = bf_arms
            .iter()
            .find(|a| a.arm == "b'['")
            .expect("b'[' is in the set asserted immediately above");
        let close = bf_arms
            .iter()
            .find(|a| a.arm == "b']'")
            .expect("b']' is in the set asserted immediately above");
        assert!(
            close.reason.contains("find_matching_open(program, pc)"),
            "b']' refusal no longer names the unsupported matching-bracket call: {}",
            close.reason
        );
        assert!(
            close.reason.contains("pc = target"),
            "b']' refusal no longer preserves the trailing green writeback: {}",
            close.reason
        );
        assert!(
            bracket.reason.contains("while need > 0"),
            "b'[' refusal no longer names the unsupported scan loop: {}",
            bracket.reason
        );

        // THE ASSERTION THAT DISCRIMINATES, and the one to replace rather than
        // re-record. `b']'` is the loop arm, so every trace that reaches the
        // loop aborts in its stub and nothing is ever compiled.
        //
        // IF THIS FAILS, the loop arm lowers and the tier has come alive. Do
        // NOT re-record it: replace this whole gate with a liveness gate
        // asserting `shape.closes_a_loop()`, which is the property a segmented
        // or aborted trace can never satisfy.
        assert_eq!(
            compiles, 0,
            "multiply({A}*{B}) compiled {compiles} loops. This crate's loop arm \
             `b']'` is an abort stub, so no trace should survive to be compiled. \
             A non-zero count means either the arm lowers now — replace this gate \
             with `assert!(shape.closes_a_loop())` — or another test entered the \
             JIT inside this probe's window without taking PROBE_LOCK"
        );

        // No assertion on `ops_after` or on the fold ratio, deliberately. With
        // nothing compiled they are both 0, and an assertion over a value that
        // is structurally 0 is the same species of oracle-that-cannot-fail this
        // gate was rebuilt to stop making. `ops_before` is reported, not pinned,
        // for the same reason.
        println!(
            "[tier-aborting] multiply({A}*{B}) = {cell} from the interpreter \
             alone, {compiles} loops compiled, {ops_before} ops recorded, \
             {ops_after} after opt, degraded {degraded:?}"
        );
    }

    #[test]
    fn jit_matches_interp_hot_loops() {
        let code = b"+++++[->+<]>++++++++[<++++++++>-]<.";
        let expected = interp::interpret(code);
        let got = run_jit(code);
        assert_eq!(got, "h");
        assert_eq!(got, expected);
    }

    #[test]
    fn jit_no_loop() {
        let output = run_jit(b"+++");
        assert_eq!(output.len(), 0);
    }
}
