/// JIT-enabled TLA interpreter via `#[jit_interp]` with `state_fields`.
///
/// RPython parity: tla.py Frame `_virtualizable_ = ['stackpos', 'stack[*]']`
/// (tla.py:98). Integer-only trace — strings cause trace abort.
///
/// Greens: [pc, bytecode]
/// Reds:   [stackpos, stack]  (tracked via state_fields)
///
/// `greens = [pc, program]` is load-bearing, not documentation: with the list
/// left empty the `CONST_INT` and `JUMP_IF` arms — the two that read an operand
/// out of `program` — do not lower and are emitted as abort stubs, so every
/// trace of the countdown loop aborts and nothing is ever compiled. Declaring
/// the greens is also what gives the merge point a green pc to report, which the
/// `; state` close needs to name a resume position.
use majit_metainterp::embed::Census;

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

/// tla.py `self.stack = [None] * 8`.
const STACK_SIZE: usize = 8;

struct TlaState {
    stackpos: i64,
    stack: Vec<i64>,
}

// ── Opcodes ──

const CONST_INT: u8 = 0;
const POP: u8 = 1;
const ADD: u8 = 2;
const RETURN: u8 = 3;
const JUMP_IF: u8 = 4;
const DUP: u8 = 5;
const SUB: u8 = 6;
const NEWSTR: u8 = 7;

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = TlaState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, initial_value: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlaState> =
        majit_metainterp::JitDriver::new(threshold);
    // Every counter the tier gate reads, plus the last body's op count and
    // shape, off the driver callbacks. `Census::begin` opens a window over them.
    Census::install(&mut driver);
    let mut pc: usize = 0;
    let stacksize: i32 = 0;
    let mut state = TlaState {
        stackpos: 1,
        stack: {
            let mut s = vec![0i64; STACK_SIZE];
            s[0] = initial_value;
            s
        },
    };

    // RPython warmspot.py:281-289 canonical-liveness install hook.
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        // `; state` selects the single-executor close: the walk's final state is
        // transferred into `state` here and the native loop resumes at the close
        // pc, instead of discarding the walk outcome and re-running the circuit
        // the walk already executed.
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        pc += 1;

        match opcode {
            CONST_INT => {
                let value = program[pc] as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos += 1;
            }
            POP => {
                state.stackpos -= 1;
            }
            DUP => {
                let v = state.stack[(state.stackpos - 1) as usize];
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
            JUMP_IF => {
                let target = program[pc] as usize;
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
            NEWSTR => break,
            RETURN => break,
            _ => {}
        }
    }

    // Also reached when the merge point broke out on a walk that already ran the
    // terminal opcode. That `break` precedes the merge point's own `pc` handoff,
    // so `pc` here still names the position the walk started from and nothing
    // below may read it; the result is rebuilt from `state` alone. `stackpos` is
    // a scalar state field and `stack` a virtualizable array field, so
    // `writeback_scalar_state_fields` / `writeback_virt_array_state_fields` have
    // already pushed the walk-final values into native `state` by here — no
    // separate `ret` field is needed.
    state.stackpos -= 1;
    state.stack[state.stackpos as usize]
}

// ── Public wrapper matching the old API ──

pub struct JitTlaInterp {
    threshold: u32,
}

impl Default for JitTlaInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl JitTlaInterp {
    pub fn new() -> Self {
        JitTlaInterp { threshold: 3 }
    }

    pub fn run(
        &mut self,
        bytecode: &[u8],
        w_arg: crate::interp::WObject,
    ) -> crate::interp::WObject {
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
        vec![DUP, CONST_INT, 1, SUB, DUP, JUMP_IF, 1, POP, RETURN]
    }

    /// For tests that assert only on the result. They still compile, so they
    /// must not run inside another test's census window — the window is what
    /// makes [`compile_probe`]'s numbers this run's rather than the process's.
    ///
    /// [`Census::begin`] is the only lock here, and it is not reentrant:
    /// [`run_jit`] and [`compile_probe`] are the only two ways a test may enter
    /// the JIT, and neither may call the other.
    fn run_jit(bc: &[u8], arg: i64) -> i64 {
        let _census = Census::begin();
        let mut jit = JitTlaInterp::new();
        jit.run(bc, interp::WObject::Int(arg)).int_value()
    }

    /// Run inside a census window, returning `(result, counts)`.
    ///
    /// The window is what separates this run's compiles from every other test's:
    /// the counters behind it are process-global, so an absolute read carries
    /// whatever the rest of the binary left behind, and a zero from it would be
    /// indistinguishable from an inherited value.
    fn compile_probe(bc: &[u8], arg: i64) -> (i64, majit_metainterp::embed::CensusCounts) {
        let census = Census::begin();
        let mut jit = JitTlaInterp::new();
        let got = jit.run(bc, interp::WObject::Int(arg)).int_value();
        (got, census.counts())
    }

    #[test]
    fn jit_countdown_5() {
        assert_eq!(run_jit(&countdown_bytecode(), 5), 5);
    }

    #[test]
    fn jit_countdown_30() {
        assert_eq!(run_jit(&countdown_bytecode(), 30), 30);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = countdown_bytecode();
        for n in [1, 2, 5, 10, 20, 30, 40] {
            let expected = interp::run(&bc, interp::WObject::Int(n));
            let got = run_jit(&bc, n);
            assert_eq!(got, expected.int_value(), "mismatch for n={n}");
        }
    }

    /// The JIT tier is alive — the one property no assertion on a *result* can
    /// establish.
    ///
    /// Both halves are needed and neither implies the other:
    ///
    /// 1. `Traces compiled` 0 → non-zero. A green suite, byte-identical output
    ///    and even an exact absolute trip count are all satisfied by the
    ///    interpreter answering alone; this example ran its entire suite green
    ///    at `Traces compiled: 0` before `greens = [pc, program]` was declared.
    /// 2. No degraded dispatch arm. An arm whose body did not lower is an abort
    ///    stub, so any trace reaching it aborts. The registry is populated at
    ///    dispatch-JitCode install time and names the arm, which an abort count
    ///    cannot: `trace action at pc=N -> Abort` reports the trace-START pc,
    ///    not the arm that caused it.
    ///
    /// The registry is process-wide, so the assertion is asked about this
    /// machine's `state = TlaState`. It runs *after* a run, because nothing
    /// installs the dispatch JitCode until the interpreter is entered.
    ///
    /// The subject is `count_to`, not `countdown`, so that the *result* assertion
    /// is itself an absolute trip count: `count_to(n)` returns the number of
    /// passes, while `countdown(n)` returns its own input whatever the trip count
    /// (`jit_trip_count_gate` states why). One compiled artifact then carries
    /// both properties — a body that ran the wrong number of times fails here on
    /// the result, not only in a separate test against a separately compiled loop.
    #[test]
    fn jit_tier_is_alive() {
        const N: i64 = 1001;
        let (got, counts) = compile_probe(&count_to_bytecode(N), 0);
        let (compiles, ops_after) = (counts.loops_compiled, counts.last_ops_after);
        let shape = counts.last_loop_body_shape;
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
            got, N,
            "count_to({N}) = {got}, so the loop the tier assertions below \
             describe ran {got} passes rather than {N}"
        );

        // `greens = [pc, program]` means a degraded arm aborts only traces that
        // reach that arm, rather than disabling the whole dispatch loop. The
        // empty pin is still the right one here: the arms this crate exercises
        // are all expected to lower, so a non-empty set is a regression and not
        // a trade-off. The pin also carries the denominator, which a bare
        // emptiness check over the registry cannot — an uninstalled portal
        // produces the same empty list a healthy one does.
        majit_metainterp::embed::assert_degraded_dispatch_arms("TlaState", &[]);

        // Zero-vs-nonzero is the property; a later change that legitimately
        // mints more than one artifact is not this regression.
        assert!(
            compiles >= 1,
            "count_to({N}) compiled {compiles} loops — the JIT tier is inert and \
             the interpreter is answering alone, which every other assertion in \
             this file would still pass"
        );
        // 24 until removals started running the postprocess callbacks the pass
        // chain had already collected for them (`send_extra_operation`,
        // optimizer.py:600-616, ends the walk but not the callbacks).
        // `postprocess_INT_SUB` synthesises up to six reverse-pure entries per
        // op via `pure_from_args2`, so running it for the ops OptPure itself
        // removes fills the 16-slot `RecentPureOps` ring: this subject's second
        // exit test recomputes `v - 254`, `v - 509` and `v - 764`, and those
        // three no longer find the first test's copies in the ring. +3 ops in
        // the preamble and +3 in the peeled body.
        assert_eq!(
            ops_after, 30,
            "compiled loop body is {ops_after} ops, not the pinned 30 — a value \
             of 1 means the body is a bare `Finish()`, i.e. a dispatch that \
             lowered nothing at all"
        );
        println!(
            "[tier-alive] count_to({N}) = {got}, compiled {compiles} loop(s) of {ops_after} ops, 0 degraded arms"
        );
    }

    /// Leave `top - value` on the stack. `CONST_INT`'s operand is one byte, so
    /// anything above 255 is subtracted in 255-sized bites.
    fn sub_const(code: &mut Vec<u8>, mut value: i64) {
        assert!(value > 0);
        while value > 0 {
            let bite = value.min(255);
            code.push(CONST_INT);
            code.push(bite as u8);
            code.push(SUB);
            value -= bite;
        }
    }

    /// Count up to `n`, one increment per pass, and return the counter.
    ///
    /// The exit test has TWO roots (`n` and `n + 1`) rather than one. TLA has no
    /// comparison opcode and no way to reach below the top of the stack, so the
    /// counter has to double as the loop condition — and a single-root test
    /// (`while v != n`) pins the answer to `n` no matter how many passes ran,
    /// which is exactly the insensitivity this gate exists to remove. Accepting
    /// `n + 1` as well makes one extra pass return `n + 1` instead of diverging.
    fn count_to_bytecode(n: i64) -> Vec<u8> {
        let mut code = Vec::new();
        // loop: (pc 0) — v += 1
        code.push(CONST_INT);
        code.push(1);
        code.push(ADD);
        // if v != n goto check2
        code.push(DUP);
        sub_const(&mut code, n);
        code.push(JUMP_IF);
        let check2_operand = code.len();
        code.push(0);
        code.push(RETURN);
        // check2: if v != n + 1 goto loop
        let check2 = code.len();
        code[check2_operand] = u8::try_from(check2).expect("jump target must fit a byte");
        code.push(DUP);
        sub_const(&mut code, n + 1);
        code.push(JUMP_IF);
        code.push(0);
        code.push(RETURN);
        code
    }

    /// Absolute trip-count gate on the JIT path.
    ///
    /// `count_to` adds 1 to the counter once per pass and leaves the loop as soon
    /// as it reaches `n` (or `n + 1`), so the returned value names the number of
    /// passes exactly: `n` passes answer `n`, and one extra pass answers `n + 1`.
    ///
    /// Agreement with `interp::run` would not settle this — the two run the same
    /// program — and neither would a byte-identical before/after output
    /// comparison: a duplicated iteration of the *compiled* loop is invisible to
    /// any check that does not assert an absolute count. `jit_countdown_30` is
    /// precisely such a check: it returns its own input whatever the trip count.
    ///
    /// Two lengths of different parity, because a peeled first iteration plus an
    /// even/odd body count is exactly the shape an off-by-one hides in.
    #[test]
    fn jit_trip_count_gate() {
        for n in [1001i64, 1002] {
            let bc = count_to_bytecode(n);
            let got = run_jit(&bc, 0);
            assert_eq!(
                got, n,
                "count_to({n}) = {got}, so the loop ran {got} passes rather than \
                 {n} — an off-by-one trip count is the signature of a terminal \
                 arm whose exit the trace dropped"
            );
            println!("[trip-count] count_to({n}) = {got} — exactly {n} passes");

            // Non-vacuity: seeding the counter at `n` leaves exactly one pass to
            // run, which overshoots onto the second root the same way a
            // duplicated compiled iteration would. It answers `n + 1`, so the
            // assertion above is one the program can fail rather than one pinned
            // to `n` by its own exit condition.
            let overshot = run_jit(&bc, n);
            assert_eq!(
                overshot,
                n + 1,
                "the second root is unreachable, so the gate above cannot \
                 distinguish {n} passes from {} passes",
                n + 1
            );
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![RETURN];
        assert_eq!(run_jit(&prog, 42), 42);
    }
}
