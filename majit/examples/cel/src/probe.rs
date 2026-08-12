//! Arithmetic-loop benchmark for the MAJIT CEL prototype.
//!
//! A serial LCG supplies data-dependent inputs so the loop cannot collapse to
//! a counter-derived closed form. The same bytecode runs through JIT, clean
//! interpreter, and JIT-disabled paths for result and timing comparisons.

use crate::common::*;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

const OP_LOAD: i64 = 0; // [LOAD, imm, dst]
const OP_ADD: i64 = 1; // [ADD, a, b, dst]
const OP_JUMP_IF_ABOVE: i64 = 2; // [JIA, a, b, target_pc]
const OP_RETURN: i64 = 3; // [RETURN, reg]
const OP_MUL: i64 = 4; // [MUL, a, b, dst]
const OP_SUB: i64 = 5; // [SUB, a, b, dst]

struct VmState {
    regs: Vec<i64>,
    /// What `OP_RETURN` hands back.
    ///
    /// The `; state` merge point leaves the loop through `break` before it
    /// assigns the walk's resume pc, so after the loop `pc` still names the
    /// position the walk started from and `program[pc + 1]` — the operand
    /// saying which register holds the result — cannot be read there. The
    /// result has to arrive in `state`.
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
    driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
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
        // `; state` selects the single-executor close: the walk's final state is
        // transferred into `state` here and the native loop resumes at the close
        // pc, instead of discarding the walk outcome and re-running the circuit
        // the walk already executed.
        jit_merge_point!(driver, program, pc; state);
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
            OP_MUL => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] * state.regs[b];
                pc += 4;
            }
            OP_SUB => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] - state.regs[b];
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
            // Stores into `ret` and then leaves through an in-arm `return`,
            // never `{ store; break }`: `classify.rs` `is_break_expr` requires
            // the arm body to be exactly `break`, so a composite body classifies
            // `Lowerable` and its tail `break` reaches `lower_stmt_fallback`,
            // which guards an enclosed `return` but not an enclosed `break` —
            // the statement is inert and is silently dropped, leaving the
            // lowered arm to fall through to the dispatch back-edge.
            OP_RETURN => {
                let r = program[pc + 1] as usize;
                state.ret = state.regs[r];
                return state.ret;
            }
            // Was `_ => break` with the panic below the loop. The loop now has a
            // second way out — the merge point's own `break` on a walk that
            // reached a terminal return — so falling out of it no longer
            // identifies a bad opcode, and the panic moves into the arm that
            // actually saw one.
            _ => panic!("fell off end of code"),
        }
    }
    // Reached only when the merge point broke out on a walk that already ran the
    // terminal opcode, so the result is whatever that opcode parked in `ret`.
    state.ret
}

/// A CLEAN interpreter of the identical bytecode: no jit_merge_point, no
/// can_enter_jit, no driver — the honest "good non-JIT implementation" baseline.
fn clean_interp(program: &Code, num_regs: usize) -> i64 {
    let mut regs = vec![0i64; num_regs];
    let mut pc = 0usize;
    loop {
        match program[pc] {
            OP_LOAD => {
                regs[program[pc + 2] as usize] = program[pc + 1];
                pc += 3;
            }
            OP_ADD => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] + regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_MUL => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] * regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_SUB => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] - regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_JUMP_IF_ABOVE => {
                let tgt = program[pc + 3] as usize;
                if regs[program[pc + 1] as usize] > regs[program[pc + 2] as usize] {
                    pc = tgt;
                } else {
                    pc += 4;
                }
            }
            OP_RETURN => return regs[program[pc + 1] as usize],
            _ => panic!("bad op"),
        }
    }
}

/// Batch program: for `i` in `0..n`, advance `x` by an LCG and accumulate
/// `f(x) = ((x*3)+7)*(x-2)` into `acc`. The LCG makes each row's input distinct
/// and data-dependent (no strength-reduction, no vectorization).
///
/// Registers: r0=i, r1=acc, r2=3, r3=7, r4=2, r5=1, r6=n, r7=x,
///            r8=LCG_A, r9=LCG_C, r10/r11=temps.
fn batch_program(n: i64) -> Vec<i64> {
    vec![
        /* setup */
        OP_LOAD,
        0,
        0, // i = 0
        OP_LOAD,
        0,
        1, // acc = 0
        OP_LOAD,
        3,
        2, // r2 = 3
        OP_LOAD,
        7,
        3, // r3 = 7
        OP_LOAD,
        2,
        4, // r4 = 2
        OP_LOAD,
        1,
        5, // r5 = 1
        OP_LOAD,
        n,
        6, // r6 = n
        OP_LOAD,
        12345,
        7, // x = seed
        OP_LOAD,
        LCG_A,
        8, // r8 = A
        OP_LOAD,
        LCG_C,
        9, // r9 = C
        // @body = pc 30
        OP_MUL,
        7,
        8,
        7, // x = x * A
        OP_ADD,
        7,
        9,
        7, // x = x + C   (LCG step: next input)
        OP_MUL,
        7,
        2,
        10, // t0 = x * 3
        OP_ADD,
        10,
        3,
        10, // t0 = t0 + 7
        OP_SUB,
        7,
        4,
        11, // t1 = x - 2
        OP_MUL,
        10,
        11,
        10, // t0 = t0 * t1  = f(x)
        OP_ADD,
        1,
        10,
        1, // acc = acc + f(x)
        OP_ADD,
        0,
        5,
        0, // i = i + 1
        OP_JUMP_IF_ABOVE,
        6,
        0,
        30, // if n > i goto @body
        OP_RETURN,
        1, // return acc
    ]
}

const NUM_REGS: usize = 12;
const BODY_PC: usize = 30;

fn time_ns_per_eval<F: Fn() -> i64>(n: i64, f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

/// The correctness half, with no timing in it: all three paths must produce the
/// identical (release-wrapped) accumulator, JIT-off must compile nothing, and
/// JIT-on must compile the hot loop.
///
/// `on_compiles == 1` says a trace was minted for the loop. It does not say the
/// trace has a real body — an empty dispatch still compiles one whose whole
/// optimized body is `Finish()` — so the count is the tier-liveness half only.
/// The shape assertion below is the other half: it reads the opcode kinds of
/// the body that count refers to, and fails when the body does not close a
/// loop.
///
/// The count is pinned exactly rather than as `>= 1`, which is what it read
/// until the equality was measured available: this program has one hot loop, so
/// a second compile inside the window is another test's, and `exclusive` in
/// `main.rs` is what makes that observable instead of tolerated. A loose bound
/// here would pass on a foreign trace exactly as `off_compiles` would have
/// failed on one.
///
/// `n` is a parameter because the gate proves the same three properties at any
/// row count past the trace threshold: [`run`] gates at the row count it then
/// times, and the test gates at a fraction of it.
pub(crate) fn run_gates(n: i64) {
    // Sanity: the body really starts at BODY_PC (the backward JIA target).
    assert_eq!(batch_program(1)[BODY_PC], OP_MUL, "BODY_PC out of sync");

    let prog = batch_program(n);

    COMPILES.store(0, Ordering::Relaxed);
    let clean = clean_interp(&prog, NUM_REGS);
    let off = mainloop(&prog, NUM_REGS, JIT_OFF);
    let off_compiles = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    LAST_HAS_JUMP.store(false, Ordering::Relaxed);
    LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
    let on = mainloop(&prog, NUM_REGS, JIT_ON);
    let on_compiles = COMPILES.load(Ordering::Relaxed);
    let shape = majit_metainterp::LoopBodyShape {
        has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
        has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
    };
    assert_eq!(clean, off, "clean vs JIT-off divergence");
    assert_eq!(clean, on, "clean vs JIT-on divergence -> miscompile");
    assert_eq!(off_compiles, 0, "JIT-off must never compile");
    assert_eq!(
        on_compiles, 1,
        "JIT-on must compile the hot loop exactly once"
    );
    // Ordered after the compile count so a tier that never ran fails on the
    // count, which names the cause. Reaching this line means a body exists,
    // so a false here is a real statement about that body — and the reset
    // values above are the failing ones, so a hook that never fired cannot
    // pass this by leaving the flags untouched.
    // `{shape:?}` carries both `LoopBodyShape` fields. `why_not()`'s string is
    // decoration: a rendered reason is a lossy encoding of a compound state,
    // and what it loses is the discrimination — so the failure output must not
    // depend on it alone.
    assert!(
        shape.closes_a_loop(),
        "JIT-on compiled {on_compiles} loop(s) but the body {} ({shape:?}) — a \
         trace was minted for a dispatch that lowers nothing",
        shape.why_not().unwrap_or("closes a loop")
    );
    println!(
        "n = {n}, acc = {on} (clean==off==on ok), compiles: off={off_compiles} on={on_compiles}, body closes a loop"
    );
}

pub fn run() {
    let n: i64 = 20_000_000;
    run_gates(n);
    let prog = batch_program(n);

    // Interleaved A/B/C, several rounds (interleave per round to average drift).
    let rounds = 9;
    let (mut a_on, mut b_clean, mut c_off) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        b_clean.push(time_ns_per_eval(n, || clean_interp(&prog, NUM_REGS)));
        c_off.push(time_ns_per_eval(n, || mainloop(&prog, NUM_REGS, JIT_OFF)));
        a_on.push(time_ns_per_eval(n, || mainloop(&prog, NUM_REGS, JIT_ON)));
    }

    let a = median(a_on);
    let b = median(b_clean);
    let c = median(c_off);
    println!("(a) majit JIT-on   : {a:.3} ns/eval");
    println!("(b) clean interp   : {b:.3} ns/eval");
    println!("(c) majit JIT-off  : {c:.3} ns/eval");
    println!();
    println!("HONEST ratio (b)/(a) clean-interp vs trace : {:.2}x", b / a);
    println!("       ratio (c)/(a) majit-interp vs trace : {:.2}x", c / a);
    println!(
        "  instrumentation (c)/(b) majit-interp cost : {:.2}x",
        c / b
    );
    println!(
        "kill bar (b)/(a) >=3x: {}",
        if b / a >= 3.0 { "PASS" } else { "FAIL" }
    );
}
