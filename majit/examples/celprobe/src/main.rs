//! cell-majit de-risk kill-test (issue #357). Contains ZERO CEL code.
//!
//! Question: does a compiled majit trace of a straight-line arithmetic body,
//! evaluated over a stream of distinct inputs, beat a CLEAN interpreter of the
//! same bytecode by the >=3x bar cometkim's cel-jit hit on simple_arithmetic
//! (8.3x)? If majit cannot clear 3x here — on its structural best case, a tight
//! arithmetic loop — applying it to cel-rust is dead before any CEL work.
//!
//! FAITHFULNESS: the "input" per row must be data-dependent, not derived from
//! the loop counter. A counter-driven polynomial lets the optimizer
//! strength-reduce / vectorize the whole loop into near-nothing (an early
//! version measured a bogus 406x this way). So each row's input `x` is advanced
//! by an LCG (`x = x*A + C`), an inherently serial recurrence the compiler
//! cannot fold or vectorize — modelling a real stream of distinct CEL inputs.
//! The body then computes `f(x) = ((x*3)+7)*(x-2)` and accumulates it.
//!
//! Three measurements, all on the identical bytecode:
//!   (a) majit JIT-on   — the compiled trace (small threshold).
//!   (b) clean interp   — a plain `match` loop, NO majit macro/instrumentation.
//!                        The honest baseline: a good non-JIT implementation.
//!   (c) majit JIT-off  — the same instrumented mainloop, compilation disabled
//!                        (`u32::MAX` threshold). Shows jit_merge_point cost.
//! The meaningful ratio is (b)/(a): clean interpreter vs compiled trace.
//!
//! RELEASE ONLY: `acc`/`x` wrap i64 and all three paths must wrap identically
//! (`+`/`*` in release), so the equality gate needs overflow checks off.

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// The env: an i64-word bytecode stream (8-byte elements).
pub type Code = [i64];

const OP_LOAD: i64 = 0; // [LOAD, imm, dst]
const OP_ADD: i64 = 1; // [ADD, a, b, dst]
const OP_JUMP_IF_ABOVE: i64 = 2; // [JIA, a, b, target_pc]
const OP_RETURN: i64 = 3; // [RETURN, reg]
const OP_MUL: i64 = 4; // [MUL, a, b, dst]
const OP_SUB: i64 = 5; // [SUB, a, b, dst]

/// Hot loops majit compiled — evidence the JIT tier traced + compiled.
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

// LCG (Knuth MMIX) constants — a serial recurrence, unfoldable/unvectorizable.
const LCG_A: i64 = 6364136223846793005;
const LCG_C: i64 = 1442695040888963407;

struct VmState {
    regs: Vec<i64>,
}

#[majit_macros::jit_interp(
    state = VmState,
    env = Code,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
fn mainloop(program: &Code, num_regs: usize, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<VmState> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let _stacksize: i32 = 0;
    let mut state = VmState {
        regs: vec![0; num_regs],
    };

    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    loop {
        jit_merge_point!();
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
            OP_RETURN => {
                let r = program[pc + 1] as usize;
                return state.regs[r];
            }
            _ => break,
        }
    }
    panic!("fell off end of code");
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
        OP_LOAD, 0, 0, // i = 0
        OP_LOAD, 0, 1, // acc = 0
        OP_LOAD, 3, 2, // r2 = 3
        OP_LOAD, 7, 3, // r3 = 7
        OP_LOAD, 2, 4, // r4 = 2
        OP_LOAD, 1, 5, // r5 = 1
        OP_LOAD, n, 6, // r6 = n
        OP_LOAD, 12345, 7, // x = seed
        OP_LOAD, LCG_A, 8, // r8 = A
        OP_LOAD, LCG_C, 9, // r9 = C
        // @body = pc 30
        OP_MUL, 7, 8, 7, // x = x * A
        OP_ADD, 7, 9, 7, // x = x + C   (LCG step: next input)
        OP_MUL, 7, 2, 10, // t0 = x * 3
        OP_ADD, 10, 3, 10, // t0 = t0 + 7
        OP_SUB, 7, 4, 11, // t1 = x - 2
        OP_MUL, 10, 11, 10, // t0 = t0 * t1  = f(x)
        OP_ADD, 1, 10, 1, // acc = acc + f(x)
        OP_ADD, 0, 5, 0, // i = i + 1
        OP_JUMP_IF_ABOVE, 6, 0, 30, // if n > i goto @body
        OP_RETURN, 1, // return acc
    ]
}

const NUM_REGS: usize = 12;
const BODY_PC: usize = 30;
const JIT_ON: u32 = 8;
const JIT_OFF: u32 = u32::MAX;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ns_per_eval<F: Fn() -> i64>(n: i64, f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

fn main() {
    // Sanity: the body really starts at BODY_PC (the backward JIA target).
    assert_eq!(batch_program(1)[BODY_PC], OP_MUL, "BODY_PC out of sync");

    let n: i64 = 20_000_000;
    let prog = batch_program(n);

    // Correctness gate: all three paths must produce the identical
    // (release-wrapped) accumulator.
    COMPILES.store(0, Ordering::Relaxed);
    let clean = clean_interp(&prog, NUM_REGS);
    let off = mainloop(&prog, NUM_REGS, JIT_OFF);
    let off_compiles = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let on = mainloop(&prog, NUM_REGS, JIT_ON);
    let on_compiles = COMPILES.load(Ordering::Relaxed);
    assert_eq!(clean, off, "clean vs JIT-off divergence");
    assert_eq!(clean, on, "clean vs JIT-on divergence -> miscompile");
    assert_eq!(off_compiles, 0, "JIT-off must never compile");
    assert!(on_compiles >= 1, "JIT-on must compile the hot loop");
    println!("n = {n}, acc = {on} (clean==off==on ok), compiles: off={off_compiles} on={on_compiles}");

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
    println!("  instrumentation (c)/(b) majit-interp cost : {:.2}x", c / b);
    println!(
        "kill bar (b)/(a) >=3x: {}",
        if b / a >= 3.0 { "PASS" } else { "FAIL" }
    );
}
