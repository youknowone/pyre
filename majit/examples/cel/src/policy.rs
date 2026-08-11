//! Slot-resolved policy benchmark for the MAJIT CEL prototype.
//!
//! It models `account.balance >= txn.amount && !account.frozen` with fields
//! pre-resolved to register slots. Skewed and unbiased data sets exercise both
//! guard-stable and frequent-guard-failure behavior across JIT, clean
//! interpreter, and JIT-disabled execution.

use crate::common::*;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

const OP_LOAD: i64 = 0; // [LOAD, imm, dst]
const OP_ADD: i64 = 1; // [ADD, a, b, dst]
const OP_SUB: i64 = 2; // [SUB, a, b, dst]
const OP_MUL: i64 = 3; // [MUL, a, b, dst]
const OP_GE: i64 = 4; // [GE, a, b, dst]  dst = (a>=b) as {0,1}
const OP_EQ: i64 = 5; // [EQ, a, b, dst]  dst = (a==b) as {0,1}
const OP_JUMP_IF_ABOVE: i64 = 6; // [JIA, a, b, target_pc]
const OP_RETURN: i64 = 7; // [RETURN, reg]

const SEED: i64 = 0x2545F4914F6CDD1D;
const LO: i64 = i64::MIN + 1; // `bal >= LO` ~always true, not foldable
const HI: i64 = i64::MAX; // `draw >= HI` ~always false, not foldable

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
    driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after, _opcodes| {
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
            OP_SUB => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] - state.regs[b];
                pc += 4;
            }
            OP_MUL => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] * state.regs[b];
                pc += 4;
            }
            OP_GE => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] >= state.regs[b] { 1 } else { 0 };
                pc += 4;
            }
            OP_EQ => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] == state.regs[b] { 1 } else { 0 };
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
            OP_SUB => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] - regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_MUL => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] * regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_GE => {
                regs[program[pc + 3] as usize] =
                    if regs[program[pc + 1] as usize] >= regs[program[pc + 2] as usize] {
                        1
                    } else {
                        0
                    };
                pc += 4;
            }
            OP_EQ => {
                regs[program[pc + 3] as usize] =
                    if regs[program[pc + 1] as usize] == regs[program[pc + 2] as usize] {
                        1
                    } else {
                        0
                    };
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

// Registers: r0=i, r1=acc, r2=n, r3=one, r4=zero, r5=x(lcg), r6=A, r7=C,
//            r8=LO, r9=HI, r10=bal, r11=amt/aux, r12=frozen, r13=t0, r14=t1.
const NUM_REGS: usize = 16;
const BODY_PC: usize = 30; // 10 LOADs * 3 words

fn setup(n: i64) -> Vec<i64> {
    vec![
        OP_LOAD, 0, 0, // i = 0
        OP_LOAD, 0, 1, // acc = 0
        OP_LOAD, n, 2, // n
        OP_LOAD, 1, 3, // one
        OP_LOAD, 0, 4, // zero
        OP_LOAD, SEED, 5, // x = seed
        OP_LOAD, LCG_A, 6, // A
        OP_LOAD, LCG_C, 7, // C
        OP_LOAD, LO, 8, // LO threshold
        OP_LOAD, HI, 9, // HI threshold
    ]
}

/// SKEWED: predicate ~always true via genuine, non-foldable comparisons against
/// extreme constants. Guard-stable = realistic skewed policy data.
fn skewed_program(n: i64) -> Vec<i64> {
    let mut p = setup(n);
    p.extend_from_slice(&[
        // @body = pc 30
        OP_MUL,
        5,
        6,
        5, //  x = x*A
        OP_ADD,
        5,
        7,
        5, //  x = x+C
        OP_SUB,
        5,
        4,
        10, // bal = x
        OP_GE,
        10,
        8,
        13, // t0 = bal >= LO      (~always true, not foldable)
        OP_MUL,
        5,
        6,
        5, //  x = x*A  (fresh draw)
        OP_ADD,
        5,
        7,
        5, //  x = x+C
        OP_GE,
        5,
        9,
        12, //  frozen = (x >= HI)  (~always false)
        OP_EQ,
        12,
        4,
        14, // t1 = (frozen == 0)  (~always true)
        OP_MUL,
        13,
        14,
        13, // pass = t0 * t1
        OP_ADD,
        1,
        13,
        1, // acc += pass
        OP_ADD,
        0,
        3,
        0, //  i += 1
        OP_JUMP_IF_ABOVE,
        2,
        0,
        BODY_PC as i64,
        OP_RETURN,
        1,
    ]);
    p
}

/// UNBIASED: bal, amt, frozen independent full-range LCG draws (~50/50 guards).
fn unbiased_program(n: i64) -> Vec<i64> {
    let mut p = setup(n);
    p.extend_from_slice(&[
        // @body = pc 30
        OP_MUL,
        5,
        6,
        5, //  x = x*A
        OP_ADD,
        5,
        7,
        5, //  x = x+C
        OP_SUB,
        5,
        4,
        10, // bal = x
        OP_MUL,
        5,
        6,
        5, //  x = x*A
        OP_ADD,
        5,
        7,
        5, //  x = x+C
        OP_SUB,
        5,
        4,
        11, // amt = x
        OP_MUL,
        5,
        6,
        5, //  x = x*A
        OP_ADD,
        5,
        7,
        5, //  x = x+C
        OP_GE,
        5,
        4,
        12, //  frozen = (x >= 0)   (~50%)
        OP_GE,
        10,
        11,
        13, // t0 = bal >= amt     (~50%)
        OP_EQ,
        12,
        4,
        14, // t1 = (frozen == 0)
        OP_MUL,
        13,
        14,
        13, // pass = t0 * t1
        OP_ADD,
        1,
        13,
        1, // acc += pass
        OP_ADD,
        0,
        3,
        0, //  i += 1
        OP_JUMP_IF_ABOVE,
        2,
        0,
        BODY_PC as i64,
        OP_RETURN,
        1,
    ]);
    p
}

const THRESH: i64 = 0; // element predicate: `elem >= 0` (~50%)

/// COMPREHENSION: `acc = sum over rows of size(list.filter(x, x >= 0))`, where
/// each row's `list` is `listlen` elements synthesized by the LCG. This is a
/// real NESTED loop — an inner comprehension loop (backward jump to pc 36)
/// inside the outer batch loop (backward jump to pc 30) — so it exercises
/// majit's nested-loop tracing (two can_enter_jit merge points).
///
/// Registers: r0=i, r1=acc, r2=n, r3=one, r4=zero, r5=x, r6=A, r7=C,
///            r8=listlen, r9=THRESH, r10=j, r11=innercount, r12=t.
fn comprehension_program(n: i64, listlen: i64) -> Vec<i64> {
    vec![
        OP_LOAD,
        0,
        0, // i = 0
        OP_LOAD,
        0,
        1, // acc = 0
        OP_LOAD,
        n,
        2, // n
        OP_LOAD,
        1,
        3, // one
        OP_LOAD,
        0,
        4, // zero
        OP_LOAD,
        SEED,
        5, // x = seed
        OP_LOAD,
        LCG_A,
        6, // A
        OP_LOAD,
        LCG_C,
        7, // C
        OP_LOAD,
        listlen,
        8, // list length
        OP_LOAD,
        THRESH,
        9, // element threshold
        // @outer_body = pc 30
        OP_LOAD,
        0,
        10, // j = 0
        OP_LOAD,
        0,
        11, // innercount = 0
        // @inner_body = pc 36
        OP_MUL,
        5,
        6,
        5, //  x = x*A
        OP_ADD,
        5,
        7,
        5, //  x = x+C          (next list element)
        OP_GE,
        5,
        9,
        12, //  t = (x >= THRESH)
        OP_ADD,
        11,
        12,
        11, // innercount += t
        OP_ADD,
        10,
        3,
        10, // j += 1
        OP_JUMP_IF_ABOVE,
        8,
        10,
        36, // if LISTLEN > j goto @inner_body
        OP_ADD,
        1,
        11,
        1, // acc += innercount
        OP_ADD,
        0,
        3,
        0, //  i += 1
        OP_JUMP_IF_ABOVE,
        2,
        0,
        30, // if n > i goto @outer_body
        OP_RETURN,
        1,
    ]
}

/// UNROLLED comprehension: same `size(list.filter(x, x >= 0))` but the inner
/// loop is UNROLLED into `listlen` inline element-checks (no inner back-edge),
/// as a real cell-majit would compile a fixed/green-length list. This collapses
/// the nested loop to a SINGLE (outer) loop — the question is whether that
/// removes the per-row deopt thrash the looped version suffers.
///
/// Registers: r0=i, r1=acc, r2=n, r3=one, r4=zero, r5=x, r6=A, r7=C,
///            r8=THRESH, r9=t.
fn comprehension_unrolled_program(n: i64, listlen: i64) -> Vec<i64> {
    let mut p = vec![
        OP_LOAD, 0, 0, // i = 0
        OP_LOAD, 0, 1, // acc = 0
        OP_LOAD, n, 2, // n
        OP_LOAD, 1, 3, // one
        OP_LOAD, 0, 4, // zero
        OP_LOAD, SEED, 5, // x = seed
        OP_LOAD, LCG_A, 6, // A
        OP_LOAD, LCG_C, 7, // C
        OP_LOAD, THRESH, 8, // element threshold
    ];
    let body_pc = p.len() as i64; // 27
    for _ in 0..listlen {
        p.extend_from_slice(&[
            OP_MUL, 5, 6, 5, // x = x*A
            OP_ADD, 5, 7, 5, // x = x+C
            OP_GE, 5, 8, 9, //  t = (x >= THRESH)
            OP_ADD, 1, 9, 1, // acc += t
        ]);
    }
    p.extend_from_slice(&[
        OP_ADD,
        0,
        3,
        0, // i += 1
        OP_JUMP_IF_ABOVE,
        2,
        0,
        body_pc,
        OP_RETURN,
        1,
    ]);
    p
}

fn time_ns<F: Fn() -> i64>(n: i64, f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

fn run_regime(name: &str, prog: &[i64], n: i64, rounds: usize) {
    COMPILES.store(0, Ordering::Relaxed);
    let clean = clean_interp(prog, NUM_REGS);
    let off = mainloop(prog, NUM_REGS, JIT_OFF);
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let on = mainloop(prog, NUM_REGS, JIT_ON);
    let on_c = COMPILES.load(Ordering::Relaxed);
    assert_eq!(clean, off, "{name}: clean vs JIT-off divergence");
    assert_eq!(
        clean, on,
        "{name}: clean vs JIT-on divergence -> miscompile"
    );
    assert_eq!(off_c, 0, "{name}: JIT-off must not compile");

    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        b.push(time_ns(n, || clean_interp(prog, NUM_REGS)));
        c.push(time_ns(n, || mainloop(prog, NUM_REGS, JIT_OFF)));
        a.push(time_ns(n, || mainloop(prog, NUM_REGS, JIT_ON)));
    }
    let (a, b, c) = (median(a), median(b), median(c));
    println!(
        "[{name}] pass-rate={:.1}%  compiles(on)={on_c}",
        100.0 * on as f64 / n as f64
    );
    println!("  (a) JIT-on {a:.3}  (b) clean {b:.3}  (c) JIT-off {c:.3}  ns/eval");
    println!(
        "  (b)/(a) clean-vs-trace = {:.2}x   kill-bar>=3x: {}",
        b / a,
        if b / a >= 3.0 { "PASS" } else { "FAIL" }
    );
}

pub fn run() {
    let n: i64 = 5_000_000;
    println!("policy: account.balance >= txn.amount && !account.frozen  (slot-resolved)\n");
    run_regime("SKEWED  ", &skewed_program(n), n, 5);
    println!();
    run_regime("UNBIASED", &unbiased_program(n), n, 5);

    // Comprehension: nested loop, ns/eval is per-ROW (each row runs `listlen`
    // inner iterations). Sweep listlen to find where the inner loop amortizes
    // its trace entry/exit. Lean (few rounds, small total work) because JIT-on
    // may thrash. Total inner work n*listlen kept ~1M.
    println!("\ncomprehension (INNER LOOP): size(list.filter(x, x >= 0))  — sweep list length\n");
    for &listlen in &[8_i64, 64, 1024] {
        let nc = (1_000_000 / listlen).max(4_000);
        run_regime(
            &format!("LOOP    len={listlen:<4}"),
            &comprehension_program(nc, listlen),
            nc,
            3,
        );
    }

    println!("\ncomprehension (UNROLLED, green length) — same work, inner loop unrolled\n");
    for &listlen in &[8_i64, 64] {
        let nc = (8_000_000 / listlen).max(50_000);
        run_regime(
            &format!("UNROLL  len={listlen:<4}"),
            &comprehension_unrolled_program(nc, listlen),
            nc,
            5,
        );
    }
}
