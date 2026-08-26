//! Column-read benchmark for the MAJIT CEL prototype.
//!
//! The programs load real external `i64` columns at data-dependent row
//! indices through `raw_load_i`, then compare JIT, clean-interpreter, and
//! JIT-disabled execution. The fixtures cover a one-column sum and a
//! two-column comparison policy.

use crate::common::*;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

const OP_LOAD: i64 = 0; // [LOAD, imm, dst]
const OP_ADD: i64 = 1; // [ADD, a, b, dst]
const OP_MUL: i64 = 2; // [MUL, a, b, dst]
const OP_JUMP_IF_ABOVE: i64 = 3; // [JIA, a, b, target_pc]
const OP_RETURN: i64 = 4; // [RETURN, reg]
const OP_COL_LOAD: i64 = 5; // [COL_LOAD, base_reg, ea_reg, dst]  dst = *(regs[base]+regs[ea])
const OP_GE: i64 = 6; // [GE, a, b, dst]  dst = (regs[a] >= regs[b]) as {0,1}

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
    driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    driver.set_on_trace_abort(|_green_key, _permanent| {
        ABORTS.fetch_add(1, Ordering::Relaxed);
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
            OP_COL_LOAD => {
                let base_reg = program[pc + 1] as usize;
                let ea_reg = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                let base = state.regs[base_reg];
                let ea = state.regs[ea_reg];
                state.regs[d] = majit_raw_load_i64(base, ea);
                pc += 4;
            }
            OP_GE => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] >= state.regs[b] { 1 } else { 0 };
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

/// Clean interpreter of the identical bytecode — the honest "good non-JIT
/// implementation" baseline. Reads columns via the same raw load a real
/// columnar interpreter would use (fair: same work, no JIT machinery).
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
            OP_COL_LOAD => {
                let base = regs[program[pc + 1] as usize];
                let ea = regs[program[pc + 2] as usize];
                regs[program[pc + 3] as usize] = majit_raw_load_i64(base, ea);
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

/// SUM: `acc = sum over i in 0..n of col[i]` at the RED row index.
/// Registers: r0=i, r1=acc, r2=n, r3=one, r4=stride(8), r5=ea, r6=v, r7=base.
const SUM_REGS: usize = 8;
const SUM_BODY_PC: usize = 18; // 6 LOADs * 3

fn sum_program(n: i64, base: i64) -> Vec<i64> {
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
        8,
        4, // stride
        OP_LOAD,
        base,
        7, // base = buffer address (loop-invariant register)
        // @body = pc 18
        OP_MUL,
        0,
        4,
        5, //  ea = i * 8
        OP_COL_LOAD,
        7,
        5,
        6, // v = *(base + ea)   [RED-index columnar read]
        OP_ADD,
        1,
        6,
        1, //  acc += v
        OP_ADD,
        0,
        3,
        0, //  i += 1
        OP_JUMP_IF_ABOVE,
        2,
        0,
        SUM_BODY_PC as i64,
        OP_RETURN,
        1,
    ]
}

/// POLICY: `acc = count of i in 0..n where col_a[i] >= col_b[i]` — a real
/// two-column cel-style predicate over actual data buffers.
/// Registers: r0=i, r1=acc, r2=n, r3=one, r4=stride, r5=ea, r6=va, r7=vb,
///            r8=base_a, r9=base_b, r10=t.
const POL_REGS: usize = 12;
const POL_BODY_PC: usize = 21; // 7 LOADs * 3

fn policy_program(n: i64, base_a: i64, base_b: i64) -> Vec<i64> {
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
        8,
        4, // stride
        OP_LOAD,
        base_a,
        8, // base_a
        OP_LOAD,
        base_b,
        9, // base_b
        // @body = pc 21
        OP_MUL,
        0,
        4,
        5, //   ea = i * 8
        OP_COL_LOAD,
        8,
        5,
        6, // va = col_a[i]
        OP_COL_LOAD,
        9,
        5,
        7, // vb = col_b[i]
        OP_GE,
        6,
        7,
        10, //   t = (va >= vb)
        OP_ADD,
        1,
        10,
        1, //  acc += t
        OP_ADD,
        0,
        3,
        0, //   i += 1
        OP_JUMP_IF_ABOVE,
        2,
        0,
        POL_BODY_PC as i64,
        OP_RETURN,
        1,
    ]
}

fn make_col(n: i64, seed: i64) -> Vec<i64> {
    let mut v = Vec::with_capacity(n as usize);
    let mut x = seed;
    for _ in 0..n {
        x = x.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        v.push(x);
    }
    v
}

fn time_ns_per_row<F: Fn() -> i64>(n: i64, f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

/// How many rows the equality gate runs. Far below the timing row count: the
/// gate's three properties hold at any length past the trace threshold.
fn gate_n() -> i64 {
    std::env::var("CELGATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300_000)
}

/// One program's correctness + tier-liveness gate, with no timing in it.
///
/// `on_c >= 1` says a trace was minted for the loop. It does not say the trace
/// has a real body — an empty dispatch still compiles one whose whole optimized
/// body is `Finish()` — so this is the liveness half only.
fn gate_program(label: &str, num_regs: usize, prog_at: &dyn Fn(i64) -> Vec<i64>) {
    let gn = gate_n();
    let gprog = prog_at(gn);
    COMPILES.store(0, Ordering::Relaxed);
    ABORTS.store(0, Ordering::Relaxed);
    let clean = clean_interp(&gprog, num_regs);
    let off = mainloop(&gprog, num_regs, JIT_OFF);
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    ABORTS.store(0, Ordering::Relaxed);
    let on = mainloop(&gprog, num_regs, JIT_ON);
    let on_c = COMPILES.load(Ordering::Relaxed);
    let on_a = ABORTS.load(Ordering::Relaxed);
    assert_eq!(clean, off, "{label}: clean vs JIT-off divergence");
    assert_eq!(
        clean, on,
        "{label}: clean vs JIT-on divergence -> miscompile"
    );
    assert_eq!(off_c, 0, "{label}: JIT-off must never compile");
    // Was a printed `!!` note and an early return, which is a diagnostic and not
    // a gate: the three assertions above are all satisfied by the interpreter
    // answering alone, so with the JIT tier inert this probe printed its warning
    // and still exited 0.
    assert!(
        on_c >= 1,
        "{label}: JIT-on compiled nothing (aborts={on_a}) — the raw_load \
         red-index read does not close the trace"
    );
    println!(
        "[{label} gate n={gn}] result={on} (clean==off==on ok)  compiles: off={off_c} on={on_c}  aborts(on)={on_a}"
    );
}

/// Run one program: the gate above, then interleaved A/B/C timing (large n).
/// `hold` keeps the backing column buffers alive.
fn run_program(
    label: &str,
    num_regs: usize,
    prog_at: &dyn Fn(i64) -> Vec<i64>,
    hold: &[&Vec<i64>],
) {
    gate_program(label, num_regs, prog_at);

    let n: i64 = 20_000_000;
    let prog = prog_at(n);
    let rounds = 9;
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        b.push(time_ns_per_row(n, || clean_interp(&prog, num_regs)));
        c.push(time_ns_per_row(n, || mainloop(&prog, num_regs, JIT_OFF)));
        a.push(time_ns_per_row(n, || mainloop(&prog, num_regs, JIT_ON)));
    }
    let (a, b, c) = (median(a), median(b), median(c));
    println!("  (a) JIT-on {a:.3}  (b) clean {b:.3}  (c) JIT-off {c:.3}  ns/row");
    println!(
        "  (b)/(a) clean-vs-trace = {:.2}x   kill-bar>=3x: {}",
        b / a,
        if b / a >= 3.0 { "PASS" } else { "FAIL" }
    );
    black_box(hold);
}

/// REAL data columns: LCG-filled, distinct, non-foldable. The LCG runs from a
/// fixed seed, so a short column is a prefix of a long one and the gate reads
/// the same rows whatever length was allocated for it.
fn make_cols(n: i64) -> (Vec<i64>, Vec<i64>) {
    (
        make_col(n, 0x2545F4914F6CDD1D),
        make_col(n, 0x9E3779B97F4A7C15u64 as i64),
    )
}

type ProgramSpec = (&'static str, usize, Box<dyn Fn(i64) -> Vec<i64>>);

/// The programs this probe covers, as `(label, num_regs, builder)`. `run` gates
/// and then times each; [`run_gates`] gates each and stops there. Both walk this
/// one list, so a program added here reaches the test as well as the binary.
fn programs(base_a: i64, base_b: i64) -> Vec<ProgramSpec> {
    vec![
        (
            "SUM   ",
            SUM_REGS,
            Box::new(move |k| sum_program(k, base_a)),
        ),
        (
            "POLICY",
            POL_REGS,
            Box::new(move |k| policy_program(k, base_a, base_b)),
        ),
    ]
}

/// Every program's gate at [`gate_n`] rows, with none of the timing. The columns
/// are allocated to the gate's own length rather than the timing length — two
/// 20M-row buffers is 320 MB the gate never reads past the first `gate_n` rows
/// of.
#[cfg(test)]
pub(crate) fn run_gates() {
    let (col_a, col_b) = make_cols(gate_n());
    let base_a = col_a.as_ptr() as i64;
    let base_b = col_b.as_ptr() as i64;
    for (label, num_regs, prog_at) in programs(base_a, base_b) {
        gate_program(label, num_regs, &prog_at);
    }
    black_box((&col_a, &col_b));
}

pub fn run() {
    let n: i64 = 20_000_000;
    let (col_a, col_b) = make_cols(n);
    let base_a = col_a.as_ptr() as i64;
    let base_b = col_b.as_ptr() as i64;

    println!("columnar red-index reads via raw_load_i (base in register file)\n");
    for (i, (label, num_regs, prog_at)) in programs(base_a, base_b).into_iter().enumerate() {
        if i > 0 {
            println!();
        }
        run_program(label, num_regs, &prog_at, &[&col_a, &col_b]);
    }
}
