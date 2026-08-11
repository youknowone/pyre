//! Float register machine — the honest fair-win probe for the float macro path.
//!
//! This machine originally fused raw_load+add+i++ into ONE opcode; that made the
//! clean interpreter a single-op memory-bandwidth-bound loop with no dispatch
//! for the JIT to eliminate, so the JIT could not win (0.62x). That was a
//! kernel artifact, not a float-JIT defect. This rewrite mirrors the `column`
//! probe's honest structure: a PER-OP bytecode machine whose clean interpreter
//! pays real dispatch cost, so the win is attributable to compilation.
//!
//! Addressing (i, n, base_a, base_b) lives in SCALAR int state fields; values
//! (column reads, accumulator) live in a `[float; virt]` register bank. Three
//! programs sweep the compute:memory ratio:
//!   ACC     — sum(a[i])              1 load,  1 float add   (memory-bound ref)
//!   DOT     — sum(a[i]*b[i])         2 loads, 1 mul + 1 add
//!   COMPUTE — sum(a*b - a*a + b*b)   2 loads, 5 float ops   (compute-bound)

use crate::common::{ABORTS, COMPILES, Code, JIT_OFF, JIT_ON, majit_raw_load_f, median};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

const OP_LOAD_A: i64 = 0; // [OP, dst]        regs[dst] = a[i]
const OP_LOAD_B: i64 = 1; // [OP, dst]        regs[dst] = b[i]
const OP_MUL: i64 = 2; // [OP, a, b, dst]  regs[dst] = regs[a] * regs[b]
const OP_ADD: i64 = 3; // [OP, a, b, dst]  regs[dst] = regs[a] + regs[b]
const OP_SUB: i64 = 4; // [OP, a, b, dst]  regs[dst] = regs[a] - regs[b]
const OP_INC: i64 = 5; // [OP]             i += 1
const OP_JUMP_IF: i64 = 6; // [OP, target]     if i < n jump to target
const OP_RETURN: i64 = 7; // [OP, reg]        return regs[reg]

const ACC: usize = 0;
const R0: usize = 1;
const R1: usize = 2;
const R2: usize = 3;
const R3: usize = 4;
const R4: usize = 5;
const R5: usize = 6;
const R6: usize = 7;
const NUM_REGS: usize = 8;

const BODY_PC: usize = 0;

struct VmState {
    i: i64,
    n: i64,
    base_a: i64,
    base_b: i64,
    regs: Vec<f64>,
}

#[majit_macros::jit_interp(
    state = VmState,
    env = Code,
    greens = [pc, program],
    state_fields = {
        i: int,
        n: int,
        base_a: int,
        base_b: int,
        regs: [float; virt],
    },
)]
fn mainloop(program: &Code, base_a: i64, base_b: i64, n: i64, threshold: u32) -> f64 {
    let mut driver: majit_metainterp::JitDriver<VmState> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    driver.set_on_trace_abort(|_green_key, _permanent| {
        ABORTS.fetch_add(1, Ordering::Relaxed);
    });

    let mut pc: usize = 0;
    let mut state = VmState {
        i: 0,
        n,
        base_a,
        base_b,
        regs: vec![0.0; NUM_REGS],
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
            OP_LOAD_A => {
                let dst = program[pc + 1] as usize;
                let ea = state.i * 8;
                state.regs[dst] = majit_raw_load_f(state.base_a, ea);
                pc += 2;
            }
            OP_LOAD_B => {
                let dst = program[pc + 1] as usize;
                let ea = state.i * 8;
                state.regs[dst] = majit_raw_load_f(state.base_b, ea);
                pc += 2;
            }
            OP_MUL => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] * state.regs[b];
                pc += 4;
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
            OP_INC => {
                state.i += 1;
                pc += 1;
            }
            OP_JUMP_IF => {
                let tgt = program[pc + 1] as usize;
                if state.n > state.i {
                    can_enter_jit!(driver, tgt, &mut state, program, || {});
                    pc = tgt;
                    continue;
                }
                pc += 2;
            }
            OP_RETURN => return state.regs[program[pc + 1] as usize],
            _ => panic!("bad opcode {opcode}"),
        }
    }
}

/// Clean interpreter of the identical bytecode — the honest per-op non-JIT
/// baseline. Same columns, same raw loads, same accumulate order (so bit-exact
/// versus the JIT), only without any JIT machinery.
fn clean_interp(program: &Code, base_a: i64, base_b: i64, n: i64) -> f64 {
    let mut i = 0i64;
    let mut regs = [0.0f64; NUM_REGS];
    let mut pc = 0usize;
    loop {
        match program[pc] {
            OP_LOAD_A => {
                regs[program[pc + 1] as usize] = majit_raw_load_f(base_a, i * 8);
                pc += 2;
            }
            OP_LOAD_B => {
                regs[program[pc + 1] as usize] = majit_raw_load_f(base_b, i * 8);
                pc += 2;
            }
            OP_MUL => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] * regs[program[pc + 2] as usize];
                pc += 4;
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
            OP_INC => {
                i += 1;
                pc += 1;
            }
            OP_JUMP_IF => {
                let tgt = program[pc + 1] as usize;
                if n > i {
                    pc = tgt;
                } else {
                    pc += 2;
                }
            }
            OP_RETURN => return regs[program[pc + 1] as usize],
            op => panic!("bad opcode {op}"),
        }
    }
}

// ── Float comparison policy machine ──────────────────────────────────────
// Mirrors the real cel flagship shape: transient float column loads, a float
// comparison MATERIALIZED to an int bool, and an int count — no float bank.
// A separate `#[jit_interp]` machine lives in its own module (the macro emits
// module-level items that would collide with the value machine above).
mod count {
    use super::*;

    const OP_COUNT_GE: i64 = 0; // if a[i] >= b[i] { count += 1 }
    const OP_JUMP_IF_C: i64 = 1; // [OP, target]  if i < n jump to target
    const OP_RETURN_COUNT: i64 = 2; // return count

    struct CountState {
        i: i64,
        n: i64,
        base_a: i64,
        base_b: i64,
        count: i64,
    }

    #[majit_macros::jit_interp(
    state = CountState,
    env = Code,
    greens = [pc, program],
    state_fields = {
        i: int,
        n: int,
        base_a: int,
        base_b: int,
        count: int,
    },
)]
    fn mainloop_count(program: &Code, base_a: i64, base_b: i64, n: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<CountState> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_green_key, _ob, _oa| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_trace_abort(|_green_key, _permanent| {
            ABORTS.fetch_add(1, Ordering::Relaxed);
        });

        let mut pc: usize = 0;
        let mut state = CountState {
            i: 0,
            n,
            base_a,
            base_b,
            count: 0,
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
                OP_COUNT_GE => {
                    let ea = state.i * 8;
                    let pa = majit_raw_load_f(state.base_a, ea);
                    let pb = majit_raw_load_f(state.base_b, ea);
                    // Float comparison materialized to an int bool, accumulated
                    // (the cel batch-policy shape: `count += policy_bool`, no
                    // conditional branch on the compare result).
                    state.count += (pa >= pb) as i64;
                    state.i += 1;
                    pc += 1;
                }
                OP_JUMP_IF_C => {
                    let tgt = program[pc + 1] as usize;
                    if state.n > state.i {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                        pc = tgt;
                        continue;
                    }
                    pc += 2;
                }
                OP_RETURN_COUNT => return state.count,
                _ => panic!("bad opcode {opcode}"),
            }
        }
    }

    fn clean_interp_count(program: &Code, base_a: i64, base_b: i64, n: i64) -> i64 {
        let mut i = 0i64;
        let mut count = 0i64;
        let mut pc = 0usize;
        loop {
            match program[pc] {
                OP_COUNT_GE => {
                    let ea = i * 8;
                    count += (majit_raw_load_f(base_a, ea) >= majit_raw_load_f(base_b, ea)) as i64;
                    i += 1;
                    pc += 1;
                }
                OP_JUMP_IF_C => {
                    let tgt = program[pc + 1] as usize;
                    if n > i {
                        pc = tgt;
                    } else {
                        pc += 2;
                    }
                }
                OP_RETURN_COUNT => return count,
                op => panic!("bad opcode {op}"),
            }
        }
    }

    fn count_program() -> Vec<i64> {
        vec![OP_COUNT_GE, OP_JUMP_IF_C, 0, OP_RETURN_COUNT]
    }

    pub(super) fn run_gate_count(label: &str, n: i64, col_a: &[f64], col_b: &[f64]) -> i64 {
        let prog = count_program();
        let base_a = col_a.as_ptr() as i64;
        let base_b = col_b.as_ptr() as i64;

        COMPILES.store(0, Ordering::Relaxed);
        ABORTS.store(0, Ordering::Relaxed);
        let clean = clean_interp_count(&prog, base_a, base_b, n);

        let off = mainloop_count(&prog, base_a, base_b, n, JIT_OFF);
        let off_c = COMPILES.load(Ordering::Relaxed);

        COMPILES.store(0, Ordering::Relaxed);
        ABORTS.store(0, Ordering::Relaxed);
        let on = mainloop_count(&prog, base_a, base_b, n, JIT_ON);
        let on_c = COMPILES.load(Ordering::Relaxed);
        let on_a = ABORTS.load(Ordering::Relaxed);

        assert_eq!(clean, off, "{label}: clean vs JIT-off count");
        assert_eq!(clean, on, "{label}: clean vs JIT-on count");
        assert_eq!(off_c, 0, "{label}: JIT-off must not compile");
        assert!(on_c >= 1, "{label}: JIT-on must compile at least one trace");
        println!("[{label} n={n}] count={on} compiles off={off_c} on={on_c} aborts on={on_a}");
        on
    }

    pub(super) fn perf_probe_count(col_a: &[f64], col_b: &[f64]) {
        let n = std::env::var("CELFLOAT_PERF_N")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(1_000_000);
        let prog = count_program();
        let base_a = col_a.as_ptr() as i64;
        let base_b = col_b.as_ptr() as i64;
        let mut clean = Vec::new();
        let mut jit = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            black_box(clean_interp_count(&prog, base_a, base_b, n));
            clean.push(t.elapsed().as_nanos() as f64 / n as f64);
            let t = Instant::now();
            black_box(mainloop_count(&prog, base_a, base_b, n, JIT_ON));
            jit.push(t.elapsed().as_nanos() as f64 / n as f64);
        }
        let clean = median(clean);
        let jit = median(jit);
        println!(
            "[perf count n={n}] jit={jit:.3} clean={clean:.3} ns/row  => JIT is {:.2}x of naive",
            clean / jit
        );
    }
} // mod count

// ── Two-bank general register machine (the cel VM shape for S3) ───────────
// The linchpin de-risk for a general float S3: a single state carrying BOTH
// an int virt-array bank (addressing, bool results, count) AND a float
// virt-array bank (column values), with a float compare crossing banks
// (float operands -> int bool). If this compiles and traces bit-exact, cel's
// VM can gain a float bank the general lowering can target.
mod twobank {
    use super::*;

    const OP_LOAD: i64 = 0; // [imm, dst]                regs[dst] = imm
    const OP_MUL: i64 = 1; // [a, b, dst]               regs[dst] = regs[a]*regs[b]
    const OP_ADD: i64 = 2; // [a, b, dst]               regs[dst] = regs[a]+regs[b]
    const OP_COL_LOAD_F: i64 = 3; // [base, ea, fdst]   fregs[fdst] = load_f(regs[base], regs[ea])
    const OP_FGE: i64 = 4; // [fa, fb, dst]             regs[dst] = (fregs[fa] >= fregs[fb]) as i64
    const OP_JIA: i64 = 5; // [a, b, tgt]               if regs[a] > regs[b] { pc = tgt }
    const OP_RETURN: i64 = 6; // [reg]                  return regs[reg]

    const R_I: usize = 0;
    const R_ACC: usize = 1;
    const R_N: usize = 2;
    const R_ONE: usize = 3;
    const R_STRIDE: usize = 4;
    const R_EA: usize = 5;
    const R_BASE_A: usize = 6;
    const R_BASE_B: usize = 7;
    const R_BOOL: usize = 8;
    const NUM_INT: usize = 9;

    const F_A: usize = 0;
    const F_B: usize = 1;
    const NUM_FLOAT: usize = 2;

    const BODY_PC: usize = 21; // 7 LOADs * 3

    struct TwoBankState {
        regs: Vec<i64>,
        fregs: Vec<f64>,
    }

    #[majit_macros::jit_interp(
        state = TwoBankState,
        env = Code,
        greens = [pc, program],
        state_fields = {
            regs: [int; virt],
            fregs: [float; virt],
        },
    )]
    fn mainloop_twobank(program: &Code, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<TwoBankState> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_g, _a, _b| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_trace_abort(|_g, _p| {
            ABORTS.fetch_add(1, Ordering::Relaxed);
        });

        let mut pc: usize = 0;
        let mut state = TwoBankState {
            regs: vec![0; NUM_INT],
            fregs: vec![0.0; NUM_FLOAT],
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
                    state.regs[program[pc + 2] as usize] = program[pc + 1];
                    pc += 3;
                }
                OP_MUL => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = state.regs[a] * state.regs[b];
                    pc += 4;
                }
                OP_ADD => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = state.regs[a] + state.regs[b];
                    pc += 4;
                }
                OP_COL_LOAD_F => {
                    let base = program[pc + 1] as usize;
                    let ea = program[pc + 2] as usize;
                    let fd = program[pc + 3] as usize;
                    state.fregs[fd] = majit_raw_load_f(state.regs[base], state.regs[ea]);
                    pc += 4;
                }
                OP_FGE => {
                    let fa = program[pc + 1] as usize;
                    let fb = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.fregs[fa] >= state.fregs[fb]) as i64;
                    pc += 4;
                }
                OP_JIA => {
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
                OP_RETURN => return state.regs[program[pc + 1] as usize],
                _ => panic!("bad opcode {opcode}"),
            }
        }
    }

    fn clean_twobank(program: &Code) -> i64 {
        let mut regs = [0i64; NUM_INT];
        let mut fregs = [0.0f64; NUM_FLOAT];
        let mut pc = 0usize;
        loop {
            match program[pc] {
                OP_LOAD => {
                    regs[program[pc + 2] as usize] = program[pc + 1];
                    pc += 3;
                }
                OP_MUL => {
                    regs[program[pc + 3] as usize] =
                        regs[program[pc + 1] as usize] * regs[program[pc + 2] as usize];
                    pc += 4;
                }
                OP_ADD => {
                    regs[program[pc + 3] as usize] =
                        regs[program[pc + 1] as usize] + regs[program[pc + 2] as usize];
                    pc += 4;
                }
                OP_COL_LOAD_F => {
                    fregs[program[pc + 3] as usize] = majit_raw_load_f(
                        regs[program[pc + 1] as usize],
                        regs[program[pc + 2] as usize],
                    );
                    pc += 4;
                }
                OP_FGE => {
                    regs[program[pc + 3] as usize] =
                        (fregs[program[pc + 1] as usize] >= fregs[program[pc + 2] as usize]) as i64;
                    pc += 4;
                }
                OP_JIA => {
                    let tgt = program[pc + 3] as usize;
                    if regs[program[pc + 1] as usize] > regs[program[pc + 2] as usize] {
                        pc = tgt;
                    } else {
                        pc += 4;
                    }
                }
                OP_RETURN => return regs[program[pc + 1] as usize],
                op => panic!("bad opcode {op}"),
            }
        }
    }

    // count rows where a[i] >= b[i]
    fn program(n: i64, base_a: i64, base_b: i64) -> Vec<i64> {
        let mut p = vec![
            OP_LOAD,
            0,
            R_I as i64,
            OP_LOAD,
            0,
            R_ACC as i64,
            OP_LOAD,
            n,
            R_N as i64,
            OP_LOAD,
            1,
            R_ONE as i64,
            OP_LOAD,
            8,
            R_STRIDE as i64,
            OP_LOAD,
            base_a,
            R_BASE_A as i64,
            OP_LOAD,
            base_b,
            R_BASE_B as i64,
        ];
        assert_eq!(p.len(), BODY_PC);
        p.extend_from_slice(&[
            OP_MUL,
            R_I as i64,
            R_STRIDE as i64,
            R_EA as i64,
            OP_COL_LOAD_F,
            R_BASE_A as i64,
            R_EA as i64,
            F_A as i64,
            OP_COL_LOAD_F,
            R_BASE_B as i64,
            R_EA as i64,
            F_B as i64,
            OP_FGE,
            F_A as i64,
            F_B as i64,
            R_BOOL as i64,
            OP_ADD,
            R_ACC as i64,
            R_BOOL as i64,
            R_ACC as i64,
            OP_ADD,
            R_I as i64,
            R_ONE as i64,
            R_I as i64,
            OP_JIA,
            R_N as i64,
            R_I as i64,
            BODY_PC as i64,
            OP_RETURN,
            R_ACC as i64,
        ]);
        p
    }

    pub(super) fn run_gate(label: &str, n: i64, col_a: &[f64], col_b: &[f64]) -> i64 {
        let base_a = col_a.as_ptr() as i64;
        let base_b = col_b.as_ptr() as i64;
        let prog = program(n, base_a, base_b);

        COMPILES.store(0, Ordering::Relaxed);
        ABORTS.store(0, Ordering::Relaxed);
        let clean = clean_twobank(&prog);

        let off = mainloop_twobank(&prog, JIT_OFF);
        let off_c = COMPILES.load(Ordering::Relaxed);

        COMPILES.store(0, Ordering::Relaxed);
        ABORTS.store(0, Ordering::Relaxed);
        let on = mainloop_twobank(&prog, JIT_ON);
        let on_c = COMPILES.load(Ordering::Relaxed);
        let on_a = ABORTS.load(Ordering::Relaxed);

        assert_eq!(clean, off, "{label}: clean vs JIT-off");
        assert_eq!(clean, on, "{label}: clean vs JIT-on");
        assert_eq!(off_c, 0, "{label}: JIT-off must not compile");
        assert!(on_c >= 1, "{label}: JIT-on must compile at least one trace");
        println!(
            "[twobank {label} n={n}] count={on} compiles off={off_c} on={on_c} aborts on={on_a}"
        );
        on
    }

    pub(super) fn perf_probe(col_a: &[f64], col_b: &[f64]) {
        let n = std::env::var("CELFLOAT_PERF_N")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(1_000_000);
        let base_a = col_a.as_ptr() as i64;
        let base_b = col_b.as_ptr() as i64;
        let prog = program(n, base_a, base_b);
        let mut clean = Vec::new();
        let mut jit = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            black_box(clean_twobank(&prog));
            clean.push(t.elapsed().as_nanos() as f64 / n as f64);
            let t = Instant::now();
            black_box(mainloop_twobank(&prog, JIT_ON));
            jit.push(t.elapsed().as_nanos() as f64 / n as f64);
        }
        let clean = median(clean);
        let jit = median(jit);
        println!(
            "[perf twobank n={n}] jit={jit:.3} clean={clean:.3} ns/row  => JIT is {:.2}x of naive",
            clean / jit
        );
    }
} // mod twobank

fn acc_program() -> Vec<i64> {
    vec![
        OP_LOAD_A,
        R0 as i64, // r0 = a[i]
        OP_ADD,
        ACC as i64,
        R0 as i64,
        ACC as i64, // acc += r0
        OP_INC,
        OP_JUMP_IF,
        BODY_PC as i64,
        OP_RETURN,
        ACC as i64,
    ]
}

fn dot_program() -> Vec<i64> {
    vec![
        OP_LOAD_A,
        R0 as i64, // r0 = a[i]
        OP_LOAD_B,
        R1 as i64, // r1 = b[i]
        OP_MUL,
        R0 as i64,
        R1 as i64,
        R2 as i64, // r2 = a*b
        OP_ADD,
        ACC as i64,
        R2 as i64,
        ACC as i64, // acc += r2
        OP_INC,
        OP_JUMP_IF,
        BODY_PC as i64,
        OP_RETURN,
        ACC as i64,
    ]
}

fn compute_program() -> Vec<i64> {
    vec![
        OP_LOAD_A,
        R0 as i64, // r0 = a[i]
        OP_LOAD_B,
        R1 as i64, // r1 = b[i]
        OP_MUL,
        R0 as i64,
        R1 as i64,
        R2 as i64, // r2 = a*b
        OP_MUL,
        R0 as i64,
        R0 as i64,
        R3 as i64, // r3 = a*a
        OP_MUL,
        R1 as i64,
        R1 as i64,
        R4 as i64, // r4 = b*b
        OP_SUB,
        R2 as i64,
        R3 as i64,
        R5 as i64, // r5 = a*b - a*a
        OP_ADD,
        R5 as i64,
        R4 as i64,
        R6 as i64, // r6 = r5 + b*b
        OP_ADD,
        ACC as i64,
        R6 as i64,
        ACC as i64, // acc += r6
        OP_INC,
        OP_JUMP_IF,
        BODY_PC as i64,
        OP_RETURN,
        ACC as i64,
    ]
}

fn make_col(n: usize, seed: u64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut x = seed;
    for k in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mant = x & ((1u64 << 52) - 1);
        let bits = (0x3ffu64 << 52) | mant;
        let v = f64::from_bits(bits) - 1.0 + (k as f64 * 0.000_000_000_001);
        out.push(if k & 1 == 0 { v } else { -v });
    }
    out
}

fn run_gate(label: &str, prog: &Code, n: i64, col_a: &[f64], col_b: &[f64]) -> f64 {
    let base_a = col_a.as_ptr() as i64;
    let base_b = col_b.as_ptr() as i64;

    COMPILES.store(0, Ordering::Relaxed);
    ABORTS.store(0, Ordering::Relaxed);
    let clean = clean_interp(prog, base_a, base_b, n);

    let off = mainloop(prog, base_a, base_b, n, JIT_OFF);
    let off_c = COMPILES.load(Ordering::Relaxed);
    let off_a = ABORTS.load(Ordering::Relaxed);

    COMPILES.store(0, Ordering::Relaxed);
    ABORTS.store(0, Ordering::Relaxed);
    let on = mainloop(prog, base_a, base_b, n, JIT_ON);
    let on_c = COMPILES.load(Ordering::Relaxed);
    let on_a = ABORTS.load(Ordering::Relaxed);

    assert_eq!(clean.to_bits(), off.to_bits(), "{label}: clean vs JIT-off");
    assert_eq!(clean.to_bits(), on.to_bits(), "{label}: clean vs JIT-on");
    assert_eq!(off_c, 0, "{label}: JIT-off must not compile");
    assert!(on_c >= 1, "{label}: JIT-on must compile at least one trace");
    println!(
        "[{label} n={n}] bits={:#018x} value={on:.17e} compiles off={off_c} on={on_c} aborts off={off_a} on={on_a}",
        on.to_bits()
    );
    on
}

fn time_ns_per_row<F: Fn() -> f64>(n: i64, f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

fn perf_probe(label: &str, prog: &Code, col_a: &[f64], col_b: &[f64]) {
    let n = std::env::var("CELFLOAT_PERF_N")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(1_000_000);
    let base_a = col_a.as_ptr() as i64;
    let base_b = col_b.as_ptr() as i64;
    let mut clean = Vec::new();
    let mut jit = Vec::new();
    for _ in 0..5 {
        clean.push(time_ns_per_row(n, || clean_interp(prog, base_a, base_b, n)));
        jit.push(time_ns_per_row(n, || {
            mainloop(prog, base_a, base_b, n, JIT_ON)
        }));
    }
    let clean = median(clean);
    let jit = median(jit);
    println!(
        "[perf {label} n={n}] jit={jit:.3} clean={clean:.3} ns/row  => JIT is {:.2}x of naive",
        clean / jit
    );
}

pub fn run() {
    let max_n = 1_100_000usize;
    let col_a = make_col(max_n, 0x2545_F491_4F6C_DD1D);
    let col_b = make_col(max_n, 0x9E37_79B9_7F4A_7C15);

    // Bit-exact correctness gate on every program, plus a guard-resume length
    // variant (steady-state accumulate never hits a float resume otherwise).
    for (label, prog) in [
        ("acc", acc_program()),
        ("dot", dot_program()),
        ("compute", compute_program()),
    ] {
        let primary = run_gate(label, &prog, 200_000, &col_a, &col_b);
        let resume = run_gate(&format!("{label}-resume"), &prog, 200_017, &col_a, &col_b);
        assert_ne!(
            primary.to_bits(),
            resume.to_bits(),
            "{label}: guard-resume variant should use a distinct length"
        );
    }

    // Float comparison policy: count rows where a[i] >= b[i]. Exercises a
    // float compare materialized to an int bool (the cel flagship shape).
    let count_primary = count::run_gate_count("count", 200_000, &col_a, &col_b);
    let count_resume = count::run_gate_count("count-resume", 200_017, &col_a, &col_b);
    assert_ne!(
        count_primary, count_resume,
        "count: guard-resume variant should use a distinct length"
    );

    perf_probe("acc", &acc_program(), &col_a, &col_b);
    perf_probe("dot", &dot_program(), &col_a, &col_b);
    perf_probe("compute", &compute_program(), &col_a, &col_b);
    count::perf_probe_count(&col_a, &col_b);

    // Two-bank de-risk: int virt-array + float virt-array in one state.
    let tb_primary = twobank::run_gate("count-ge", 200_000, &col_a, &col_b);
    let tb_resume = twobank::run_gate("count-ge-resume", 200_017, &col_a, &col_b);
    assert_ne!(tb_primary, tb_resume, "twobank: distinct lengths");
    twobank::perf_probe(&col_a, &col_b);

    black_box(col_a);
    black_box(col_b);
}
