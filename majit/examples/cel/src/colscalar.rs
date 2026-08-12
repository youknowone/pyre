//! Diagnostic: does a SCALAR `int` state-field base for raw_load compile, and
//! does WRITING that field every iteration (as the wasmi kernel does with
//! `state.mem_base = __mb`) change the outcome vs a READ-ONLY scalar field?
//!
//! The `column` probe proved base-in-REGISTER compiles (13-16x). This isolates
//! base-in-SCALAR-FIELD. Two programs, identical except one re-writes the
//! scalar field each iteration:
//!   * RDONLY  — `state.col_base` set once at init, only READ in the loop.
//!   * WRITTEN — `state.col_base` re-assigned from a register every iteration.
//! Prints compiles/aborts for each. RELEASE ONLY.
//!
//! What it measured until the `VirtualizableConfig::identity_input_index` fix:
//! RDONLY reported `aborts(on)=37499 -> ABORTS (never compiles)` while WRITTEN
//! compiled, and that asymmetry was read as "a scalar-field base trips
//! VirtualStatesCantMatch". It was not about the base at all. The tracker
//! probed input-arg slot 0 for the virtualizable identity — the legacy PyFrame
//! `[frame, scalars.., items..]` layout — and in the state-field layout slot 0
//! is the first `int` scalar. `col_base` at slot 0 is loop-DEAD in RDONLY, so
//! the poisoned slot reached the loop-close Jump as its own inputarg and the
//! type cross rejected it; WRITTEN reassigns it every iteration, hands the Jump
//! the recomputed value, and never reads the poison. Both compile now
//! (`aborts(on)=0`), so this probe no longer separates the two cases and its
//! remaining value is as a regression canary for that fix. The predicate is
//! pinned by `majit-metainterp/tests/jit_interp_two_virt_arrays_with_scalar.rs`.

use crate::common::*;
use std::sync::atomic::Ordering;

const OP_LOAD: i64 = 0; // [LOAD, imm, dst]
const OP_ADD: i64 = 1; // [ADD, a, b, dst]
const OP_MUL: i64 = 2; // [MUL, a, b, dst]
const OP_JUMP_IF_ABOVE: i64 = 3; // [JIA, a, b, target_pc]
const OP_RETURN: i64 = 4; // [RETURN, reg]
const OP_COL_LOAD_SF: i64 = 5; // [COL_LOAD_SF, ea_reg, dst]  dst=*(col_base+regs[ea])
const OP_SET_BASE: i64 = 6; // [SET_BASE, src_reg]  state.col_base = regs[src]

struct VmState {
    regs: Vec<i64>,
    col_base: i64,
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
        col_base: int,
        ret: int,
    },
)]
fn mainloop(program: &Code, num_regs: usize, col_base: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<VmState> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_g, _a, _b, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    driver.set_on_trace_abort(|_g, _p| {
        ABORTS.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let _stacksize: i32 = 0;
    let mut state = VmState {
        regs: vec![0; num_regs],
        col_base,
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
            OP_COL_LOAD_SF => {
                let ea_reg = program[pc + 1] as usize;
                let d = program[pc + 2] as usize;
                let ea = state.regs[ea_reg];
                state.regs[d] = majit_raw_load_i64(state.col_base, ea);
                pc += 3;
            }
            OP_SET_BASE => {
                let src = program[pc + 1] as usize;
                state.col_base = state.regs[src];
                pc += 2;
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

fn clean_interp(program: &Code, num_regs: usize, col_base: i64) -> i64 {
    let mut regs = vec![0i64; num_regs];
    let mut base = col_base;
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
            OP_COL_LOAD_SF => {
                let ea = regs[program[pc + 1] as usize];
                regs[program[pc + 2] as usize] = majit_raw_load_i64(base, ea);
                pc += 3;
            }
            OP_SET_BASE => {
                base = regs[program[pc + 1] as usize];
                pc += 2;
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

// Registers: r0=i, r1=acc, r2=n, r3=one, r4=stride, r5=ea, r6=v, r7=base_val.
const NUM_REGS: usize = 8;

/// RDONLY: col_base set once at state init, only READ in the loop.
fn rdonly_program(n: i64) -> Vec<i64> {
    let body_pc = 15i64; // 5 LOADs * 3
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
        // @body = 15
        OP_MUL,
        0,
        4,
        5, //  ea = i*8
        OP_COL_LOAD_SF,
        5,
        6, // v = *(col_base + ea)
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
        body_pc,
        OP_RETURN,
        1,
    ]
}

/// WRITTEN: re-assign col_base from a register (holding the base value) every
/// iteration — replicating the wasmi kernel's `state.mem_base = __mb`.
fn written_program(n: i64, base: i64) -> Vec<i64> {
    let body_pc = 18i64; // 6 LOADs * 3
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
        7, // r7 = base value
        // @body = 18
        OP_SET_BASE,
        7, // state.col_base = r7   (write scalar field each iter)
        OP_MUL,
        0,
        4,
        5, //  ea = i*8
        OP_COL_LOAD_SF,
        5,
        6, // v = *(col_base + ea)
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
        body_pc,
        OP_RETURN,
        1,
    ]
}

fn make_col(n: i64) -> Vec<i64> {
    let mut v = Vec::with_capacity(n as usize);
    let mut x = 0x2545F4914F6CDD1Di64;
    for _ in 0..n {
        x = x.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        v.push(x);
    }
    v
}

/// One case's correctness + tier-liveness gate. There is no timing half in this
/// probe — it reports compile/abort counts, not ns/row — so this is the whole
/// of the work `run` does.
///
/// `on_c >= 1` says a trace was minted for the case's loop. It does not say the
/// trace has a real body (an empty dispatch still compiles a bare `Finish()`),
/// so it is the liveness half only.
fn gate_case(label: &str, prog: &[i64], col_base: i64) {
    COMPILES.store(0, Ordering::Relaxed);
    ABORTS.store(0, Ordering::Relaxed);
    let clean = clean_interp(prog, NUM_REGS, col_base);
    let off = mainloop(prog, NUM_REGS, col_base, JIT_OFF);
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    ABORTS.store(0, Ordering::Relaxed);
    let on = mainloop(prog, NUM_REGS, col_base, JIT_ON);
    let on_c = COMPILES.load(Ordering::Relaxed);
    let on_a = ABORTS.load(Ordering::Relaxed);
    assert_eq!(clean, off, "{label}: clean vs off");
    assert_eq!(clean, on, "{label}: clean vs on -> miscompile");
    assert_eq!(off_c, 0, "{label}: JIT-off must never compile");
    // Was a printed verdict — "COMPILES" / "ABORTS (never compiles)" — because
    // RDONLY genuinely never compiled before the `identity_input_index` fix the
    // header describes. Both cases compile now, so the outcome the probe exists
    // to watch is a fixed one and belongs in an assertion: a printed verdict
    // cannot fail a run, and the two equality gates above stay green with the
    // JIT tier entirely inert.
    assert!(
        on_c >= 1,
        "{label}: JIT-on compiled nothing (aborts={on_a}) — the scalar-field \
         base no longer closes the trace"
    );
    println!("[{label}] compiles: off={off_c} on={on_c}  aborts(on)={on_a}");
}

/// Both cases' gates. This is the whole probe; `run` adds only the banner.
pub(crate) fn run_gates() {
    let n: i64 = std::env::var("CELN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300_000);
    let col = make_col(n);
    let base = col.as_ptr() as i64;
    gate_case("RDONLY ", &rdonly_program(n), base);
    gate_case("WRITTEN", &written_program(n, base), base);
    std::hint::black_box(&col);
}

pub fn run() {
    println!("scalar-int-state-field base for raw_load: read-only vs written\n");
    run_gates();
}
