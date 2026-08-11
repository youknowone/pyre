//! Ensures a label-entered compiled loop deoptimizes to the program counter
//! reached by the blackhole interpreter.
//!
//! Resuming at the loop header would repeat non-idempotent instructions between
//! the header and the blackhole's green program counter.

use core::sync::atomic::{AtomicU32, Ordering};

pub type Bytecode = [i64];

const OP_LOAD: i64 = 1; // [LOAD, imm, dst]                     pc += 3
const OP_ADD: i64 = 2; // [ADD, a, b, dst]  regs[dst] = a + b    pc += 4
const OP_JIA: i64 = 3; // [JIA, a, b, target] jump while a > b   pc += 4
const OP_RETURN: i64 = 4; // [RETURN, reg]                       terminal

const R_I: i64 = 0; // trip counter
const R_N: i64 = 1; // trip limit
const R_ACC: i64 = 2; // accumulator, advanced once per trip
const R_ONE: i64 = 3;
const NUM_REGS: usize = 4;

/// `[prologue] [header: acc += 1; i += 1; jump while n > i] return acc`.
///
/// The prologue is four `LOAD`s, so the loop header sits at word 12 and the
/// terminal `RETURN` at word 24 — the two pcs a deopt could resume at.
fn count_program(n: i64) -> Vec<i64> {
    let mut p = vec![
        OP_LOAD, 0, R_I, //
        OP_LOAD, n, R_N, //
        OP_LOAD, 0, R_ACC, //
        OP_LOAD, 1, R_ONE, //
    ];
    let header = p.len() as i64;
    assert_eq!(header, 12);
    p.extend_from_slice(&[
        OP_ADD, R_ACC, R_ONE, R_ACC, //
        OP_ADD, R_I, R_ONE, R_I, //
        OP_JIA, R_N, R_I, header, //
        OP_RETURN, R_ACC, //
    ]);
    p
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

struct VmState {
    regs: Vec<i64>,
    ret: i64,
}

#[majit_macros::jit_interp(
    state = VmState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
        ret: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<VmState> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = VmState {
        regs: vec![0; NUM_REGS],
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

        let opcode = program[pc];
        match opcode {
            OP_LOAD => {
                state.regs[program[pc + 2] as usize] = program[pc + 1];
                pc += 3;
            }
            OP_ADD => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] + state.regs[b];
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
            // Store then in-arm `return`, never `{ store; break }`: `classify.rs`
            // `is_break_expr` requires the arm body to be exactly `break`, so a
            // composite body classifies `Lowerable` and its tail `break` reaches
            // `lower_stmt_fallback`, which guards an enclosed `return` but not an
            // enclosed `break` — the statement is inert and silently dropped,
            // leaving the lowered arm to fall through to the dispatch back-edge.
            OP_RETURN => {
                state.ret = state.regs[program[pc + 1] as usize];
                return state.ret;
            }
            _ => panic!("bad opcode {opcode}"),
        }
    }
    state.ret
}

/// The same bytecode with no driver, no merge point and no `can_enter_jit`.
fn clean_interp(program: &Bytecode) -> i64 {
    let mut regs = vec![0i64; NUM_REGS];
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
            OP_JIA => {
                if regs[program[pc + 1] as usize] > regs[program[pc + 2] as usize] {
                    pc = program[pc + 3] as usize;
                    continue;
                }
                pc += 4;
            }
            OP_RETURN => return regs[program[pc + 1] as usize],
            op => panic!("bad opcode {op}"),
        }
    }
}

#[test]
fn a_label_entered_deopt_resumes_at_the_green_pc() {
    for n in [1_000i64, 1_001] {
        let program = count_program(n);
        COMPILES.store(0, Ordering::Relaxed);

        let clean = clean_interp(&program);
        assert_eq!(clean, n, "fixture: the clean interpreter must run n trips");

        let got = mainloop(&program, 3);

        assert!(
            COMPILES.load(Ordering::Relaxed) >= 1,
            "n={n}: nothing compiled, so the machine never reached the \
             label-entered deopt this test exists to exercise"
        );
        assert_eq!(
            got, clean,
            "n={n}: the JIT tier answered {got} where the clean interpreter \
             answered {clean}. A result of n+1 means the deopt resumed at the \
             loop header and re-ran the header..green-pc span on state that had \
             already passed it"
        );
    }
}

/// The same machine with the JIT effectively off, so a failure above can be
/// attributed to the compiled tier rather than to the bytecode or the fixture.
#[test]
fn the_same_machine_without_tracing_answers_n() {
    for n in [1_000i64, 1_001] {
        let program = count_program(n);
        COMPILES.store(0, Ordering::Relaxed);

        let got = mainloop(&program, u32::MAX);

        assert_eq!(
            COMPILES.load(Ordering::Relaxed),
            0,
            "n={n}: an unreachable threshold must not compile anything"
        );
        assert_eq!(got, n, "n={n}: the untraced dispatch loop answered {got}");
    }
}
