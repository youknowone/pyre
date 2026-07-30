//! A tracing abort taken mid-source-opcode must not skip that opcode's
//! remaining effects on a `[.. ; virt]` state.
//!
//! `pyjitpl.py:2949 run_blackhole_interp_to_cancel_tracing` answers an abort by
//! running `blackhole.py:1799 convert_and_run_from_pyjitpl`, which finishes the
//! aborting framestack in the blackhole and resumes from the merge point it
//! reaches. Without that consumer the abort hands back the walk's root-frame
//! source pc, which dispatch advanced *before* the opcode arm ran — so a
//! two-store opcode like `PUSH` (`stack[stackpos] = v; stackpos += 1`) can
//! commit one half and lose the other.
//!
//! This fixture is the virt-array shape (`stackpos: int, stack: [int; virt]`)
//! that every majit example interpreter uses. It lives in its own integration
//! test binary because it sets `MAJIT_STEP_LIMIT`, and the knob is a
//! process-wide `LazyLock` (`lib.rs step_limit`) — one test per process.
//!
//! Short-circuiting the conversion so the abort falls back to that source pc
//! makes the same run panic with `index out of bounds: the len is N but the
//! index is N` from `PUSH`, or reach a `stackpos` of `usize::MAX` from `POP`:
//! a torn stack pointer.

pub type Bytecode = [u8];

const PUSH: u8 = 2; // [PUSH, imm]: push a signed-byte immediate
const POP: u8 = 3; // pop top
const SWAP: u8 = 4; // swap the top two
const PICK: u8 = 6; // [PICK, i]: duplicate stack[stackpos - i - 1]
const ADD: u8 = 8; // pop a, b; push b + a
const SUB: u8 = 9; // pop a, b; push b - a
const BR_COND: u8 = 18; // [BR_COND, off]: pop cond; if cond != 0 jump
const RETURN: u8 = 21; // return top
const PUSHARG: u8 = 22; // push the input argument

struct StackState {
    stackpos: i64,
    stack: Vec<i64>,
}

#[majit_macros::jit_interp(
    state = StackState,
    env = Bytecode,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<StackState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = StackState {
        stackpos: 0,
        stack: vec![0i64; program.len()],
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos = state.stackpos + 1;
            }
            POP => {
                state.stackpos = state.stackpos - 1;
            }
            SWAP => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 1) as usize] = b;
                state.stack[(state.stackpos - 2) as usize] = a;
            }
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let v = state.stack[(state.stackpos as usize) - i - 1];
                state.stack[state.stackpos as usize] = v;
                state.stackpos = state.stackpos + 1;
            }
            ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos = state.stackpos - 1;
            }
            SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos = state.stackpos - 1;
            }
            BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                state.stackpos = state.stackpos - 1;
                let jump = state.stack[state.stackpos as usize] != 0;
                if jump {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            RETURN => break,
            PUSHARG => {
                state.stack[state.stackpos as usize] = inputarg;
                state.stackpos = state.stackpos + 1;
            }
            _ => {}
        }
    }

    state.stackpos = state.stackpos - 1;
    state.stack[state.stackpos as usize]
}

/// `sum(N) = N + (N-1) + ... + 1`, a hot loop over the virt stack.
fn sum_program() -> Vec<u8> {
    vec![
        PUSH, 0,       // 0
        PUSHARG, // 2
        PICK, 0, // 3  (loop header)
        BR_COND, 2,      // 5  -> body @9
        POP,    // 7
        RETURN, // 8
        SWAP,   // 9
        PICK, 1,    // 10
        ADD,  // 12
        SWAP, // 13
        PUSH, 1, SUB, // 14
        PUSH, 1, // 17
        BR_COND, 238, // 19 -> back to the header @3
    ]
}

#[test]
fn mid_opcode_abort_preserves_virt_stack() {
    // Small enough that every trace attempt is cut off inside an opcode arm,
    // which is the `run_to_end` runaway backstop's own mid-opcode `Abort` —
    // one of the exits the conversion exists to cover. Set before any JIT call
    // because `step_limit()` latches on first read.
    unsafe {
        std::env::set_var("MAJIT_STEP_LIMIT", "200");
    }

    let program = sum_program();
    for n in [1_i64, 2, 3, 5, 10, 20, 50, 100, 200] {
        let got = mainloop(&program, n, 3);
        assert_eq!(
            got,
            n * (n + 1) / 2,
            "sum({n}) diverged under a forced mid-opcode abort",
        );
    }
}
