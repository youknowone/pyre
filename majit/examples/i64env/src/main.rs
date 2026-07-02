//! `[i64]`-env regression example.
//!
//! Proves the `#[jit_interp]` macro reads an `env` whose element type is wider
//! than a byte (`pub type Code = [i64];`) at the correct stride. The macro
//! lowers every `program[pc + N]` read with a descr whose `item_size` matches
//! the env element (`size_of::<<Code as Index<usize>>::Output>()` = 8), so the
//! load scales the index by 8 instead of reading a stray byte. A byte-wide
//! descr (the previous hardcoding) would read the wrong word and miscompile.
//!
//! The register file is `[int; virt]` because it is loop-carried: a plain
//! `[int]` array element is not restored on a CloseLoop guard deopt. (See the
//! macro's loop-carried-plain-array diagnostic.)

use std::sync::atomic::{AtomicUsize, Ordering};

/// The env: an i64-word bytecode stream. The whole point of this example is
/// that the element is 8 bytes wide, not 1.
pub type Code = [i64];

// Opcodes and operands are full i64 words. Values can exceed a byte to make a
// byte-stride miscompile observable.
const OP_LOAD: i64 = 0; // [LOAD, imm, dst]
const OP_ADD: i64 = 1; // [ADD, a, b, dst]
const OP_JUMP_IF_ABOVE: i64 = 2; // [JIA, a, b, target_pc]
const OP_RETURN: i64 = 3; // [RETURN, reg]

/// Hot loops majit compiled — evidence the JIT tier traced + compiled.
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

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
    let mut stacksize: i32 = 0;
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

/// `r0` counts up by `step` while `n` > `r0`, exiting at `r0 >= n`. `step` and
/// `n` are full i64 words (intentionally > 255 in tests).
fn count_program(n: i64, step: i64) -> Vec<i64> {
    vec![
        OP_LOAD,
        step,
        1, // r1 = step
        OP_LOAD,
        n,
        2, // r2 = n
        OP_LOAD,
        0,
        0, // r0 = 0
        // @l1 = pc 9
        OP_ADD,
        0,
        1,
        0, // r0 = r0 + r1
        OP_JUMP_IF_ABOVE,
        2,
        0,
        9, // if r2 > r0 goto @l1
        OP_RETURN,
        0, // return r0
    ]
}

fn run(program: &Code, num_regs: usize, threshold: u32) -> i64 {
    mainloop(program, num_regs, threshold)
}

fn main() {
    let result = run(&count_program(1000, 1), 3, 3);
    println!("count to 1000 (step 1) = {result}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The env element is 8 bytes wide and the immediate `n = 1000` does not
    /// fit a byte. A byte-stride descr would read the wrong word and never
    /// reach 1000.
    #[test]
    fn i64_env_reads_wide_immediates() {
        COMPILES.store(0, Ordering::Relaxed);
        let program = count_program(1000, 1);
        let result = run(&program, 3, 3);
        assert_eq!(result, 1000, "i64-env loop must compute 1000");
        assert!(
            COMPILES.load(Ordering::Relaxed) >= 1,
            "majit should have compiled the hot loop at least once"
        );
    }

    /// Correctness across inputs (each exercises the CloseLoop guard-exit
    /// deopt), all with byte-overflowing immediates.
    #[test]
    fn i64_env_varies_n() {
        for n in [300_i64, 500, 1000, 4096, 100_000] {
            let r = run(&count_program(n, 1), 3, 3);
            assert_eq!(r, n, "count to {n} mismatch");
        }
    }

    /// A step > 255 proves the ADD operand word is read at the right stride
    /// too: counting by 7 up to a multiple of 7 lands exactly on `n`.
    #[test]
    fn i64_env_wide_step() {
        let r = run(&count_program(7 * 300, 7), 3, 3);
        assert_eq!(r, 7 * 300);
    }
}
