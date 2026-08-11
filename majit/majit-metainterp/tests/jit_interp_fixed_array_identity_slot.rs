//! Verifies virtualizable identity handling when a fixed integer array appears
//! before a virtualizable array in the flattened input layout.
//!
//! Runtime-sized fixed arrays prevent the macro from declaring a constant
//! identity slot. The optimizer must decline identity tracking instead of
//! falling back to slot zero, which contains an integer in this fixture.

use core::sync::atomic::{AtomicU32, Ordering};

pub type Bytecode = [u8];

const OP_ACC: u8 = 1; // iregs[2] += cells[0]  (the fixed array, read-only)
const OP_INC: u8 = 2; // iregs[0] += 1
const OP_JUMP_BACK: u8 = 3; // [OP_JUMP_BACK, target]: loop while iregs[0] < iregs[1]
const OP_RETURN: u8 = 4; // yield iregs[0]

/// `iregs = [counter, limit, accumulator]`.
fn count_program() -> Vec<u8> {
    vec![OP_ACC, OP_INC, OP_JUMP_BACK, 0, OP_RETURN]
}

/// What every machine here must return for a given limit.
fn expected(limit: i64) -> i64 {
    limit
}

/// A fixed `[int]` array between the loop-dead `int` scalar and the
/// `[int; virt]` array. `extract_live` emits `[ret, cells[0], cells[1],
/// identity]`, so the identity is at flat slot 3 — not at `num_scalars` (1),
/// and not at 0.
mod fixed_array_before_virt_array {
    use super::{AtomicU32, Bytecode, OP_ACC, OP_INC, OP_JUMP_BACK, OP_RETURN, Ordering};
    use majit_metainterp::JitState as _;

    static COMPILES: AtomicU32 = AtomicU32::new(0);

    struct FixedAndVirt {
        ret: i64,
        cells: Vec<i64>,
        iregs: Vec<i64>,
    }

    #[majit_macros::jit_interp(
        state = FixedAndVirt,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            ret: int,
            cells: [int],
            iregs: [int; virt],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn mainloop(program: &Bytecode, limit: i64, threshold: u32) -> i64 {
        let mut driver: majit_metainterp::JitDriver<FixedAndVirt> =
            majit_metainterp::JitDriver::new(threshold);
        driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        let mut pc: usize = 0;
        let mut state = FixedAndVirt {
            ret: 0,
            cells: vec![0i64, 0i64],
            iregs: vec![0i64, limit, 0i64],
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
                // Read-only: a MUTATED plain `[int]` is refused outright (it is
                // not restored on deopt), so the fixed array is loop-carried but
                // never written.
                OP_ACC => state.iregs[2] = state.iregs[2] + state.cells[0],
                OP_INC => state.iregs[0] = state.iregs[0] + 1,
                OP_JUMP_BACK => {
                    let target = program[pc] as usize;
                    pc += 1;
                    if state.iregs[0] < state.iregs[1] {
                        if target < pc {
                            can_enter_jit!(driver, target, &mut state, program, || {});
                        }
                        pc = target;
                        continue;
                    }
                }
                OP_RETURN => {
                    state.ret = state.iregs[0];
                    return state.ret;
                }
                _ => panic!("bad opcode {opcode}"),
            }
        }
        state.ret
    }

    /// Ground truth for the slot the identity occupies, in the same shape
    /// `jit_interp_float_state_field.rs`
    /// `the_entry_contract_carries_no_element_in_its_red_block` pins: int
    /// scalars, then every fixed array's cells, then the identity. No `iregs`
    /// element belongs in the red block.
    #[test]
    fn the_identity_sits_past_the_fixed_array_cells() {
        let state = FixedAndVirt {
            ret: 0,
            cells: vec![0i64, 0i64],
            iregs: vec![0i64, 4, 0i64],
        };
        let program: &Bytecode = &super::count_program();
        let meta = state.build_meta(0, program);

        assert_eq!(
            state.live_value_types(&meta),
            vec![
                majit_ir::Type::Int, // ret
                majit_ir::Type::Int, // cells[0]
                majit_ir::Type::Int, // cells[1]
                majit_ir::Type::Ref, // __vable_identity
            ],
            "the identity must follow the fixed array's cells, so its flat slot \
             is `num_scalars + cells.len()` = 3 — neither `num_scalars` (1) nor \
             the legacy frame-first 0",
        );
        assert_eq!(state.extract_live(&meta).len(), 4);
    }

    #[test]
    fn a_fixed_array_beside_a_virt_array_compiles_the_hot_loop() {
        let program = super::count_program();
        let limit = 50i64;
        COMPILES.store(0, Ordering::Relaxed);

        let got = mainloop(&program, limit, 3);

        assert_eq!(
            got,
            super::expected(limit),
            "scalar carried the wrong result out"
        );
        assert!(
            COMPILES.load(Ordering::Relaxed) >= 1,
            "declaring a fixed `[int]` array beside a `[int; virt]` one stopped \
             the hot loop from compiling at all: the macro emits no \
             `identity_live_index` for that combination and the optimizer fell \
             back to flat slot 0, which here is the loop-dead `ret` int scalar, \
             so every trace aborted with VirtualStatesCantMatch (expected Ref, \
             got Int) on slot 0"
        );
    }
}
