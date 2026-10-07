//! A flattened `[int]` whose runtime length exceeds the identity prefix
//! pinned at lowering (`FLAT_INT_ARRAY_IDENTITY_RESERVE` per array) cannot
//! be encoded: `assembler.py` `emit_const` refuses a constant outside
//! `0..256`, and the codewriter already assigned every red a register.
//! Install skips the dispatch JitCode; the interpreter still runs.

use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const CELL_LEN: usize = 20;
const N: i64 = 40;

struct OversizedCells {
    acc: i64,
    cells: Vec<i64>,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = OversizedCells,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, cells: [int], pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn oversized_cells_count(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<OversizedCells> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = OversizedCells {
        acc: 0,
        cells: vec![1i64; CELL_LEN],
        pos: 0,
        n,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.acc = state.acc + state.cells[0];
        state.pos = state.pos + 1i64;
    }
    state.acc
}

#[test]
fn cells_longer_than_the_identity_reserve_still_interpret() {
    let state = OversizedCells {
        acc: 0,
        cells: vec![1i64; CELL_LEN],
        pos: 0,
        n: N,
    };
    let mut driver: JitDriver<OversizedCells> = JitDriver::new(2);
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, &PROGRAM)
            .install_canonical_liveness(&mut driver);
    }
    assert!(
        driver.dispatch_jitcode().is_none(),
        "CELL_LEN={CELL_LEN} exceeds one array's identity reserve, so install \
         skips register_dispatch_jitcode"
    );

    let got = oversized_cells_count(&PROGRAM, 2, N);
    assert_eq!(got, N, "each step adds cells[0]=1, so acc == n");
}
