//! Scalar-only residual loop.
//!
//! `ScalarState` has only one scalar red. `TICK` calls a
//! `#[dont_look_inside]` residual and `LOOP` jumps back to the header while
//! the counter is live.

type Code = [u8];

const TICK: u8 = 1;
const LOOP: u8 = 2;

#[cfg(test)]
static SCALAR_TICKS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

#[majit_macros::dont_look_inside]
extern "C" fn scalar_tick(left: i64) {
    #[cfg(test)]
    SCALAR_TICKS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let _ = left;
}

struct ScalarState {
    left: i64,
}

#[majit_macros::jit_interp(
    state = ScalarState,
    env = Code,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        left: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn scalar_mainloop(program: &Code, n: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<ScalarState> =
        majit_metainterp::JitDriver::new(threshold);
    majit_metainterp::embed::Census::install(&mut driver);
    let mut pc: usize = 0;
    let stacksize: i32 = 0;
    let mut state = ScalarState { left: n };

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
            TICK => {
                scalar_tick(state.left);
                state.left -= 1;
            }
            LOOP => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                if state.left > 0 {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            _ => break,
        }
    }
    state.left
}

/// `TICK; LOOP` back to pc 0.
///
///   0: TICK
///   1: LOOP
///   2: offset byte. target = 2 + (-3) + 1 = 0
fn tick_loop_program() -> Vec<u8> {
    vec![TICK, LOOP, 253]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::Ordering;
    use majit_metainterp::embed::Census;

    fn scalar_probe(n: i64) -> (i64, u32, usize) {
        let census = Census::begin();
        SCALAR_TICKS.store(0, Ordering::Relaxed);
        let got = scalar_mainloop(&tick_loop_program(), n, 3);
        (
            got,
            SCALAR_TICKS.load(Ordering::Relaxed),
            census.counts().loops_compiled,
        )
    }

    #[test]
    fn scalar_only_residual_closes_loop() {
        for n in [50_i64, 500] {
            let (got, ticks, compiles) = scalar_probe(n);
            assert!(
                compiles >= 1,
                "scalar-only loop never compiled for n={n} (ticks={ticks}, left={got})"
            );
            assert_eq!(
                ticks, n as u32,
                "scalar residual fired {ticks}× for n={n}, compiles={compiles}"
            );
            assert_eq!(got, 0, "scalar counter did not reach 0 for n={n}");
        }
    }
}
