/// JIT-enabled two-tape interpreter using `#[jit_interp]` + two `[int; virt]`
/// state arrays.
///
/// Greens: [pc, program]
/// Reds:   [pa, a, pb, b]  (two virtualizable tapes, tracked via state_fields)
///
/// Both tapes are virtualized: during tracing their cells are tracked as
/// symbolic OpRefs and carried through the loop header as `virtualizable_boxes`
/// element shadows. The loop-close splices every tape's `<arr>_ptr`/`<arr>_len`
/// header first, then the concatenated element block, matching the trace-entry
/// Label for any number of arrays.

pub type Bytecode = [u8];

const TAPE_SIZE: usize = 8;
const DEFAULT_THRESHOLD: u32 = 3;

struct DualState {
    pa: i64,
    a: Vec<i64>,
    pb: i64,
    b: Vec<i64>,
}

#[majit_macros::jit_interp(
    state = DualState,
    env = Bytecode,
    state_fields = {
        pa: int,
        a: [int; virt],
        pb: int,
        b: [int; virt],
    },
)]
fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<DualState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = DualState {
        pa: 0,
        a: vec![0i64; TAPE_SIZE],
        pb: 0,
        b: vec![0i64; TAPE_SIZE],
    };

    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    loop {
        if pc >= program.len() {
            break;
        }
        jit_merge_point!();
        let ch = program[pc];

        match ch {
            b'>' => {
                state.pa += 1;
                pc += 1;
            }
            b'<' => {
                state.pa -= 1;
                pc += 1;
            }
            b'+' => {
                state.a[state.pa as usize] += 1;
                pc += 1;
            }
            b'-' => {
                state.a[state.pa as usize] -= 1;
                pc += 1;
            }
            b'}' => {
                state.pb += 1;
                pc += 1;
            }
            b'{' => {
                state.pb -= 1;
                pc += 1;
            }
            b'*' => {
                state.b[state.pb as usize] += 1;
                pc += 1;
            }
            b'/' => {
                state.b[state.pb as usize] -= 1;
                pc += 1;
            }
            b'[' => {
                if state.a[state.pa as usize] == 0 {
                    let mut need: i32 = 1;
                    let mut p = pc + 1;
                    while need > 0 {
                        if program[p] == b']' {
                            need -= 1;
                        } else if program[p] == b'[' {
                            need += 1;
                        }
                        p += 1;
                    }
                    pc = p;
                } else {
                    pc += 1;
                }
            }
            b']' if state.a[state.pa as usize] != 0 => {
                let target = find_matching_open(program, pc);
                if target < pc {
                    can_enter_jit!(driver, target, &mut state, program, || {});
                }
                pc = target;
                continue;
            }
            _ => {
                pc += 1;
            }
        }
    }

    state.a.iter().sum::<i64>() + state.b.iter().sum::<i64>()
}

/// Find the matching '[' for a ']' at the given position.
fn find_matching_open(code: &[u8], close_pos: usize) -> usize {
    let mut need: i32 = 1;
    let mut p = close_pos - 1;
    while need > 0 {
        if code[p] == b']' {
            need += 1;
        } else if code[p] == b'[' {
            need -= 1;
        }
        if need > 0 {
            p -= 1;
        }
    }
    p
}

pub struct JitDualInterp {
    threshold: u32,
}

impl JitDualInterp {
    pub fn new() -> Self {
        JitDualInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn run(&mut self, code: &[u8]) -> i64 {
        mainloop(code, self.threshold)
    }
}

impl Default for JitDualInterp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    fn check(code: &[u8]) {
        let expected = interp::interpret(code);
        let mut jit = JitDualInterp::new();
        let got = jit.run(code);
        assert_eq!(
            got, expected,
            "JIT result {got} != interp {expected} for {code:?}"
        );
    }

    #[test]
    fn jit_matches_interp_dual_loop() {
        // a[0]=10; loop 10x mutating both tapes. Runs hot enough to compile
        // the inner loop, then guard-fails on exit and reconstructs both tapes.
        check(b"++++++++++[->+<*}*{]");
    }

    #[test]
    fn jit_matches_interp_only_tape_b_in_body() {
        // Loop counts on tape a but the body only touches tape b.
        check(b"+++++++[-*}*{]");
    }

    #[test]
    fn jit_matches_interp_wider_elements() {
        // Spread writes across several cells of each tape so more than one
        // element box per array is live across the loop header.
        check(b"++++++++[->+>+<<*}*}*{{]");
    }

    #[test]
    fn jit_matches_interp_no_loop() {
        check(b"+++>+<*}*");
    }
}
