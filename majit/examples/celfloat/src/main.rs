use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub type Code = [i64];

const OP_COL_ACC_F: i64 = 0;
const OP_JUMP_IF_ABOVE: i64 = 1;
const OP_RETURN: i64 = 2;

const BODY_PC: usize = 0;
const ACC: usize = 0;
const NUM_REGS: usize = 1;

const JIT_ON: u32 = 8;
const JIT_OFF: u32 = u32::MAX;

pub static COMPILES: AtomicUsize = AtomicUsize::new(0);
pub static ABORTS: AtomicUsize = AtomicUsize::new(0);

fn majit_raw_load_f(base: i64, ea: i64) -> f64 {
    unsafe { core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const f64) }
}

struct VmState {
    i: i64,
    n: i64,
    base: i64,
    regs: Vec<f64>,
}

#[majit_macros::jit_interp(
    state = VmState,
    env = Code,
    greens = [pc, program],
    state_fields = {
        i: int,
        n: int,
        base: int,
        regs: [float; virt],
    },
)]
fn mainloop(program: &Code, base: i64, n: i64, threshold: u32) -> f64 {
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
        base,
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
            OP_COL_ACC_F => {
                let ea = state.i * 8;
                let v = majit_raw_load_f(state.base, ea);
                state.regs[ACC] = state.regs[ACC] + v;
                state.i += 1;
                pc += 1;
            }
            OP_JUMP_IF_ABOVE => {
                if state.n > state.i {
                    can_enter_jit!(driver, BODY_PC, &mut state, program, || {});
                    pc = BODY_PC;
                    continue;
                }
                pc += 1;
            }
            OP_RETURN => return state.regs[ACC],
            _ => panic!("bad opcode {opcode}"),
        }
    }
}

fn clean_interp(program: &Code, base: i64, n: i64) -> f64 {
    let mut i = 0i64;
    let mut regs = vec![0.0f64; NUM_REGS];
    let mut pc = 0usize;
    loop {
        match program[pc] {
            OP_COL_ACC_F => {
                let ea = i * 8;
                let v = majit_raw_load_f(base, ea);
                regs[ACC] = regs[ACC] + v;
                i += 1;
                pc += 1;
            }
            OP_JUMP_IF_ABOVE => {
                if n > i {
                    pc = BODY_PC;
                } else {
                    pc += 1;
                }
            }
            OP_RETURN => return regs[ACC],
            op => panic!("bad opcode {op}"),
        }
    }
}

fn program() -> Vec<i64> {
    vec![OP_COL_ACC_F, OP_JUMP_IF_ABOVE, OP_RETURN]
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

fn run_gate(label: &str, n: i64, col: &[f64]) -> f64 {
    let prog = program();
    let base = col.as_ptr() as i64;

    COMPILES.store(0, Ordering::Relaxed);
    ABORTS.store(0, Ordering::Relaxed);
    let clean = clean_interp(&prog, base, n);

    let off = mainloop(&prog, base, n, JIT_OFF);
    let off_c = COMPILES.load(Ordering::Relaxed);
    let off_a = ABORTS.load(Ordering::Relaxed);

    COMPILES.store(0, Ordering::Relaxed);
    ABORTS.store(0, Ordering::Relaxed);
    let on = mainloop(&prog, base, n, JIT_ON);
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

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn perf_probe(col: &[f64]) {
    let n = std::env::var("CELFLOAT_PERF_N")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(1_000_000);
    let prog = program();
    let base = col.as_ptr() as i64;
    let mut clean = Vec::new();
    let mut jit = Vec::new();
    for _ in 0..5 {
        clean.push(time_ns_per_row(n, || clean_interp(&prog, base, n)));
        jit.push(time_ns_per_row(n, || mainloop(&prog, base, n, JIT_ON)));
    }
    let clean = median(clean);
    let jit = median(jit);
    println!(
        "[perf n={n}] jit={jit:.3} clean={clean:.3} ns/row clean/jit={:.2}x",
        clean / jit
    );
}

fn main() {
    let max_n = 1_100_000usize;
    let col = make_col(max_n, 0x2545_F491_4F6C_DD1D);

    let primary = run_gate("primary", 200_000, &col);
    let resume = run_gate("guard-resume", 200_017, &col);
    assert_ne!(
        primary.to_bits(),
        resume.to_bits(),
        "guard-resume variant should use a distinct length"
    );

    perf_probe(&col);
    black_box(col);
}
