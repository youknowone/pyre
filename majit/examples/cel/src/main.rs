//! cel-majit de-risk probes (issue #357) — one binary gathering the
//! meta-tracing kill-tests / prototypes that used to be separate examples
//! (celprobe / celpolicy / celcolumn / celcolscalar / celfloat).
//!
//! Each probe is a self-contained `#[jit_interp]` register machine over an
//! i64-word bytecode, gated 3-way (clean interp == JIT-off == JIT-on) and then
//! timed clean-vs-trace. NONE contain CEL text; they model the cel batch shapes.
//! Because the `#[jit_interp]` macro emits module-level items that collide
//! within a single module, every machine lives in its own module.
//!
//!   Run all probes:  cargo run -p cel --release --features cranelift
//!   Run one probe:   cargo run -p cel --release --features cranelift -- <probe>
//!   <probe> ∈ { probe, policy, column, colscalar, float }
//!
//!   * probe     — straight-line arithmetic batch (foundational >=3x kill-test)
//!   * policy    — slot-resolved policy predicate + comprehension (nested loop)
//!   * column    — columnar red-index reads via raw_load_i (base in register file)
//!   * colscalar — scalar int state-field base: read-only vs written (diagnostic)
//!   * float     — float register machines (acc/dot/compute, count, two-bank)
//!
//! RELEASE ONLY: the i64-wrap and bit-exact-float equality gates need overflow
//! checks off.

mod colscalar;
mod column;
mod float;
mod policy;
mod probe;

/// Helpers shared by every probe: the bytecode word type, the LCG that
/// synthesizes distinct per-row inputs, the raw native-memory load intrinsics
/// the `#[jit_interp]` macro recognizes, the JIT thresholds, the compile/abort
/// counters the drivers bump, and the timing median.
pub mod common {
    use std::sync::atomic::AtomicUsize;

    /// The env: an i64-word bytecode stream (8-byte elements).
    pub type Code = [i64];

    /// LCG (Knuth MMIX) constants — a serial recurrence the compiler cannot
    /// fold or vectorize, so each row's synthesized input is genuinely distinct.
    pub const LCG_A: i64 = 6364136223846793005;
    pub const LCG_C: i64 = 1442695040888963407;

    pub const JIT_ON: u32 = 8;
    pub const JIT_OFF: u32 = u32::MAX;

    /// Hot loops majit compiled / traces it aborted — evidence the JIT tier ran.
    pub static COMPILES: AtomicUsize = AtomicUsize::new(0);
    pub static ABORTS: AtomicUsize = AtomicUsize::new(0);

    /// Raw native-memory load intrinsics recognized by the `#[jit_interp]` proc
    /// macro (lowered to `raw_load_i` / `raw_load_f`); at the interpreter tier
    /// these real fns run. `base`/`ea` are an address and a byte offset.
    pub fn majit_raw_load_i64(base: i64, ea: i64) -> i64 {
        unsafe {
            core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const i64)
        }
    }

    pub fn majit_raw_load_f(base: i64, ea: i64) -> f64 {
        unsafe {
            core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const f64)
        }
    }

    pub fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
}

fn main() {
    let which = std::env::args().nth(1);
    let selected = |name: &str| which.as_deref().is_none_or(|w| w == name);
    let mut ran = false;
    for (name, run) in [
        ("probe", probe::run as fn()),
        ("policy", policy::run),
        ("column", column::run),
        ("colscalar", colscalar::run),
        ("float", float::run),
    ] {
        if selected(name) {
            println!("\n═══════════════════════════ {name} ═══════════════════════════");
            run();
            ran = true;
        }
    }
    if !ran {
        eprintln!(
            "unknown probe {:?}; expected one of: probe policy column colscalar float",
            which.unwrap_or_default()
        );
        std::process::exit(2);
    }
}
