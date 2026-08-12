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
//!   Run all probes:  cargo run -p cel --release
//!   Run one probe:   cargo run -p cel --release -- <probe>
//!   <probe> ∈ { probe, policy, column, colscalar, float }
//!
//!   * probe     — straight-line arithmetic batch (foundational >=3x kill-test)
//!   * policy    — slot-resolved policy predicate + comprehension (nested loop)
//!   * column    — columnar red-index reads via raw_load_i (base in register file)
//!   * colscalar — scalar int state-field base: read-only vs written (diagnostic)
//!   * float     — float register machines (acc/dot/compute, count, two-bank)
//!
//! RELEASE ONLY: the i64-wrap and bit-exact-float equality gates need overflow
//! checks off. That applies to the `#[test]`s at the bottom of this file as much
//! as to the binary, so they self-ignore under `debug_assertions` and CI runs
//! them from a dedicated `cargo test --release -p cel` step.

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
    use std::sync::atomic::{AtomicBool, AtomicUsize};

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

    /// Shape of the most recently compiled loop body, from
    /// `LoopBodyShape::of(opcodes)` in the `on_compile_loop` hook.
    ///
    /// `COMPILES` counts that a trace was minted; these two say whether the
    /// body it minted does anything. A dispatch that lowers nothing still
    /// compiles a loop whose whole optimized body is `Finish()`, so a count
    /// alone cannot separate a working tier from a dead one.
    pub static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
    pub static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

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

/// Each probe's correctness + tier-liveness gate, run as a test.
///
/// Every gate the five probes carry already existed; until this module they were
/// reachable only through `main`, so `cargo test -p cel` compiled the crate and
/// ran nothing. These tests call the gate halves only — the timing loops
/// (`probe` alone times 9 interleaved rounds over 20M rows) stay in the binary.
///
/// The gates assert three-way agreement (clean interpreter == JIT-off ==
/// JIT-on), that JIT-off compiles nothing, and that JIT-on compiles at least one
/// loop. That last one is tier liveness and nothing more: `compiles >= 1` counts
/// TRACES, not work, and an empty dispatch still compiles a trace whose whole
/// optimized body is `Finish()`. No assertion here inspects a compiled body.
#[cfg(test)]
mod tests {
    /// `common::COMPILES` / `common::ABORTS` are process-global and every gate
    /// brackets its runs with `store(0)` … `load()`, so two gates on libtest's
    /// parallel threads read each other's compiles. The guard is held across the
    /// whole gate call, which is what puts the counter loads — they happen
    /// inside the gate — inside the lock. A load taken after the guard dropped
    /// would observe a concurrent test's compile.
    ///
    /// Poison is discarded: a gate that fails an assertion panics with the lock
    /// held, and a poisoned mutex would turn one real failure into five.
    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The unroll-retry channel, read as a delta inside `PROBE_LOCK` — the same
    /// window the gates bracket their own counters in. The counters are
    /// process-global and CUMULATIVE, so an absolute read would carry every
    /// other gate's events, which is what the lock exists to prevent.
    ///
    /// WARNING: `unroll_cancelled_invalid_loop` and `unroll_free_retry_rescued` must
    /// not be summed: they are two events on one ladder, and which of them
    /// answers "how often was the unrolled compile abandoned" is decided by
    /// `max_unroll_loops`. That reading rule is stated once, beside the
    /// counters themselves in `MC_DIAG_LABELS`; a second copy here would be
    /// free to drift from the ladder it describes.
    ///
    /// The slots are looked up by LABEL and their indices fall out, because an
    /// index re-stated beside a hand-written name is free to drift from it and
    /// these indices have moved before. The printed name comes from
    /// `MC_DIAG_LABELS` for the same reason, as does the gate name from
    /// libtest's thread name — a harness that does not name its threads reports
    /// `unknown` rather than a plausible-looking guess.
    const CENSUS_SLOTS: [usize; 3] = [
        mc_diag_slot("unroll_cancelled_invalid_loop"),
        mc_diag_slot("unroll_free_retry_rescued"),
        mc_diag_slot("unroll_free_retry_failed"),
    ];

    /// The index of `label` in `MC_DIAG_LABELS`, resolved at compile time.
    ///
    /// A label renamed or removed upstream fails the BUILD here. The census
    /// prints `MC_DIAG_LABELS[slot]` beside every count, so a wrong index reads
    /// back as internally consistent and cannot announce itself at runtime.
    const fn mc_diag_slot(label: &str) -> usize {
        let mut slot = 0;
        while slot < majit_metainterp::MC_DIAG_LABELS.len() {
            if str_eq(majit_metainterp::MC_DIAG_LABELS[slot], label) {
                return slot;
            }
            slot += 1;
        }
        panic!("no MC_DIAG slot carries this label");
    }

    /// `str` carries no const `==`.
    const fn str_eq(a: &str, b: &str) -> bool {
        let a = a.as_bytes();
        let b = b.as_bytes();
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    fn exclusive<T>(gate: impl FnOnce() -> T) -> T {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let before = CENSUS_SLOTS.map(majit_metainterp::mc_diag);
        let r = gate();
        let after = CENSUS_SLOTS.map(majit_metainterp::mc_diag);
        let thread = std::thread::current();
        let counts = CENSUS_SLOTS
            .iter()
            .enumerate()
            .map(|(i, &slot)| {
                format!(
                    "{}={}",
                    majit_metainterp::MC_DIAG_LABELS[slot],
                    after[i] - before[i]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "[unroll-census] gate={} {counts}",
            thread.name().unwrap_or("unknown")
        );

        // #90's degraded-arm gate.  Sited here, once, rather than in each of
        // the five gates: cel's seven `#[jit_interp]` mainloops carry seven
        // `state = T` names, and a per-gate assertion would have to restate
        // that list beside them, free to drift the moment a sixth probe module
        // lands.  Nothing but cel's own jitcodes installs in cel's test binary,
        // so the whole registry IS cel's set and no `interp` filter is needed
        // to scope it.
        //
        // An EMPTY-SET EQUALITY, not one of the cause gates the crates with
        // degraded arms carry: there is nothing here to classify, and a cause
        // gate over an empty set classifies a population of zero.  The set is
        // empty BY CONSTRUCTION rather than by luck — the macro's own per-arm
        // census (`MAJIT_MACRO_DEBUG=1 cargo check -p cel --no-default-features
        // --features dynasm`, which needs a cold `CARGO_TARGET_DIR` because it
        // is absent from any rebuild that hits the incremental cache) reads 46
        // arms over the seven loops: 45 `Lowerable`, every one of them
        // `inlined=true`, and one `Halt`.  `record_degraded_dispatch_arm` is
        // reached only from the NOT-inlined `Lowerable` paths and from
        // `Unsupported`, while `Nop`/`Halt`/`AbortPermanent` are declared
        // outcomes that deliberately record nothing, so cel's expansion emits
        // no call to the recorder at all.
        //
        // What it grades is the day that stops holding: an arm that stops
        // inlining, or a new arm that never lowered.  It cannot notice a
        // dispatch loop disappearing — an empty expectation agrees with an
        // empty subject — and no equality over an empty set can.
        let degraded: Vec<(&str, &str, &str)> = majit_metainterp::degraded_dispatch_arms()
            .iter()
            .map(|a| (a.interp, a.arm, a.reason))
            .collect();
        assert_eq!(
            degraded,
            Vec::new(),
            "dispatch arms degraded to abort stubs, seen after gate {}: \
             {degraded:?}. The registry is cumulative and never cleared, so the \
             gate named is where this was NOTICED and not necessarily the one \
             that installed the arm — every later gate reports it too.",
            thread.name().unwrap_or("unknown")
        );
        r
    }

    // Why every test below self-ignores under `debug_assertions`: the gates
    // compare i64 results that wrap and floats that must match bit-for-bit
    // across three execution paths, and in a debug profile the arithmetic panics
    // on overflow instead of wrapping — see the module header. `cargo test --all`
    // is debug, so that leg reports these ignored; the dedicated release step in
    // `.github/workflows/pyre-ci.yml` is what actually runs them. Left out of CI
    // the crate is back to being unable to fail, only now with a green badge.
    //
    // The reason string is spelled out at each site rather than shared through a
    // `macro_rules!`: `#[ignore = mac!()]` compiles and then silently reports a
    // bare `ignored` with no reason at all.

    #[test]
    #[cfg_attr(debug_assertions, ignore = "release only: wrapping i64 arithmetic")]
    fn probe_gates() {
        // 20M rows in the binary; the gate proves the same three properties at
        // any row count past the trace threshold of 8.
        exclusive(|| crate::probe::run_gates(300_000));
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "release only: wrapping i64 arithmetic")]
    fn policy_gates() {
        exclusive(crate::policy::run_gates);
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "release only: wrapping i64 arithmetic")]
    fn column_gates() {
        exclusive(crate::column::run_gates);
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "release only: wrapping i64 arithmetic")]
    fn colscalar_gates() {
        exclusive(crate::colscalar::run_gates);
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release only: bit-exact float and wrapping i64 arithmetic"
    )]
    fn float_gates() {
        exclusive(crate::float::run_gates);
    }
}
