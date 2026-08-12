//! Allocation-site census for a real pyre run.
//!
//! #90, #117 and #123 all stalled on the same absent capability: pyre has no
//! way to count allocations and say *where* they came from. cel has one
//! (`cel-jit/cel/examples/rca88b.rs`), but cel is blind to every site those
//! tasks care about — it sets no driver descriptor, and `#[jit_interp]`
//! overrides the trait defaults pyre's hand-written `JitState` does not use.
//!
//! ```text
//! cargo run --release --example allocsites -- script.py
//! cargo run --release --no-default-features --features cranelift \
//!     --example allocsites -- script.py
//! ```
//!
//! WARNING: The `#[global_allocator]` below is declared in this example, not in
//! `pyrex`'s library, so nothing a user runs links this counter.
//!
//! It also does not collide with `mimalloc`, despite that being a default
//! feature: `mimalloc`'s `#[global_allocator]` is declared in the three **bin**
//! targets (`src/main.rs`, `src/bin/pyre-{cranelift,dynasm}.rs`) and never in
//! `lib.rs`, and an example links the library, not another binary's `main`. So
//! the allocator slot here is free under any feature set. `--no-default-features`
//! is still the honest configuration to *compare against* a `pyre` run, since a
//! default `pyre` measures mimalloc's behaviour and this example measures
//! `System`'s; the manifest names that case directly — *"Keep the feature
//! separate so allocator-sensitive diagnostics can still use
//! `--no-default-features`."* Counts of `alloc` calls are unaffected either way.
//!
//! # Three things that are not obvious, and each of which silently returns zero
//!
//! 1. **`main_entry` runs the interpreter on a spawned thread** (256 MiB
//!    stack), not the process's original one. Every counter here is a
//!    process-global atomic for that reason. Thread-locals — which is what
//!    cel's instrument uses, because cel runs inline on the calling thread —
//!    would report zero for the whole run.
//!
//! 2. **A backtrace per allocation cannot be afforded here.** cel's instrument
//!    captures unconditionally because a settled cel call is ~27 allocations; a
//!    pyre startup is many orders more, at ~10 µs a capture. So capture is off
//!    unless asked for, and when asked for it is *windowed* (`_AFTER`) and
//!    *budgeted* (`_BUDGET`). The totals are always exact — only attribution is
//!    sampled, and the report prints the window it used.
//!
//! 3. **`real_main` exits through `std::process::exit` on several paths**
//!    (`finalize_system_exit`, the sandbox controller, every usage error), and
//!    an exit skips this report. `pyrex` has no `libc` dependency to hang an
//!    `atexit` on. The failure mode is therefore *no report at all*, never a
//!    truncated one — if you do not see `=== allocsites ===`, the run left
//!    through an exit and nothing below was measured. Scripts that end by
//!    falling off the last line return normally; `sys.exit()` does not.
//!
//! # STOP: DISCARD THE FIRST RUN OF ANY BINARY — NEW NAME *OR* FRESHLY REBUILT
//!
//! The first run of a given binary costs about **+48,000 allocations (+6.9%)**
//! against every run after it. Measured three ways on `loop.py`, `n=200000`:
//!
//! ```text
//! fresh name, same dir : 749,183   701,177   701,122
//! after a plain rebuild: 749,422   701,353   701,386
//! ```
//!
//! The delta is not diffuse. Diffing the cold and warm site tables (this
//! instrument, exhaustive capture) attributes **all 48,095 of it** to
//! `pyre_interpreter::importing::load_source_module` under `dunder_import` —
//! module import work the warm run does not repeat.
//!
//! Two caches sit in `lib-python/3/__pycache__/` and only one of them is
//! shared: `<mod>.pyre-314.pyc` is tagged by implementation and common to every
//! binary, while
//! `frozen.importlib__bootstrap{,_external}.<BINARY>.opt-0.marshalcache` is
//! keyed by the **executable's basename** — `pyre`, `pyre-cranelift`,
//! `pyre-dynasm` and this example each carry their own pair, and a copy under a
//! new name mints another. A rebuild under the *same* name re-arms the cost too,
//! so the name is not the whole key; treat "first run of this binary" as the
//! rule and do not rely on a finer one.
//!
//! STOP: Any A/B **between two differently-named binaries** is wrong by that margin
//! unless both are warmed, and the error is one-sided — the newer name always
//! looks worse. Comparing a purpose-built control binary against the shipped one
//! is exactly the shape that hits it, and `__pycache__` already holds a
//! `pyre-dynasm-freshllbc` pair from such a comparison.
//!
//! WARNING: The executable's *location* matters too, and more: the same binary copied
//! outside the repo read **846,431** (+20.7%), because stdlib discovery is
//! relative to the executable. Compare like with like, from the same directory.
//!
//! WARNING: One caveat on comparing this example's totals to a `pyre` binary's:
//! `pyrex`'s `[dev-dependencies]` enable `pyre-interpreter/test-hooks`, and
//! dev-dependencies apply to examples. Feature unification means this target
//! runs with that hook available where the shipped binary does not. It is a
//! hash-hook install, so it should not move allocation counts — but "should
//! not" is a prediction, and a cross-target delta is the first place to look
//! if one appears.
//!
//! # Attribution is by site, not one aggregate
//!
//! The consumers want different cuts (#90 green-key sites, #117 cell counters,
//! #123 per-back-edge), so one total would send each of them back to
//! re-instrument. Allocations are grouped by their captured stack, filtered to
//! frames naming this system.
//!
//! STOP: Grouping by *size* — which is what an earlier instrument did — is what
//! this deliberately avoids: a size-keyed bucket that stops matching reports a
//! zero, and a zero from a missed filter is indistinguishable from a real one.
//!
//! # Environment
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `PYRE_ALLOCSITES` | unset | set to `1` to capture stacks at all |
//! | `PYRE_ALLOCSITES_AFTER` | 0 | skip this many allocations first (startup) |
//! | `PYRE_ALLOCSITES_BUDGET` | 20000 | stop capturing after this many stacks |
//! | `PYRE_ALLOCSITES_EVERY` | 1 | capture 1 in N of the eligible allocations |
//! | `PYRE_ALLOCSITES_ROWS` | 40 | how many site rows to print |

use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};

/// Every allocation the program makes, whatever the configuration. Exact.
static TOTAL_ALLOCS: AtomicU64 = AtomicU64::new(0);
/// Allocations made by the capture machinery itself, kept out of `TOTAL_ALLOCS`
/// so that turning capture on does not change the number capture is measuring.
static INSTRUMENT_ALLOCS: AtomicU64 = AtomicU64::new(0);
/// Allocations that fell inside the capture window and were eligible to be
/// attributed (before `_EVERY` sampling).
static ELIGIBLE: AtomicU64 = AtomicU64::new(0);
/// Stacks actually recorded. `<= ELIGIBLE`.
static CAPTURED: AtomicU64 = AtomicU64::new(0);
/// Samples lost because another thread held the site map. Reported: a silent
/// drop would understate a site by an unknown amount.
static LOST_CONTENDED: AtomicU64 = AtomicU64::new(0);

static CAPTURE_ON: AtomicBool = AtomicBool::new(false);
static AFTER: AtomicU64 = AtomicU64::new(0);
static BUDGET: AtomicU64 = AtomicU64::new(20_000);
static EVERY: AtomicU64 = AtomicU64::new(1);
static ROWS: AtomicUsize = AtomicUsize::new(40);

/// Site signature -> count. A `Mutex`, not a thread-local, for reason (1)
/// above; `try_lock` keeps the allocator non-blocking.
static SITES: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

std::thread_local! {
    /// `Backtrace::force_capture`, `format!` and `HashMap::insert` all
    /// allocate, and those allocations re-enter `alloc` immediately. Without
    /// this the first capture recurses until the stack ends. Re-entrancy is
    /// per-thread even though the counters are not. `const`-initialised so
    /// first access does not itself allocate.
    static IN_CAPTURE: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

/// True while this thread is inside the capture machinery.
fn in_capture() -> bool {
    IN_CAPTURE.try_with(Cell::get).unwrap_or(true)
}

#[cold]
#[inline(never)]
fn record_site() {
    if IN_CAPTURE.try_with(|c| c.replace(true)).unwrap_or(true) {
        return;
    }
    let bt = Backtrace::force_capture();
    let sig = interesting_frames(&format!("{bt}"));
    if !sig.is_empty() {
        match SITES.try_lock() {
            Ok(mut guard) => {
                if let Some(map) = guard.as_mut() {
                    *map.entry(sig).or_insert(0) += 1;
                    CAPTURED.fetch_add(1, Relaxed);
                }
            }
            Err(_) => {
                LOST_CONTENDED.fetch_add(1, Relaxed);
            }
        }
    }
    let _ = IN_CAPTURE.try_with(|c| c.set(false));
}

/// Decide whether this allocation is inside the capture window. Runs on every
/// allocation once capture is on, so it is loads and one modulo.
#[inline]
fn maybe_capture(n: u64) {
    if n <= AFTER.load(Relaxed) {
        return;
    }
    if CAPTURED.load(Relaxed) >= BUDGET.load(Relaxed) {
        return;
    }
    let eligible = ELIGIBLE.fetch_add(1, Relaxed) + 1;
    let every = EVERY.load(Relaxed);
    if every > 1 && eligible % every != 0 {
        return;
    }
    record_site();
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if in_capture() {
            INSTRUMENT_ALLOCS.fetch_add(1, Relaxed);
        } else {
            let n = TOTAL_ALLOCS.fetch_add(1, Relaxed) + 1;
            if CAPTURE_ON.load(Relaxed) {
                maybe_capture(n);
            }
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A realloc counts as an allocation: it is exactly what an unsized
        // `Vec` pays per growth step, which is the cost #123 site 3 is about.
        if in_capture() {
            INSTRUMENT_ALLOCS.fetch_add(1, Relaxed);
        } else {
            let n = TOTAL_ALLOCS.fetch_add(1, Relaxed) + 1;
            if CAPTURE_ON.load(Relaxed) {
                maybe_capture(n);
            }
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// Keep the frames naming this system; drop libstd, the allocator shim and the
/// capture machinery. A raw backtrace here is ~60 frames.
///
/// `allocsites_probe_marker` is kept deliberately — it is what the non-vacuity
/// check below looks for, and filtering the instrument's own frames out
/// entirely would make that check unable to observe itself.
fn interesting_frames(bt: &str) -> String {
    bt.lines()
        .map(str::trim)
        .filter(|l| l.contains("::"))
        .filter(|l| {
            (l.contains("majit") || l.contains("pyre") || l.contains("allocsites_probe_marker"))
                && !l.contains("record_site")
                && !l.contains("maybe_capture")
                && !l.contains("interesting_frames")
                && !l.contains("Backtrace")
                && !l.contains("backtrace")
        })
        .take(6)
        .collect::<Vec<_>>()
        .join(" <- ")
}

/// The allocations the non-vacuity check makes. Named so its frame survives
/// `interesting_frames`, which is how the check proves the *whole* chain works
/// — hook, backtrace, filter, map insert — and not merely the counter.
#[inline(never)]
fn allocsites_probe_marker(sink: &mut Vec<String>) {
    for i in 0..PROBE_ALLOCS {
        sink.push(format!("nonvacuity-{i}"));
    }
    std::hint::black_box(sink);
}

const PROBE_ALLOCS: u64 = 64;

/// ⭐ In-band non-vacuity check, printed every run.
///
/// An allocation counter that reads a plausible number while measuring nothing
/// is the failure this instrument exists to avoid, and a plausible number looks
/// exactly like a real one. So the harness makes a **known** number of
/// allocations and reports whether it saw them.
///
/// STOP: It runs at *report* time, not at startup, and deliberately so: the
/// configuration that matters is the one in force during the measured run. A
/// probe fired before `_AFTER` allocations had elapsed would sit outside the
/// capture window and fail for a reason that says nothing about the run.
fn nonvacuity() -> (u64, u64, bool) {
    let total_before = TOTAL_ALLOCS.load(Relaxed);
    let captured_before = CAPTURED.load(Relaxed);
    let mut sink: Vec<String> = Vec::with_capacity(PROBE_ALLOCS as usize);
    allocsites_probe_marker(&mut sink);
    let counted = TOTAL_ALLOCS.load(Relaxed) - total_before;
    let attributed = CAPTURED.load(Relaxed) - captured_before;
    // The site half can only be asserted when capture is on and unexhausted.
    let site_chain_testable = CAPTURE_ON.load(Relaxed)
        && CAPTURED.load(Relaxed) < BUDGET.load(Relaxed)
        && EVERY.load(Relaxed) == 1;
    (counted, attributed, site_chain_testable)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // Read configuration and build the map BEFORE arming: `std::env::var` and
    // `HashMap::new` allocate, and doing this under a live counter would
    // attribute the instrument's own setup to the program.
    let capture = std::env::var("PYRE_ALLOCSITES").is_ok_and(|v| v == "1");
    AFTER.store(env_u64("PYRE_ALLOCSITES_AFTER", 0), Relaxed);
    BUDGET.store(env_u64("PYRE_ALLOCSITES_BUDGET", 20_000), Relaxed);
    EVERY.store(env_u64("PYRE_ALLOCSITES_EVERY", 1).max(1), Relaxed);
    ROWS.store(env_u64("PYRE_ALLOCSITES_ROWS", 40) as usize, Relaxed);
    SITES.lock().unwrap().replace(HashMap::with_capacity(1024));
    CAPTURE_ON.store(capture, Relaxed);

    // The real CLI. argv is this example's own, so everything after `--`
    // reaches `real_main` unchanged.
    //
    // WARNING: Does not return if the script exits through `sys.exit()` — see the
    // module header. Nothing below runs in that case, and the absence of the
    // report is the signal.
    pyrex::main_entry("pyre-allocsites");

    let (probe_counted, probe_attributed, site_testable) = nonvacuity();
    CAPTURE_ON.store(false, Relaxed);

    let total = TOTAL_ALLOCS.load(Relaxed);
    let instrument = INSTRUMENT_ALLOCS.load(Relaxed);
    let eligible = ELIGIBLE.load(Relaxed);
    let captured = CAPTURED.load(Relaxed);
    let lost = LOST_CONTENDED.load(Relaxed);

    println!("\n=== allocsites ===");

    // The non-vacuity verdict is printed FIRST, so a reader cannot reach the
    // numbers without passing it.
    let counter_ok = probe_counted >= PROBE_ALLOCS;
    let sites_ok = !site_testable || probe_attributed > 0;
    let verdict = match (counter_ok, sites_ok) {
        (true, true) if site_testable => "PASS (counter + site chain)",
        (true, true) => "PASS (counter only; site chain not testable in this config)",
        (false, _) => {
            "STOP: FAIL — the counter did not see allocations it just made. \
                       Every number below is VOID, not small."
        }
        (_, false) => {
            "STOP: FAIL — allocations were counted but none were attributed. \
                       Site rows below are VOID."
        }
    };
    println!(
        "non-vacuity: made {PROBE_ALLOCS} allocations, counted {probe_counted}, \
         attributed {probe_attributed}  [{verdict}]"
    );

    let window = if capture {
        format!(
            "after={} budget={} every={}",
            AFTER.load(Relaxed),
            BUDGET.load(Relaxed),
            EVERY.load(Relaxed)
        )
    } else {
        "off (set PYRE_ALLOCSITES=1)".to_string()
    };
    println!("capture window:      {window}");
    println!("total allocations:   {total}   (exact, independent of the window)");
    println!(
        "WARNING: discard run 1 of any new binary NAME: the frozen-importlib marshal cache is\n\
         \x20 keyed by basename and costs ~48k extra allocations to build, once."
    );
    println!("instrument's own:    {instrument}   (excluded from the total above)");
    println!("eligible in window:  {eligible}");
    println!("stacks recorded:     {captured}");
    if lost > 0 {
        println!("WARNING: samples lost to lock contention: {lost}");
    }
    if capture && captured >= BUDGET.load(Relaxed) {
        println!(
            "WARNING: BUDGET EXHAUSTED — the rows below describe the first {captured} \
             eligible allocations only, not the run."
        );
    }

    let map = SITES.lock().unwrap().take().unwrap_or_default();
    let mut rows: Vec<(String, u64)> = map.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let limit = ROWS.load(Relaxed);
    println!("\ntop {limit} allocation sites (majit/pyre frames, innermost first):");
    for (sig, count) in rows.iter().take(limit) {
        println!("  {count:>9}  {sig}");
    }
    if rows.is_empty() {
        println!("  (none)");
    } else {
        println!("\ndistinct sites: {}", rows.len());
    }
}
