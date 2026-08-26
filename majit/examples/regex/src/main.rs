//! Marked-regex matching, from "An Efficient and Elegant Regular Expression
//! Matcher in Python" and "A JIT for Regular Expression Matching".
//!
//! A regex is a tree; each node holds one mutable `marked` bit, and one
//! `shift(c, mark)` per input character propagates the marks left to right.
//! Two things a reader should not have to guess: **the marks live in the
//! nodes**, not in the matcher, and `empty` — does this subexpression accept
//! the empty string? — is **computed once, at lowering time**, so the matcher
//! only reads it.
//!
//! `regex` builds and lowers the tree; `interp` is the plain matcher over the
//! lowered graph; `jit_interp` and `shortcircuit` are the two JIT portals.
//!
//! This file is also the benchmark, and its measurement discipline is half of
//! what it has to say. Every number printed below without a disclaimer was
//! measured in this process, in this run: each row is the median of
//! [`REPEATS`] timed runs with its min-max range beside it, taken at three
//! input lengths, and every ratio is one measured row divided by another.
//! Nothing is a constant carried over from an earlier session. What this run
//! could not measure — the post's own 2010 figures, the Python rows recorded
//! separately — is printed under a heading that says so, and no ratio above it
//! is computed from any of it.

pub mod interp;
pub mod jit_interp;
pub mod regex;
pub mod shortcircuit;

use regex::{NodeRec, bench_regex, bench_regex_left, count, depth, lower, nonmatching, vectors};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Timed runs per row. The example's own history is why this is not 1: the
/// same binary on this machine produced a 4.8x and a 5.5x minutes apart, and a
/// single run cannot tell which of those the machine actually does.
const REPEATS: usize = 5;

/// The post benchmarks "a random string (of varying lengths)". Three lengths
/// two orders of magnitude apart is enough to see whether a row reports a
/// per-character rate at all, which is the only reading under which the rows
/// can be compared to each other or to anybody else.
const LENGTHS: [usize; 3] = [1 << 12, 1 << 16, 1 << 20];

/// The `{N}` of the post's benchmark regex `(a|b)*a(a|b){20}a(a|b)*`.
const N: usize = 20;

/// The generator seed. Fixed, so every run of this binary — and every port
/// that reimplements `nonmatching` — scans exactly the same bytes.
const SEED: u64 = 42;

/// Back-edge count before a portal records. The tests in `jit_interp` grade
/// the loop compiled at this threshold, so the benchmark times that same loop.
const THRESHOLD: u32 = 3;

/// A row whose fastest and slowest length differ by more than this is not
/// reporting a per-character rate, and the difference is a finding rather than
/// noise.
const FLAT_ENOUGH: f64 = 1.10;

const NAMES: [&str; 3] = [
    "Rust interp over NodeRec, no JIT",
    "majit JIT, regex promoted",
    "majit JIT, short-circuit and/or",
];
const R_INTERP: usize = 0;
const R_JIT: usize = 1;

/// Characters scanned per second.
fn rate(chars: usize, secs: f64) -> f64 {
    chars as f64 / secs
}

fn commas(x: f64) -> String {
    let n = x.round() as u64;
    let d = n.to_string();
    let mut out = String::new();
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// What the run can say about the machine without shelling out: the target the
/// binary was built for, plus the two runtime facts `std` will hand over.
///
/// The post records its own hardware — "Intel Core2 Duo P8400 2.26GHz" — and a
/// chars/s figure means nothing without it.
fn machine() -> String {
    let cores =
        std::thread::available_parallelism().map_or_else(|_| "?".to_string(), |n| n.to_string());
    let backend = if cfg!(feature = "dynasm") {
        "dynasm"
    } else if cfg!(feature = "cranelift") {
        "cranelift"
    } else {
        "none"
    };
    let profile = if cfg!(debug_assertions) {
        "debug — NOT a benchmark build"
    } else {
        "release"
    };
    format!(
        "machine, read at runtime: {} {}, {} logical cores, {}-bit pointers, \
         majit backend {backend}, {profile} build",
        std::env::consts::ARCH,
        std::env::consts::OS,
        cores,
        usize::BITS,
    )
}

/// One implementation measured at one input length.
struct Row {
    name: &'static str,
    /// chars/s, one per timed run, ascending.
    rates: Vec<f64>,
    /// Loops the JIT compiled across the timed runs. 0 for a row that never
    /// enters the JIT.
    compiles: usize,
}

impl Row {
    /// The reported number: median, not mean.
    ///
    /// Benchmark noise is one-sided. A scheduler preemption, an interrupt, a
    /// page fault or a thermal step can only ever make a run slower — nothing
    /// makes a run faster than the machine can go. A mean folds every such
    /// excursion into the answer, so one bad run moves it and the figure
    /// reported is a run that never happened. The median is a run that did
    /// happen, and it is the one a rerun reproduces. The min-max range is
    /// printed beside it so the spread is visible instead of taken on trust: a
    /// median with a 2x range under it is not a measurement either, and the
    /// reader should be able to see that without being told.
    fn median(&self) -> f64 {
        let n = self.rates.len();
        if n % 2 == 1 {
            self.rates[n / 2]
        } else {
            (self.rates[n / 2 - 1] + self.rates[n / 2]) / 2.0
        }
    }
    fn min(&self) -> f64 {
        self.rates[0]
    }
    fn max(&self) -> f64 {
        self.rates[self.rates.len() - 1]
    }
}

/// Time `run` over `s`, [`REPEATS`] times.
///
/// One untimed run first: the first pass over a freshly built buffer takes the
/// page faults and the cold branch predictors, and those are not what the row
/// is about.
///
/// Every run's answer is asserted, timed ones included. That assertion is also
/// the hygiene this file used to spell as "a fresh tree per row": a mark left
/// behind by an earlier run would surface here as a spurious match, because
/// the input is built not to match.
fn bench(
    name: &'static str,
    root: *mut NodeRec,
    s: &[u8],
    counter: Option<&AtomicUsize>,
    run: impl Fn(*mut NodeRec, &[u8]) -> bool,
) -> Row {
    assert!(
        !run(root, s),
        "{name}: the benchmark input is supposed NOT to match"
    );
    let before = counter.map_or(0, |c| c.load(Ordering::Relaxed));
    let mut rates = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let hit = run(root, s);
        let secs = t0.elapsed().as_secs_f64();
        assert!(!hit, "{name}: the benchmark input is supposed NOT to match");
        rates.push(rate(s.len(), secs));
    }
    let compiles = counter.map_or(0, |c| c.load(Ordering::Relaxed) - before);
    rates.sort_by(f64::total_cmp);
    Row {
        name,
        rates,
        compiles,
    }
}

/// The peeled loop body of a recorded trace: everything from the last `Label`
/// on. A compiled loop is preamble plus peeled body and only the body runs per
/// input character, so grading the whole thing charges the body for the reads
/// the preamble hoisted out of it.
fn peeled(body: &[majit_ir::OpCode]) -> &[majit_ir::OpCode] {
    match body.iter().rposition(|op| *op == majit_ir::OpCode::Label) {
        Some(i) => &body[i..],
        None => body,
    }
}

/// The census a compiled body is read by. `getfield_gc_r` is the subject and
/// zero is the claim — the tree walk's pointer reads are gone — and the total
/// is printed beside it because a body of a handful of ops would also have no
/// pointer reads and would mean the walk never entered the trace at all.
fn census(body: &[majit_ir::OpCode]) -> String {
    use majit_ir::OpCode;
    let n = |op: OpCode| body.iter().filter(|o| **o == op).count();
    format!(
        "{:>4} ops: {} getfield_gc_r, {} getfield_gc_i, {} setfield_gc, {} int_eq, {} guard_value",
        body.len(),
        n(OpCode::GetfieldGcR),
        n(OpCode::GetfieldGcI),
        n(OpCode::SetfieldGc),
        n(OpCode::IntEq),
        n(OpCode::GuardValue),
    )
}

fn main() {
    let started = Instant::now();
    println!("{}", machine());
    println!();

    // ── correctness first: a fast wrong matcher is not a result ────────────
    let mut bad = 0;
    let total = vectors().len();
    for (re, s, want) in vectors() {
        let got = interp::matches(lower(&re), s.as_bytes());
        if got != want {
            bad += 1;
            println!("  FAIL {s:?}: got {got}, want {want}");
        }
    }
    println!("vectors: {}/{total} pass", total - bad);

    let balanced = lower(&bench_regex(N));
    let left_assoc = lower(&bench_regex_left(N));
    println!(
        "(a|b)*a(a|b){{{N}}}a(a|b)*  balanced: {} nodes, depth {}",
        count(balanced),
        depth(balanced)
    );
    println!(
        "(a|b)*a(a|b){{{N}}}a(a|b)*  left    : {} nodes, depth {}",
        count(left_assoc),
        depth(left_assoc)
    );

    // One tree per row, for the whole sweep. Two rows may never share a tree:
    // a mark is node state, so whatever one row left behind the other would
    // read. Re-lowering per length would only leak more graphs — `matches`
    // clears the marks it leaves, and `bench` asserts every answer, which is
    // the thing that breaks first if that ever stops being true.
    let roots = [
        lower(&bench_regex(N)),
        lower(&bench_regex(N)),
        lower(&bench_regex(N)),
    ];

    println!();
    println!("timed rows: {REPEATS} runs each, median (min - max).");
    println!("the input is nonmatching(len, {N}, {SEED}) — it matches nothing, so the matcher");
    println!("must read every character and no early exit can hide the per-character cost.");

    let mut sweep: Vec<(usize, [Row; 3])> = Vec::new();
    for len in LENGTHS {
        let s = nonmatching(len, N, SEED);
        assert!(
            !interp::matches(balanced, &s),
            "the benchmark input is supposed NOT to match"
        );

        let rows = [
            bench(NAMES[0], roots[0], &s, None, |r, s| interp::matches(r, s)),
            bench(
                NAMES[1],
                roots[1],
                &s,
                Some(&jit_interp::COMPILES),
                |r, s| jit_interp::matches(r, s, THRESHOLD),
            ),
            bench(
                NAMES[2],
                roots[2],
                &s,
                Some(&shortcircuit::COMPILES),
                |r, s| shortcircuit::matches(r, s, THRESHOLD),
            ),
        ];

        println!();
        println!("  {} chars, no match as intended:", commas(len as f64));
        for row in &rows {
            let compiled = if row.compiles > 0 {
                format!(
                    "   {:.1} loops compiled per run",
                    row.compiles as f64 / REPEATS as f64
                )
            } else {
                String::new()
            };
            println!(
                "    {:<32}: {:>12} chars/s  ({:>12} - {:>12}){}",
                row.name,
                commas(row.median()),
                commas(row.min()),
                commas(row.max()),
                compiled,
            );
        }
        sweep.push((len, rows));
    }

    // ── is any of it a per-character rate? ─────────────────────────────────
    println!();
    println!("rate against input length, median chars/s. The algorithm is O(m*n) with no");
    println!("per-character allocation, so a flat row is the expectation and a row that is not");
    println!("flat is reporting something other than the cost of one character.");
    print!("  {:<32}", "implementation");
    for (len, _) in &sweep {
        print!("{:>14}", commas(*len as f64));
    }
    println!("{:>9}", "spread");

    // Per row: the spread across the lengths, and which length was fastest —
    // the direction is what separates the two explanations below.
    let mut shape: Vec<(f64, usize)> = Vec::new();
    for (i, name) in NAMES.iter().enumerate() {
        print!("  {name:<32}");
        let (mut lo, mut hi, mut best) = (f64::MAX, 0.0f64, 0usize);
        for (j, (_, rows)) in sweep.iter().enumerate() {
            let m = rows[i].median();
            lo = lo.min(m);
            if m > hi {
                hi = m;
                best = j;
            }
            print!("{:>14}", commas(m));
        }
        println!("{:>8.2}x", hi / lo);
        shape.push((hi / lo, best));
    }

    let not_flat: Vec<usize> = (0..NAMES.len())
        .filter(|i| shape[*i].0 > FLAT_ENOUGH)
        .collect();
    if !not_flat.is_empty() {
        println!();
        println!("  FINDING: not every row reports a per-character rate.");
        for i in &not_flat {
            println!(
                "    {:<32} spread {:>6.2}x, fastest at {:>9} chars",
                NAMES[*i],
                shape[*i].0,
                commas(sweep[shape[*i].1].0 as f64),
            );
        }
        println!("    A row fastest at the LONGEST input is amortizing a cost that is fixed per");
        println!(
            "    `matches` call. `loops compiled per run` above is the candidate: each portal"
        );
        println!("    builds its `JitDriver` inside `mainloop`, so the compiled-loop cache dies");
        println!("    with the call and the next call records and compiles the loop again. Read");
        println!("    such a row at the shortest length as a matcher-plus-compiler rate, and at");
        println!("    the longest as the compiled loop's own.");
        println!("    A row fastest at the SHORTEST input has two candidates and this sweep does");
        println!("    not separate them. The tree is 93 nodes at every length, but the input");
        println!("    buffer grows from 4 KiB to 1 MiB, so the longest length is not reading out");
        println!("    of the same level of cache as the shortest. And a machine doing other work");
        println!("    at the same time costs a long run more than a short one — that one shows in");
        println!("    the min-max ranges above, so read the range before the median. An otherwise");
        println!("    idle machine is the control that tells the two apart.");
    }

    // ── what the JIT rows actually compiled ────────────────────────────────
    let jit_body = jit_interp::LAST_BODY.lock().unwrap().clone();
    let sc_body = shortcircuit::last_peeled_body();
    if !jit_body.is_empty() || !sc_body.is_empty() {
        println!();
        println!(
            "the peeled loop body of the last loop each portal compiled in this run — this is"
        );
        println!("what runs per input character, and 0 getfield_gc_r is the experiment's claim:");
        if !jit_body.is_empty() {
            println!("  {:<32}: {}", NAMES[1], census(peeled(&jit_body)));
        }
        if !sc_body.is_empty() {
            println!("  {:<32}: {}", NAMES[2], census(&sc_body));
        }
    }

    // ── the summary, in the post's shape ───────────────────────────────────
    let (len, rows) = sweep.last().expect("LENGTHS is not empty");
    let slowest = rows
        .iter()
        .min_by(|a, b| a.median().total_cmp(&b.median()))
        .expect("three rows");
    println!();
    println!(
        "SUMMARY — (a|b)*a(a|b){{{N}}}a(a|b)* over {} chars, median of {REPEATS} runs.",
        commas(*len as f64)
    );
    println!("Every ratio divides this run's numbers by this run's own slowest measured row —");
    println!("the post's own speedup column divides by its slowest row too, which for it was");
    println!("pure Python. Here the denominator is:");
    println!("  {}", slowest.name);
    println!(
        "  {:<32}{:>14}{:>12}",
        "implementation", "chars/s", "speedup"
    );
    for row in rows {
        println!(
            "  {:<32}{:>14}{:>11.1}x",
            row.name,
            commas(row.median()),
            row.median() / slowest.median()
        );
    }
    println!("  {:<32}{:>26}", "CPython `re`", "not measured by this run");

    // ── the two ratios the post makes its claims from ──────────────────────
    let jit = rows[R_JIT].median();
    let aot = rows[R_INTERP].median();
    println!();
    println!("the two ratios the post's headline claims are made of:");
    println!(
        "  majit JIT / the same matcher compiled ahead of time : {:>5.1}x",
        jit / aot
    );
    println!("      the comparable quantity in the post is its own 16,500,000 / 720,000 = 22.9x,");
    println!("      on its 2010 machine. Both sides divide a JIT that specialized on the regex");
    println!("      by the same algorithm compiled ahead of time — here `rustc -O`, there");
    println!("      RPython translated to C. The absolute rows are not comparable across");
    println!("      sixteen years and two instruction sets; the ratio is.");
    println!("  majit JIT / CPython `re`                            : not measured by this run");
    println!("      nothing in this process ran `re`; this binary times the Rust rows only. It");
    println!("      would not be a like-for-like ratio even once measured: `re` may bail out of");
    println!("      a non-match early, while the marked matcher always scans the whole string.");

    // ── everything below here this run did NOT measure ─────────────────────
    println!();
    println!("NOT MEASURED BY THIS RUN — this binary times the Rust rows only, and quotes no");
    println!("number it did not take itself. The pure-Python marked matcher, the same matcher");
    println!("in C++ and Java, CPython `re` and Google re2 are the rest of the post's table,");
    println!("and `comparisons/run.sh` measures them, printing each row's load beside it:");
    println!(
        "  majit/examples/regex/comparisons/run.sh {} {REPEATS} {N}",
        LENGTHS[2]
    );
    println!();
    println!("NOT MEASURED BY THIS RUN, AND NOT THIS MACHINE — the 2010 post's own figures, on");
    println!("its own hardware (Intel Core2 Duo P8400, 2.26GHz). Sixteen years and a different");
    println!("instruction set away; only its internal ratios are comparable to anything above:");
    println!("  pure Python                      :       12,200 chars/s");
    println!("  RPython translated to C          :      720,000 chars/s");
    println!("  CPython `re`                     :    2,500,000 chars/s");
    println!("  RPython + JIT, regex green       :   16,500,000 chars/s");
    println!("  its JIT-over-no-JIT ratio        :         22.9x");
    println!("  its JIT-over-`re` ratio          :          6.6x");

    println!();
    println!(
        "this run took {:.1}s of wall clock, all of it inside the rows above.",
        started.elapsed().as_secs_f64()
    );

    if bad != 0 {
        std::process::exit(1);
    }
}
