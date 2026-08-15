//! Pins the trace SHAPE a hot loop compiles to, read from the `MAJIT_LOG` dump.
//!
//! `loops_compiled` counts *compiles*, and a straight-line FINISH trace is one
//! of them — so it reads `1` for a peeled loop and `1` for a program that
//! compiled no loop at all. Nothing else in the tree distinguishes the two:
//! `pyre/check.py`'s jit-stats gate compares counters, and an equality gate
//! cannot fire on `1 -> 1`. The trace dump is the only instrument that can, and
//! it needs no new counter because the emitter already labels each artifact.
//!
//! Only three dump headers identify a shape. The generic before/after headers
//! occur on every compile path and are rejected explicitly by [`classify`].
//!
//! | header | what it tells you |
//! |---|---|
//! | `--- peeled trace (assembled) ---`      | loop, peeled   |
//! | `--- simple loop trace (after opt) ---` | loop, unpeeled |
//! | `--- finish trace (...) ---`            | **flat** — straight line, no loop |
//! | `--- trace (before opt) ---`             | generic; rejected |
//! | `--- trace (after opt) ---`              | generic; rejected |
//!
//! The test locates the transition within a declared window instead of pinning
//! a trip count, because the hotness threshold and loop-body size can change.
//!
//! The scan is linear because a binary search would require compilation to be
//! monotone in `n`. When the boundary leaves the window, the test says so.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the target
//! directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary Cargo already
//! built for this integration test.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset — which would be a compile error rather than a skip. Compile this file
// to nothing in that configuration instead.
#![cfg(feature = "dynasm")]

use std::process::Command;
use std::sync::OnceLock;

/// The dynasm-backend `pyre` binary cargo built for this test. See the module
/// doc: resolving it this way rather than building it is what keeps the test
/// from deadlocking against its own parent cargo.
const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// Trip counts scanned to locate the boundary, inclusive.
///
/// The window must straddle the transition: it needs at least one `n` below the
/// boundary that compiles nothing, and at least one above it. Both edges are
/// checked at run time (see `locate`), because a window that no longer
/// straddles the boundary would otherwise report its own edge as the answer.
const WINDOW_LO: u32 = 1030;
const WINDOW_HI: u32 = 1050;

/// What one scan observed. Fields are observations, not assertions — the tests
/// do the asserting, so a failure message can show what was actually seen.
struct Boundary {
    /// Smallest `n` in the window that compiled anything at all.
    n: u32,
    shape_at_n: Shape,
    shape_above: Shape,
    loops_at_n: Option<u64>,
    loops_above: Option<u64>,
    headers_at_n: Vec<String>,
    headers_above: Vec<String>,
}

/// The shape of the artifact a run compiled, as named by the dump itself.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Shape {
    /// Straight-line FINISH trace: no loop was compiled.
    Flat,
    Peeled,
    SimpleLoop,
    /// Nothing was compiled at all — distinct from `Flat`, which DID compile.
    None,
}

/// Classify a run's stderr by the dump's own artifact-kind header.
///
/// Generic before/after headers are ignored because every compile path emits
/// them and they carry no shape information.
fn classify(stderr: &str) -> Shape {
    let mut shape = Shape::None;
    for line in stderr.lines() {
        let s = if line.starts_with("--- peeled trace") {
            Shape::Peeled
        } else if line.starts_with("--- simple loop trace") {
            Shape::SimpleLoop
        } else if line.starts_with("--- finish trace") {
            Shape::Flat
        } else {
            continue;
        };
        // A loop header wins over a finish header: a peeled loop also emits
        // finish traces for its bridges once guards start failing.
        if shape == Shape::None || matches!(s, Shape::Peeled | Shape::SimpleLoop) {
            shape = s;
        }
    }
    shape
}

/// Every artifact-kind header a run emitted, for failure messages. Includes the
/// generic ones precisely so a failure shows what WAS there.
fn headers(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .filter(|l| l.starts_with("--- "))
        .map(str::to_owned)
        .collect()
}

fn loops_compiled(stderr: &str) -> Option<u64> {
    stderr
        .split_whitespace()
        .filter_map(|t| t.strip_prefix("loops_compiled="))
        .last()
        .and_then(|v| v.parse().ok())
}

/// Run a bare `while` loop of `n` trips and return the run's stderr. A bare
/// loop is the smallest thing that can peel, with no calls to complicate the
/// trace; `-c` keeps the probe out of the filesystem, so parallel tests cannot
/// race on a shared scratch file.
fn run_probe(n: u32) -> String {
    // A raw string, NOT a `\`-continued one: Rust's line continuation strips
    // the next line's leading whitespace, which would silently dedent `i += 1`
    // out of the loop body and change the program being measured.
    let src = format!(
        r#"
def f(n):
    s = 0
    i = 0
    while i < n:
        s += i
        i += 1
    return s

print(f({n}))
"#
    );
    let out = Command::new(PYRE)
        .args(["-c", &src])
        .env("MAJIT_LOG", "1")
        .env("MAJIT_STATS", "1")
        .output()
        .expect("spawn pyre-dynasm");
    assert!(
        out.status.success(),
        "probe at n={n} exited {:?}\nstderr tail:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .join("\n"),
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Scan the window and return the first trip count that compiles anything,
/// together with what it and its successor produced.
///
/// Panics only when the window can no longer answer the question — never
/// because the boundary moved *within* it. Those two cases are different and
/// the messages say which.
fn locate() -> Boundary {
    let mut first = None;
    for n in WINDOW_LO..=WINDOW_HI {
        let err = run_probe(n);
        if classify(&err) != Shape::None {
            first = Some((n, err));
            break;
        }
    }

    let (n, err_at_n) = first.unwrap_or_else(|| {
        panic!(
            "no trip count in [{WINDOW_LO},{WINDOW_HI}] compiled anything at all. \
             The boundary has left the window from ABOVE, or compilation is off \
             in this build. Widen the window and re-derive; do not delete the \
             assert."
        )
    });

    // Require observations on both sides; otherwise the result is merely a
    // window edge rather than a located transition.
    assert!(
        n > WINDOW_LO,
        "the window floor n={WINDOW_LO} already compiles, so the boundary is at \
         or below it and this scan never saw the non-compiling side. Lower \
         WINDOW_LO and re-derive.",
    );
    assert!(
        n < WINDOW_HI,
        "the boundary landed on the window ceiling n={WINDOW_HI}, so there is no \
         n+1 left to compare against. Raise WINDOW_HI and re-derive.",
    );

    let err_above = run_probe(n + 1);

    // Reported, not asserted: this is the movable fact.
    println!(
        "[jit_trace_shape] located boundary: n={n} flat / n={} peeled",
        n + 1
    );

    Boundary {
        n,
        shape_at_n: classify(&err_at_n),
        shape_above: classify(&err_above),
        loops_at_n: loops_compiled(&err_at_n),
        loops_above: loops_compiled(&err_above),
        headers_at_n: headers(&err_at_n),
        headers_above: headers(&err_above),
    }
}

/// One scan shared by both tests — the scan is ~20 subprocess runs and both
/// tests need the same observation.
fn boundary() -> &'static Boundary {
    static BOUNDARY: OnceLock<Boundary> = OnceLock::new();
    BOUNDARY.get_or_init(locate)
}

#[test]
fn first_compiling_trip_count_is_flat_and_the_next_one_peels() {
    let b = boundary();

    assert_eq!(
        b.shape_at_n,
        Shape::Flat,
        "THE TRACE SHAPE AT THE BOUNDARY CHANGED — re-derive it, do not delete \
         or widen this assert.\n\
         n={} is the first trip count in [{WINDOW_LO},{WINDOW_HI}] that compiled \
         anything, so it should be the FLAT cell — the threshold trips with the \
         loop already exhausted, the tracer never sees a back edge, and the tail \
         is recorded as a straight line. It classified as {:?} instead.\n\
         What to do: run the recipe in the WINDOW_LO doc above, confirm which \
         trip counts now produce which shapes, and update the recorded boundary \
         in that doc. If the JIT legitimately no longer emits a flat cell at \
         all, update the test to document the new transition.\n\
         headers at n={}: {:?}",
        b.n,
        b.shape_at_n,
        b.n,
        b.headers_at_n,
    );
    assert_eq!(
        b.shape_above,
        Shape::Peeled,
        "THE TRACE SHAPE ABOVE THE BOUNDARY CHANGED — re-derive it, do not \
         delete or widen this assert.\n\
         n={} was expected to compile a peeled loop but classified as {:?}. \
         The flat cell at n={} was found as expected, so the boundary itself is \
         intact and only the shape above it moved — check the peeler before \
         assuming the threshold slid.\n\
         headers at n={}: {:?}",
        b.n + 1,
        b.shape_above,
        b.n,
        b.n + 1,
        b.headers_above,
    );
}

#[test]
fn loops_compiled_cannot_tell_the_two_shapes_apart() {
    let b = boundary();

    // The equality below is meaningful only when the two runs have different
    // shapes; otherwise it compares a shape with itself.
    assert!(
        b.shape_at_n != b.shape_above,
        "n={} and n={} both compiled {:?}, so there is no shape change here for \
         a counter to be blind TO. This test cannot say anything until the \
         shapes differ.",
        b.n,
        b.n + 1,
        b.shape_at_n,
    );

    let flat_n = b
        .loops_at_n
        .expect("no loops_compiled at the flat trip count");
    let peeled_n = b
        .loops_above
        .expect("no loops_compiled at the peeled trip count");

    // Both sides must actually have compiled something, or the equality below
    // is satisfied by two zeros and asserts nothing.
    assert!(
        flat_n > 0 && peeled_n > 0,
        "one side compiled nothing (flat={flat_n}, peeled={peeled_n}); this \
         test's equality would then hold vacuously",
    );
    assert_eq!(
        flat_n,
        peeled_n,
        "loops_compiled is expected to be BLIND to the shape change at \
         n={} -> {}, but it moved {flat_n} -> {peeled_n}.\n\
         If a counter now distinguishes flat from peeled, `pyre/check.py`'s \
         jit-stats gate can see shape regressions and the note beside \
         JITSTATS_SNAPSHOT_FIELDS saying it cannot should be revisited.",
        b.n,
        b.n + 1,
    );
}
