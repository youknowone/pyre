//! Pins the trace SHAPE a hot loop compiles to, read from the `MAJIT_LOG` dump.
//!
//! `loops_compiled` counts *compiles*, and a straight-line FINISH trace is one
//! of them — so it reads `1` for a peeled loop and `1` for a program that
//! compiled no loop at all. Nothing else in the tree distinguishes the two:
//! `pyre/check.py`'s jit-stats gate compares counters, and an equality gate
//! cannot fire on `1 -> 1`. The trace dump is the only instrument that can, and
//! it needs no new counter because the emitter already labels each artifact.
//!
//! STOP: ONLY THREE OF THE DUMP'S HEADERS NAME A SHAPE. The other two fire on
//! every compile path regardless of shape:
//!
//! | header | what it tells you |
//! |---|---|
//! | `--- peeled trace (assembled) ---`      | loop, peeled   |
//! | `--- simple loop trace (after opt) ---` | loop, unpeeled |
//! | `--- finish trace (...) ---`            | **flat** — straight line, no loop |
//! | `--- trace (before opt) ---`            | STOP: NOTHING — generic |
//! | `--- trace (after opt) ---`             | STOP: NOTHING — generic |
//!
//! A test that matched `trace (after opt)` would match on every shape and pass
//! forever. That is the exact failure this test exists to prevent, so the
//! generic spellings are rejected explicitly in `classify` rather than merely
//! left out.
//!
//! ## ⭐ This test LOCATES the boundary; it does not assert a literal
//!
//! An earlier revision asserted `N_FLAT == 1040`. That is an **observed
//! constant, not an invariant** — it is a fact about today's back-edge
//! threshold and today's loop-body op count, both of which are expected to move
//! for entirely legitimate reasons. A live gate that reds on a *correct* change
//! trains people to bless it away, and the first thing blessed away is the
//! finding it encodes.
//!
//! So the test scans a declared window, locates the boundary, and asserts the
//! **structure** around it — reporting the located trip count as output rather
//! than asserting it. The structural claim is what #137 actually found and it
//! does not move when the threshold does.
//!
//! STOP: The scan is LINEAR, deliberately. A binary search would assume
//! "compiles anything" is monotone in `n`, which nobody has measured. A window
//! scan assumes nothing, and when the boundary leaves the window it says so.
//!
//! STOP: DO NOT REINTRODUCE A NESTED `cargo build` HERE. An earlier revision of
//! this file shelled out to `cargo build --release -p pyrex --bin pyre-dynasm`
//! from inside the test. That DEADLOCKS: `cargo test` holds the lock on the
//! target directory for the whole command — including while the test binaries
//! run — so a nested cargo writing into the same target directory waits on a
//! lock its own parent is holding, forever. It is not slowness and no timeout
//! fixes it. `CARGO_BIN_EXE_pyre-dynasm` is the supported route: cargo has
//! already built the binary for this integration test, with known features and
//! a known profile, before the test starts.

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
/// boundary that compiles NOTHING, and at least one above it. Both edges are
/// checked at run time (see `locate`), because a window that no longer
/// straddles the boundary would otherwise report its own edge as the answer.
///
/// Last observed boundary: **1040 flat / 1041 peeled**, on "majit-metainterp:
/// consult the hotness counter before building the tracing arm's arguments"
/// and on "majit-metainterp: borrow the virtualizable name in sync_before
/// instead of cloning it" (which carries "majit-metainterp: add typed creator
/// forms for attach_procedure_to_interp and mark_force_finish_tracing"'s
/// rewrite of seven `WarmEnterState` method bodies), in both the `dev` and
/// `release` profiles.
///
/// WARNING: Those three are cited by SUBJECT because the shas the readings were taken
/// at are gone. Their rebased successors are `3f9b12db219`, `d45ea243ef8` and
/// `f3d502f0852` as of 2026-08-11 — same patches, but rebased onto a different
/// base, so no surviving commit has the tree either reading was taken on. Find
/// them with `git log -1 --fixed-strings --grep='<subject>'`, and treat the
/// shas as dated annotations that the next rebase invalidates again.
///
/// WARNING: The `release` reading came from a binary whose tree was NOT recorded, so
/// profile and tree are CONFOUNDED in that comparison. The honest claim is
/// "agreed across a profile change and a tree change together", NOT
/// "profile-independent" — do not upgrade that sentence without a `release`
/// build at a named tree.
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
/// STOP: Deliberately ignores `--- trace (before opt) ---` / `--- trace (after
/// opt) ---`: those are emitted on every compile path and carry no shape
/// information. Matching them is how this test would become vacuous.
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

    // STOP: The window must STRADDLE the boundary. If the very first n scanned
    // already compiles, we never observed the non-compiling side, and `n` is
    // the window's edge rather than a located boundary — the scan would be
    // reporting its own floor as a finding.
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
         all, that RETIRES #137's finding and this file should say so rather \
         than be deleted quietly.\n\
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

    // STOP: THE SHAPES MUST DIFFER BEFORE "THE COUNTER CANNOT TELL THEM APART"
    // MEANS ANYTHING. Without this, two same-shaped runs make the equality
    // below hold trivially and the test reports blindness it never observed —
    // it would be comparing a thing to itself. This assert was added after a
    // negative control (pinning the flat n to the peeled one) left this test
    // GREEN while the shape test correctly failed.
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
