//! A tuple receiver survives a collection raised by its own slice bounds.
//!
//! `binary_slice_values` converts both bounds through `eval_slice_index`,
//! which runs a user `__index__` — arbitrary Python, so an arbitrary number of
//! collections — and only then fetches the items. The tuple header comes from
//! the nursery, so the receiver moves across that call and the arm has to
//! reread it from its root slot rather than carry the word in.
//!
//! No unit test over `binary_slice_values` can arrange this. The defect needs
//! a real collection to land between the arm's type test and its fetch loop,
//! which needs a real nursery filling up under a real allocating `__index__`;
//! calling the helper directly with a hand-made tuple reproduces nothing.
//!
//! Nor does it fail loudly at the point of the mistake. `w_tuple_getitem_known`
//! ends in `else { debug_assert!(is _oo); ... }`, and that assertion is
//! compiled out of a release build, so the stale word is read as an arity-2
//! specialised tuple. `w_tuple_len` has already answered a garbage capacity
//! through the same word, so the bounds guard lets index 2 through and the
//! failure surfaces as `unreachable!("bounds guard above")` — a message about
//! the index, in a receiver that is no longer a tuple.
//!
//! What is asserted: the program's own arithmetic. A stale word can still
//! answer a type test and hand back items, so the sum of the slices is the
//! observable, not merely a zero exit status. Both are checked, because a
//! harness that reads output alone scores a killed process as a pass.
//!
//! Both dispatch loops run it. The interpreter reaches the helper from
//! `PyFrame::binary_slice` and the JIT reaches it from `bh_binary_slice_fn`;
//! at this size the JIT arm compiles the loop rather than falling back to the
//! interpreter, so neither route stands for the other.
//!
//! `PYPY_GC_NURSERY` is not what makes the defect appear — the default-nursery
//! arm reproduced it too, on the host this was written on. It is what makes it
//! deterministic: on a host whose default nursery is large enough to hold the
//! whole run, only the small-nursery arms would still collect inside the
//! window.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the
//! target directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary
//! Cargo already built for this integration test.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset — which would be a compile error rather than a skip. Compile this file
// to nothing in that configuration instead.
#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

/// The dynasm-backend `pyre` binary cargo built for this test.
const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// A multiple of 5, so every residue the program slices at appears equally
/// often and [`EXPECTED`] closes. Enough for the JIT arm to compile the loop
/// (four loops and three bridges, measured), and small enough that the arms
/// stay quick under the unoptimized build `cargo test` runs by default.
const ROUNDS: usize = 2_000;

/// `tup[a:a + 3]` over `range(10)` sums to `3a + 3`, and `a` cycles through
/// `0..5`, so each group of five contributes `3*(0+1+2+3+4) + 15 = 45` —
/// nine per round.
const EXPECTED: usize = 9 * ROUNDS;

/// Slices a tuple with bounds whose `__index__` allocates.
///
/// The comprehension's result is discarded: it is there to fill the nursery
/// while the receiver sits in the arm's local, not for its value. Written as a
/// `while` loop so the only iterator in the program is the one under test.
const PROGRAM: &str = "\
class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        [object() for _ in range(200)]
        return self.v


tup = tuple(range(10))
total = 0
i = 0
while i < ROUNDS:
    a = i % 5
    total += sum(tup[Idx(a):Idx(a + 3)])
    i += 1
print(total)
";

/// Run [`PROGRAM`] under `env` and return its stdout, asserting it exited
/// cleanly. Named by path rather than passed with `-c` so a failure reports a
/// real filename, and per arm so concurrent arms cannot read each other's
/// half-written file.
fn run(label: &str, env: &[(&str, &str)]) -> String {
    let src = PROGRAM.replace("ROUNDS", &ROUNDS.to_string());
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("tuple_slice_index_rooting_{label}.py"));
    std::fs::write(&path, &src).expect("write the program under test");

    let mut cmd = Command::new(PYRE);
    cmd.arg(&path);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run the program under test");
    assert!(
        out.status.success(),
        "{label}: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn default_nursery_slices_correctly() {
    assert_eq!(run("default", &[]), EXPECTED.to_string());
}

/// The JIT loop reaches the helper from `bh_binary_slice_fn`.
#[test]
fn small_nursery_jit_slices_correctly() {
    assert_eq!(
        run("jit", &[("PYPY_GC_NURSERY", "64K")]),
        EXPECTED.to_string(),
    );
}

/// The interpreter reaches the same helper from `PyFrame::binary_slice`.
#[test]
fn small_nursery_interpreter_slices_correctly() {
    assert_eq!(
        run("nojit", &[("PYPY_GC_NURSERY", "64K"), ("PYRE_NO_JIT", "1")],),
        EXPECTED.to_string(),
    );
}
