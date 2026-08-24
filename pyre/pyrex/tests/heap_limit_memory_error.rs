//! A bounded heap surfaces as a `MemoryError` before it aborts.
//!
//! incminimark.py's module docstring states the `PYPY_GC_MAX` policy as a
//! ladder: the GC *"will first collect more often, then raise an RPython
//! MemoryError, and if that is not enough, crash the program with a fatal
//! error"*. The middle rung is the only one a program can act on, and
//! `major_collection_step` is where upstream raises it.
//!
//! Pyre reaches that rung differently, and this test exists because the
//! difference is invisible to any unit test. Upstream's `raise MemoryError`
//! unwinds out of whichever driver ran the collection; pyre's collections are
//! mostly driven from the dispatch-loop safepoint, which returns `()` and
//! cannot raise, so the collector defers the exception through the eval-breaker
//! word and the dispatch loop raises it. Only an end-to-end run exercises that
//! hand-off — a collector unit test can assert the flag and a dispatch-loop
//! unit test can assert the raise, and both can pass while nothing joins them.
//!
//! What is asserted, and what is deliberately not. The middle rung's promise is
//! that the limit *becomes a Python exception*; it is not a promise that the
//! program survives one. After the first breach the live set is by definition
//! at the limit, so a handler that allocates before it frees can reach the rung
//! below — upstream says as much where it raises, and aborts on the second
//! arrival for exactly that reason. Whether any particular handler fits in the
//! remaining headroom is a question about collection scheduling, not about the
//! raise, and it is not stable enough to gate: sweeping the limit over
//! 80/96/112/128 MiB moves a surviving handler to an aborting one and back.
//! The two properties below were stable over every limit and both dispatch
//! loops when measured, so they are what this pins.
//!
//! `--heapsize` rather than `PYPY_GC_MAX` because it also covers the launcher
//! plumbing (`take_heapsize_option` -> `gc_set_max_heap_size`), which the env
//! var bypasses.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the target
//! directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary Cargo already
//! built for this integration test.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset — which would be a compile error rather than a skip. Compile this file
// to nothing in that configuration instead.
#![cfg(feature = "dynasm")]

use std::process::{Command, Output};

/// The dynasm-backend `pyre` binary cargo built for this test.
const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// 96 MiB — comfortably above what interpreter startup needs, so a breach is
/// the program's doing, and far below what `ROUNDS` objects need.
const HEAP_LIMIT: usize = 96 * 1024 * 1024;

/// 8 KiB of items per object.
const WORDS: usize = 1024;

/// Allocations attempted: `ROUNDS * WORDS * 8` = 200 MiB, twice the limit, so
/// the limited arms breach well before the end and the unbounded control still
/// finishes in under a second. Bounded rather than `while True` so an arm that
/// never raises terminates with a diagnosis instead of hanging.
const ROUNDS: usize = 25_000;

/// 160 KiB of items per object, which puts each one over the collector's
/// `large_object_threshold` of `(16384 + 512) * 8` bytes and so onto a
/// different allocation path from `WORDS` — see the large-object test.
const WORDS_LARGE: usize = 20_000;

/// `ROUNDS_LARGE * WORDS_LARGE * 8` = 320 MiB, again well past the limit.
const ROUNDS_LARGE: usize = 2_000;

/// Keeps every object reachable, so the live set really does grow — a dead one
/// would be reclaimed and the limit never reached.
const GROW: &str = "\
data = []
for _ in range(ROUNDS):
    data.append([0] * WORDS)
print('completed')
";

fn run(env: &[(&str, &str)], args: &[&str]) -> Output {
    run_shape(env, args, ROUNDS, WORDS)
}

fn run_shape(env: &[(&str, &str)], args: &[&str], rounds: usize, words: usize) -> Output {
    let src = GROW
        .replace("ROUNDS", &rounds.to_string())
        .replace("WORDS", &words.to_string());
    let mut argv: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    argv.push("-c".to_string());
    argv.push(src);
    let mut cmd = Command::new(PYRE);
    cmd.args(&argv);
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn pyre-dynasm")
}

fn report(what: &str, out: &Output) -> String {
    format!(
        "{what} exited {:?}\n--- stdout ---\n{}\n--- stderr tail ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// The control. Without a limit the same program runs to completion, so a
/// `MemoryError` in the limited arms is attributable to the limit and not to
/// the allocation volume, the host's memory, or the script.
#[test]
fn the_same_program_completes_when_the_heap_is_unbounded() {
    let out = run(&[], &[]);
    assert!(out.status.success(), "{}", report("control", &out));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("completed"),
        "{}",
        report("control did not finish", &out)
    );
}

/// The middle rung, on the JIT dispatch loop (the default).
///
/// The program has no handler, so the exception it is owed unwinds its frames
/// and is reported. That report is the whole point: before the breach was
/// routed to a dispatch loop, this run's only sign of the limit was
/// `out_of_memory("using too much memory, aborting")` on the *second* breach,
/// with the rung that names a Python exception skipped entirely.
#[test]
fn a_bounded_heap_reports_a_memory_error_rather_than_only_aborting() {
    let limit = HEAP_LIMIT.to_string();
    let out = run(&[], &["--heapsize", &limit]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("completed"),
        "{}",
        report("the limited run was not bounded at all", &out)
    );
    assert!(
        stderr.contains("MemoryError"),
        "{}",
        report("the limit never became a Python exception", &out)
    );
}

/// Objects past `large_object_threshold` are not exempt from the limit.
///
/// They take a different allocation path — `alloc_with_type_slow` hands them
/// to `alloc_in_oldgen_clear` and returns, ahead of the point where a nursery
/// allocation would consult `oom_pending` — so "the policy is enforced" does
/// not follow from the two tests above, which only ever allocate under the
/// threshold. It holds for a different reason, and that reason is worth
/// pinning: the breach is detected in the collector, by the safepoint-driven
/// collection that observes the old generation growing, rather than in an
/// allocator's return path. Routing the exception through the eval-breaker
/// word is what makes that detection point able to report at all, so this
/// shape would regress with the small one if the channel were removed —
/// but it would also regress alone if delivery were moved back into the
/// nursery allocator, which is the case the tests above cannot see.
///
/// Both dispatch loops in one test: the seam they share is already covered
/// per-loop above, and what is new here is only the allocation path.
#[test]
fn objects_over_the_large_object_threshold_are_bounded_too() {
    // Its own control, for the reason the small shape has one, plus one it
    // does not: this shape asks the host for 320 MiB, and a host that cannot
    // supply that would fail the assertions below for a reason that has
    // nothing to do with `--heapsize`.
    let control = run_shape(&[], &[], ROUNDS_LARGE, WORDS_LARGE);
    assert!(
        String::from_utf8_lossy(&control.stdout).contains("completed"),
        "{}",
        report("large objects, unbounded control did not finish", &control)
    );

    let limit = HEAP_LIMIT.to_string();
    for env in [&[][..], &[("PYRE_JIT", "0")][..]] {
        let out = run_shape(env, &["--heapsize", &limit], ROUNDS_LARGE, WORDS_LARGE);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let what = if env.is_empty() {
            "large objects, jit loop"
        } else {
            "large objects, plain loop"
        };
        assert!(
            !String::from_utf8_lossy(&out.stdout).contains("completed"),
            "{}",
            report(
                &format!("{what}: the limited run was not bounded at all"),
                &out
            )
        );
        assert!(
            stderr.contains("MemoryError"),
            "{}",
            report(
                &format!("{what}: the limit never became a Python exception"),
                &out
            )
        );
    }
}

/// The same, on the plain evaluator.
///
/// The two dispatch loops carry the delivery separately, so a fix landing in
/// one and not the other reads as green here unless both are run. `PYRE_JIT=0`
/// selects the plain one.
#[test]
fn the_plain_evaluator_reports_it_too() {
    let limit = HEAP_LIMIT.to_string();
    let out = run(&[("PYRE_JIT", "0")], &["--heapsize", &limit]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("completed"),
        "{}",
        report("the limited run was not bounded at all", &out)
    );
    assert!(
        stderr.contains("MemoryError"),
        "{}",
        report("the limit never became a Python exception", &out)
    );
}
