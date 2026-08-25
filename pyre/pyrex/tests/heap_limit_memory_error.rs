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
//! that the limit *becomes a Python exception the program can act on*; it is
//! not a promise that the program survives one. After the first breach the live
//! set is by definition at the limit, so a handler that allocates before it
//! frees can reach the rung below — upstream says as much where it raises, and
//! aborts on the second arrival for exactly that reason. Whether any particular
//! handler fits in the remaining headroom is a question about collection
//! scheduling, not about the raise, and it is not stable enough to gate.
//!
//! The interpreter's own traceback printer is such a handler, which is why the
//! scripts below carry an `except MemoryError` and report through it rather
//! than leaving the exception to unwind and be printed. `eprint_exception`
//! renders the whole report into a buffer and writes nothing until it is
//! complete, and for a `-c` program the filename is `<string>`, so
//! `read_registered_source_line` fetches each frame's line by calling
//! app-level `linecache` — a nested dispatch loop whose first safepoint drives
//! `do_collect_oldgen_nonmoving`, a whole major cycle, over a heap the breach
//! left at its limit. The abort unwinds out of the half-built buffer and stderr
//! receives nothing at all. Measured: the same program read from a file prints
//! its traceback (the filesystem branch runs no Python), and a `-c` run of it
//! prints nothing — while both raise, deliver and reach the top-level handler
//! identically.
//!
//! Freeing the live set before reporting is not enough either, and that cost a
//! CI round to learn: dropping it only makes it collectable, and an in-flight
//! cycle that already marked it sweeps nothing, so the completion still breaches
//! and the abort still discards a `print` that a pipe had block buffered. The
//! only report that cannot be lost is one that writes nothing and allocates
//! nothing — the exit status, via `os._exit`.
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
///
/// The handler reports by exiting, and `os._exit` rather than a `print` or a
/// `raise`, because every other way of reporting is itself an allocation on a
/// heap that is still at its limit. A `print` reaches a pipe, so it is block
/// buffered and the buffer is discarded if anything later aborts; dropping the
/// live set first does not help, because the in-flight cycle may already have
/// marked it and so sweeps nothing. `os._exit` writes nothing, allocates
/// nothing beyond the call, and skips finalization, which is precisely the
/// "quit cleanly" the rung exists to offer. The status is the evidence.
const GROW: &str = "\
import os
data = []
try:
    for _ in range(ROUNDS):
        data.append([0] * WORDS)
except MemoryError:
    os._exit(CAUGHT)
print('completed')
";

/// The exit status the handler reports with. Distinct from anything the
/// runtime itself exits with, so a bounded arm cannot pass by dying.
const CAUGHT: i32 = 17;

/// [`run_shape`] at the small shape, which is what every arm but the
/// large-object one wants.
fn run(env: &[(&str, &str)], args: &[&str]) -> Output {
    run_shape(env, args, ROUNDS, WORDS)
}

/// Run [`GROW`] at the given shape, as `-c` source, and collect the child's
/// output. `env` and `args` select the dispatch loop and the heap limit.
fn run_shape(env: &[(&str, &str)], args: &[&str], rounds: usize, words: usize) -> Output {
    let src = GROW
        .replace("ROUNDS", &rounds.to_string())
        .replace("WORDS", &words.to_string())
        .replace("CAUGHT", &CAUGHT.to_string());
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

/// A failure message carrying the child's status, its stdout, and the tail of
/// its stderr — an assertion here fails in a subprocess, so the parent has to
/// quote it or the reason is lost.
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

/// Assert a `--heapsize` arm reached the middle rung: the loop did not finish,
/// and the handler ran.
fn assert_caught(what: &str, out: &Output) {
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("completed"),
        "{}",
        report(
            &format!("{what}: the limited run was not bounded at all"),
            out
        )
    );
    assert_eq!(
        out.status.code(),
        Some(CAUGHT),
        "{}",
        report(
            &format!("{what}: the limit never became a Python exception"),
            out
        )
    );
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
/// Reaching the handler is the whole point: before the breach was routed to a
/// dispatch loop, this run's only sign of the limit was
/// `out_of_memory("using too much memory, aborting")` on the *second* breach,
/// with the rung that names a Python exception skipped entirely.
#[test]
fn a_bounded_heap_reports_a_memory_error_rather_than_only_aborting() {
    let limit = HEAP_LIMIT.to_string();
    let out = run(&[], &["--heapsize", &limit]);
    assert_caught("jit loop", &out);
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
        let what = if env.is_empty() {
            "large objects, jit loop"
        } else {
            "large objects, plain loop"
        };
        assert_caught(what, &out);
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
    assert_caught("plain loop", &out);
}
