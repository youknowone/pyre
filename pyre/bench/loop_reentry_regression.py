# Regression guard: a compiled hot loop must be re-entered on every call of its
# enclosing function, not only on the call that compiled it.
#
# KNOWN-FAILING on the single-pass walker JIT as of this commit. Not wired into
# check.py's run list precisely because it currently fails; run it directly:
#
#     python pyre/bench/loop_reentry_regression.py                 # PASS (CPython)
#     target/release/pyre-cranelift pyre/bench/loop_reentry_regression.py   # FAIL
#
# Root cause: the only path that enters a compiled loop is
# JitDriver::try_resume_into_compiled_loop (majit/majit-metainterp/src/
# jitdriver.rs), gated on single_pass_compiled_key, which is set once at
# CloseLoop compile and consumed by .take(). The #[jit_interp] merge-point
# expansion (majit/majit-macros/src/jit_interp/mod.rs) calls it only inside
# `if driver.is_tracing()` on the post-compile snapshot, and its own comment
# notes "the native back-edges never key the walk's merge-point header, so the
# compiled inner loop is otherwise unreachable". So the loop is entered exactly
# once (right after it compiles) and a fresh activation of the same function
# runs it fully interpreted. back_edge_internal / can_enter_jit_keyed already
# implement compiled-loop entry but are not wired to the native hot back-edge.
#
# Symptom the check detects: call 1 runs the loop compiled (~sub-ns/iter), calls
# 2+ run it interpreted (~1000 ns/iter). The margin is ~1000x, so the 20x gate
# is not timing-flaky.

import time

N = 3_000_000
SLOWDOWN_LIMIT = 20.0  # calls 2+ may be at most 20x slower than call 1


def loop(n):
    i = 0
    total = 0
    while i < n:
        total = total + i
        i = i + 1
    return total


def per_call_ns():
    t0 = time.perf_counter()
    loop(N)
    return (time.perf_counter() - t0) / N * 1e9


first = per_call_ns()          # compiles mid-call, then runs compiled
rest = [per_call_ns() for _ in range(3)]
worst = max(rest)
ratio = worst / first if first > 0 else float("inf")

print(f"call1={first:.2f}ns rest={[round(x, 2) for x in rest]} worst_ratio={ratio:.1f}x")
if ratio > SLOWDOWN_LIMIT:
    print(f"FAIL re-entry broken: later calls {ratio:.0f}x slower than the compiling call")
    raise SystemExit(1)
print("PASS re-entry preserved")
