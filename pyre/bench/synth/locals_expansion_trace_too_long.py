# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# pyre-check: spec-folds=builtin_locals_trace_limit_cut
# The `locals()` expansion's own trace_limit cut, witnessed.
#
# `pyjitpl.py` `MetaInterp._interpret` asks `blackhole_if_trace_too_long()`
# after every jitcode step, so the `@jit.unroll_safe` `fast2locals` it looks
# into is interrupted between its own steps.  Both arms of pyre's expansion
# record the whole unroll inside ONE Python opcode, and `mod.rs` asks only
# once that opcode returns, so without a cut inside the unroll a single
# `locals()` overshoots `trace_limit` by the frame's own `co_nlocals`.
#
# `locals_expansion_cut_if_too_long` runs the same `history.length() >
# trace_limit` question per slot and aborts the trace from there.  The trace is
# discarded whole, so what this fixture guards is that the discard is CLEAN:
# the half-built expansion is never published, and the answer the interpreter
# then produces is the one `locals_in_wide_portal_frame` records for the same
# frame.
#
# The limit is 200 and the frame is 44 slots wide, so one expansion is worth
# roughly 88 ops -- enough to cross it partway through.  Measured on release
# dynasm:
#
#                       builtin_locals   ..._trace_limit_cut   abrt_too_long
#   with the cut        consulted=10     consulted=5 fired=5   5
#                       fired=5
#   cut suppressed      consulted=10     suppressed=80         5
#                       fired=10
#
# Five of the ten expansions now end inside the unroll instead of emitting all
# 44 slots past the limit.  `abrt_too_long` does not move, because the walk was
# going to abort either way -- what moves is how far past the limit it got
# first.  The suppressed row is asked once per remaining slot rather than once
# per expansion, because a suppressed cut returns and the unroll goes on
# reaching it.
#
# The answer is the same on both sides, which is half the point: a trace
# discarded mid-expansion must publish nothing.  The other half is that the
# answer being the same is exactly why the result alone cannot gate the cut --
# so `spec-folds` does.  `builtin_locals_trace_limit_cut` fires only where the
# cut fires, so removing the calls to `locals_expansion_cut_if_too_long` reads
# as `declared fold(s) never fired` on every backend rather than as a pass.
#
# The two rows come from ONE binary: `PYRE_FBW_NO_SPECIALIZE` suppresses the
# row and with it the cut, which is what that lever is for.
#
# Only `trace_limit` is lowered.  `function_threshold=1` was measured to stop
# the fold firing at all (`consulted=5 fired=0` at every limit), which would
# leave the fixture guarding nothing, and a module-level `try:` around the
# `pypyjit` import is avoided for the same class of reason.
#
# `selfcheck-interpreted` is the measured state, not a relaxation: at this
# limit every trace this file starts is aborted, so `loops_compiled` reads 0
# and there is no compiled shape to name.  That IS half the guard -- the answer
# has to survive a trace discarded partway through the expansion.
#
# `builtin_locals` itself is deliberately NOT declared: at this limit whether
# that fold completes is what the cut decides, so pinning it would pin the
# thing under test to one side of it.
import pypyjit

pypyjit.set_param("trace_limit=200")

import sys

N = 2000

EXPECTED = [
    "a00",
    "a01",
    "a02",
    "a03",
    "a04",
    "a05",
    "a06",
    "a07",
    "a08",
    "a09",
    "a10",
    "a11",
    "a12",
    "a13",
    "a14",
    "a15",
    "a16",
    "a17",
    "a18",
    "a19",
    "a20",
    "a21",
    "a22",
    "a23",
    "a24",
    "a25",
    "a26",
    "a27",
    "a28",
    "a29",
    "a30",
    "a31",
    "a32",
    "a33",
    "a34",
    "a35",
    "a36",
    "a37",
    "a38",
    "a39",
    "i",
    "n",
    "total",
    "width",
]

WIDTH = 44


def wide(n):
    a00 = 0
    a01 = 1
    a02 = 2
    a03 = 3
    a04 = 4
    a05 = 5
    a06 = 6
    a07 = 7
    a08 = 8
    a09 = 9
    a10 = 10
    a11 = 11
    a12 = 12
    a13 = 13
    a14 = 14
    a15 = 15
    a16 = 16
    a17 = 17
    a18 = 18
    a19 = 19
    a20 = 20
    a21 = 21
    a22 = 22
    a23 = 23
    a24 = 24
    a25 = 25
    a26 = 26
    a27 = 27
    a28 = 28
    a29 = 29
    a30 = 30
    a31 = 31
    a32 = 32
    a33 = 33
    a34 = 34
    a35 = 35
    a36 = 36
    a37 = 37
    a38 = 38
    a39 = 39
    total = 0
    width = 0
    for i in range(n):
        total += locals()["a39"]
        width = len(locals())
    return total, width, sorted(locals())


def main():
    total, width, names = wide(N)
    if width != WIDTH:
        print(f"FAIL width: {width} (expected {WIDTH})")
        return 1
    if names != EXPECTED:
        print(f"FAIL name set: {names}")
        print(f"  expected: {EXPECTED}")
        return 1
    if total != 39 * N:
        print(f"FAIL total: {total} (expected {39 * N})")
        return 1
    print("PASS locals expansion trace_limit cut")
    return 0


sys.exit(main())
