# OPEN DEFECT reproducer, kept out of the runner glob until it passes.
#
# A loop-bearing callee resumed at its own `for` header with the operand-stack
# height left over from the walk instead of the height that header executes
# against.  `rec` has three locals (n, acc, ch) and its loop header is the
# FOR_ITER at py_pc 18, whose static operand-stack depth is 1 (the iterator).
# The frame reaches the header advertising `valuestackdepth == stack_base`, so
# its TOS is the last LOCAL and the resumed FOR_ITER calls `next()` on it:
#
#     TypeError: 'int' object is not an iterator
#
#     python3 pyre/bench/synth/_pending/loop_callee_for_header_resume.py   # PASS
#     target/release/pyre-dynasm pyre/bench/synth/_pending/loop_callee_for_header_resume.py
#
# Deterministic on dynasm; `PYRE_NO_JIT=1` passes, and the failure survives
# `PYRE_TRACE_EAGERNESS=1000000` (bridging off), `PYRE_WALKABORT_OFF=1` and
# `PYRE_FORITER_CALL_BODY=1`.  The two loops in `main` are both needed at these
# trip counts; a single loop needs ~1200 trials to reach it.
#
# Traced writer (`PYRE_FBW_DEBUG_ABORT` census against an instrumented build):
# `maybe_publish_inline_callee_last_instr_concrete`
# (pyre-jit-trace/src/jitcode_dispatch/residual_call.rs) publishes the EXECUTING
# `last_instr = callee_py_pc` onto the inline callee's concrete frame and writes
# no `valuestackdepth` beside it; `LiveLastInstrGuard` then captures that pair
# as its saved state and restores it verbatim, so the pc reads back as a resume
# coordinate with the walk's mid-flight depth.

DATA = tuple(range(40))
STEP = sum(DATA)


def rec(n):
    if n <= 0:
        return 0
    acc = 0
    for ch in DATA:
        acc = acc + ch
    return acc + rec(n - 1)


def expected(n):
    return STEP * n if n > 0 else 0


def main():
    total = 0
    bad = 0
    for trial in range(50):
        n = trial % 50
        for value in [rec(i) for i in range(n)]:
            total += value
    for trial in range(60):
        n = trial % 20
        for i in range(n):
            value = rec(i)
            if value != expected(i):
                bad += 1
                if bad < 4:
                    print("MISMATCH i=%d got=%r want=%r" % (i, value, expected(i)))
            total += value
    if bad:
        print("FAIL %d wrong rec() results" % bad)
        raise SystemExit(1)
    if total != 17955600:
        print("FAIL total=%d want=17955600" % total)
        raise SystemExit(1)
    print("PASS loop-callee for-header resume total=%d" % total)


main()
