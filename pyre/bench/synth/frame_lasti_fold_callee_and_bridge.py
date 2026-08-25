# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=root:caller_view,root:leaf
# pyre-check: spec-folds=frame_lasti
# The `root:` arm is measured, not a relaxation: this fixture's loop aborts
# five times with ABORT_ESCAPE and what reaches the JIT is the root trace
# `finish_and_compile` attaches. The declining arm is the positive-depth-at-
# an-inline-level preflight in `try_walker_specialize_sys_getframe`, whose
# `!next_op_is_f_locals_for_getframe_result(..)` disjunct admits only a result
# consumed immediately by `f_locals`; binding the frame to a name puts a
# STORE_FAST in between, so this fixture misses it twice over. Widening it
# would fall the follow-on getter through to the generic heap getter, measured
# returning three different `f_lasti` values for one program point.
# ⚠ `[getframe-decline] non-vref hop` in the log is NOT this: it fires once
# against five aborts, from a different walk that compiles.
# Self-checking guard for the `f_lasti` coordinate on the two frames a walk
# owns that are NOT the loop's own portal: the frame of an inlined callee, and
# the caller's frame read from inside one.
#
# The virtualizable boxes describe the PORTAL frame only, so a read inside an
# inlined callee cannot answer from them -- `pyjitpl.py` keeps one MIFrame per
# inlined call and each carries its own coordinate.  pyre's walker mirrors that
# with a per-level source: an inline sub-walk resolves the callee's own jitcode
# pc through its own metadata, and the caller keeps the CALL boundary it is
# suspended at.  Two sources behind one witness, so both need a site here.
#
# The cold arm compiles as a guard-failure bridge off the hot loop, which
# records its `f_lasti` read at a pc the loop trace never held.
#
# Every site collects a SET: the loop compiles part-way through, so an
# interpreted answer and a compiled one that disagree appear as a second
# element rather than as one shifted value.  `co_positions()` indexed by
# `f_lasti // 2` is the oracle for the coordinate, the way a `dis` consumer
# reads it.
import sys

N = 20000

FIRST = sys._getframe().f_lineno


def leaf(x):
    f = sys._getframe()
    return (f.f_code, f.f_lasti)                             # +5


def caller_view():
    f = sys._getframe(1)
    return (f.f_code, f.f_lasti)


def main():
    inlined = set()
    caller = set()
    cold = set()
    total = 0
    for i in range(N):
        inlined.add(leaf(i))
        caller.add(caller_view())                            # +20
        if i % 997 == 0:
            f = sys._getframe()
            cold.add((f.f_code, f.f_lasti))                  # +23
        total += i

    for label, seen, want_line in (
        ("inlined-callee", inlined, 5),
        ("caller-from-callee", caller, 20),
        ("bridge", cold, 23),
    ):
        if len(seen) != 1:
            print(f"FAIL {label} diverged: {sorted(v for _, v in seen)}")
            return 1
        code, lasti = next(iter(seen))
        if lasti < 0 or lasti % 2 != 0:
            print(f"FAIL {label} f_lasti not an even byte offset: {lasti}")
            return 1
        row = list(code.co_positions())[lasti // 2][0]
        if row is None or row - FIRST != want_line:
            print(f"FAIL {label} f_lasti={lasti} names line {row} not +{want_line}")
            return 1
    if total != sum(range(N)):
        print(f"FAIL dropped iteration: total={total}")
        return 1
    print("PASS f_lasti callee and bridge coordinates")
    return 0


sys.exit(main())
