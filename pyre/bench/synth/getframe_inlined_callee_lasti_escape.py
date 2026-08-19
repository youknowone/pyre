# pyre-check: selfcheck
# An inlined callee returns its own escaped frame.  PyPy's dispatch loop writes
# `last_instr` for both the `_getframe` call and the later return opcode.  The
# JitCode walker bypasses that loop, so both transitions must be emitted on the
# callee's own red frame.  Without them compiled iterations expose the frame
# constructor's `last_instr = -1` sentinel (`f_lasti == -2`) alongside the
# interpreted return coordinate.
import sys

N = 20000


at_getframe = set()


def leaf(x):
    frame = sys._getframe(0)
    # The `_getframe` transition, read while the callee still owns the frame.
    # The return coordinate `main` checks cannot stand in for it: a missing
    # publication here is overwritten by the return one before `main` looks.
    at_getframe.add(frame.f_lasti)
    return frame


def main():
    seen = set()
    total = 0
    for i in range(N):
        frame = leaf(i)
        seen.add(frame.f_lasti)
        total += i

    if total != sum(range(N)):
        print(f"FAIL dropped iteration: total={total} expected={sum(range(N))}")
        return 1
    if len(at_getframe) != 1 or next(iter(at_getframe)) < 0:
        print(f"FAIL inlined frame f_lasti at _getframe: {sorted(at_getframe)}")
        return 1
    if len(seen) != 1 or next(iter(seen)) < 0:
        print(f"FAIL inlined frame f_lasti diverged: {sorted(seen)}")
        return 1
    print("PASS inlined callee f_lasti")
    return 0


sys.exit(main())
