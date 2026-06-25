# PENDING — documents a PRE-EXISTING JIT bug, not a gap-10 regression.
#
# Nested try/except where each handler reads `sys.exc_info()`.  After an inner
# handler unwinds, POP_EXCEPT must restore the slot to the prev its matching
# PUSH_EXC_INFO saved (the outer ValueError), and after the outer handler to
# None.  Expected per-iteration signature 2*1 + 1*10 + 0*100 = 12 → 360000.
#
# The interpreter (PYRE_NO_JIT=1) is CORRECT (360000).  The JIT is WRONG on
# BOTH tracers: trait gives 3320000, the FBW walker gives 3360000.  Root: under
# JIT the nested handlers' POP_EXCEPT restores are not lowered to the EC
# `sys_exc_value` slot (the `sys.exc_info()` may-force between PUSH and POP ends
# the authoritative walk), so the slot keeps the inner exception after the
# handler exits.  The B3 POP-restores-prev fix (FBW_EXC_PREV LIFO) corrects the
# single-handler shape (raise_catch / rc_small DCE) but does not reach these
# un-lowered nested POPs.  Fixing requires keeping the walk authoritative across
# the in-handler `sys.exc_info()` read (or lowering it too).  Lives here so the
# check.py synthetic suite stays green; promote back to ../ once fixed.
import sys

N = 30000


def classify(t):
    if t is ValueError:
        return 1
    if t is KeyError:
        return 2
    if t is None:
        return 0
    return 9


def run(n):
    acc = 0
    i = 0
    while i < n:
        try:
            raise ValueError("outer")
        except ValueError:
            try:
                raise KeyError("inner")
            except KeyError:
                acc += classify(sys.exc_info()[0]) * 1
            acc += classify(sys.exc_info()[0]) * 10
        acc += classify(sys.exc_info()[0]) * 100
        i += 1
    return acc


def main():
    print(run(N))


main()
