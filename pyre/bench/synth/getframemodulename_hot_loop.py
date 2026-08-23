# pyre-check: selfcheck
# Self-checking regression guard for `sys._getframemodulename` read from a hot
# loop, at depth 0 and from one frame further in.
#
# It is the third forcing reader of a calling frame in the `sys` module, beside
# `getframe` and `_current_frames`, and it had no coverage anywhere -- neither
# a synthetic fixture nor a snippet.  It cannot join `current_frames_hot_loop`,
# which is an ordinary synthetic fixture and therefore also run under PyPy for
# the ratio: PyPy does not implement `_getframemodulename`, and the run dies
# with AttributeError before it reaches any assertion.  A self-checking fixture
# runs on pyre alone, which is what makes this surface reachable at all.
#
# The answer is invariant across the loop, so it is collected as a SET rather
# than compared per iteration: a route that went stale, or that started
# answering per-iteration once the loop compiled, shows up as a second element.
# A scalar comparison against the expected value would accept a value that
# oscillated and happened to be right on the last iteration.
#
# Depth 1 is read from a callee the tracer can inline, so the answer has to
# name the CALLER's module rather than the frame the walk is standing in.
import sys

N = 200000
EXPECTED = "__main__"


def inner_depth1(seen):
    seen.add(sys._getframemodulename(1))


def hot(n):
    at0 = set()
    at1 = set()
    for _ in range(n):
        at0.add(sys._getframemodulename(0))
        inner_depth1(at1)
    return at0, at1


def main():
    at0, at1 = hot(N)
    if at0 != {EXPECTED}:
        print("FAIL _getframemodulename(0) not invariant:", sorted(at0, key=str))
        return 1
    if at1 != {EXPECTED}:
        print("FAIL _getframemodulename(1) from an inlinable callee:",
              sorted(at1, key=str))
        return 1
    print("PASS _getframemodulename hot loop")
    return 0


sys.exit(main())
