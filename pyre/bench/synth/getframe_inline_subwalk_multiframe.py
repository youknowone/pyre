# Frame-identity regression for two `_getframe` operations executed from an
# inlined `leaf` MIFrame.  The depth-zero lookup must return `leaf`'s own red
# frame, and its `f_locals` proxy must retain that frame so reading `"x"`
# observes the live callee local.  The positive-depth lookup must start from the
# same callee frame and reach `main`, where `"base"` is stored.
#
# Collapsing either lookup onto the portal frame changes the namespace: the
# first read loses `x`, while the second lands on the module frame and loses
# `base`.  Residualizing the depth-zero lookup instead forces the published
# callee during tracing and prevents this loop from compiling.  The specialized
# path therefore records the callee frame and creates its `FrameLocalsProxy`
# without forcing the outer standard virtualizable; positive-depth lookup stays
# on the established virtual-reference walk until its per-level lowering is
# available.
#
# A real PyPy run compiles one loop with no bridges, forcings, virtualizable
# forcings, or aborts.  Pyre must print the same value, compile one loop, and
# report no escape abort or frame-blackhole adoption for this fixture.
import sys


def leaf(x):
    own_x = sys._getframe(0).f_locals["x"]
    return sys._getframe(2).f_locals["base"] + own_x


def mid(x):
    return leaf(x) + 1


def main():
    base = 7
    total = 0
    for i in range(20000):
        total += mid(i)
    print(total, base)


main()
