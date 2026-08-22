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
# without forcing the outer standard virtualizable.  The positive-depth walk
# must carry each frame's own red box, close the corresponding live
# `virtual_ref` pair around `jit_force_virtual`, and let the optimizer forward
# the force to that pair's virtual frame.
#
# A real PyPy run compiles one loop with no bridges, forcings, virtualizable
# forcings, or aborts.  Pyre must print the same value, compile one loop, and
# report no escape abort or frame-blackhole adoption for this fixture.
#
# The remaining single guard failure is also present in the depth-zero-only
# control; the positive-depth frame walk itself compiles without an abort,
# bridge, forcing, or blackhole adoption.  In the optimized loop, the temporary
# `JIT_FORCE_VIRTUAL`/`GUARD_NOT_FORCED` pair emitted for the orthodox residual
# bracket is gone: `VIRTUAL_REF_FINISH` lets the optimizer forward it to the
# paired virtual frame, just as in PyPy.
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
