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
#
# The positive-depth clause has a measured cost, and the recorded counters
# encode it rather than endorse it.  `leaf`'s `_getframe(2)` reaches `main`'s
# frame, which is the virtualizable of `main`'s compiled loop, and materializes
# it.  `mid(i)` stays a residual `CallMayForce` in that loop, so the
# `GuardNotForced` behind it fails on every machine-code entry: `MAJIT_STATS=1`
# reports `mc_entered` equal to `guard_failures`, with `back_edge_polls=0` and
# `bridges_compiled=0`, so the compiled loop leaves for the blackhole before it
# ever crosses its back edge.  The attribution is a one-run check -- a `leaf`
# holding only the depth-zero read records `guard_failures=1`, one holding only
# the `_getframe(2)` read reproduces the full count.  The recorded
# `guard_failures` is therefore a known shortfall of the positive-depth walk,
# not a number to preserve; it is expected to fall once that walk gets its
# per-level lowering.  `loops_compiled` counts two traces here: `main`'s loop,
# and a linear `Finish` trace for the inlined `mid` -> `leaf` chain, whose own
# four `GuardNotForced`s never fail.  The wasm baseline records
# `guard_failures=1` instead because that backend always materializes the
# virtualizable, which makes `GuardNotForced` a no-op there.
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
