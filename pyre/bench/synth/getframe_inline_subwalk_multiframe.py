# Coverage for the multi-frame blackhole build path: an INLINED callee that
# forces an outer frame while the walk is already inside a residual call.
#
# The walker executes a residual call concretely, so that level gets a real
# frame from the interpreter's own call sequence; an inline push did not run
# that sequence, so its level had none. A force fired from the inlined body
# therefore built a frame chain that mixed the two, rooted at the intermediate
# residual frame rather than the walked frame, and
# `try_adopt_multi_frame_blackhole`'s chain-root identity gate declined it. What
# that decline wanted was the `jit.virtual_ref` emit at the inline push
# (`executioncontext.py:89`); `walker_ec_enter` / `walker_ec_leave` landed it.
# The gate no longer fires here: the shape now adopts once per build and returns
# the same result as before. This fixture pins that result.
#
# One `sys._getframe(1)` level does NOT reach the build: the chain needs a
# residual level under the walked frame and an inlined level under that, so
# the force has to reach two frames up. No other fixture in the corpus gets
# here -- in the historical gate-enabled sweep with `PYRE_FBW_DEBUG_ABORT=1`,
# 0 of 310 reached `BUILT multi-frame`, so without this one the path has no
# repro at all.
#
# The multi-frame image is built unconditionally when the latch conditions hold,
# so this is both an output guard and build-path coverage: 5 builds, and now 5
# adopts with zero chain-root declines (`PYRE_FBW_DEBUG_ABORT=1` prints both
# tallies; the other 5 escapes in the run have `inline_subwalk=false` and take
# the single-frame arm).
#
# What the decline used to hold back, measured by lifting it before the
# execution-context push landed: the resumed chain shifted every
# `sys._getframe(n)` up exactly one level, so the read below landed on the
# module frame and raised `KeyError: 'base'`. Variants of this shape that cannot
# raise returned a wrong number instead, silently -- `f_locals.get("base", -1)`
# scored -1 for 7, `len(f_locals)` scored the module globals' 12 for this
# frame's 3, `len(f_code.co_name)` scored `<module>`'s 8 for a 9-character
# caller name. That is the failure mode this fixture still guards: every outcome
# arm was wrong, not only the one carrying a resume coordinate.
#
# Deliberately carries no `# pyre-check: max-pypy-ratio=` header: this guards
# an output, and the forcing read makes it a poor perf subject.
import sys


def leaf(x):
    return sys._getframe(2).f_locals["base"] + x


def mid(x):
    return leaf(x) + 1


def main():
    base = 7
    total = 0
    for i in range(20000):
        total += mid(i)
    print(total, base)


main()
