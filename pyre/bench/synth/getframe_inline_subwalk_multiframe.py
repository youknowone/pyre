# Coverage for the multi-frame blackhole build path: an INLINED callee that
# forces an outer frame while the walk is already inside a residual call.
#
# The walker executes a residual call concretely, so that level gets a real
# frame from the interpreter's own call sequence; an inline push never runs
# that sequence, so its level has none. A force fired from the inlined body
# therefore builds a frame chain that mixes the two, and the chain's root is
# the intermediate residual frame rather than the walked frame -- which is
# what `try_adopt_multi_frame_blackhole` declines on today (it wants the
# `jit.virtual_ref` emit at the inline push, `executioncontext.py:89`). This
# fixture pins that the declined path still returns byte-identical results.
#
# One `sys._getframe(1)` level does NOT reach the build: the chain needs a
# residual level under the walked frame and an inlined level under that, so
# the force has to reach two frames up. No other fixture in the corpus gets
# here -- swept with `PYRE_FBW_DEBUG_ABORT=1 PYRE_FBW_MULTIFRAME=1`, 0 of 310
# reach `BUILT multi-frame`, so without this one the path has no repro at all.
#
# With the gate at its default the image is not built either, so under a plain
# run this is an output guard; the coverage it adds is for `PYRE_FBW_MULTIFRAME=1`
# (5 builds), which is how the gate's owner exercises the path.
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
