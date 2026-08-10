# CPython-suite gap: frame-line tests cannot inspect nested pyre inlined JIT frames.
# parity-tests reason: this guards JIT caller coordinates across residual calls.

# The line number a frame reports while a residual runs inside a call chain
# that was inlined more than one level deep.
#
# A residual executed inside an inlined callee temporarily publishes
# `last_instr` onto the outer traced frame, so that a frame reader running
# during the call (`sys._getframe().f_lineno`, a warning's registry key, a
# traceback) sees the line the call was made from rather than whatever the last
# resume point left behind.  That coordinate is derived from the CALL
# instruction's JitCode offset, which indexes the outer frame's JitCode only
# while the walk is that frame's own.  One level down, the offset belongs to the
# intermediate callee, and mapping it through the outer frame's pc tables names
# whatever line that byte happens to land on — here the `def` line's body start
# instead of the call.
#
# `driver` runs the traced loop, inlines `mid`, which inlines `leaf`; the
# `sys._getframe` residual inside `leaf` is what publishes the coordinate.  The
# loop collects every line `driver` reports across the run, so a single wrong
# iteration is caught: the set must hold exactly the one call line.

import sys

N = 3000


def leaf(k):
    # Frame depths: 0 = leaf, 1 = mid, 2 = driver.
    return sys._getframe(2).f_lineno


def mid(k):
    return leaf(k)


def driver(n):
    seen = set()
    i = 0
    while i < n:
        seen.add(mid(i))  # <-- the only line `driver` may ever report
        i += 1
    return sorted(seen)


CALL_LINE = driver.__code__.co_firstlineno + 4
observed = driver(N)
assert observed == [CALL_LINE], (CALL_LINE, observed)

print("OK")
