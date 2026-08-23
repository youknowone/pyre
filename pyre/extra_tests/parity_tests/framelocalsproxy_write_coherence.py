# CPython-suite gap: the suite exercises FrameLocalsProxy writes only against
# cold frames, never against one whose surrounding loop the JIT has compiled.
# parity-tests reason: this guards the coherence of a write through the running
# frame's `f_locals` against the value a compiled loop holds for that local.

"""A write through the running frame's ``f_locals`` must be visible to a later
read of the same local, including after the loop around the write is compiled.

The proxy is a live view over the frame's fast locals, so ``proxy["x"] = i`` has
to reach the storage a subsequent ``LOAD_FAST x`` reads.  A compiled loop that
keeps ``x`` in its virtualizable shadow answers the value the local held when
the loop was compiled, which makes the result depend on the compilation
threshold rather than on the program.

Both directions are covered.  The reverse one -- a ``STORE_FAST`` inside the
compiled loop followed by a proxy read -- already agrees, and is kept here so a
fix to the write direction cannot silently break it.
"""

import sys

ROUNDS = 4000


def write_through_proxy(rounds):
    x = -1
    proxy = sys._getframe(0).f_locals
    for i in range(rounds):
        proxy["x"] = i
    return x


def store_then_read_proxy(rounds):
    proxy = sys._getframe(0).f_locals
    x = -1
    for i in range(rounds):
        x = i
    return proxy["x"]


def write_through_caller_proxy(rounds):
    def setter(value):
        sys._getframe(1).f_locals["x"] = value

    x = -1
    for i in range(rounds):
        setter(i)
    return x


print("write_through_proxy", write_through_proxy(ROUNDS))
print("store_then_read_proxy", store_then_read_proxy(ROUNDS))
print("write_through_caller_proxy", write_through_caller_proxy(ROUNDS))
