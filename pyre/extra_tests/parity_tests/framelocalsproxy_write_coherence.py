# CPython-suite gap: the suite exercises FrameLocalsProxy access only against
# cold frames, never against one whose surrounding loop the JIT has compiled.
# parity-tests reason: this guards the coherence of the running frame's
# `f_locals` against the values a compiled loop holds for those locals.

"""An access through the running frame's ``f_locals`` must agree with the fast
local it views, including after the loop around the access is compiled.

The proxy is a live view over the frame's fast locals, so ``proxy["x"] = i`` has
to reach the storage a subsequent ``LOAD_FAST x`` reads, and ``proxy["x"]`` has
to answer what the most recent ``STORE_FAST x`` wrote.  A compiled loop that
keeps ``x`` in its virtualizable shadow diverges from the frame array in both
directions, which makes the result depend on the compilation threshold rather
than on the program.

Every arm puts its store in the loop *body*.  That is what makes them witnesses:
a store placed before the loop happens while the frame is still interpreted, so
the two sides are still equal at the moment the loop compiles and the arm passes
whether or not the defect is present.  ``store_then_read_proxy`` keeps that
weaker shape deliberately, as a control.
"""

import sys

ROUNDS = 4000


def write_through_proxy(rounds):
    x = -1
    proxy = sys._getframe(0).f_locals
    for i in range(rounds):
        proxy["x"] = i
    return x


def read_after_store_proxy(rounds):
    # The read direction, through the getitem scan: the body's `STORE_FAST x`
    # goes to the shadow once compiled while `proxy["x"]` reads the array.
    #
    # This reports the LAST value rather than a running total, which scopes it
    # to the standing divergence.  A total also counts the single update lost at
    # the one iteration where the loop compiles -- a separate defect, present on
    # the write side too and visible with this one removed, so it does not
    # belong to the arm that guards this one.
    x = -1
    proxy = sys._getframe(0).f_locals
    last = None
    for i in range(rounds):
        x = i
        last = proxy["x"]
    return last


def snapshot_after_store_proxy(rounds):
    # The same read direction through the snapshot rather than the scan.
    x = -1
    proxy = sys._getframe(0).f_locals
    last = None
    for i in range(rounds):
        x = i
        last = dict(proxy)["x"]
    return last


def store_then_read_proxy(rounds):
    # Control: the store precedes the compile, so this agrees either way.
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
print("read_after_store_proxy", read_after_store_proxy(ROUNDS))
print("snapshot_after_store_proxy", snapshot_after_store_proxy(ROUNDS))
print("store_then_read_proxy", store_then_read_proxy(ROUNDS))
print("write_through_caller_proxy", write_through_caller_proxy(ROUNDS))
