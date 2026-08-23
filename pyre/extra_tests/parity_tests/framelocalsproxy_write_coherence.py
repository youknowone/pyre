# CPython-suite gap: the suite exercises FrameLocalsProxy access only against
# cold frames, never against one whose surrounding loop the JIT has compiled.
# parity-tests reason: this guards the coherence of the running frame's
# `f_locals` against the values a compiled loop holds for those locals.

"""A write through the running frame's ``f_locals`` must be visible to a later
read of the same local, including after the loop around the write is compiled.

The proxy is a live view over the frame's fast locals, so ``proxy["x"] = i`` has
to reach the storage a subsequent ``LOAD_FAST x`` reads.  A compiled loop that
keeps ``x`` in its virtualizable shadow answers the value the local held when
the loop was compiled, which makes the result depend on the compilation
threshold rather than on the program.

``write_through_proxy`` puts its write in the loop *body*.  That is what makes
it a witness: a write placed before the loop happens while the frame is still
interpreted, so the two sides are still equal at the moment the loop compiles
and the arm passes whether or not the defect is present.  The two controls keep
shapes that agree either way, each for its own reason.

Only the write direction is covered.  The read direction -- a ``STORE_FAST`` in
the body followed by a proxy read -- diverges the same way and is not fixed
here: forcing on a read would also force the reads a trace has inlined, which
upstream never does, and there is no way to tell those apart at the accessor
yet.
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
    # Control: the store precedes the compile, so this agrees either way.
    proxy = sys._getframe(0).f_locals
    x = -1
    for i in range(rounds):
        x = i
    return proxy["x"]


def write_through_caller_proxy(rounds):
    # Second control: this one agrees either way because `setter` builds a
    # fresh proxy per iteration, so the getter's own force fires every time.
    # What the arms above need is a proxy that OUTLIVES that force.
    def setter(value):
        sys._getframe(1).f_locals["x"] = value

    x = -1
    for i in range(rounds):
        setter(i)
    return x


def check(name, got):
    # Every arm drives the same local to ``ROUNDS - 1``; a stale answer reports
    # the value it held when the loop compiled, which is well short of that.
    if got != ROUNDS - 1:
        print(f"{name}: got {got!r}, want {ROUNDS - 1!r}")
        return False
    return True


results = [
    check("write_through_proxy", write_through_proxy(ROUNDS)),
    check("store_then_read_proxy", store_then_read_proxy(ROUNDS)),
    check("write_through_caller_proxy", write_through_caller_proxy(ROUNDS)),
]
if all(results):
    print("OK")
