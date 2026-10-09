# pyre-check: pypy-diverges: pins the PEP 667 write-through f_locals proxy, which pypy3 does not have -- its f_locals is a snapshot dict, so the callee's write never reaches the caller's fast local
# CPython-suite gap: the suite exercises FrameLocalsProxy writes only against
# cold frames, never against a frame whose surrounding loop is being traced at
# the moment the write happens.
# parity-tests reason: the write is made by a call the tracer runs for real,
# against the frame the tracer is stepping a copy of, so it is the tracer's
# handling of that call -- not the proxy -- that decides whether the write
# survives.
# pyre-check: regresses-under: PYRE_FBW_NO_ADOPT_RESIDUAL_LOCALS=1 -- without the adopt the walk keeps the box the local held before the call, and both traced iterations lose the write

"""A callee's write through the caller's ``f_locals`` survives tracing.

``setter`` forces the caller and writes its fast-local array while the tracer
holds a boxed copy. ``adopt_residual_locals_writes`` must read the slot back for
both the concretely called and later inlined recordings. The bound reaches both;
the ``regresses-under`` arm proves the shape remains live.
"""

import sys

ROUNDS = 4000


def run(rounds):
    def setter(value):
        sys._getframe(1).f_locals["x"] = value

    x = -1
    lost = []
    for i in range(rounds):
        setter(i)
        if x != i:
            lost.append((i, x))
    return lost


lost = run(ROUNDS)
# Report a few, not all of them: a runtime without the write-through loses
# every iteration, and the whole list buries the runner's output.
assert not lost, f"{len(lost)} of {ROUNDS}, first: {lost[:4]}"
print("OK")
