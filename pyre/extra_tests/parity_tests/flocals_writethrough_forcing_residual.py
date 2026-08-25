# pyre-check: pypy-diverges: pins the PEP 667 write-through f_locals proxy, which pypy3 does not have -- its f_locals is a snapshot dict, so the callee's write never reaches the caller's fast local
# CPython-suite gap: the suite exercises FrameLocalsProxy writes only against
# cold frames, never against a frame whose surrounding loop is being traced at
# the moment the write happens.
# parity-tests reason: the write is made by a call the tracer runs for real,
# against the frame the tracer is stepping a copy of, so it is the tracer's
# handling of that call -- not the proxy -- that decides whether the write
# survives.

"""A write made through ``f_locals`` by a *called* function must be visible to
the caller's next read, including on the iteration the caller's loop is traced.

``setter`` reaches up with ``sys._getframe(1)`` and stores into the caller's
fast local.  Executing that call forces the caller's virtualizable, and the
store then lands on the frame the force just wrote.  The tracer walks a copy of
that frame and holds each local as a box, so unless it reads the stored slot
back out of the array, the value it carries -- and the value its loop closes on
-- is the one the local held before the call.

The bound matters.  The write is lost only on an iteration where the loop is
being recorded, so a loop that never gets that far cannot witness it, and the
first recording is what this pins.  A second recording, reached much later,
inlines ``setter`` instead of calling it; the store is then a residual that
forces nothing at all, and that arm is a separate defect this does not cover.
"""

import sys

ROUNDS = 1500


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
