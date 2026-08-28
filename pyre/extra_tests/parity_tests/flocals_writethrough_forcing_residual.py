# pyre-check: pypy-diverges: pins the PEP 667 write-through f_locals proxy, which pypy3 does not have -- its f_locals is a snapshot dict, so the callee's write never reaches the caller's fast local
# CPython-suite gap: the suite exercises FrameLocalsProxy writes only against
# cold frames, never against a frame whose surrounding loop is being traced at
# the moment the write happens.
# parity-tests reason: the write is made by a call the tracer runs for real,
# against the frame the tracer is stepping a copy of, so it is the tracer's
# handling of that call -- not the proxy -- that decides whether the write
# survives.
# pyre-check: regresses-under: PYRE_FBW_NO_ADOPT_RESIDUAL_LOCALS=1 -- without the adopt the walk keeps the box the local held before the call, and both traced iterations lose the write

"""A write made through ``f_locals`` by a *called* function must be visible to
the caller's next read, including on the iteration the caller's loop is traced.

``setter`` reaches up with ``sys._getframe(1)`` and stores into the caller's
fast local.  Executing that call forces the caller's virtualizable, and the
store then lands on the frame the force just wrote.  The tracer walks a copy of
that frame and holds each local as a box, so unless it reads the stored slot
back out of the array, the value it carries -- and the value its loop closes on
-- is the one the local held before the call.

The bound matters, and it has to clear TWO recordings.  The write is lost only
on an iteration where the loop is being recorded, so a loop that stops short
witnesses nothing.  The first recording calls ``setter`` for real: that call
forces the caller's virtualizable, and the store lands on the frame the force
just wrote.  The second, reached much later, inlines ``setter`` instead, so the
store itself is the residual and nothing forces at all -- the proxy's own force
is gated on the live frame's ``vable_token``, which is zero while the walk is
tracing.  The two lose the write through different paths and are pinned here at
iterations 1040 and 2105, so the bound sits well past the later of them.

Both are reached only because ``adopt_residual_locals_writes`` reads the slot
back; the passing run says nothing about that on its own, since a script that
stopped reaching either iteration would pass too.  The ``regresses-under``
header is what separates the two: with the adopt switched off this script must
report ``2 of 4000, first: [(1040, 1039), (2105, 2104)]`` and exit non-zero, so
a change that stops reaching the shape is reported here rather than read as a
pass.
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
