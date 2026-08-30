# pyre-check: pypy-diverges: pins the PEP 667 f_locals proxy as a LIVE view, which pypy3 does not have -- its f_locals is a snapshot dict, so a proxy bound before the loop keeps answering with the value the local held at the bind
# CPython-suite gap: the suite reads FrameLocalsProxy only against cold frames,
# never against a frame whose surrounding loop is being traced at the moment of
# the read.
# parity-tests reason: the read is served by a call the tracer runs for real,
# against the frame the tracer is stepping a copy of, so it is the tracer's
# handling of that call -- not the proxy -- that decides which value comes back.

"""A proxy bound before a loop reads every subsequent ``STORE_FAST``.

The trace keeps the local in a box while the proxy reads
``locals_cells_stack_w``. The residual subscript and bound-``get`` paths must
write the live box back first; otherwise both freeze at the recording value and
the accumulated lag becomes nonzero.
"""

import sys

ROUNDS = 4000


def through_subscript(rounds):
    x = -1
    p = sys._getframe(0).f_locals
    lag = 0
    for i in range(rounds):
        x = i
        lag += i - p["x"]
    return lag


def through_get(rounds):
    x = -1
    p = sys._getframe(0).f_locals
    lag = 0
    for i in range(rounds):
        x = i
        lag += i - p.get("x")
    return lag


for read in (through_subscript, through_get):
    lag = read(ROUNDS)
    assert lag == 0, f"{read.__name__}: {lag} behind over {ROUNDS} iterations"
print("OK")
