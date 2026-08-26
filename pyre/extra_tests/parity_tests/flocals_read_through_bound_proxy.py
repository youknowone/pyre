# pyre-check: pypy-diverges: pins the PEP 667 f_locals proxy as a LIVE view, which pypy3 does not have -- its f_locals is a snapshot dict, so a proxy bound before the loop keeps answering with the value the local held at the bind
# CPython-suite gap: the suite reads FrameLocalsProxy only against cold frames,
# never against a frame whose surrounding loop is being traced at the moment of
# the read.
# parity-tests reason: the read is served by a call the tracer runs for real,
# against the frame the tracer is stepping a copy of, so it is the tracer's
# handling of that call -- not the proxy -- that decides which value comes back.

"""A read through an ``f_locals`` proxy bound BEFORE a loop must see the stores
the loop body makes, including once the loop is compiled.

The proxy is a live view: it reads ``locals_cells_stack_w`` at each access
rather than copying out of it at the bind.  A traced body holds each local as a
box instead, and a ``STORE_FAST`` in the body updates only that box -- nothing
puts the value into the array the proxy reads.  So the compiled loop answers
every ``p["x"]`` with whatever the array happened to hold when the trace was
recorded.

Binding the proxy OUTSIDE the loop is what makes it show.  Spelled
``sys._getframe(0).f_locals["x"]`` the getter is re-entered every iteration and
each entry materializes the frame, so the array is current by construction;
bound once, nothing materializes it again for the rest of the loop.

Both read spellings are here because they reach the array through different
residual calls -- ``__getitem__`` on the proxy, and the bound ``get`` method --
and a write-back attached to only one of them leaves the other reading the
array as the trace left it.

The bodies carry no call and no branch besides the read under test, so every
iteration records the same trace and the total below counts iterations, not
paths.  Without the write-back the compiled loop freezes at the value the local
held when the trace closed, so the total grows quadratically past that point
rather than by one per iteration.
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
