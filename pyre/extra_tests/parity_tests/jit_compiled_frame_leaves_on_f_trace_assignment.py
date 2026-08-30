# CPython-suite gap: test_sys_settrace assigns `f_trace` only from a tracer the
# `call` event already installed, on a frame that has run a handful of
# bytecodes; nothing in the suite assigns it to a frame that has been looping
# long enough to be compiled, which is the only order in which a compiled trace
# can outlive the assignment.
# parity-tests reason: `bytecode_only_trace` opens on
# `frame.get_w_f_trace() is None`, and compiled code makes that call rather than
# running it, so a frame handed its own trace function mid-loop keeps running
# compiled and reports no `line` event until the walker's `w_f_trace` guard
# leaves the trace.

"""Assigning `f_trace` to a running compiled frame has to leave the trace.

The loop is warmed with a global tracer already installed that returns `None`
from every `call` event, so it claims no frame: every `w_f_trace` stays null
and the loop compiles, with the callee emitted as a residual rather than
inlined.  The callee then reaches back with `sys._getframe(1)` and hands the
looping frame its own trace function, which is what `_trace` reads for every
event but `'call'` — from that iteration on, `line` events are due.
"""
import sys

WARM = 4000
N = 2000

seen = []


def watcher(frame, event, arg):
    # Claims nothing: a `None` return leaves every frame's `f_trace` null, so
    # the loop below keeps compiling while this is installed.
    return None


def recorder(frame, event, arg):
    seen.append(event)
    return recorder


def claim(i, at):
    if i == at:
        sys._getframe(1).f_trace = recorder
    return i


def hot(n, at):
    total = 0
    for i in range(n):
        total += claim(i, at)
    return total


sys.settrace(watcher)
try:
    hot(WARM, -1)
    hot(N, N // 2)
finally:
    sys.settrace(None)

lines = seen.count("line")
print("line events after the assignment:", lines >= N // 4)
assert lines >= N // 4, "hot reported %d line events, expected at least %d" % (
    lines,
    N // 4,
)
print("OK")
