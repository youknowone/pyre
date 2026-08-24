# pyre-check: selfcheck
# pyre-check: selfcheck-loops=2
# Self-checking guard for local tracing armed on a frame that is ALREADY
# running compiled code.
#
# `dispatch_bytecode` (pyopcode.py) opens its loop with a `jit.we_are_jitted()`
# arm that reads the frame's own `debugdata` and fires `ec.bytecode_only_trace`
# when `_d.w_f_trace is not None`.  `debugdata` is one of the `_virtualizable_`
# fields (`interp_jit.py PyFrame._virtualizable_`), so that read is recorded
# into the trace and every compiled loop carries a guard on it; arming
# `frame.f_trace` mid-run fails that guard and the events keep arriving from
# the interpreter.  Reading a per-ExecutionContext flag instead produces no
# such guard, and the compiled loop then runs to completion with the tracer
# installed and silent.
#
# THE TAIL IS THE TEST.  The loss the guard prevents is total and permanent,
# not partial: `settrace` forces every frame, so the compiled loop deopts once
# at the moment of arming and reports that single iteration -- and then, with
# nothing testing the frame again, runs the whole remaining tail in compiled
# code and reports nothing, at any length.  Measured before the guard: a
# 100 000-iteration tail reported the same handful of events a 9-iteration tail
# did.  So the tail here is long enough that the one-shot deopt cannot come
# close to satisfying it, and a fix that merely moves or repeats that deopt
# fails just as loudly as no fix at all.
#
# The expectation is stated per source line, not as a total, so an
# implementation that keeps firing but at the wrong instruction fails too.  The
# two loop-body lines are exact -- one event per remaining iteration, no more
# and no fewer, which cpython 3.14.6 and pypy3 7.3.22 both hold.  The loop
# HEADER is a lower bound, because the rule `run_trace_func` fires on is
# `frame.last_instr < d.instr_prev_plus_one` and an implementation whose back
# edge lands lower fires the header a second time: at this tail cpython reads
# exactly TAIL and pypy3 reads 3599, and neither is wrong.
#
# Both `sys` entry points that install a trace function are exercised, because
# the defect is in the compiled loop rather than in either setter and a fix
# pinned on one spelling is pinned on nothing.  `threading.settrace_all_threads`
# is the same path -- it stores the hook and delegates to
# `sys._settraceallthreads` -- so it is left out rather than paying the
# `threading` import in a synthetic fixture.
#
# `f_trace` is armed from a callee via `sys._getframe(1)`, so the traced frame
# is mid-execution and no `call` event ever fires for it.
import sys

N = 20000  # past `threshold` (1039) many times over before ARM
TAIL = 2000  # iterations left after arming

events = []


def tracer(frame, event, arg):
    events.append((event, frame.f_lineno - frame.f_code.co_firstlineno))
    return tracer


def arm(setter):
    frame = sys._getframe(1)
    frame.f_trace = tracer
    frame.f_trace_lines = True
    setter(tracer)


def hot(n, arm_at, setter, unsetter):   # +0
    acc = 0                             # +1
    for i in range(n):                  # +2
        acc += i                        # +3
        if i == arm_at:                 # +4
            arm(setter)                 # +5
    unsetter(None)                      # +6
    return acc                          # +7


def run(setter, unsetter):
    del events[:]
    hot(N, N - TAIL, setter, unsetter)
    return list(events)


# One `line` event per body line for each of the `TAIL - 1` iterations that
# follow the arming one, then the line after the loop.  The `+2` header entry
# is checked separately as a lower bound.
EXACT = {("line", 3): TAIL - 1, ("line", 4): TAIL - 1, ("line", 6): 1}
HEADER = ("line", 2)

SETTERS = [("sys.settrace", sys.settrace, sys.settrace)]
_all_threads = getattr(sys, "_settraceallthreads", None)
if _all_threads is not None:
    SETTERS.append(("sys._settraceallthreads", _all_threads, _all_threads))


def main():
    failures = []
    for name, setter, unsetter in SETTERS:
        counts = {}
        for event in run(setter, unsetter):
            counts[event] = counts.get(event, 0) + 1
        header = counts.pop(HEADER, 0)
        if counts != EXACT:
            failures.append(
                f"{name}: got {sorted(counts.items())!r}, want {sorted(EXACT.items())!r}"
            )
        if header < TAIL:
            failures.append(
                f"{name}: loop header fired {header!r} times, want at least {TAIL!r}"
            )
    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS f_trace armed mid-loop")
    return 0


sys.exit(main())
