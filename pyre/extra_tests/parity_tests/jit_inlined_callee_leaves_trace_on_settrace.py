# CPython-suite gap: test_sys_settrace installs its tracer before the traced
# function has run enough to be compiled, so every case it covers is the
# already-hooked order; nothing in the suite warms a loop first and installs a
# tracer afterwards, which is the only order in which a compiled trace can
# outlive the hook's installation.
# parity-tests reason: `sys.settrace` does not change the portal green — `call_trace`
# sets `is_being_profiled` only for a *profile* function — so an already
# compiled trace keeps running, and the callee it inlined reports no `call`
# event at all until a guard on `ExecutionContext.w_tracefunc` leaves it.

"""A compiled trace has to leave when `sys.settrace` is installed.

The tracer returns `None` from every `call` event, so it claims no frame and
`frame_tracing_active` lets the loop keep compiling — what this measures is the
compiled path's own frame events rather than the interpreter's.

The loop is warmed with no hook installed, so its callee is inlined into a
trace recorded against a null `w_tracefunc`.  Installing a tracer afterwards
does not change the portal green — `call_trace` sets `is_being_profiled` only
when a *profile* function is installed — so nothing but a guard on that field
can stop the recorded trace from running on, and an inlined callee that runs
inside compiled code reports no `call` event at all.
"""
import sys

WARM = 4000
N = 2000

seen = []


def tracer(frame, event, arg):
    if frame.f_code.co_name == "callee":
        seen.append(event)
    return None


def callee(x):
    return x + 1


def hot(n):
    total = 0
    for i in range(n):
        total += callee(i)
    return total


hot(WARM)
sys.settrace(tracer)
try:
    hot(N)
finally:
    sys.settrace(None)

calls = seen.count("call")
print("callee call events:", calls)
assert calls == N, "callee reported %d call events, expected %d" % (calls, N)
print("OK")
