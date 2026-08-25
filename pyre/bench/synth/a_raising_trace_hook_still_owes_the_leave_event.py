# pyre-check: selfcheck
# A profiler and a trace function are both installed on a loop the JIT already
# compiled, and the trace callback raises.
#
# `pyframe.py execute_frame` is two nested `try`s under one `finally`:
# `call_trace` sits in the outer `try`, `return_trace` in the inner `finally`,
# and `ec.leave` -- whose profile arm is the `_trace(frame, 'leaveframe', ...)`
# that becomes `setprofile`'s `return` event -- in the outer `finally`.  So a
# `call_trace` that raises skips both the eval body and `return_trace` and STILL
# owes the leave event, and a `return_trace` that raises owes it too.
#
# pyre's portal carries that bracket itself rather than reaching `execute_frame`
# (`eval::eval_with_jit_inner`), so the nesting has to be carried with it: an
# early return on either of the first two hooks drops the third.  Measured
# before the fix, with the same two arms below: `call` reported no profile event
# at all and `return` reported only `call`.
#
# Both arms are exact and both are checked, because they fail separately -- the
# first drops the leave event from a path that never ran the body, the second
# from a path that did.  cpython 3.14.6 and pypy3 agree on both.
import sys

WARM = 20000  # past the loop threshold (1039) many times over
EXPECTED = {'call': ['return'], 'return': ['call', 'return']}


def hot(n):
    total = 0
    for _ in range(n):
        total = (total + 1) % 1000003
    return total


def run(raise_at):
    seen = []

    def profiler(frame, event, arg):
        if frame.f_code.co_name == 'hot':
            seen.append(event)

    def tracer(frame, event, arg):
        if event == raise_at and frame.f_code.co_name == 'hot':
            raise RuntimeError('boom at ' + raise_at)
        # Returning the hook arms `f_trace`, which is what makes the `return`
        # arm reachable at all -- `return_trace` fires on `gettrace()`.
        return tracer

    # Compile the loop first, with nothing installed, so the arms below
    # interrupt compiled code rather than a cold frame.
    hot(WARM)
    sys.setprofile(profiler)
    sys.settrace(tracer)
    raised = None
    try:
        hot(100)
    except RuntimeError as exc:
        raised = str(exc)
    finally:
        sys.settrace(None)
        sys.setprofile(None)
    return raised, seen


def main():
    failures = []
    for where, expected in sorted(EXPECTED.items()):
        raised, seen = run(where)
        if raised != 'boom at ' + where:
            failures.append('%s: the callback did not escape, got %r' % (where, raised))
        if seen != expected:
            failures.append(
                '%s: profile events for the loop frame were %r, expected %r — a '
                'hook that raised took the leave event down with it'
                % (where, seen, expected)
            )
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a raising trace hook still owes the profiler its leave event')
    return 0


sys.exit(main())
