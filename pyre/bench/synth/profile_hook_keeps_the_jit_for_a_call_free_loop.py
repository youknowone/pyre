# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=pure,root:_callee
# `pure` is the call-free loop of the title.  `builtin_only` and `py_callee`
# compile no loop -- the walker declines at the call, which is the rule this
# fixture states -- so only `_callee` reaching the JIT as a root trace is
# declared for them.
# A frame under `sys.setprofile` is no longer handed to the plain evaluator.
#
# `interp_jit.py` carries `is_being_profiled` as a portal green rather than as
# a reason to stop compiling, and `executioncontext.py setprofile` never turns
# the JIT off.  What a profiled frame owes is answered per event: its own
# `call` by the `ec.call_trace` the portal brackets the frame with, its own
# `return` by `leave`'s `_trace('leaveframe')` — `return_trace` carries only
# the `gettrace()` one — and `c_call` / `c_return` by the walker declining a
# trace at the call itself, because it DECIDES calls where upstream's tracer
# traced through the arm that reports them.
#
# The three shapes below are the whole of that rule, and each is checked by
# EQUALITY rather than by a floor.  A narrowed test that merely stopped
# declining reports `call` 200022 against 200001 owed at a 200000 tail, so a
# count that is too HIGH is the failure this fixture exists to catch; one that
# is too low is the other.
#
#   * `pure` calls nothing, so it compiles and still reports its own pair.
#   * `builtin_only` owes one `c_call` / `c_return` per iteration.  The walk
#     aborts at the fold, the frame runs interpreted, and the counts stand.
#   * `py_callee` owes a `call` / `return` for the callee's own frame at every
#     iteration, plus the one for the loop frame.
#
# Measured equal on cpython 3.14.6 and on pypy3 (7.3.22 / 3.11), which agree
# with each other and with pyre on every count below.
import sys

# Past the loop threshold (1039) several times over, so the measured pass is
# the compiled one for the shape that compiles.
N = 3000


def pure(n):
    total = 0
    for i in range(n):
        total += i
    return total


def builtin_only(n):
    total = 0
    text = 'abc'
    for _ in range(n):
        total += len(text)
    return total


def _callee(x):
    return x + 1


def py_callee(n):
    total = 0
    for i in range(n):
        total += _callee(i)
    return total


def measure(body):
    counts = {}

    def hook(frame, event, arg):
        key = event
        if event.startswith('c_'):
            key = '%s:%s' % (event, getattr(arg, '__name__', repr(arg)))
        counts[key] = counts.get(key, 0) + 1

    sys.setprofile(hook)
    try:
        body(N)
    finally:
        sys.setprofile(None)
    # `sys.setprofile` is itself a builtin called from this frame, and the
    # arming call is inside the window it opens.
    counts.pop('c_call:setprofile', None)
    return counts


def check(counts, expected, arm, failures):
    for key, want in sorted(expected.items()):
        got = counts.get(key, 0)
        if got != want:
            failures.append('%s: %s = %d, expected %d' % (arm, key, got, want))
    for key in sorted(counts):
        if key not in expected:
            failures.append(
                '%s: %s = %d, expected no event' % (arm, key, counts[key])
            )


def main():
    failures = []
    cases = [
        ('a loop that calls nothing', pure, {'call': 1, 'return': 1}),
        (
            'a loop calling a builtin',
            builtin_only,
            {
                'call': 1,
                'return': 1,
                'c_call:len': N,
                'c_return:len': N,
            },
        ),
        (
            'a loop calling a Python callee',
            py_callee,
            {'call': N + 1, 'return': N + 1},
        ),
    ]
    for arm, body, expected in cases:
        check(measure(body), expected, arm, failures)
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a profiled frame reports exactly what it owes, compiled or not')
    return 0


sys.exit(main())
