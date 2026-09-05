# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,rec
# A recursion that stays inside the portal must still answer
# `sys.getrecursionlimit()`.
#
# This is an intentional 3.14-spec difference: measured CPython 3.14.6 raises
# RecursionError for the deep call below, while pypy3 7.3.22 returns normally.
# The implementation stays at PyPy's own activation seam: `PyFrame.execute_frame`
# carries `insert_stack_check_here`, and this fixture covers a compiled door
# that synthesizes that entry without executing the source graph itself.
#
# `pyframe.py` carries `execute_frame.insert_stack_check_here = True`, and
# `insert_ll_stackcheck` (`rpython/translator/transform.py`) puts the check at
# operation 0 of that graph — ahead of the `ec.enter(self)` the activation is
# otherwise made of.  The ordinary recursive CALL_ASSEMBLER arm already
# reproduces that entry through `record_activation_charge`; the loop-callee
# CALL_ASSEMBLER arm was a second door into the same portal, but skipped the
# charge and therefore skipped the limit as long as recursion stayed compiled.
#
# THE CALLEE'S OWN LOOP IS THE TEST, not decoration: it is what makes the fold
# take `emit_walker_loop_callee_call_assembler` and reach the callee through
# the portal shim.  A loopless recursion folds through the self-recursive arm
# and executes the ordinary `call_fn` residual, which lands in the
# interpreter's own `execute_frame` — checked either way, so a loopless shape
# passes whether or not the portal doors carry the check.
#
# THE WARMUP DEPTH IS SHALLOW ON PURPOSE: `rec` has to be compiled BEFORE the
# deep call, or the deep call is interpreted and every level reaches
# `execute_frame`.
#
# THE LIMIT IS THE ONE UNDER TEST, not the native stack: `LIMIT` is far below
# `DEPTH` but `DEPTH` is small enough that the C stack survives it, so a tree
# without the check answers with a VALUE rather than by crashing — which is the
# reading this fixture turns into a failure.
import sys

WARM = 3000
INNER = 30
WARM_DEPTH = 4
LIMIT = 150
DEPTH = 400


def rec(n, m):
    total = 0
    for i in range(m):
        total += i
    if n <= 0:
        return total
    return rec(n - 1, m) + total


def hot(k, depth):
    caught = 0
    returned = 0
    wrong = []
    for _ in range(k):
        try:
            rec(depth, INNER)
            returned += 1
        except RecursionError:
            caught += 1
        except Exception as exc:
            if len(wrong) < 3:
                wrong.append('%s: %s' % (type(exc).__name__, exc))
    return caught, returned, wrong


def main():
    failures = []
    # Compile `hot` and `rec` at a depth the budget accommodates, so the deep
    # call below runs against compiled code rather than a cold frame.
    caught, returned, wrong = hot(WARM, WARM_DEPTH)
    if caught or wrong or returned != WARM:
        failures.append(
            'warmup at depth %d: %d returned, %d raised RecursionError, %r — '
            'the warmup must not be near the limit or the deep call below '
            'tests nothing' % (WARM_DEPTH, returned, caught, wrong)
        )

    saved = sys.getrecursionlimit()
    sys.setrecursionlimit(LIMIT)
    try:
        caught, returned, wrong = hot(1, DEPTH)
    finally:
        sys.setrecursionlimit(saved)

    if wrong:
        failures.append('deep call raised something else: %r' % (wrong,))
    if returned:
        failures.append(
            'rec(%d) returned a value under a recursion limit of %d — the '
            'loop-callee portal door skipped the activation charge and its '
            'limit check, so the recursion ran %d levels past it'
            % (DEPTH, LIMIT, DEPTH - LIMIT)
        )
    if caught != 1:
        failures.append(
            'the deep call raised RecursionError %d times, owed 1' % caught
        )

    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a recursion limit binds a portal-driven recursion')
    return 0


sys.exit(main())
