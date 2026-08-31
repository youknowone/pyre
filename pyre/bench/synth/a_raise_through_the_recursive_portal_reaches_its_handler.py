# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,rec
# A recursive callee that raises past its own loop header owes the caller an
# exception, not a value.
#
# `warmspot.py ll_portal_runner` raises into RPython's single ll exception
# state, and every executor of the portal reads that one state: compiled
# `CALL_ASSEMBLER` through `pos_exception()` / `pos_exc_value()`
# (`llmodel.py`), `bhimpl_recursive_call_*` through the propagated RPython
# exception (`blackhole.py:1110-1116`), and the trace-time execution that
# `do_recursive_call` performs through `metainterp.execute_raised`
# (`executor.py:52-78`).
#
# pyre spells that one state as two carriers — `BH_LAST_EXC_VALUE` and the
# backend `_store_exception` cells — and the portal shim wrote only the
# second.  The walker's residual executor reads only the first, so a raise out
# of the portal was invisible to it: it recorded `GUARD_NO_EXCEPTION` over the
# raise, wrote the shim's NULL result into the destination slot, and left the
# backend cells armed for an unrelated guard.
#
# THE LOOP IN THE CALLEE IS THE TEST, not decoration.  The fold that runs the
# portal shim at trace time is the one an inlined callee sub-walk takes when it
# reaches the callee's OWN loop header
# (`emit_walker_loop_callee_call_assembler`).  A loopless recursion folds
# through the self-recursive arm instead, which executes the ordinary `call_fn`
# residual — that helper publishes both carriers, so a loopless shape passes
# whether or not the portal shim is correct.
#
# THE CALL SITE MUST SIT OUTSIDE THE `try`, for the same reason: the fold
# declines on a call site inside a protected region, so the handler belongs one
# level up, in the driver loop.
#
# Measured before the fix on dynasm: 5 of 3000 calls surfaced
# `TypeError: unsupported operand type(s) for +: 'object' and 'int'` — the
# swallowed raise's NULL reaching the `+` — where `PYRE_JIT=0` and cpython 3.14
# both deliver 3000 `ValueError`s.  One wrong answer is the failure; the count
# only tracks when the callee's loop compiles.
import sys

WARM = 3000
DEPTH = 4
INNER = 30


def rec(n, m):
    total = 0
    for i in range(m):
        total += i
    if n <= 0:
        raise ValueError('bottom')
    return rec(n - 1, m) + total


def hot(k):
    caught = 0
    wrong = 0
    seen = []
    for _ in range(k):
        try:
            rec(DEPTH, INNER)
        except ValueError:
            caught += 1
        except BaseException as exc:
            wrong += 1
            if len(seen) < 3:
                seen.append('%s: %s' % (type(exc).__name__, exc))
    return caught, wrong, seen


def main():
    caught, wrong, seen = hot(WARM)
    failures = []
    if wrong:
        failures.append(
            '%d of %d calls surfaced a foreign exception instead of the '
            'ValueError the callee raised — the portal published the raise to '
            'one carrier and the walker reads the other, so its NULL result '
            'flowed on: %r' % (wrong, WARM, seen)
        )
    if caught != WARM:
        failures.append(
            'the handler ran %d times, owed %d' % (caught, WARM)
        )
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a raise through the recursive portal reaches its handler')
    return 0


sys.exit(main())
