# pyre-check: pypy-diverges: pins 3.14's hard cutoff at the limit; pypy3's is
# approximate, so it answers rec(1000) under a limit of 500 and raises only
# once the native stack budget runs out. Its jitcodes are built before
# `insert_ll_stackcheck` runs -- `driver.py`'s `pyjitpl_lltype` task depends on
# `rtype` alone while `stackcheckinsertion_lltype` follows `backendopt` -- so
# compiled code there answers to the backend's native stack probe and to
# nothing else.
# CPython-suite gap: test_sys exercises `sys.setrecursionlimit` on cold frames
# only, so nothing there grades a recursion whose levels all run as compiled
# code.
# parity-tests reason: a self-recursive function warmed past the function-entry
# threshold runs its recursion through the trace's own `CALL_ASSEMBLER`. Every
# level below the first is an activation the trace mints, so it reaches no
# interpreter door, and the logical half of the recursion check has no other
# place to run.
# `recursion_limit_survives_jit_warmup.py` sits next to this one and does not
# cover it: it asks that the depth reached stay at or under the limit, which is
# satisfied whenever the native byte budget cuts in first -- the very case its
# own docstring records. A limit the compiled path ignores entirely still
# passes there, so the cutoff has to be graded against a depth the budget
# cannot reach.

"""A warm recursive function still answers to `sys.setrecursionlimit`.

`rec` is entered thousands of times before the limit is lowered, so the JIT
owns it by the time the deep call arrives. The limit has to bound that call
the same way it bounds `cold_rec`, which is the same function text run once.

The module body deliberately keeps the deep call in a helper of its own and
does not wrap it in a `try`/`finally`: both change which of the module frame's
statements run compiled, and the recursion has to start from compiled code for
this to grade anything.
"""

import sys


def rec(n):
    return 0 if n <= 0 else rec(n - 1) + 1


def cold_rec(n):
    return 0 if n <= 0 else cold_rec(n - 1) + 1


def warm_overflows(depth):
    try:
        rec(depth)
    except RecursionError:
        return True
    return False


def cold_overflows(depth):
    try:
        cold_rec(depth)
    except RecursionError:
        return True
    return False


# Warm past the function-entry threshold, so `rec` is entered as compiled code
# rather than through the interpreter's own activation door.
for _ in range(3000):
    rec(20)

sys.setrecursionlimit(500)
assert rec(100) == 100
assert warm_overflows(1000), "a warm recursion outran a limit of 500"
assert cold_overflows(1000)
# The budget the raise consumed is given back, so the next call answers.
assert rec(100) == 100
sys.setrecursionlimit(1000)

print("OK")
