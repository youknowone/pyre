# pyre-check: max-pypy-ratio=8
# pyre-check: min-pypy-ratio=1.28
# A loop that defines a function in its own body and calls it.
#
# `pyopcode.py:1457 MAKE_FUNCTION` runs `function.py:47-57 Function.__init__`
# per execution, so `grab` is a fresh object every iteration.  Nothing here
# escapes it: the call is inlined and the object is dead by the next
# `MAKE_FUNCTION`.  Building the function inline — `NewWithVtable` plus one
# `SetfieldGc` per slot the constructor writes — is what lets the optimizer see
# that and drop the allocation; behind the residual call it could only ever
# allocate.  The gate measures whether it was dropped: the residual costs
# hundreds of ns per iteration against pypy's fraction of one.
#
# What the inline call depends on is the
# `_immutable_fields_ = ['code?', 'w_func_globals?', 'closure?[*]', 'defs_w?[*]']`
# set (function.py:34-42), all loop-invariant here, so guarding those fields off
# the live function keeps the trace across the fresh objects.  `guard_failures`
# in the committed `.jitstats` baseline is the signal: it is 1 with the field
# guards and 9480 with an identity guard, which also compiled 47 bridges here
# (2497 at N=1000000) before `make_a_counter_per_value` (regalloc.py:496-499)
# reached the dynasm and wasm backends.  With the identity guard gone this
# fixture no longer exercises that bucketing — `bridges_compiled` stays 0.
#
# `N` is sized by the GATE, not by the loop.  The gate compares
# startup-subtracted times with the baseline floored at 5ms, so below ~11M
# iterations pypy lands on that floor and the comparison becomes a fixed
# ~45ms budget for pyre — which is the stable regime, since only pyre's own
# startup estimate is then left to miss.  Above it the ceiling is CPython:
# ~195ns/iteration here against pypy's fraction of one, and this fixture must
# not become the slowest baseline run in the suite.
N = 10000000


def run(n):
    acc = 0
    i = 0
    while i < n:
        def grab():
            return 1

        acc += grab() & 7
        i += 1
    return acc


print(run(N))
