# pyre-check: max-pypy-ratio=8
# A loop that defines a function in its own body and calls it.
#
# `MAKE_FUNCTION` hands back a fresh `Function` every iteration, so an inline
# specialized on the callable's OBJECT IDENTITY can only ever hold by address
# coincidence.  It never does.  What the inline actually depends on is the
# `_immutable_fields_ = ['code?', 'w_func_globals?', 'closure?[*]', 'defs_w?[*]']`
# set (function.py:34-42), all of which are loop-invariant here, so guarding
# those fields off the live function keeps the trace.
#
# `guard_failures` in the committed `.jitstats` baseline is the signal: it is 1
# with the field guards and 9480 with the identity guard, which also compiled
# 47 bridges here (2497 at N=1000000) before `make_a_counter_per_value`
# (regalloc.py:496-499) reached the dynasm and wasm backends.  With the
# identity guard gone this fixture no longer exercises that bucketing —
# `bridges_compiled` stays 0 either way.
N = 1500000


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
