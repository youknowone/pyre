# A loop that defines a function in its own body and calls it.
#
# The inline lever specializes the call on the callable's OBJECT IDENTITY, so
# the guard it emits can only ever hold if MAKE_FUNCTION happens to hand back
# the same address twice.  It never does: every compiled entry deopts, one per
# two Python iterations.
#
# That is the frontend half.  The half this fixture pins is the backend one:
# `make_a_counter_per_value` (regalloc.py:496-499) buckets the jitcounter by
# (guard, failing value), so a guard whose value never repeats never reaches
# `trace_eagerness` and compiles no bridge.  Without it every 200th failure
# compiled another bridge, without bound — 47 here, and 2497 at N=1000000.
#
# `bridges_compiled` in the committed `.jitstats` baseline is the signal.
N = 20000


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
