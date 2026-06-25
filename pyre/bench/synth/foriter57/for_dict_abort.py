def f():
    cnt = {}
    seen = []
    for x in range(500):
        # `cnt[0] = ...` is a concrete NON-journaled, NON-idempotent heap
        # mutation (a dict accumulate via the residual `store_subscr_fn`, not
        # the list[int] setitem journal) that COMMITS before a later op in the
        # same iteration aborts the trace.  `seen.append` is the abort trigger
        # (its inline sub-walk declines); it exists only to force the in-body
        # abort AFTER the dict accumulate has committed.
        #
        # The OLD R1 guard checked only the symbolic-decline flag, so it
        # delivered the in-flight FOR_ITER item and re-ran the whole body,
        # DOUBLING the accumulate (cnt[0] over-counts by the number of aborts).
        # The hardened guard refuses delivery whenever a body effect committed
        # since the consume, so the accumulate is counted exactly once.
        #
        # Only `cnt[0]` is asserted: it commits on every iteration (including
        # the aborted ones, whose bypass still advances the loop), so the
        # conservative refuse-and-drop fallback leaves it EXACT.  A re-run
        # double, by contrast, is observable as 500 + (number of aborts).
        cnt[0] = cnt.get(0, 0) + 1
        seen.append(x)
    return cnt[0]
print(f())
