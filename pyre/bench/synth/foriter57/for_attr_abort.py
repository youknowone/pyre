class Obj:
    pass


def f():
    o = Obj()
    o.n = 0
    seen = []
    for x in range(500):
        # `o.n += 1` is a concrete NON-journaled, NON-idempotent heap mutation
        # (a STORE_ATTR via the residual `store_attr_fn`, which carries
        # `PyreHelperKind::None` and is covered by NO journal) that COMMITS
        # before a later op in the same iteration aborts the trace.
        # `seen.append` is the abort trigger (its inline sub-walk declines);
        # it exists only to force the in-body abort AFTER the attr store has
        # committed.
        #
        # The OLD R1 guard flagged a body effect ONLY for a tiny allow-list of
        # `PyreHelperKind`s (`StoreSubscr` / `CallFn` / `SetCurrentException`),
        # so this `None`-tagged store was never flagged: the guard delivered the
        # in-flight FOR_ITER item and re-ran the whole body, DOUBLING the
        # accumulate (o.n over-counts by the number of aborts → 500 + #aborts).
        # The inverted Finding #1 predicate flags ANY residual that is not
        # provably side-effect-free, so the store is counted exactly once.
        #
        # Only `o.n` is asserted: it commits on every iteration (including the
        # aborted ones, whose bypass still advances the loop), so the
        # conservative refuse-and-drop fallback leaves it EXACT.  A re-run
        # double, by contrast, is observable as 500 + (number of aborts).
        o.n += 1
        seen.append(x)
    return o.n


print(f())
