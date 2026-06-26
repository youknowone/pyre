def make():
    n = 0

    def f():
        nonlocal n
        seen = []
        for x in range(500):
            # `n += 1` on a closure free variable lowers to STORE_DEREF, whose
            # `store_deref_value` residual is a VALUE-returning (`Ref`) in-place
            # cell write carrying `PyreHelperKind::StoreDeref`.  It commits a
            # NON-journaled heap mutation before the later `seen.append` aborts
            # the trace.
            #
            # The Void-result write proxy in the #57 Option C body-effect guard
            # cannot see this residual (its result_type is `Ref`, not `Void`),
            # so without the `StoreDeref` helper tag the guard DELIVERS the
            # in-flight FOR_ITER item, re-runs the body, and DOUBLES the cell
            # write (n = 500 + #aborts).  The tag flags it as a body effect so
            # delivery is refused and the count stays EXACT.
            n += 1
            seen.append(x)
        return n

    return f


print(make()())
