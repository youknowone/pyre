def f():
    # `acc += delta` is a bare BINARY_OP NB_INPLACE_ADD (no CALL) that extends
    # the Integer-strategy list `acc` IN PLACE — a non-journaled heap mutation
    # that COMMITS during the walk.  A mid-body abort (the `x < 250` branch's
    # unrestorable kept stack) re-delivers the in-flight FOR_ITER item; without
    # journaling the in-place extend, the re-run DOUBLES it (len 501) and a plain
    # drop would instead lose that iteration's `n += 1` (n 249).  The fix journals
    # the extend (append-journal pre-length rewind) so the abort rollback undoes
    # the one walk-committed extend and the deliver re-applies it exactly once.
    #
    # Asserts BOTH the length (no double) AND the branch counter `n` (no dropped
    # tail) — `for_dict_abort` deliberately checks only the committed effect.
    acc = []
    delta = [1]
    n = 0
    for x in range(500):
        acc += delta
        if x < 250:
            n += 1
    return len(acc), n


print(f())
