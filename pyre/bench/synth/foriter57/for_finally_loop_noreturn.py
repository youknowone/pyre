def inner():
    # Same finally-duplicated loop shape as `for_finally_loop`, but the `try`
    # does not `return`, so the exhaustion side-exit does not take the poisoned
    # guard path and the original case happened to stay correct.  The decline
    # gate (`for_iter_frame_is_finally_duplicated`) intentionally over-covers this
    # harmless shape — both copies are declined together — so this bench locks in
    # that declining it keeps the result correct (it must not regress).
    try:
        x = 1
    finally:
        acc = 0
        for x in range(2000):
            acc += x
        r2 = ("FIN", acc)
    return r2


for _ in range(300):
    r = inner()
print(r)
