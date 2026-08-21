# pyre-check: max-pypy-ratio=6
# The ceiling sits between the two measured states: folded this runs 2.6x
# pypy, and with `subscr_specialised_pair` suppressed about 198x.
# #171/#11 Approach C, SUBSCRIPT slice: canonical-tuple `t[i]` emits a PURE
# getarrayitem in the JIT walker (OptPure CSEs / const-folds the element load).
#
# Case A exercises the canonical array-backed `W_TupleObject` (arity > 2) on the
# hot path — the pure element load is the point of this slice.
#
# Case B reads a 2-element literal tuple in a loop. A 2-int literal tuple may be
# stored as SPECIALISED_TUPLE_II (inline value0/value1, NO wrappeditems block);
# the trace-time `ob_type == &TUPLE_TYPE` gate declines it to the non-pure /
# residual path. It MUST NOT SIGSEGV and MUST stay correct.
#
# Case C builds a fresh arity-2 pair per iteration and reads both halves, which
# is the `subscr_specialised_pair` fold's own shape. Suppressing that one fold
# measures 46.3x here (0.092s -> 4.257s), the largest single-fold effect in the
# corpus, so this loop is what keeps the ceiling below honest.


def main():
    # Case A: canonical array-backed tuple (arity 5 > 2).
    t = (10, 20, 30, 40, 50)
    s = 0
    for _ in range(200000):
        s += t[0] + t[1] + t[2] + t[3] + t[4]
    print(s)

    # Case B: 2-element tuple (specialised-tuple path must be safe + correct).
    t2 = (1, 2)
    s2 = 0
    for _ in range(200000):
        s2 += t2[0] + t2[1]
    print(s2)


main()


def hot_specialised_pair(n):
    """Case C: a fresh 2-tuple per iteration, both halves read."""
    s = 0
    i = 0
    while i < n:
        p = (i, i + 1)
        s += p[0] + p[1]
        i += 1
    return s


print(hot_specialised_pair(20000000))
