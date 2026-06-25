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
