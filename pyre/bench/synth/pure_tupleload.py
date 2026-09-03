# pyre-check: max-pypy-ratio=9
# The ceiling moves for the startup-subtraction reason recorded on fib_recursive
# in `check.py`: a pyre backend subtracts pypy's startup rather than its own, so
# what a pyre process spends above pypy to reach the first bytecode stays inside
# every pyre reading.  This fixture was not among the 41 that commit lengthened,
# and its pypy execution is 0.01s-0.05s, so that fixed deficit is several times
# the baseline it divides.  Over the twelve CI runs created after 6aabe927ce1 the
# cranelift leg tripped three times -- runs 33691104038 and 33702184169 on
# ubuntu and 33704998320 on macos, at (exec, pypy) of (0.150, 0.0175),
# (0.140, 0.0145) and (0.100, 0.0107) -- and the widest needs 6.92 to pass.  9
# covers it with 30% headroom.  The floor it derives is capped at parity and is
# inert here either way: every reading is marked `?` or `~`, so no leg has a
# baseline the floor gate will act on.
#
# Lengthening is the fix this does not take.  Tripling the trip count would put
# pypy's execution over FLOOR_GATE_MIN_BASELINE_S and make the ratio a
# measurement rather than a quantisation of a 0.01s denominator, but it also
# re-records three jit-stats baselines, one of them wasm.
# The ceiling sits between the two measured states: served this runs 2.6x pypy,
# and with the arity-2 reader off the loop pays the opaque residual (about 198x
# when the retired `subscr_specialised_pair` fold was the only reader).
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
# Case C builds a fresh arity-2 pair per iteration and reads both halves. This
# is the shape `subscr_tuple_descent` serves by descending `w_tuple_getitem`;
# the hand-written `subscr_specialised_pair` reader it replaced measured 46.3x
# here when suppressed, the largest single-fold effect in the corpus, so this
# loop is what keeps the ceiling below honest.


def main():
    # Case A: canonical array-backed tuple (arity 5 > 2).
    t = (10, 20, 30, 40, 50)
    s = 0
    for _ in range(1600000):
        s += t[0] + t[1] + t[2] + t[3] + t[4]
    print(s)

    # Case B: 2-element tuple (specialised-tuple path must be safe + correct).
    t2 = (1, 2)
    s2 = 0
    for _ in range(1600000):
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


print(hot_specialised_pair(160000000))
