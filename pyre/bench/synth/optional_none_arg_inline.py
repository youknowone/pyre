# pyre-check: max-pypy-ratio=8
# A hot call into a callee that branches on `arg is None` -- the shape every
# optional argument in the standard library takes.  The tested local holds the
# default `None`, a Ref; the other parameter is an int.
#
# The gate is `loops_compiled` first: declined, the callee takes a trace of
# its own and the count reads 2; inlined, it folds into the caller's and reads
# 1.  That census is host-independent, which the ratio at this size is not:
# the body is a single add, so pypy runs the whole loop well under
# `EXEC_TIME_FLOOR_S` and the comparison marks it -- only the ceiling applies.
# Sized up until pypy cleared the floor the fixture cost cpython a second, so
# the counter stays the gate here, as it is for `kwonly_default_callee_inline`.
#
# Its sibling `is_none_unboxed_operand_decline` is this fixture with the
# default changed from `None` to an int, and stays declined.  The pair is what
# shows the `is`-against-None scan asking about the operand the branch tests
# rather than about the signature as a whole.
#
# `clamp` is called once more off the hot path so the printed total depends on
# the arm the trace guards away as well as the one it keeps: an identity test
# answered the wrong way changes the sum rather than the timing.
N = 175933800


def clamp(v, lo=None):
    if lo is None:
        return v
    return lo


def main():
    total = 0
    for i in range(N):
        total += clamp(i)
    total += clamp(1, 7)
    print(total)


main()
