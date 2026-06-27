# Sequence-iterator pickle protocol parity.  check.py's correctness oracle is
# PyPy, so this bench asserts only behaviour where 3.14 and PyPy AGREE.
#
# `iter(list)` yields the generic sequence iterator (W_AbstractSeqIterObject),
# whose `__reduce__` / `__setstate__` recreate the cursor (iterobject.py:32-45):
#   * a live cursor reduces to `(iter, (seq,), index)`;
#   * an exhausted cursor (`w_seq is None`) reduces to `_empty_iterable` =
#     `(iter, ((),))` (iterobject.py:251-253), so it restores empty;
#   * `__setstate__` restores the index only while the sequence is still live.
# A negative restored index DIVERGES (3.14 leaves the iterator exhausted; PyPy
# clamps the index to 0) so it is exercised only in the implementation, not
# asserted here; every case below agrees between 3.14 and PyPy.


def drive():
    out = []

    # Live cursor: __reduce__ carries the index; replaying it resumes mid-list.
    it = iter([10, 20, 30, 40, 50])
    next(it)
    next(it)
    r = it.__reduce__()
    it2 = r[0](*r[1])
    if len(r) > 2 and r[2] is not None:
        it2.__setstate__(r[2])
    out.append(("partial", list(it2), len(r)))

    # Exhausted cursor reduces to the empty-iterable shape (no index element).
    ex = iter([1, 2])
    list(ex)
    r2 = ex.__reduce__()
    it3 = r2[0](*r2[1])
    out.append(("exhausted", list(it3), len(r2)))

    # __setstate__ on an already-exhausted cursor is a no-op (seq is None).
    g = iter([1, 2, 3])
    list(g)
    g.__setstate__(0)
    out.append(("exhausted_setstate", list(g)))

    # Drive a reduce/replay hot so a compiled trace exercises the cursor path.
    total = 0
    k = 0
    while k < 20000:
        h = iter([1, 2, 3, 4])
        next(h)
        rr = h.__reduce__()
        h2 = rr[0](*rr[1])
        h2.__setstate__(rr[2])
        total += sum(h2)
        k += 1
    out.append(("hot", total))
    return out


for row in drive():
    print(row)
