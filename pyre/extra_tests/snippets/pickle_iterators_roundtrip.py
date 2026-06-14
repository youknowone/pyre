# Protocol-sweep round-trip for the iterator reducers: every iterator
# family is advanced past its first element(s), pickled, and unpickled at
# each protocol 0-5, then must resume at the saved position.  Complements
# pickle_roundtrip.py (which pins the exact reduce shapes at the default
# protocol) by exercising the full protocol range, including the text
# protocols 0/1 that route through copyreg._reduce_ex.
import pickle
from itertools import islice


def resume(make, advance, proto):
    it = make()
    for _ in range(advance):
        next(it)
    return pickle.loads(pickle.dumps(it, proto))


# every finite iterator family resumes at its saved position across protocols
for proto in range(0, 6):
    assert list(resume(lambda: iter([1, 2, 3, 4]), 2, proto)) == [3, 4], proto
    assert list(resume(lambda: iter((1, 2, 3, 4)), 1, proto)) == [2, 3, 4], proto
    assert list(resume(lambda: iter("abcd"), 3, proto)) == ["d"], proto
    assert list(resume(lambda: iter(range(5)), 2, proto)) == [2, 3, 4], proto
    assert list(resume(lambda: iter({"a": 1, "b": 2}.keys()), 1, proto)) == ["b"], proto
    assert list(resume(lambda: enumerate([10, 20, 30]), 1, proto)) == [(1, 20), (2, 30)], proto
    # long-range iterator: a 10**30-element range stays lazy, so take the
    # first few with islice rather than materialising the whole thing.
    big = resume(lambda: iter(range(10 ** 30)), 2, proto)
    assert list(islice(big, 3)) == [2, 3, 4], proto

print("pickle_iterators_roundtrip OK")
