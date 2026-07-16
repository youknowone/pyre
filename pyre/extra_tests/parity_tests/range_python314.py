"""Python 3.14/PyPy structural parity for range and its iterators."""

import operator


required = {
    "__doc__",
    "__new__",
    "__repr__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__reversed__",
    "__reduce__",
    "__contains__",
    "__eq__",
    "__hash__",
    "__bool__",
    "count",
    "index",
    "start",
    "stop",
    "step",
}
assert required <= set(range.__dict__)

r = range(2, 13, 3)
assert repr(r) == "range(2, 13, 3)"
assert range.__repr__(r) == "range(2, 13, 3)"
assert (r.start, r.stop, r.step) == (2, 13, 3)
assert len(r) == 4 and r.__len__() == 4 and bool(r)
assert list(r) == [2, 5, 8, 11]
assert r[1] == 5 and r[-1] == 11
assert r[::-1] == range(11, 1, -3)
assert list(reversed(r)) == [11, 8, 5, 2]
assert 8 in r and 9 not in r
assert r.count(8) == 1 and r.count(8.0) == 1
assert r.index(8) == 2 and r.index(8.0) == 2
assert range(0, 3, 2) == range(0, 4, 2)
assert hash(range(0, 3, 2)) == hash(range(0, 4, 2))
assert range.__eq__(r, object()) is NotImplemented
assert r.__reduce__() == (range, (2, 13, 3))
assert not range(0)

try:
    r.start = 9
except AttributeError:
    pass
else:
    raise AssertionError("range fields must be read-only")

huge = range(10**30)
try:
    huge.__len__()
except OverflowError:
    pass
else:
    raise AssertionError("range.__len__ must overflow Py_ssize_t")

for source in (range(5), range(10**30, 10**30 + 4)):
    iterator = iter(source)
    iterator_type = type(iterator)
    assert {
        "__iter__",
        "__length_hint__",
        "__next__",
        "__reduce__",
        "__setstate__",
    } <= set(iterator_type.__dict__)
    first = next(iterator)
    assert first == source[0]
    remaining = list(source)[1:]
    assert operator.length_hint(iterator) == len(remaining)
    reduced = iterator.__reduce__()
    assert reduced[0] is iter and reduced[2] is None

    iterator.__setstate__(-1)
    assert list(iterator) == remaining

    iterator = iter(source)
    next(iterator)
    iterator.__setstate__(1)
    assert list(iterator) == remaining[1:]

    iterator = iter(source)
    next(iterator)
    iterator.__setstate__(99)
    assert list(iterator) == []

iterator = iter(range(5))
try:
    iterator.__setstate__(10**40)
except OverflowError:
    pass
else:
    raise AssertionError("machine range iterator state must fit a C long")

print("OK")
