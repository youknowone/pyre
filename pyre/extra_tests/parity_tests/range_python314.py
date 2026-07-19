"""Python 3.14/PyPy structural parity for range and its iterators."""

import operator


expected_surface = {
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
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__hash__",
    "__bool__",
    "count",
    "index",
    "start",
    "stop",
    "step",
}
assert set(range.__dict__) == expected_surface

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


class AlwaysEqual:
    def __eq__(self, other):
        return True


class EqualIntSubclass(int):
    def __eq__(self, other):
        return True


assert range(10).count(AlwaysEqual()) == 10
assert EqualIntSubclass(11) in range(10)
assert list(range(1, 9, 3)[-1:-3:-1]) == [7, 4]
assert list(range(8, 0, -3)[-1:-3:-1]) == [2, 5]
for method_name in ("count", "index"):
    method = getattr(range(3), method_name)
    try:
        method()
    except TypeError as error:
        assert str(error) == (
            f"range.{method_name}() takes exactly one argument (0 given)"
        )
    else:
        raise AssertionError(f"range.{method_name}() must require one argument")

assert range(0, 3, 2) == range(0, 4, 2)
assert hash(range(0, 3, 2)) == hash(range(0, 4, 2))
assert range.__eq__(r, object()) is NotImplemented
assert range.__ne__(r, range(2, 13, 3)) is False
assert range.__ne__(r, range(2, 15, 3)) is True
assert range.__ne__(r, object()) is NotImplemented
for name in ("__lt__", "__le__", "__gt__", "__ge__"):
    assert getattr(range, name)(r, range(2, 15, 3)) is NotImplemented
    assert getattr(range, name)(r, object()) is NotImplemented
assert r.__reduce__() == (range, (2, 13, 3))
assert not range(0)

for compare in (
    lambda: r < range(2, 15, 3),
    lambda: r <= range(2, 15, 3),
    lambda: r > range(2, 15, 3),
    lambda: r >= range(2, 15, 3),
):
    try:
        compare()
    except TypeError:
        pass
    else:
        raise AssertionError("range ordering must remain unsupported")

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

small_type = type(iter(range(1)))
long_type = type(iter(range(10**30, 10**30 + 1)))
assert small_type.__name__ == "range_iterator"
assert long_type.__name__ == "longrange_iterator"
assert small_type is not long_type

iterator_surface = {
    "__iter__",
    "__length_hint__",
    "__next__",
    "__reduce__",
    "__setstate__",
    "__doc__",
}
assert set(small_type.__dict__) == iterator_surface
assert set(long_type.__dict__) == iterator_surface

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

for iterator_type, foreign in (
    (small_type, iter(range(10**30, 10**30 + 1))),
    (long_type, iter(range(1))),
):
    for name in ("__iter__", "__next__", "__length_hint__", "__reduce__", "__setstate__"):
        descriptor = iterator_type.__dict__[name]
        args = (foreign, 0) if name == "__setstate__" else (foreign,)
        try:
            descriptor(*args)
        except TypeError as exc:
            owner = iterator_type.__name__
            received = type(foreign).__name__
            if name in ("__iter__", "__next__"):
                expected = (
                    f"descriptor '{name}' requires a '{owner}' object "
                    f"but received a '{received}'"
                )
            else:
                expected = (
                    f"descriptor '{name}' for '{owner}' objects "
                    f"doesn't apply to a '{received}' object"
                )
            assert str(exc) == expected
        else:
            raise AssertionError("range iterator descriptors must validate their receiver")

        try:
            descriptor()
        except TypeError as exc:
            if name in ("__iter__", "__next__"):
                expected = f"descriptor '{name}' of '{owner}' object needs an argument"
            else:
                expected = f"unbound method {owner}.{name}() needs an argument"
            assert str(exc) == expected
        else:
            raise AssertionError("range iterator descriptors must require a receiver")

iterator = iter(range(5))
try:
    iterator.__setstate__(10**40)
except OverflowError:
    pass
else:
    raise AssertionError("machine range iterator state must fit a C long")

class IndexState:
    def __index__(self):
        return 2

iterator = iter(range(5))
iterator.__setstate__(IndexState())
assert next(iterator) == 2

for state in (True, IndexState()):
    iterator = iter(range(10**30))
    try:
        iterator.__setstate__(state)
    except TypeError as exc:
        assert str(exc) == "state must be an int, not " + type(state).__name__
    else:
        raise AssertionError("longrange_iterator state must be an exact int")

print("OK")
