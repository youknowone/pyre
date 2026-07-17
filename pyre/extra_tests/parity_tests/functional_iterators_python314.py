"""PyPy functional iterator typedefs with Python 3.14 state/strict slots."""

import operator


expected = {
    enumerate: {"__doc__", "__new__", "__iter__", "__next__", "__reduce__", "__class_getitem__"},
    reversed: {"__doc__", "__new__", "__iter__", "__next__", "__length_hint__", "__reduce__", "__setstate__"},
    map: {"__doc__", "__new__", "__iter__", "__next__", "__reduce__", "__setstate__"},
    filter: {"__doc__", "__new__", "__iter__", "__next__", "__reduce__"},
    zip: {"__doc__", "__new__", "__iter__", "__next__", "__reduce__", "__setstate__"},
}
for typ, names in expected.items():
    assert names <= set(typ.__dict__)
    assert typ.__doc__

e = enumerate(["a", "b"], 10)
assert iter(e) is e and next(e) == (10, "a")
reduced = e.__reduce__()
assert reduced[0] is enumerate and reduced[1][1] == 11
alias = enumerate[str]
assert alias.__origin__ is enumerate and alias.__args__ == (str,)


class HugeIndex:
    def __index__(self):
        return 10**30


for start, expected in [
    (10**30, 10**30),
    (-(10**30), -(10**30)),
    (HugeIndex(), 10**30),
]:
    e = enumerate(["x", "y"], start)
    assert next(e) == (expected, "x")
    reduced = e.__reduce__()
    assert reduced[0] is enumerate
    assert reduced[1][1] == expected + 1
    assert next(e) == (expected + 1, "y")

# Crossing the machine-word boundary promotes the counter to the same bigint
# state used when construction starts outside the fast range.
e = enumerate(iter([1, 2]), 2**63 - 1)
assert next(e) == (2**63 - 1, 1)
assert next(e) == (2**63, 2)
assert e.__reduce__()[1][1] == 2**63 + 1

try:
    enumerate([], None)
except TypeError:
    pass
else:
    raise AssertionError("an explicit None start must be passed through __index__")

for name in ("__iter__", "__next__", "__reduce__"):
    try:
        getattr(enumerate, name)(42)
    except TypeError:
        pass
    else:
        raise AssertionError(f"enumerate.{name} must validate its receiver")


class EnumerateSubclass(enumerate):
    pass


assert EnumerateSubclass([1]).__reduce__()[0] is EnumerateSubclass

r = reversed([1, 2, 3])
assert iter(r) is r and operator.length_hint(r) == 3
assert next(r) == 3 and operator.length_hint(r) == 2
r.__setstate__(0)
assert list(r) == [1]


class Sequence:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


seq = Sequence([0, 1, 2])
r = reversed(seq)
assert type(r) is reversed
assert r.__reduce__()[0] is reversed
assert r.__reduce__()[1] == (seq,)
assert r.__reduce__()[2] == 2
assert operator.length_hint(r) == 3
assert next(r) == 2
r.__setstate__(99)
assert operator.length_hint(r) == 3 and next(r) == 2
r.__setstate__(-99)
assert operator.length_hint(r) == 0
try:
    next(r)
except StopIteration:
    pass
else:
    raise AssertionError("reversed negative state must clamp to exhaustion")

# The length hint re-reads the live sequence length, as required by PyPy and
# CPython's dedicated regression test.
seq = Sequence([0, 1, 2])
r = reversed(seq)
seq.values[:] = [0]
assert operator.length_hint(r) == 0


class ReversedSubclass(reversed):
    pass


r = ReversedSubclass((1, 2))
assert r.__reduce__()[0] is ReversedSubclass
assert list(r) == [2, 1]
assert r.__reduce__()[0] is ReversedSubclass

for name, call_args in [
    ("__iter__", (42,)),
    ("__next__", (42,)),
    ("__length_hint__", (42,)),
    ("__reduce__", (42,)),
    ("__setstate__", (42, 0)),
]:
    try:
        getattr(reversed, name)(*call_args)
    except TypeError:
        pass
    else:
        raise AssertionError(f"reversed.{name} must validate its receiver")


class IndexState:
    def __index__(self):
        return 1


class IntState:
    def __int__(self):
        return 1


for state in (IndexState(), IntState(), 1.25):
    try:
        reversed(Sequence([1])).__setstate__(state)
    except TypeError:
        pass
    else:
        raise AssertionError("reversed state must be a concrete Python int")

try:
    reversed(Sequence([1])).__setstate__(10**30)
except OverflowError:
    pass
else:
    raise AssertionError("reversed state must fit Py_ssize_t")

for non_reversible in (iter(range(3)), iter([1, 2])):
    try:
        reversed(non_reversible)
    except TypeError:
        pass
    else:
        raise AssertionError("iterators are not reversible in Python 3.14")


class DisabledReversed(Sequence):
    __reversed__ = None


try:
    reversed(DisabledReversed([1]))
except TypeError:
    pass
else:
    raise AssertionError("__reversed__ = None must disable sequence fallback")

m = map(lambda x: x + 1, [1, 2])
assert iter(m) is m and next(m) == 2
assert m.__reduce__()[0] is map
assert list(m) == [3]


class MapSubclass(map):
    pass


m = MapSubclass(str, [1])
assert m.__reduce__()[0] is MapSubclass
m.__setstate__([1])
assert m.__reduce__()[2] is True
m.__setstate__([])
assert len(m.__reduce__()) == 2

for name, call_args in [
    ("__iter__", (42,)),
    ("__next__", (42,)),
    ("__reduce__", (42,)),
    ("__setstate__", (42, True)),
]:
    try:
        getattr(map, name)(*call_args)
    except TypeError:
        pass
    else:
        raise AssertionError(f"map.{name} must validate its receiver")

f = filter(None, [0, 1, "", "x"])
assert iter(f) is f and f.__reduce__()[0] is filter
assert list(f) == [1, "x"]


class FilterSubclass(filter):
    pass


assert FilterSubclass(None, [1]).__reduce__()[0] is FilterSubclass


class FilterInitSubclass(filter):
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs


f = FilterInitSubclass(None, [1], marker=42)
assert f.init_args == (None, [1]) and f.init_kwargs == {"marker": 42}
try:
    FilterSubclass(None, [1], marker=42)
except TypeError:
    pass
else:
    raise AssertionError("filter subclass without __init__ override must reject keywords")

for name in ("__iter__", "__next__", "__reduce__"):
    try:
        getattr(filter, name)(42)
    except TypeError:
        pass
    else:
        raise AssertionError(f"filter.{name} must validate its receiver")

z = zip([1, 2], [3, 4])
assert iter(z) is z and next(z) == (1, 3)
assert z.__reduce__()[0] is zip
assert list(z) == [(2, 4)]


class ZipSubclass(zip):
    pass


z = ZipSubclass([1], [2])
assert z.__reduce__()[0] is ZipSubclass
z.__setstate__([1])
assert z.__reduce__()[2] is True
z.__setstate__([])
assert len(z.__reduce__()) == 2

for name, call_args in [
    ("__iter__", (42,)),
    ("__next__", (42,)),
    ("__reduce__", (42,)),
    ("__setstate__", (42, True)),
]:
    try:
        getattr(zip, name)(*call_args)
    except TypeError:
        pass
    else:
        raise AssertionError(f"zip.{name} must validate its receiver")

try:
    list(zip([1], [2, 3], strict=True))
except ValueError:
    pass
else:
    raise AssertionError("zip(strict=True) must detect unequal lengths")

try:
    list(map(lambda x, y: x + y, [1], [2, 3], strict=True))
except ValueError:
    pass
else:
    raise AssertionError("map(strict=True) must detect unequal lengths")

print("OK")
