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

r = reversed([1, 2, 3])
assert iter(r) is r and operator.length_hint(r) == 3
assert next(r) == 3 and operator.length_hint(r) == 2
r.__setstate__(0)
assert list(r) == [1]

m = map(lambda x: x + 1, [1, 2])
assert iter(m) is m and next(m) == 2
assert m.__reduce__()[0] is map
assert list(m) == [3]

f = filter(None, [0, 1, "", "x"])
assert iter(f) is f and f.__reduce__()[0] is filter
assert list(f) == [1, "x"]

z = zip([1, 2], [3, 4])
assert iter(z) is z and next(z) == (1, 3)
assert z.__reduce__()[0] is zip
assert list(z) == [(2, 4)]

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
