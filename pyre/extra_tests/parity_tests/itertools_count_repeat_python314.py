import itertools


assert isinstance(itertools.count, type)
assert isinstance(itertools.repeat, type)
assert itertools.count.__module__ == "itertools"
assert itertools.count.__name__ == "count"
assert itertools.repeat.__module__ == "itertools"
assert itertools.repeat.__name__ == "repeat"

assert itertools.count.__doc__ == (
    "Return a count object whose .__next__() method returns consecutive values.\n\n"
    "Equivalent to:\n"
    "    def count(firstval=0, step=1):\n"
    "        x = firstval\n"
    "        while 1:\n"
    "            yield x\n"
    "            x += step"
)
assert itertools.repeat.__doc__ == (
    "repeat(object [,times]) -> create an iterator which returns the object\n"
    "for the specified number of times.  If not specified, returns the object\n"
    "endlessly."
)

c = itertools.count()
assert iter(c) is c
assert repr(c) == "count(0)"
assert next(c) == 0
assert repr(c) == "count(1)"

c = itertools.count(start=2, step=3)
assert repr(c) == "count(2, 3)"
assert [next(c), next(c), next(c)] == [2, 5, 8]

for args in [(1, 2, 3), ("not a number",), ([], 1)]:
    try:
        itertools.count(*args)
    except TypeError:
        pass
    else:
        raise AssertionError(args)

try:
    itertools.count(unknown=1)
except TypeError:
    pass
else:
    raise AssertionError("unknown count keyword accepted")


class CountSubclass(itertools.count):
    pass


c = CountSubclass(4, 2)
assert type(c) is CountSubclass
assert repr(c) == "CountSubclass(4, 2)"
assert next(c) == 4

r = itertools.repeat("x")
assert iter(r) is r
assert repr(r) == "repeat('x')"
assert next(r) == "x"
try:
    r.__length_hint__()
except TypeError as exc:
    assert str(exc) == "len() of unsized object"
else:
    raise AssertionError("infinite repeat reported a finite length")

r = itertools.repeat(object="x", times=3)
assert repr(r) == "repeat('x', 3)"
assert r.__length_hint__() == 3
assert next(r) == "x"
assert r.__length_hint__() == 2
assert list(r) == ["x", "x"]

r = itertools.repeat("x", -10)
assert repr(r) == "repeat('x', 0)"
assert r.__length_hint__() == 0
assert list(r) == []


class Index:
    def __index__(self):
        return 2


assert list(itertools.repeat("y", Index())) == ["y", "y"]

for args in [(), (1, 2, 3), (1, None), (1, 2.0)]:
    try:
        itertools.repeat(*args)
    except TypeError:
        pass
    else:
        raise AssertionError(args)


class RepeatSubclass(itertools.repeat):
    pass


r = RepeatSubclass("z", 2)
assert type(r) is RepeatSubclass
assert repr(r) == "RepeatSubclass('z', 2)"
assert list(r) == ["z", "z"]

# Python 3.14 removed the old concrete __reduce__ slots retained by the
# bundled PyPy 3.11 source.  Generic object reduction therefore rejects
# the exact builtin objects.
assert "__reduce__" not in itertools.count.__dict__
assert "__reduce__" not in itertools.repeat.__dict__
for obj in (itertools.count(), itertools.repeat(None)):
    try:
        obj.__reduce__()
    except TypeError:
        pass
    else:
        raise AssertionError("Python 3.14 non-picklable iterator became picklable")

print("itertools count/repeat Python 3.14 parity ok")
