import gc
import itertools


assert isinstance(itertools.permutations, type)
assert "__reduce__" not in itertools.permutations.__dict__
assert "__setstate__" not in itertools.permutations.__dict__
assert "__sizeof__" in itertools.permutations.__dict__

assert list(itertools.permutations("ABC", 2)) == [
    ("A", "B"),
    ("A", "C"),
    ("B", "A"),
    ("B", "C"),
    ("C", "A"),
    ("C", "B"),
]
assert list(itertools.permutations("ABC")) == [
    ("A", "B", "C"),
    ("A", "C", "B"),
    ("B", "A", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("C", "B", "A"),
]
assert list(itertools.permutations("AB", None)) == [("A", "B"), ("B", "A")]
assert list(itertools.permutations("AB", 0)) == [()]
assert list(itertools.permutations("", 0)) == [()]
assert list(itertools.permutations("AB", 3)) == []
assert list(itertools.permutations(iterable="ABC", r=2)) == [
    ("A", "B"),
    ("A", "C"),
    ("B", "A"),
    ("B", "C"),
    ("C", "A"),
    ("C", "B"),
]

# Snapshot only the input pool; do not recursively materialize P(n, r).
large = itertools.permutations(range(1000), 2)
assert next(large) == (0, 1)
assert next(large) == (0, 2)


class IndexOnly:
    def __index__(self):
        return 2


try:
    itertools.permutations("ABC", IndexOnly())
except TypeError as exc:
    assert str(exc) == "Expected int as r"
else:
    raise AssertionError("permutations accepted an index-only r")


class IntSubclass(int):
    pass


assert next(itertools.permutations("ABC", IntSubclass(2))) == ("A", "B")

try:
    itertools.permutations("ABC", -1)
except ValueError as exc:
    assert str(exc) == "r must be non-negative"
else:
    raise AssertionError("negative permutations r accepted")

# CPython creates the pool before checking the object supplied as r.
events = []


class Source:
    def __iter__(self):
        events.append("iter")
        return iter("AB")


try:
    itertools.permutations(Source(), IndexOnly())
except TypeError:
    pass
else:
    raise AssertionError("invalid permutations r accepted")
assert events == ["iter"]

owned = itertools.permutations([object(), object(), object()], 2)
gc.collect()
first = next(owned)
second = next(owned)
third = next(owned)
assert first[0] is second[0]
assert first[0] is third[1]
assert first[1] is third[0]

assert itertools.permutations([], 0).__sizeof__() < itertools.permutations(
    range(3), 2
).__sizeof__()


class IterOverride(itertools.permutations):
    def __iter__(self):
        return iter(((99,),))


class NextOverride(itertools.permutations):
    def __next__(self):
        return (88,)


assert list(IterOverride("abc", 2)) == [(99,)]
assert next(NextOverride("abc", 2)) == (88,)

finalized = []


def make_subtype():
    def finalize(self):
        finalized.append("permutations")

    return type(
        "FinalizingPermutations",
        (itertools.permutations,),
        {"__del__": finalize},
    )


subtype = make_subtype()
obj = subtype("abc", 2)
assert type(obj) is subtype
assert next(obj) == ("a", "b")
del obj, subtype
gc.collect()
assert finalized == ["permutations"]

print("OK")
