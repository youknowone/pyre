import gc
import itertools


assert isinstance(itertools.combinations, type)
assert "__reduce__" not in itertools.combinations.__dict__
assert "__setstate__" not in itertools.combinations.__dict__
assert "__sizeof__" in itertools.combinations.__dict__

assert list(itertools.combinations("ABCD", 3)) == [
    ("A", "B", "C"),
    ("A", "B", "D"),
    ("A", "C", "D"),
    ("B", "C", "D"),
]
assert list(itertools.combinations("AB", 0)) == [()]
assert list(itertools.combinations("AB", 3)) == []
assert list(itertools.combinations(iterable="ABC", r=2)) == [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
]

# Construction snapshots the input pool, but does not recursively materialize
# all C(n, r) outputs.
large = itertools.combinations(range(1000), 2)
assert next(large) == (0, 1)
assert next(large) == (0, 2)


class Index:
    def __index__(self):
        return 2


assert list(itertools.combinations("ABC", Index())) == [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
]
try:
    itertools.combinations("ABC", -1)
except ValueError as exc:
    assert str(exc) == "r must be non-negative"
else:
    raise AssertionError("negative combinations r accepted")
try:
    itertools.combinations("ABC", 1.5)
except TypeError:
    pass
else:
    raise AssertionError("non-index combinations r accepted")

# The pool contents and mutable index/result lists are retained solely by the
# W_Combinations descriptor across a moving collection.
owned = itertools.combinations([object(), object(), object()], 2)
gc.collect()
first = next(owned)
second = next(owned)
third = next(owned)
assert first[0] is second[0]
assert first[1] is third[0]
try:
    next(owned)
except StopIteration:
    pass
else:
    raise AssertionError("combinations did not stop")

assert itertools.combinations([], 0).__sizeof__() < itertools.combinations([], 2).__sizeof__()


class IterOverride(itertools.combinations):
    def __iter__(self):
        return iter(((99,),))


class NextOverride(itertools.combinations):
    def __next__(self):
        return (88,)


assert list(IterOverride("abc", 2)) == [(99,)]
assert next(NextOverride("abc", 2)) == (88,)

finalized = []


def make_subtype():
    def finalize(self):
        finalized.append("combinations")

    return type(
        "FinalizingCombinations",
        (itertools.combinations,),
        {"__del__": finalize},
    )


subtype = make_subtype()
obj = subtype("abc", 2)
assert type(obj) is subtype
assert next(obj) == ("a", "b")
del obj, subtype
gc.collect()
assert finalized == ["combinations"]

print("OK")
