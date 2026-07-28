import gc
import itertools


cwr = itertools.combinations_with_replacement
assert isinstance(cwr, type)
assert "__reduce__" not in cwr.__dict__
assert "__setstate__" not in cwr.__dict__
assert "__sizeof__" in cwr.__dict__

assert list(cwr("ABC", 2)) == [
    ("A", "A"),
    ("A", "B"),
    ("A", "C"),
    ("B", "B"),
    ("B", "C"),
    ("C", "C"),
]
assert list(cwr("AB", 0)) == [()]
assert list(cwr("", 0)) == [()]
assert list(cwr("", 1)) == []
assert list(cwr(iterable="ABC", r=2)) == [
    ("A", "A"),
    ("A", "B"),
    ("A", "C"),
    ("B", "B"),
    ("B", "C"),
    ("C", "C"),
]

# Construction snapshots the input pool, but does not recursively materialize
# all repeated combinations.
large = cwr(range(2), 1000)
assert next(large) == (0,) * 1000
second = next(large)
assert second[:-1] == (0,) * 999
assert second[-1] == 1


class Index:
    def __index__(self):
        return 2


assert list(cwr("AB", Index())) == [
    ("A", "A"),
    ("A", "B"),
    ("B", "B"),
]
try:
    cwr("ABC", -1)
except ValueError as exc:
    assert str(exc) == "r must be non-negative"
else:
    raise AssertionError("negative combinations_with_replacement r accepted")
try:
    cwr("ABC", 1.5)
except TypeError:
    pass
else:
    raise AssertionError("non-index combinations_with_replacement r accepted")

owned = cwr([object(), object()], 2)
gc.collect()
first = next(owned)
second = next(owned)
third = next(owned)
assert first[0] is first[1]
assert first[0] is second[0]
assert second[1] is third[0]
assert third[0] is third[1]
try:
    next(owned)
except StopIteration:
    pass
else:
    raise AssertionError("combinations_with_replacement did not stop")

assert cwr([], 0).__sizeof__() < cwr([], 2).__sizeof__()


class IterOverride(cwr):
    def __iter__(self):
        return iter(((99,),))


class NextOverride(cwr):
    def __next__(self):
        return (88,)


assert list(IterOverride("abc", 2)) == [(99,)]
assert next(NextOverride("abc", 2)) == (88,)

finalized = []


def make_subtype():
    def finalize(self):
        finalized.append("combinations_with_replacement")

    return type(
        "FinalizingCombinationsWithReplacement",
        (cwr,),
        {"__del__": finalize},
    )


subtype = make_subtype()
obj = subtype("ab", 2)
assert type(obj) is subtype
assert next(obj) == ("a", "a")
del obj, subtype
gc.collect()
assert finalized == ["combinations_with_replacement"]

print("OK")
