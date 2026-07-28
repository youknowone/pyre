import gc
import itertools


assert isinstance(itertools.groupby, type)
assert isinstance(itertools._grouper, type)
assert "__reduce__" not in itertools.groupby.__dict__
assert "__setstate__" not in itertools.groupby.__dict__
assert "__reduce__" not in itertools._grouper.__dict__

grouped = itertools.groupby("AAABBC")
key_a, group_a = next(grouped)
assert key_a == "A"
assert list(group_a) == ["A", "A", "A"]
key_b, group_b = next(grouped)
assert key_b == "B"
assert list(group_b) == ["B", "B"]
key_c, group_c = next(grouped)
assert key_c == "C"
assert list(group_c) == ["C"]
try:
    next(grouped)
except StopIteration:
    pass
else:
    raise AssertionError("groupby did not stop")

assert [(key, list(group)) for key, group in itertools.groupby([1, 1, 2])] == [
    (1, [1, 1]),
    (2, [2]),
]
assert [
    (key, list(group))
    for key, group in itertools.groupby(
        iterable=["a", "A", "b"],
        key=str.lower,
    )
] == [("a", ["a", "A"]), ("b", ["b"])]

# Advancing the parent invalidates the previous child iterator.
parent = itertools.groupby("AABB")
_, old_group = next(parent)
_, new_group = next(parent)
assert list(old_group) == []
assert list(new_group) == ["B", "B"]

# Construction and grouping are lazy: only iter() happens at construction.
events = []


class Source:
    def __init__(self):
        self.values = iter((10, 10))

    def __iter__(self):
        events.append("iter")
        return self

    def __next__(self):
        events.append("next")
        return next(self.values)


lazy = itertools.groupby(Source())
assert events == ["iter"]
key, child = next(lazy)
assert key == 10
assert events == ["iter", "next"]
assert next(child) == 10
assert events == ["iter", "next"]
assert next(child) == 10
assert events == ["iter", "next", "next"]

# The child strongly retains the parent and shared source cursor across GC.
owned_parent = itertools.groupby([object(), object()], key=lambda value: 1)
owned_key, owned_child = next(owned_parent)
assert owned_key == 1
del owned_parent
gc.collect()
owned_values = list(owned_child)
assert len(owned_values) == 2
assert owned_values[0] is not owned_values[1]

# CPython 3.14 exposes the internal final type.
try:
    class BadGrouper(itertools._grouper):
        pass
except TypeError:
    pass
else:
    raise AssertionError("_grouper unexpectedly accepted subclassing")


class IterOverride(itertools.groupby):
    def __iter__(self):
        return iter((("override", iter((99,))),))


class NextOverride(itertools.groupby):
    def __next__(self):
        return ("override", iter((88,)))


assert [(key, list(group)) for key, group in IterOverride("abc")] == [
    ("override", [99])
]
key, group = next(NextOverride("abc"))
assert key == "override"
assert list(group) == [88]

finalized = []


def make_subtype():
    def finalize(self):
        finalized.append("groupby")

    return type(
        "FinalizingGroupBy",
        (itertools.groupby,),
        {"__del__": finalize},
    )


subtype = make_subtype()
obj = subtype("aa")
assert type(obj) is subtype
key, child = next(obj)
assert key == "a"
assert type(child) is itertools._grouper
del child, obj, subtype
gc.collect()
assert finalized == ["groupby"]

print("OK")
