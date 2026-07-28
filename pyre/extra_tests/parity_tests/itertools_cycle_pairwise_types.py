import gc
import itertools


assert isinstance(itertools.cycle, type)
assert isinstance(itertools.pairwise, type)
assert list(itertools.pairwise([1, 2, 3])) == [(1, 2), (2, 3)]

cyclic = itertools.cycle([1, 2])
assert [next(cyclic) for _ in range(5)] == [1, 2, 1, 2, 1]

finalized = []


def make_finalizing(name, base, iterable):
    def finalize(self):
        finalized.append(name)

    subtype = type("Finalizing_" + name, (base,), {"__del__": finalize})
    obj = subtype(iterable)
    assert type(obj) is subtype
    return obj


objects = [
    make_finalizing("cycle", itertools.cycle, []),
    make_finalizing("pairwise", itertools.pairwise, []),
]
del objects
gc.collect()

assert sorted(finalized) == ["cycle", "pairwise"]

for constructor in (itertools.cycle, itertools.pairwise):
    try:
        constructor(iterable=[])
    except TypeError:
        pass
    else:
        raise AssertionError("positional-only input accepted as a keyword")

# Python 3.14 permits keywords intended for an overridden subtype __init__,
# while still rejecting a second positional argument in __new__.  This is a
# deliberate delta from the older fixed PyPy pairwise gateway signature.
for constructor in (itertools.cycle, itertools.pairwise):
    class KeywordSubtype(constructor):
        def __init__(self, iterable, *, marker):
            self.marker = marker

    assert KeywordSubtype([], marker=42).marker == 42
    try:
        KeywordSubtype([], 42)
    except TypeError:
        pass
    else:
        raise AssertionError("surplus positional argument accepted")

print("OK")
