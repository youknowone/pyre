import gc
import itertools


assert isinstance(itertools.batched, type)
assert "__reduce__" not in itertools.batched.__dict__
assert "__setstate__" not in itertools.batched.__dict__


class Source:
    def __init__(self):
        self.value = 0
        self.pulls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.value == 7:
            raise StopIteration
        value = self.value
        self.value += 1
        self.pulls += 1
        return value


# CPython's batchedobject owns the source and fills only the batch requested
# by each __next__ call.
source = Source()
batches = itertools.batched(source, 3)
assert source.pulls == 0
assert iter(batches) is batches
assert next(batches) == (0, 1, 2)
assert source.pulls == 3
assert next(batches) == (3, 4, 5)
assert source.pulls == 6
assert next(batches) == (6,)
assert source.pulls == 7
try:
    next(batches)
except StopIteration:
    pass
else:
    raise AssertionError("exhausted batched iterator produced another batch")

assert list(itertools.batched([], 2)) == []
assert list(itertools.batched(range(6), 2)) == [(0, 1), (2, 3), (4, 5)]
assert list(itertools.batched(iterable=range(5), n=2)) == [(0, 1), (2, 3), (4,)]
for args, kwargs in [
    (([], 2, False), {}),
    (([], 2), {"n": 3}),
    (([], 2), {"unknown": True}),
]:
    try:
        itertools.batched(*args, **kwargs)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid batched call shape accepted")

strict = itertools.batched(range(5), 2, strict=True)
assert next(strict) == (0, 1)
assert next(strict) == (2, 3)
try:
    next(strict)
except ValueError as exc:
    assert str(exc) == "batched(): incomplete batch"
else:
    raise AssertionError("strict batched accepted an incomplete batch")
try:
    next(strict)
except StopIteration:
    pass
else:
    raise AssertionError("strict batched did not latch exhaustion")


class Index:
    def __index__(self):
        return 2


assert list(itertools.batched(range(3), Index())) == [(0, 1), (2,)]
for n in (0, -1):
    try:
        itertools.batched([], n)
    except ValueError as exc:
        assert str(exc) == "n must be at least one"
    else:
        raise AssertionError("invalid batch size accepted")


class Broken:
    def __init__(self):
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls == 1:
            return 1
        if self.calls == 2:
            raise RuntimeError("broken source")
        return 2


broken = itertools.batched(Broken(), 2)
try:
    next(broken)
except RuntimeError as exc:
    assert str(exc) == "broken source"
else:
    raise AssertionError("source exception was swallowed")
try:
    next(broken)
except StopIteration:
    pass
else:
    raise AssertionError("source exception did not latch exhaustion")

# The source is reachable solely through W_Batched across a moving collection.
owned_source = itertools.batched(iter(range(5)), 2)
gc.collect()
assert list(owned_source) == [(0, 1), (2, 3), (4,)]


class IterOverride(itertools.batched):
    def __iter__(self):
        return iter(((99,),))


class NextOverride(itertools.batched):
    def __next__(self):
        return (88,)


assert list(IterOverride([], 2)) == [(99,)]
assert next(NextOverride([], 2)) == (88,)

finalized = []


def make_subtype():
    def finalize(self):
        finalized.append("batched")

    return type("FinalizingBatched", (itertools.batched,), {"__del__": finalize})


subtype = make_subtype()
obj = subtype(range(3), 2)
assert type(obj) is subtype
assert next(obj) == (0, 1)
del obj, subtype
gc.collect()
assert finalized == ["batched"]

print("OK")
