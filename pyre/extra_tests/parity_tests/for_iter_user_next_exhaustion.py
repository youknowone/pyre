# CPython-suite gap: the suite does not exhaust hot user iterators through
# StopIteration subclasses or nested Python `__next__` frames.
# parity-tests reason: this guards pyre's inlined FOR_ITER exhaustion mapping
# across explicit, deep, and residual StopIteration paths.

"""User-defined iterator exhaustion reaches the owning FOR_ITER exactly once."""


class MyStop(StopIteration):
    pass


class SubclassStoppingIterator:
    def __init__(self, limit):
        self.index = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise MyStop
        self.index += 1
        return self.index


class DeepStoppingIterator:
    def __init__(self, limit):
        self.index = 0
        self.limit = limit

    def __iter__(self):
        return self

    def _advance(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        return self.index

    def __next__(self):
        return self._advance()


def count_explicit(iterator):
    count = 0
    for value in iterator:
        assert value == count + 1
        count += 1
    return count


for _ in range(12):
    assert count_explicit(SubclassStoppingIterator(1600)) == 1600
    assert count_explicit(DeepStoppingIterator(1600)) == 1600


class Wrap:
    def __init__(self, inner):
        self._it = inner

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


def sum_wrapped(n):
    total = 0
    for value in Wrap(iter(range(n))):
        total += value
    return total


expected = 20000 * 19999 // 2
for _ in range(12):
    assert sum_wrapped(20000) == expected


def count_nested(n):
    count = 0
    for _ in Wrap(Wrap(iter(range(n)))):
        count += 1
    return count


for _ in range(12):
    assert count_nested(4000) == 4000


class StopEarly:
    def __init__(self, inner, stop_at):
        self._it = inner
        self._stop_at = stop_at
        self._seen = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self._it)
        self._seen += 1
        if value == self._stop_at:
            raise StopIteration
        return value


for _ in range(12):
    early = StopEarly(iter(range(20000)), 1500)
    assert sum(1 for _ in early) == 1500
    assert early._seen == 1501

print("OK")
