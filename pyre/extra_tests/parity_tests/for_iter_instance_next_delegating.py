# CPython-suite gap: the suite does not run a hot delegating __next__ whose
# exhaustion arrives from a residual call rather than an explicit raise.
# parity-tests reason: FOR_ITER owns the StopIteration-to-exhaustion mapping,
# so a callee that merely propagates it must still end the loop.


class Wrap:
    def __init__(self, inner):
        self._it = inner

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


def consume(n):
    total = 0
    for value in Wrap(iter(range(n))):
        total += value
    return total


expected = 20000 * 19999 // 2
for _ in range(12):
    assert consume(20000) == expected


# The same shape one level deeper: the inner iterator is itself a wrapper, so
# the exhaustion propagates through two Python __next__ frames.
class Doubled:
    def __init__(self, inner):
        self._it = inner

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


def consume_nested(n):
    count = 0
    for _ in Doubled(Wrap(iter(range(n)))):
        count += 1
    return count


for _ in range(12):
    assert consume_nested(4000) == 4000


# A delegating __next__ that also raises its own StopIteration on a sentinel
# must end the loop at the sentinel, not at the inner iterator's exhaustion.
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
