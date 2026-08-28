# CPython-suite gap: the suite does not exhaust a hot user iterator through a
# nested Python frame below __next__.
# parity-tests reason: a deep StopIteration must reach the caller FOR_ITER
# handler instead of leaking from an inlined callee chain.


class DelegatingIterator:
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


def consume(limit):
    count = 0
    for value in DelegatingIterator(limit):
        assert value == count + 1
        count += 1
    return count


for _ in range(12):
    assert consume(1600) == 1600
print("OK")
