# CPython-suite gap: the suite does not deopt a hot user-defined __next__ at a
# raising residual after the iterator has advanced.
# parity-tests reason: a pending non-StopIteration exception must propagate
# once without replaying the current state transition.

advances = []


class RaisingIterator:
    def __init__(self, limit, raise_at):
        self.index = 0
        self.limit = limit
        self.raise_at = raise_at

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        advances.append(self.index)
        if self.index == self.raise_at:
            int("not-an-integer")
        return self.index


def consume(limit, raise_at):
    count = 0
    for _ in RaisingIterator(limit, raise_at):
        count += 1
    return count


for _ in range(8):
    assert consume(1600, 2000) == 1600

advances.clear()
errors = 0
k = 1300
try:
    consume(1600, k)
except ValueError:
    errors += 1

assert errors == 1
assert len(advances) == k
print("OK")
