# CPython-suite gap: the suite does not exercise a JIT guard inside an inlined
# user-defined __next__ after an observable effect.
# parity-tests reason: caller-boundary FOR_ITER resume must not apply the
# current iterator step twice when the hot branch changes direction.

effects = []


class SwitchingIterator:
    def __init__(self, limit, switch_at):
        self.index = 0
        self.limit = limit
        self.switch_at = switch_at

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        value = self.index
        self.index += 1
        effects.append(value)
        if value < self.switch_at:
            return value + 1
        return value - 1


def consume(limit, switch_at):
    total = 0
    for value in SwitchingIterator(limit, switch_at):
        total += value
    return total


rounds = 12
limit = 1600
for _ in range(rounds):
    consume(limit, 1200)

assert len(effects) == rounds * limit
print("OK")
