# pyre-check: gate=1
"""set.update/union/difference_update read every operand once, up front.

Each operand is drained through a user `__hash__`/`__iter__`, so an operand
still waiting its turn must be reachable afterwards rather than re-read from
whatever the call started with.
"""


class H:
    def __init__(self, n):
        self.n = n

    def __hash__(self):
        for _ in range(200):
            [0] * 64
        return self.n

    def __eq__(self, other):
        return isinstance(other, H) and other.n == self.n


for _ in range(50):
    first = [H(i) for i in range(8)]
    second = [H(100 + i) for i in range(8)]

    s = set()
    s.update(first, second)
    assert len(s) == 16, len(s)

    u = set().union(first, second)
    assert len(u) == 16, len(u)

    d = set(first) | set(second)
    d.difference_update(first, second)
    assert len(d) == 0, len(d)
