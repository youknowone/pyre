# pyre-check: gate=1
"""User-key symmetric_difference: equality, a raising probe, a mutating probe.

The hot loop is what the JIT traces. The per-key contains/insert stay residual;
this loop is ordinary Python on top of that merge.
"""


class Key:
    def __init__(self, k):
        self.k = k

    def __hash__(self):
        return hash(self.k)

    def __eq__(self, other):
        return isinstance(other, Key) and self.k == other.k


class Boom:
    def __init__(self, k, explode=False):
        self.k = k
        self.explode = explode

    def __hash__(self):
        return 7

    def __eq__(self, other):
        if self.explode or getattr(other, "explode", False):
            raise ValueError("boom in __eq__")
        return isinstance(other, Boom) and self.k == other.k


class Mut:
    def __init__(self, k, target=None):
        self.k = k
        self.target = target
        self.n = 0

    def __hash__(self):
        return hash(self.k)

    def __eq__(self, other):
        self.n += 1
        if self.target is not None and self.n == 1:
            self.target.clear()
        return isinstance(other, Mut) and self.k == other.k


def sym(a, b):
    return a.symmetric_difference(b)


def keys_of(items):
    return sorted(item.k for item in items)


acc = 0
for i in range(400):
    acc += len(sym({1, 2, 3, i}, {2, 3, 4}))
print("hot", acc)

print("user", keys_of(sym({Key(1), Key(2), Key(3)}, {Key(2), Key(4)})))

try:
    sym({Boom(1), Boom(3)}, {Boom(2, explode=True), Boom(4)})
    print("raise", "swallowed")
except ValueError as exc:
    print("raise", str(exc))


def mutated():
    left = {Mut(1), Mut(2)}
    right = {Mut(2), Mut(3)}
    for item in list(left):
        if item.k == 1:
            item.target = right
    return keys_of(sym(left, right))


print("mutate", mutated())
