# CPython-suite gap: no suite test drives one constructor hot enough to inline.
# parity-tests reason: this guards the replay safety of an inlined `__init__`.

"""An inlined `__init__` may write only the instance the call just allocated.

A guard inside an inlined constructor resumes at the caller's CALL and re-runs
the instantiation, so a store the first attempt made must not survive.  That
holds for `self.x = v` on the fresh instance and for nothing else, so the three
shapes below have to keep answering the same as the interpreter: the plain
constructor, a method mutating a receiver that outlives the call, and a
constructor that publishes `self` before it stores.
"""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

ROUNDS = 3000


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.total = x + y


total = 0
for i in range(ROUNDS):
    p = Point(i, i + 1)
    assert p.x == i, (i, p.x)
    assert p.y == i + 1, (i, p.y)
    assert p.total == 2 * i + 1, (i, p.total)
    total += p.total
assert total == sum(2 * i + 1 for i in range(ROUNDS)), total


class Counter:
    def __init__(self):
        self.n = 0

    def bump(self, by):
        # The receiver outlives the call, so this write is NOT an
        # initialization: re-running it after a rewind would double it.
        self.n += by
        return self.n


c = Counter()
for i in range(ROUNDS):
    c.bump(1)
assert c.n == ROUNDS, c.n

acc = Counter()
seen = []
for i in range(ROUNDS):
    seen.append(acc.bump(2))
assert acc.n == 2 * ROUNDS, acc.n
assert seen == list(range(2, 2 * ROUNDS + 2, 2)), seen[:4]

REGISTRY = []


class Published:
    def __init__(self, v):
        # `self` escapes before the store, so a rewind no longer discards it.
        REGISTRY.append(self)
        self.v = v


for i in range(ROUNDS):
    Published(i)
assert len(REGISTRY) == ROUNDS, len(REGISTRY)
assert [o.v for o in REGISTRY[:5]] == [0, 1, 2, 3, 4], [o.v for o in REGISTRY[:5]]
assert REGISTRY[-1].v == ROUNDS - 1, REGISTRY[-1].v

print("OK")
