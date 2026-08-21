# pyre-check: gate=1
"""`x in <set>` keeps the set alive across the element's `__hash__`.

`CONTAINS_OP` pops the container off the operand stack before dispatching, so
on a set that is only a temporary nothing refers to it while the element
hashes. A set is an old-gen allocation and never moves, but an unreachable one
is still collected, so it has to be pinned for liveness even though there is
no address to reload.
"""

import gc

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class Key:
    def __init__(self, n):
        self.n = n

    def __hash__(self):
        gc.collect()
        churn()
        return self.n

    def __eq__(self, other):
        return isinstance(other, Key) and other.n == self.n


ROUNDS = 30

for _round in range(ROUNDS):
    print(repr(Key(9) in {i for i in range(20)}))

for _round in range(ROUNDS):
    print(repr(Key(9) in frozenset(range(20))))

for _round in range(ROUNDS):
    print(repr(Key(9) not in {i for i in range(20)}))

# A named receiver is rooted by the frame and was never the failing shape;
# it is here so the pair stays visible.
for _round in range(ROUNDS):
    numbers = {i for i in range(20)}
    assert (Key(9) in numbers) is False
