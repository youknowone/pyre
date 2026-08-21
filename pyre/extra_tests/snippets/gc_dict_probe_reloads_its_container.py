# pyre-check: gate=1
"""A dict operation re-reads its receiver after the key's `__hash__`.

Every checked dict entry point holds the dictionary in a native local while it
hashes the key and while it promotes a typed strategy to the object strategy.
Both allocate, and a dictionary still in the nursery moves when they do, so the
probe that follows has to name the address the collector left behind rather
than the one the call started with.

Each operation gets its own loop: the crash needs the receiver to still be
young at the probe, and the surrounding allocation is what decides that, so
sequencing several probes over one dictionary hides them.
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


ROUNDS = 40

for _round in range(ROUNDS):
    numbers = {i: i for i in range(20)}
    print(repr(Key(9) in numbers))

for _round in range(ROUNDS):
    numbers = {i: i for i in range(20)}
    assert numbers.get(Key(9)) is None

for _round in range(ROUNDS):
    numbers = {i: i for i in range(20)}
    assert numbers.get(Key(9), "absent") == "absent"

for _round in range(ROUNDS):
    numbers = {i: i for i in range(20)}
    assert numbers.pop(Key(9), "absent") == "absent"

for _round in range(ROUNDS):
    numbers = {i: i for i in range(20)}
    assert numbers.setdefault(Key(9), "fresh") == "fresh"
    assert numbers[9] == 9

for _round in range(ROUNDS):
    words = {str(i): i for i in range(20)}
    assert words.get(Key(3), "absent") == "absent"
