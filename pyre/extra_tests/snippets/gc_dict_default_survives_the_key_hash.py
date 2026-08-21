# pyre-check: gate=1
"""`dict.get` / `dict.pop` return their default after hashing the key.

Hashing is user code, and the default is only consumed once it has run, so a
default that is itself a list or a dict has to come back from the root stack
rather than from the argument copy the call started with.
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


for _round in range(20):
    numbers = {i: i for i in range(20)}

    assert numbers.get(Key(7), [1, 2, 3]) == [1, 2, 3]
    assert numbers.get(Key(7), {"a": 1}) == {"a": 1}
    assert numbers.pop(Key(7), [4, 5, 6]) == [4, 5, 6]
    assert numbers.pop(Key(7), {"b": 2}) == {"b": 2}

    fallback = [7, 8, 9]
    assert numbers.get(Key(7), fallback) is fallback
    assert numbers.setdefault(Key(7), [1]) == [1]
    assert numbers[Key(7)] == [1]

    numbers[Key(9)] = [3, 2, 1]
    assert numbers.get(Key(9), None) == [3, 2, 1]
