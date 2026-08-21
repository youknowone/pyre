# pyre-check: gate=1
"""`list +=` and `dict |=` hand back the receiver at its current address.

Both slots delegate to the shared update helper and then return the receiver
out of the gateway's stack copy of the arguments.  The helper runs user code —
a `__next__` for the list, `keys`/`__getitem__`/`__hash__` for the dict — and a
minor rewrites the shadow slots rather than that copy, so what came back was a
pre-move word.  The interpreter then stores it into the assignment target,
which is how a root ends up pointing at a stale header.

`extend` and `update` are unaffected: they answer `None`, so there is no
receiver to hand back.  That is why only the operator forms fail.
"""

import gc

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]
    gc.collect()
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class Counting:
    """An iterator whose `__next__` collects between yields."""

    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        churn()
        if self.n > 3:
            raise StopIteration
        return self.n


class Mapping:
    """A mapping whose `keys` and `__getitem__` collect."""

    def keys(self):
        churn()
        return ["a", "b"]

    def __getitem__(self, key):
        churn()
        return 1


for _ in range(25):
    seq = [1, 2]
    seq += Counting()
    print(len(seq), seq)

    mapping = {1: 2}
    mapping |= Mapping()
    print(len(mapping), sorted(map(str, mapping)))

    # The method forms answer None and take a different route to the same
    # helper; they are here so a regression in the helper itself shows up too.
    seq2 = [1, 2]
    seq2.extend(Counting())
    mapping2 = {1: 2}
    mapping2.update(Mapping())
    print(len(seq2), len(mapping2))
