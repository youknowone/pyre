# pyre-check: gate=1
"""`set ^ set` keeps its receiver alive while the walk runs `__eq__`.

`union` and `difference` copy the receiver before any Python runs, so it is
never held across a hop.  `symmetric_difference` resolves the *operand* first —
which drains an iterable — and only then walks the receiver, and `COMPARE_OP`
popped that receiver off the value stack.  Nothing moves and no answer changes,
so a weakref-able cell placed in the set as its only referrer is what makes it
observable.
"""

import gc
import weakref

REF = None
KEEP = None
SEEN = set()


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]
    gc.collect()
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class Cell:
    pass


def tracked():
    global REF
    c = Cell()
    REF = weakref.ref(c)
    return c


class Probe:
    def __hash__(self):
        return 7

    def __eq__(self, other):
        churn()
        SEEN.add(REF() is not None)
        return False


for _ in range(25):
    print(len({Probe(), tracked()} ^ {Probe()}))
    print(len({Probe(), tracked()}.symmetric_difference([Probe()])))
    print(len(frozenset({Probe(), tracked()}) ^ frozenset({Probe()})))

print("receiver liveness observed:", sorted(SEEN))
