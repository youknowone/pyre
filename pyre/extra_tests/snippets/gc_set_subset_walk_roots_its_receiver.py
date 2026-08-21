# pyre-check: gate=1
"""A set subset walk keeps its receiver alive across an element's `__eq__`.

`set_is_subset_of` probes with the digest each element was stored under, so
the walk hashes nothing — but a bucket match still runs the elements'
`__eq__`, and `COMPARE_OP` popped both operands before dispatching.  A set is
old-gen and never moves, so nothing crashes and no answer changes: the
receiver is simply collected while the walk is still using it.

A crash probe cannot see that, because it needs the freed memory reused before
the dangling read.  This asks the collector instead — a weakref-able cell is
placed inside the set as its only referrer, so if the set dies during the hop
the weakref clears.
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
    """A fresh cell whose only referrer will be the set under test."""
    global REF
    c = Cell()
    REF = weakref.ref(c)
    return c


class Probe:
    """Collects, then records the receiver's liveness from inside the hop."""

    def __hash__(self):
        return 7

    def __eq__(self, other):
        churn()
        SEEN.add(REF() is not None)
        return False


# `<=` takes the subset walk whatever the lengths are, while `==`
# short-circuits on the length the tracking cell changes.  Which element the
# walk reaches first — and so whether `__eq__` runs at all for a given form —
# follows set iteration order, so the observations are pooled across every form
# and every round and reported once.  What is asserted is that no form ever
# reaches `__eq__` with the receiver already collected, not how often any
# particular one gets there.
for _ in range(20):
    print(frozenset({Probe(), tracked()}) <= {Probe(), 1, 2})
    print({Probe(), tracked()} <= {Probe(), 1, 2})
    print({Probe(), tracked()}.issubset([Probe(), 1, 2]))
    print({Probe(), tracked()} == {Probe(): 1, 2: 2}.keys())

print("receiver liveness observed during the subset walks:", sorted(SEEN))
