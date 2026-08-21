# pyre-check: gate=1
"""`list.remove` addresses the live receiver after its equality scan.

The scan runs the elements' `__eq__`, which is a collection point, and a
`W_ListObject` header moves.  The scan helper pins and reloads for its own
loop, but that scope drops when it returns, so the length read and the pop
below it addressed a pre-move header.

The element has to actually match: with an `__eq__` that always reports unequal
the scan raises before the pop is ever reached, and the defective line does not
run.
"""

import gc

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]
    gc.collect()
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class Match:
    """Collects, then reports equal, so `remove` reaches its pop."""

    def __eq__(self, other):
        churn()
        return True


for _ in range(30):
    victim = [1, 2, 3, 4, 5]
    victim.remove(Match())
    print(len(victim), victim)
