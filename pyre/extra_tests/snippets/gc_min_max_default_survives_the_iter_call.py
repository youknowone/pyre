# pyre-check: gate=1
"""`min`/`max` slot their `default` before running the argument's `__iter__`.

`iter(sequence)` runs a `__iter__` written in Python, which is a collection
point.  `default` and `key` are copies taken out of the pinned argument array,
and a minor rewrites the array rather than the copies, so both named pre-move
headers by the time they were pinned.

Pinning a stale word is worse than reading one: the next collection walks the
root slot as if it named an object, which is why this shows up as a bare
segfault rather than a wrong answer.  That also makes a NON-EMPTY iterable
enough to trigger it -- the defective root is published whether or not
`default` is the value that comes back.
"""

import gc

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]
    gc.collect()
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class CollectingIter:
    """Collects from `__iter__`, before `min`/`max` has slotted its default."""

    def __init__(self, items):
        self.items = items

    def __iter__(self):
        churn()
        return iter(self.items)


held = []
for i in range(30):
    held.append(max(CollectingIter([1, 2, 3]), default=[]))
    held.append(min(CollectingIter([1, 2, 3]), default=[]))
    held.append(max(CollectingIter([]), default=[i]))
    held.append(min(CollectingIter([]), default=[i]))

churn()
for value in held:
    if type(value) is list:
        value.append(0)
churn()
print(len(held), held[:4], held[-2:])
