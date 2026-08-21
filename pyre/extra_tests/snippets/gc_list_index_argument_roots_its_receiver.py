# pyre-check: gate=1
"""A list method reads its receiver back after the index argument runs Python.

The gateway hands the body a stack copy of the arguments.  A minor collection
rewrites the shadow-stack slots, not that copy, so the receiver read out of it
after an `__index__` dispatch is a pre-move address.
"""

import gc

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class Idx:
    def __init__(self, n):
        self.n = n

    def __index__(self):
        gc.collect()
        churn()
        return self.n


numbers = list(range(8))
assert numbers.pop(Idx(2)) == 2
assert numbers == [0, 1, 3, 4, 5, 6, 7], numbers

numbers.insert(Idx(2), 99)
assert numbers == [0, 1, 99, 3, 4, 5, 6, 7], numbers

numbers = [1, 2, 3]
numbers *= Idx(2)
assert numbers == [1, 2, 3, 1, 2, 3], numbers

assert [1, 2, 3] * Idx(2) == [1, 2, 3, 1, 2, 3]
assert Idx(2) * [1, 2, 3] == [1, 2, 3, 1, 2, 3]

buf = bytearray(b"abcdefgh")
assert buf.pop(Idx(2)) == 99
buf.insert(Idx(2), 67)
assert buf == bytearray(b"abCdefgh"), buf
