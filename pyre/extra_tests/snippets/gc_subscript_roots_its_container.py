# pyre-check: gate=1
"""Item access keeps its container reachable across the key's `__index__`.

`BINARY_SUBSCR` / `STORE_SUBSCR` / `DELETE_SUBSCR` pop every operand before
the dispatch reaches the by-layout arm, so a container the operand stack was
the only holder of has nothing rooting it while `__index__` runs Python.  A
list is nursery-allocated on top of that: a collection there moves it, and the
address the arm started with is the pre-move one.
"""

import gc

KEEP = None


def churn():
    """Take the freed boxes, so a sweep that frees is not invisible."""
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


class Idx:
    def __init__(self, n):
        self.n = n

    def __index__(self):
        gc.collect()
        churn()
        return self.n


# Temporaries: nothing but the operand stack ever held these containers.
assert [10, 20, 30][Idx(1)] == 20
assert (10, 20, 30)[Idx(1)] == 20
assert "abcd"[Idx(1)] == "b"
assert b"abcd"[Idx(1)] == 98
assert bytearray(b"abcd")[Idx(1)] == 98
assert range(10, 20)[Idx(1)] == 11
assert list(range(8))[Idx(2) :] == [2, 3, 4, 5, 6, 7]
assert bytearray(b"abcdefgh")[Idx(2) : Idx(6)] == bytearray(b"cdef")

# The assigned value is read after the key's `__index__` has already run.
target = bytearray(b"abcd")
target[Idx(1)] = 65
assert target == bytearray(b"aAcd"), target

held = [10, 20, 30]
held[Idx(1)] = 99
assert held == [10, 99, 30], held

held = [10, 20, 30]
del held[Idx(1)]
assert held == [10, 30], held

held = list(range(12))
del held[Idx(2) : Idx(8)]
assert held == [0, 1, 8, 9, 10, 11], held

held = list(range(12))
del held[Idx(2) : Idx(10) : Idx(3)]
assert held == [0, 1, 3, 4, 6, 7, 9, 10, 11], held

held = bytearray(b"abcdefgh")
held[Idx(2) : Idx(4)] = b"ZZZ"
assert held == bytearray(b"abZZZefgh"), held
