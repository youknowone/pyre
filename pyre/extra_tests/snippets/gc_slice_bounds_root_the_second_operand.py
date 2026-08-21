# pyre-check: gate=1
"""A start/stop pair is converted one bound at a time.

The first bound's `__index__` runs Python, so the second must be reachable
across it.  A window call that supplies only `start` is the sharp case: the
implicit `stop` is a freshly built int nothing else holds.
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


for _round in range(30):
    numbers = list(range(40))
    assert numbers.index(5, Idx(2)) == 5
    assert numbers.index(5, Idx(2), Idx(30)) == 5

    pairs = tuple(range(40))
    assert pairs.index(5, Idx(2)) == 5

    text = "abcdefgh" * 4
    assert text.find("c", Idx(2)) == 2
    assert text.count("a", Idx(2)) == 3
    assert text.startswith("c", Idx(2)) is True
    assert text.rfind("a", Idx(2)) == 24

    raw = b"abcdefgh" * 4
    assert raw.find(b"c", Idx(2)) == 2
    assert raw.count(b"a", Idx(2)) == 3
    assert raw.endswith(b"h", Idx(2)) is True

    buf = bytearray(b"abcdefgh" * 4)
    assert buf.find(b"c", Idx(2)) == 2
