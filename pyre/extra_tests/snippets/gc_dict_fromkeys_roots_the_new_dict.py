# pyre-check: gate=1
"""`dict.fromkeys` builds its dict before it drains the iterable.

Draining runs the iterable's own Python, and a dict is nursery-allocated, so
the dict has to be rooted for that window and read back afterwards — a store
into the pre-move address hands the write barrier an object that is no longer
there.
"""

import gc

KEEP = None


def churn():
    global KEEP
    KEEP = [[i] * 24 for i in range(60)] + [bytearray(b"Q" * 96) for _ in range(30)]


def keys(items):
    for item in items:
        gc.collect()
        churn()
        yield item



class Sub(dict):
    pass


for _round in range(30):
    built = dict.fromkeys(keys(["a", "b", "c"]))
    assert sorted(built) == ["a", "b", "c"], built
    assert all(value is None for value in built.values()), built

    built = dict.fromkeys(keys([1, 2, 3]), "v")
    assert built == {1: "v", 2: "v", 3: "v"}, built


    built = Sub.fromkeys(keys([1, 2, 3]), 0)
    assert isinstance(built, Sub) and built == {1: 0, 2: 0, 3: 0}, built
