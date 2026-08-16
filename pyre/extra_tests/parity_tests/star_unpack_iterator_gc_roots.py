# CPython-suite gap: no suite test collects inside __next__ while a call
# unpacks that iterator.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""Unpacking ``*iterable`` must survive a collection inside ``__next__``.

A star-argument that is neither a tuple nor a list drains through
``unpackiterable``, which grows the argument set one ``next()`` at a time.  A
generator takes that function's ``unpack_into`` fast path and everything else
takes its unknown-length loop, so both branches are exercised here.  Every step
runs Python and collects, moving the accumulator underneath the drain.

The payloads are lists because those relocate; an instance of a Python class is
born at a stable address and would never exercise this.
"""

import gc


def collect_now():
    garbage = [[index] for index in range(4000)]
    assert len(garbage) == 4000
    gc.collect()


class CollectingIterator:
    """Produce ``count`` payloads, forcing a moving collection per step."""

    def __init__(self, count):
        self.remaining = count
        self.produced = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining == 0:
            raise StopIteration
        collect_now()
        item = [self.remaining]
        self.remaining -= 1
        self.produced.append(item)
        return item


def collecting_generator(count):
    """The generator branch of the same unpack."""
    for index in range(count):
        collect_now()
        yield [index]


def sink(*args):
    return args


for width in (1, 2, 3, 8):
    for _ in range(3):
        source = CollectingIterator(width)
        unpacked = sink(*source)
        # A dropped or truncated drain shows up as a short argument tuple.
        assert len(unpacked) == width, (width, len(unpacked))
        # A stale accumulator entry shows up as the wrong object in that slot.
        assert list(unpacked) == source.produced, (width, unpacked)
        assert [item[0] for item in unpacked] == list(range(width, 0, -1))

        yielded = sink(*collecting_generator(width))
        assert len(yielded) == width, (width, len(yielded))
        assert list(yielded) == [[index] for index in range(width)], (width, yielded)

# The same drain backs the list constructor and sequence unpacking.
for width in (1, 2, 3, 8):
    source = CollectingIterator(width)
    assert list(source) == source.produced, width
    assert list(collecting_generator(width)) == [[i] for i in range(width)], width

print("OK")
