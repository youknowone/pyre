# pyre-check: gate=1
"""`gc.get_count` answers with the minor collections run since the last major.

3.14's three elements are not the same kind of number: element 0 counts tracked
containers allocated and not yet freed, while elements 1 and 2 count
*collections* -- generation-0 collections since generation 1 was collected, and
generation-1 collections since generation 2 was.  Collecting a generation zeroes
its own count and every younger one.

Under the generation mapping this module already publishes, a minor collection
is the generation-0 one and a full collection covers both older generations at
once.  So element 1 is the minors since the last full collection, element 2 is
exactly zero because no generation-1-only collection exists here, and element 0
is zero because the allocation seam carries no tracked-container count -- see
the `get_count` entry in `module/gc` for why one cannot be maintained.
"""

import gc

counts = gc.get_count()
assert isinstance(counts, tuple), counts
assert len(counts) == 3, counts
assert all(isinstance(value, int) for value in counts), counts

# A full collection is the generation-2 collection, which zeroes all three.
gc.collect()
assert gc.get_count() == (0, 0, 0), gc.get_count()

# Allocating past the nursery runs minors.  The objects are kept so that the
# allocation escapes: a value the loop drops can be removed outright, and then
# no collection is reached at all.  Sampling rather than testing every
# iteration keeps this to a handful of extra calls.
keep = []
saw_minor = False
for i in range(200000):
    keep.append((i, i))
    if i % 1000 == 0:
        counts = gc.get_count()
        assert counts[0] == 0, counts
        assert counts[2] == 0, counts
        if counts[1] > 0:
            saw_minor = True
            break

assert saw_minor, gc.get_count()

# ...and the next full collection zeroes it again.
gc.collect()
assert gc.get_count() == (0, 0, 0), gc.get_count()
