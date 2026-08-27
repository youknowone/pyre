# pyre-check: gate=1
"""`gc.get_count`'s second element counts collections of the youngest generation.

3.14's three elements are not the same kind of number: element 0 counts tracked
containers allocated and not yet freed, while elements 1 and 2 count
*collections* -- generation-0 collections since generation 1 was collected, and
generation-1 collections since generation 2 was.  Collecting a generation zeroes
its own count and every younger one, so the counts move under `gc.collect(n)`
without depending on how much the script happened to allocate.

Only element 1 is pinned here.  Element 0 is an allocation count and element 2
needs a generation-1-only collection, and an implementation is free to answer
either differently -- see the `get_count` entry in `module/gc` for what pyre
answers and why.
"""

import gc

counts = gc.get_count()
assert isinstance(counts, tuple), counts
assert len(counts) == 3, counts
assert all(isinstance(value, int) for value in counts), counts

# The default generation is the oldest one, and collecting it zeroes every
# count below it as well.
gc.collect()
assert gc.get_count() == (0, 0, 0), gc.get_count()

# One collection of the youngest generation is one thing for element 1 to
# count.  Reading it back is what says the argument was honoured: a `collect`
# that ignored its generation and ran the oldest one would leave this at zero.
gc.collect(0)
assert gc.get_count()[1] == 1, gc.get_count()

gc.collect(0)
assert gc.get_count()[1] == 2, gc.get_count()

# ...and collecting the oldest generation zeroes it again.
gc.collect()
assert gc.get_count() == (0, 0, 0), gc.get_count()
