# pyre-check: gate=1
"""`gc.is_finalized` reports an object whose own finalizer already ran.

A fresh object answers False, so the divergence is only visible once a
finalizer has resurrected its own receiver: the object is reachable again and
the answer has to be True.  The finalizer still runs at most once.

Only an object that resurrects *itself* can observe this.  A finalizer that
resurrects some *other* member of the same garbage set is not a test of the
flag: incminimark keeps a finalizer-reachable object alive for the collection
that queued it, so the other member is never finalized at all, and answering
False for it is correct.
"""

import gc

storage = []


class Lazarus:
    def __del__(self):
        storage.append(self)


lazarus = Lazarus()
assert gc.is_finalized(lazarus) is False

del lazarus
gc.collect()

lazarus = storage.pop()
assert gc.is_finalized(lazarus) is True

# The finalizer does not run a second time for the resurrected object.
del lazarus
gc.collect()
assert storage == [], storage


# A generator's finalizer is the collector's too: closing it counts as the run,
# so a generator that resurrects itself from a `finally` block reads back the
# same way.  The cycle is generator -> frame -> box -> generator, so only the
# collector can reach it.
def resurrect(box):
    try:
        yield 1
    finally:
        storage.append(box[0])


box = []
gen = resurrect(box)
next(gen)
box.append(gen)
assert gc.is_finalized(gen) is False

del gen, box
gc.collect()

gen = storage.pop()
assert gc.is_finalized(gen) is True

# An object no collector tracks is never finalized.
assert gc.is_finalized(3) is False
