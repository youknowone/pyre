# pyre-check: gate=1
"""`gc.is_finalized` reports an object whose `__del__` already ran.

A fresh object answers False, so the divergence is only visible once a
finalizer has resurrected its own receiver: the object is reachable again and
the answer has to be True.  The finalizer still runs at most once.
"""

import gc

storage = []


class Lazarus:
    def __del__(self):
        storage.append(self)


lazarus = Lazarus()
print(gc.is_finalized(lazarus))

del lazarus
gc.collect()

lazarus = storage.pop()
print(gc.is_finalized(lazarus))

# The finalizer does not run a second time for the resurrected object.
del lazarus
gc.collect()
print(len(storage))

# A generator's finalizer is the collector's too: `close` counts as the run, so
# a generator that resurrects itself from a `finally` block reads back the same
# way.  The cycle is generator -> frame -> box -> generator, so only the
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
print(gc.is_finalized(gen))

del gen, box
gc.collect()

gen = storage.pop()
print(gc.is_finalized(gen))

# An object no collector tracks is never finalized.
print(gc.is_finalized(3))
