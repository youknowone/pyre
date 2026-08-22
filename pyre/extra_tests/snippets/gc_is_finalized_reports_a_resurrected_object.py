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

# An object no collector tracks is never finalized.
print(gc.is_finalized(3))
