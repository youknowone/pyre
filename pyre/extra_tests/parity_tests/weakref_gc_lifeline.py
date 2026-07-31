"""Weakref lifelines remain owned by the GC across collection.

PyPy stores each rweakref inside the GC-traced W_Weakref/WeakrefLifeline
object.  Fill the nursery while both the referents and their weakref objects
are live, then finalize them together; callbacks allocate while the detached
lifeline callback path is active.
"""

import gc
import weakref


class Target:
    pass


callback_count = 0


def callback(dead):
    global callback_count
    assert dead() is None
    callback_count += 1
    # Allocations during a callback can fill the nursery; the detached
    # lifeline must not be revisited by an intervening collection.
    [Target() for _ in range(8)]


count = 2_000
targets = [Target() for _ in range(count)]
refs = [weakref.ref(target, callback) for target in targets]

# Keep every weakref object alive while their referents die as one group.
del targets
gc.collect()

assert callback_count == count
assert all(ref() is None for ref in refs)

print("OK")
