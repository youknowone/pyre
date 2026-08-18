# `id()` must survive the object moving.  A minor collection copies a surviving
# nursery object to the old generation, so an `id()` that reports the current
# address hands out a different value before and after — and every id()-keyed
# memo in the pure-Python stdlib (`pickle._Pickler.memo`, `copy.deepcopy`'s
# `memo`, `json.encoder`'s `markers`) then misses on a collection landing
# mid-walk, emitting a shared or recursive reference twice so it round-trips as
# two distinct objects.  `incminimark.py id_or_identityhash` answers a nursery
# object with its shadow — the old-gen address the next minor collection will
# copy it into — and an out-of-nursery object with its own address, which does
# not move under mark-sweep.
#
# The payload must be lists: instances never move here, so an instance-based
# workload reports a stable id even when the mechanism is missing.  200 lists
# are held live across 40 rounds of nursery churn plus a full collection, which
# is enough to move every one of them.

import gc


live = [[i] for i in range(200)]
before = [id(obj) for obj in live]

for _ in range(40):
    junk = [[j] for j in range(2000)]
    del junk

gc.collect()
after = [id(obj) for obj in live]
changed = sum(1 for old, new in zip(before, after) if old != new)
assert changed == 0, "id changed for %d/200 objects" % changed

print("OK")
