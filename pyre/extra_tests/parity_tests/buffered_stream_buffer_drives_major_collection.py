# CPython-suite gap: test_bz2's testOpenDel drops 10000 BZ2File objects, but
# reference counting closes each one at its `del`, so the suite never asks
# whether dropped buffered streams are reclaimed without an explicit collect.
# parity-tests reason: the buffered streams' bytearray buffer lives outside
# the GC heap, and only its accounting toward the major-collection threshold
# (`rgc.add_memory_pressure`) lets a loop of dropped streams reach a major.

"""Dropped buffered streams must not outlive the descriptor table.

Each `open` below allocates a 1 MiB buffer and nothing else of size.  With
the buffer charged to the major-collection threshold, a major runs every few
dozen iterations and the finalizers close the dropped files; uncharged, the
loop allocates too little GC memory to reach one, and the open files pile up
until the descriptor table is exhausted.
"""

import os
import weakref

path = os.path.abspath(__file__)
refs = []
peak = 0
for i in range(3000):
    f = open(path, "rb", buffering=1 << 20)
    f.read(1)
    refs.append(weakref.ref(f.raw))
    del f
    if i % 100 == 0:
        alive = sum(1 for r in refs if r() is not None)
        peak = max(peak, alive)

assert peak < 256, peak
print("OK")
