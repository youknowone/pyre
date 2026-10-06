# CPython-suite gap: test_bz2's testOpenDel drops 10000 BZ2File objects, but
# reference counting closes each one at its `del`, so the suite never asks
# whether dropped buffered streams are reclaimed without an explicit collect.
# parity-tests reason: the buffered streams' bytearray buffer lives outside
# the GC heap, and only its accounting toward the major-collection threshold
# (`rgc.add_memory_pressure`) lets a loop of dropped streams reach a major.
# parity-env: PYPY_GC_MIN=8M
# parity-env: PYPY_GC_MAX_DELTA=4M
# parity-env: PYPY_GC_INCREMENT_STEP=32M

"""Dropped buffered streams must not outlive the descriptor table.

Each `open` below allocates a 1 MiB buffer and nothing else of size.  With
the buffer charged to the major-collection threshold, a major runs every few
iterations and the finalizers close the dropped files; uncharged, the
loop allocates too little GC memory to reach one, and the open files pile up
until the descriptor table is exhausted.

`PYPY_GC_MIN` is the first-major floor (`incminimark` `post_setup`); the
default `nursery*8` on a large-RAM host never trips inside this loop.
`PYPY_GC_MAX_DELTA` caps how far the next-major threshold may sit above the
live size (default 1/8 of RAM). `PYPY_GC_INCREMENT_STEP` is the mark budget
per nursery collection; the default `nursery*4` can leave the first
incremental major unfinished across hundreds of opens, so the peak samples
the in-flight set rather than the reclaimed one.
"""

import os
import weakref

path = os.path.abspath(__file__)
refs = []
peak = 0
for i in range(400):
    f = open(path, "rb", buffering=1 << 20)
    f.read(1)
    refs.append(weakref.ref(f.raw))
    del f
    if i % 10 == 0:
        alive = sum(1 for r in refs if r() is not None)
        peak = max(peak, alive)

assert peak < 256, peak
print("OK")
