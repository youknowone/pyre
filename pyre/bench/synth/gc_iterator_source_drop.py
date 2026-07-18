# Immortal iterator/sequence wrappers (enumerate, itertools.filterfalse,
# takewhile) hold their source iterator solely through the wrapper object.  The
# marker never scans an immortal wrapper, so unless its child offsets are
# registered the source is not forwarded across a collection: a moving GC driven
# by a hot inner loop then frees it, and the next __next__ reads a garbage
# source ("not an iterator") or yields wrong values.  Each case builds a wrapper,
# partially consumes it, forces collections while it is the sole root of its
# source, then drains the rest and folds it into a checksum.
from itertools import filterfalse, takewhile

N = 4000
ALLOC = 250


def churn():
    s = 0
    for i in range(ALLOC):
        x = [i, i + 1, i + 2]
        s += x[0] + x[2]
    return s


def enumerate_hold():
    acc = 0
    for k in range(N):
        src = [k * 10 + j for j in range(8)]
        it = enumerate(src)
        first = [next(it), next(it), next(it)]
        churn()
        rest = list(it)
        for idx, val in first + rest:
            acc = (acc + idx * 31 + val) % 1000003
    return acc


def filterfalse_hold():
    acc = 0
    for k in range(N):
        src = list(range(k, k + 12))
        it = filterfalse(lambda v: v % 2 == 0, src)
        first = next(it)
        churn()
        rest = list(it)
        for val in [first] + rest:
            acc = (acc + val) % 1000003
    return acc


def takewhile_hold():
    acc = 0
    for k in range(N):
        src = list(range(k, k + 20))
        it = takewhile(lambda v: v < k + 15, src)
        first = next(it)
        churn()
        rest = list(it)
        for val in [first] + rest:
            acc = (acc + val * 7) % 1000003
    return acc


print("enumerate_hold", enumerate_hold())
print("filterfalse_hold", filterfalse_hold())
print("takewhile_hold", takewhile_hold())
