# A FOR_ITER trace must keep compiling when an inlined callee allocates a
# fresh list.  The allocation helper is replay-safe and has no reason to abort
# the caller's loop trace.


def chain(item):
    out = []
    return len(out) + (item - item)


def drive(n):
    last = -1
    for i in range(n):
        last = chain(i)
    return last


print(drive(20000))
