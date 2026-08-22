# pyre-check: max-pypy-ratio=63
# The ceiling is twice the slowest ratio observed.
# A type change on one instance freezes unboxing for the whole class
# (mapdict.py:623). Instances created before the freeze keep an unboxed slot
# until something reads it: `_direct_read` migrates them off unboxed storage
# (mapdict.py:594-596). A folded read that performs `_prim_direct_read` alone
# would skip that migration and leave unboxed and boxed instances mixed under
# one promoted-map guard.
# Sized against FLOOR_GATE_MIN_BASELINE_S, not EXEC_TIME_FLOOR_S. The two are a
# factor of ten apart, and it is the larger one that decides whether a ratio is
# judged: below it the floor gate declines the baseline as too small while the
# ceiling still fires, so the fixture is gated on a number the comparison table
# itself marks `?`. At the previous 406399 pypy measured 0.02s -- four times the
# floor, and 2.5x short of the gate minimum -- so whether this fixture passed
# came down to how loaded the runner was. Here pypy measures 0.17s.
N = 2000000


class C:
    def __init__(self, v):
        self.x = v


def build(n):
    # Created while unboxing is still allowed, so each gets an unboxed int
    # slot, and none of them has been read yet.
    objs = [C(i) for i in range(n)]
    freeze = C(0)
    freeze.x = 1.5
    return objs, freeze.x


def first_reads(objs, n):
    total = 0
    i = 0
    while i < n:
        total += objs[i].x
        i += 1
    return total


def reread(objs, n):
    # Every instance has migrated to boxed storage by now; the same loop must
    # keep reading the same values.
    total = 0
    i = 0
    while i < n:
        total += objs[i].x
        i += 1
    return total


objs, frozen_value = build(N)
print(first_reads(objs, N))
print(reread(objs, N))
print(frozen_value)
