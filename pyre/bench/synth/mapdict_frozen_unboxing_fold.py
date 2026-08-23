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
#
# It is still the runner that decides, though not through load. Four Linux CI
# legs measured pypy at 0.31s, 0.35s, 0.35s and 0.35s; one measured 0.05s, and
# on that leg cpython read 1.20s against 1.36-1.42s and cranelift 4.49s against
# 7.28-7.40s. Everything ran faster there and pypy by far the most, because
# pypy's time here is almost all the two-million-object build: its read loops
# JIT down to nothing, so what it measures is the cost of first-touching the
# pages, which is the part that moves most between hosts. The ratio reached
# 115x against the 63x ceiling with the numerator *smaller* than usual, and
# check.py's own margin put it at 3.06x pypy startup -- so not a subtraction
# artefact. Raising the ceiling to cover it would have to reach 230x, which
# would stop the gate seeing a real 3x regression on every other runner, so the
# ceiling stays where the other four legs put it.
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
