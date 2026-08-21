# pyre-check: max-pypy-ratio=45
# A callee whose hot arm is branch-free and whose cold arm the loop never
# enters.  `Getter.__call__`'s `self._single` branch is True for the whole run,
# so the comprehension arm below it is dead — but it is what the callee's body
# CONTAINS, and the replay-safety scan reads the body, not the path.
#
# The comprehension compiles to a `GET_ITER` / `FOR_ITER` pair plus the list
# fill, and the scan accepts none of the three: the iterator advance is
# irreversible, so replaying the callee from the caller's CALL boundary would
# double-consume it.  Collapsing that to one verdict for the whole body made an
# untaken branch decide the admission for the branch every iteration takes, and
# the callee residualized once per call.
#
# The scan reports those pcs instead, the admission takes the body, and the walk
# refuses only on arriving at one.  Nothing here ever does, so what the fixture
# measures is the hot arm inlining with the cold arm present.  Deleting the
# `else` arm entirely is the control: it measures the same hot path with nothing
# to poison, so the two ratios should sit together.
#
# `co_cellvars` is empty by construction.  A generator expression would close
# over `obj` and make it a freevar, which declines at a different gate
# (`callee_code.cellvars` non-empty) and would measure that gate instead of this
# one.  The list comprehension keeps the cell out.
# Sized off pypy rather than off this backend.  The body is a single subscript,
# so pypy runs it at well under a nanosecond an iteration and its execution-only
# time is what has to clear `FLOOR_GATE_MIN_BASELINE_S` (ten times the 5ms exec
# floor).  At 20000000 it read 0.014s, inside the band where check.py applies
# the ceiling but declines the floor gate as too small to judge and prints the
# ratio with a `?`.  This reads 0.054s, which clears the gate but not by much;
# a host that drifts back under it prints `?` and keeps the ceiling, which is
# the benign direction.  Raising N further is bounded by the wasm leg, not by
# this one.  The ratio is flat in N once the callee is admitted -- what the gate
# watches for is the admission being LOST, which moves it by two orders of
# magnitude, not by a few percent.
N = 80000000
T = (7, 8, 9)


class Getter:
    __slots__ = ("_single", "_idx")

    def __init__(self, idx):
        self._single = True
        self._idx = idx

    def __call__(self, obj):
        if self._single:
            return obj[self._idx]
        return sum([obj[i] for i in self._idx])


def main():
    g = Getter(0)
    total = 0
    for _ in range(N):
        total += g(T)
    print(total)


main()
