# Regression guard: the multi-frame carrier drain must not replay callee effects.
# The drain sub-walk concrete-executes the reconstructed callee and then aborts to
# the blackhole, which replays that callee from the guard; without a non-commit
# rollback every eager store stands and is applied a second time. The 2-level
# inline chain with a data-dependent branch in the middle function drives the
# carrier resume, and the innermost callee's list setitem is journaled, so hits[0]
# counts one bump per iteration exactly. hits[0] != N means the drain doubled.
N = 120000
hits = [0]


def add3(a, b, c):
    hits[0] = hits[0] + 1
    return a + b + c


def mix(a, b):
    if a & 1:
        return add3(a, b, 7)
    return add3(b, a, -3)


i = 0
acc = 0
while i < N:
    acc = acc + mix(i, acc & 255)
    i = i + 1
print(acc, hits[0])
