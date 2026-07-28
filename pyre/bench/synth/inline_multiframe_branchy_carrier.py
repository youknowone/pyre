# pyre-check: max-pypy-ratio=12
# Module-scope hot loop inlining a 2-level call chain whose middle function has
# a data-dependent branch — regression guard for the branchy inlined-callee
# multi-frame carrier miscompile, in a pure and a journaled shape.
#
# The drain sub-walk concrete-executes the reconstructed callee and then aborts
# to the blackhole, which replays that callee from the guard; without a
# non-commit rollback every eager store stands and is applied a second time.
# The pure loop surfaces a replay in `acc`; the journaled loop makes it
# countable — the innermost callee's list setitem is journaled, so hits[0]
# counts one bump per iteration exactly and hits[0] != N means the drain
# doubled.
N = 120000
hits = [0]


def add3(a, b, c):
    return a + b + c


def mix(a, b):
    if a & 1:
        return add3(a, b, 7)
    return add3(b, a, -3)


def add3_journaled(a, b, c):
    hits[0] = hits[0] + 1
    return a + b + c


def mix_journaled(a, b):
    if a & 1:
        return add3_journaled(a, b, 7)
    return add3_journaled(b, a, -3)


i = 0
acc = 0
while i < N:
    acc = acc + mix(i, acc & 255)
    i = i + 1
print(acc)

i = 0
acc = 0
while i < N:
    acc = acc + mix_journaled(i, acc & 255)
    i = i + 1
print(acc, hits[0])
