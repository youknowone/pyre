# pyre-check: max-pypy-ratio=3
# `descr_iter` picks a shape from `promote_step`, not from `step == 1`, so the
# loops below compile three different iterators over the same span: `range(n)`
# and `range(a, b)` carry an immutable `stop` and no countdown, while
# `range(a, b, 1)` keeps the general shape's `remaining` and `step` fields.
#
# The ceiling is host headroom, not the guard.  The three shapes emit the same
# jit-stats and differ only by a per-element field read, so the margin between
# them is smaller than the spread across runners and no threshold separates
# them everywhere.  What this file guards is behaviour: the shapes must agree
# with each other, and an iterator must survive a pickle round trip as its own
# type.
import pickle
import sys

N = 25000000
MAX = sys.maxsize


def one_arg(n):
    t = 0
    for i in range(n):
        t += i
    return t


def step_one(a, b):
    t = 0
    for i in range(a, b):
        t += i
    return t


def explicit_step(a, b, s):
    t = 0
    for i in range(a, b, s):
        t += i
    return t


print(one_arg(N - 1))
print(step_one(1, N))
print(explicit_step(1, N, 1))
print(explicit_step(0, N, 2))

# The three shapes must agree with each other.
assert one_arg(1000) == step_one(0, 1000) == explicit_step(0, 1000, 1)
assert list(range(5)) == list(range(0, 5)) == list(range(0, 5, 1))
assert list(range(5, 2)) == list(range(5, 2, 1)) == []
assert list(range(1)) == [0] and list(range(4, 5)) == [4]

# `__reduce__` rebuilds the walk as `range(current, current + length * step,
# step)`, so a span whose one-past bound leaves the machine word cannot be
# carried by the word iterator: the rebuilt range would need a bigint stop and
# the round trip would come back a different type.  Selecting the word shape
# for these is therefore only sound if `__reduce__` stops naming that bound,
# and this pins the two halves together.
for t in [(MAX - 1, MAX, 2), (1, 2, MAX), (MAX - 5, MAX, 2), (13, 21, 3)]:
    r = range(*t)
    it = iter(r)
    assert type(pickle.loads(pickle.dumps(it, 0))) is type(it), t
    assert list(pickle.loads(pickle.dumps(iter(r), 0))) == list(r), t
assert list(range(MAX - 5, MAX, 2)) == [MAX - 5, MAX - 3, MAX - 1]
assert next(iter(range(MAX - 1, MAX, 2))) == MAX - 1
