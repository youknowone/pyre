# pyre-check: max-pypy-ratio=60
# The ceiling sits between the two measured states: folded this runs 29x
# pypy -- pypy turns the whole iterator into a tight int loop, which pyre
# does not -- and with `instance_next` suppressed about 975x.
# FOR_ITER body with a CALL: the JIT must handle calls inside for-loop bodies
# correctly without replaying the last iteration on deopt.  `accumulating`
# additionally threads the running total through the call, so the inline
# sub-walk decline has to handle a callee whose argument is loop-carried.
# A hot FOR_ITER over a user-defined iterator rides along, entering a Python
# `__next__` per item: without the `instance_next` fold the callee stops being
# inlined and `ForIterNext` stays an opaque residual, 23.5x on its own
# (0.191s -> 4.485s).


def g(x):
    return x * 2


def add(a, b):
    return a + b


def main():
    total = 0
    n = 0
    while n < 20000:
        for x in range(10):
            total += g(x)
        n += 1
    return total


def accumulating():
    total = 0
    n = 0
    while n < 100:
        for j in range(200):
            total = add(total, n * j)
        n += 1
    return total


print(main())
# Expected: 20000 * sum(2*x for x in range(10)) = 20000 * 90 = 1800000
print(accumulating())
# Expected: sum(n for n in range(100)) * sum(j for j in range(200))
#         = 4950 * 19900 = 98505000


class NextIter:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i


def hot_instance_next(n):
    """Hot FOR_ITER entering a Python `__next__`, the `instance_next` fold."""
    s = 0
    for v in NextIter(n):
        s += v
    return s


print(hot_instance_next(20000000))
