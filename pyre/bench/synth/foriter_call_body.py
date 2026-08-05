# pyre-check: max-pypy-ratio=20
# pyre-check: min-pypy-ratio=2.72
# FOR_ITER body with a CALL: the JIT must handle calls inside for-loop bodies
# correctly without replaying the last iteration on deopt.  `accumulating`
# additionally threads the running total through the call, so the inline
# sub-walk decline has to handle a callee whose argument is loop-carried.


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
