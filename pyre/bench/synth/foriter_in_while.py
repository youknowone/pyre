# pyre-check: max-pypy-ratio=6
# FOR_ITER inside a while-loop: the common real-world pattern.
# The while-loop is the JIT entry; the for-loop body must handle the FOR_ITER
# liveness scoping correctly.  `outer_var` additionally reads the outer
# loop's variable from inside the for-body.


def main():
    total = 0
    n = 0
    while n < 10000:
        for x in range(100):
            total += x
        n += 1
    return total


def outer_var():
    total = 0
    n = 0
    while n < 1000:
        for i in range(200):
            total += i * n
        n += 1
    return total


print(main())
# Expected: 10000 * sum(range(100)) = 10000 * 4950 = 49500000
print(outer_var())
# Expected: sum(n for n in range(1000)) * sum(i for i in range(200))
#         = 499500 * 19900 = 9940050000
