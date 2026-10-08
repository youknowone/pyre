# pyre-check: max-pypy-ratio=80
# Exact-int BINARY_SLICE on a list: the walker descends
# `binary_slice_values_inner` (no residual CallMayForceR).
N = 50000


def slice_loop(n):
    xs = list(range(32))
    total = 0
    i = 0
    while i < n:
        a = i & 3
        b = a + 7
        total = total + len(xs[a:b])
        i = i + 1
    return total


assert slice_loop(N) == 7 * N
print("OK")
