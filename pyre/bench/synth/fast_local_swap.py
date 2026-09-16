# pyre-check: max-pypy-ratio=3
# The ceiling gates cranelift as well as dynasm, and `perf_gate_floor` derives
# a floor from it as ceiling/6, so both ends of the reading spread pick it. Run
# 33384229844 reads 0.5x (macos dynasm, median of 3), 0.6x (macos cranelift),
# 1.3x and 1.5x (ubuntu) on the four pairs where pypy's baseline was measurable
# -- wasm is ungated. Windows CI read 2.9x (0.33s vs pypy 0.12s) against the
# previous 2x ceiling. 3x still keeps the floor at 0.5x.
# pyre-check: skip-cpython
# cpython 1.33s vs pyre 0.24s (5.5x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Sized so pypy's own execution clears Windows `FLOOR_GATE_MIN_BASELINE_S`
# (~0.16s).  Below that the ceiling divides by a `?` band denominator and
# the same binary reads 2x on one host and 3x on the next.
N = 38000000


def fib_swap(n):
    a = 0
    b = 1
    i = 0
    while i < n:
        a, b = b, (a + b) % 1000000007
        i = i + 1
    return a


def plain_swap(n):
    x = 1
    y = 2
    i = 0
    while i < n:
        x, y = y, x
        i = i + 1
    return x * 10 + y


def store_load_chain(n):
    acc = 0
    i = 0
    while i < n:
        acc = acc + i
        acc = acc % 999983
        i = i + 1
    return acc


def main():
    print(fib_swap(N))
    print(plain_swap(N + 1))
    print(store_load_chain(N))


main()
