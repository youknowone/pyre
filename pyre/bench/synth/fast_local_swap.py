# pyre-check: max-pypy-ratio=2
# The ceiling gates cranelift as well as dynasm, and `perf_gate_floor` derives
# a floor from it as ceiling/6, so both ends of the reading spread pick it. Run
# 33384229844 reads 0.5x (macos dynasm, median of 3), 0.6x (macos cranelift),
# 1.3x and 1.5x (ubuntu) on the four pairs where pypy's baseline was measurable
# -- wasm is ungated and windows read a clamped baseline. Sizing off the slow
# end alone lands the floor on the fast end: 3x derives exactly 0.5x. 2x clears
# both by a third.
# pyre-check: skip-cpython
# cpython 1.33s vs pyre 0.24s (5.5x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 15162700


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
