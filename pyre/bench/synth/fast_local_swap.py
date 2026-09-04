# pyre-check: max-pypy-ratio=2.6
# The ceiling gates cranelift as well as dynasm, and `perf_gate_floor` derives
# a floor from it as ceiling/6, so both ends of the reading spread pick it. Run
# 33384229844 reads 0.5x (macos dynasm, median of 3), 0.6x (macos cranelift),
# 1.3x and 1.5x (ubuntu) on the four pairs where pypy's baseline was measurable
# -- wasm is ungated and windows read a clamped baseline. Run 33860926996 later
# read 2.2x on ubuntu with pyre at 0.33s, the same as main's 0.34s in run
# 33849050216; pypy's denominator moved instead. 2.6 is that cross-host high
# plus 15%, while its 0.433x derived floor remains below the measured 0.5x.
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
