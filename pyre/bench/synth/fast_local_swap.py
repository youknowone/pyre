# pyre-check: max-pypy-ratio=4.5
# The ceiling gates cranelift as well as dynasm, and `perf_gate_floor` derives
# a floor from it as ceiling/6, so both ends of the reading spread pick it.
# Fitted over the 38 CI jobs of 2026-09-03: macos reads 1.3x-1.6x, ubuntu
# 1.6x-2.2x and windows 2.0x-3.1x.  The reading the old ceiling of 2 was sized
# against -- 0.5x on macos dynasm -- no host produces any more, and windows
# crossed 2 three times.  4.5 clears the widest by 45% and derives a 0.75x
# floor the narrowest clears by 73%.
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
