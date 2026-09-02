# pyre-check: max-pypy-ratio=4.2
# Ubuntu run 33279264115: 1.3-2.1x; the ceiling is twice the slowest,
# rounded up to one decimal place.
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
