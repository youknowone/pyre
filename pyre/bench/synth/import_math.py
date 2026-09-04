# pyre-check: max-pypy-ratio=2
# Ubuntu run 33279264115: 0.7-1x; the ceiling is twice the slowest,
# rounded up to one decimal place.
# pyre-check: skip-cpython
# 56372764 puts pypy at 142ms; a count cpython could finish inside its
# reference timeout leaves pypy back on the floor, so cpython is dropped
# deliberately rather than by spending the whole timeout to discover the same
# drop.
import math

# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 56372764


def main():
    i = 0
    acc = 0
    while i < N:
        x = math.sqrt((i & 255) + 1)
        acc = acc + int(x * 1000.0)
        i = i + 1
    print(acc)


main()
