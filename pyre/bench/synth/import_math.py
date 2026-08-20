# pyre-check: max-pypy-ratio=8
# pyre-check: skip-cpython
# 56372764 puts pypy at 142ms; a count cpython could finish inside its
# reference timeout leaves pypy back on the floor, so cpython is dropped
# deliberately rather than by spending the whole timeout to discover the same
# drop.
# pypy's side is a measurement at this count and the loop reads 1.3-1.9x, so 8
# is twice the slowest reading. A ceiling at or above PERF_GATE_FLOOR_DIVISOR
# would pin the derived floor to parity, which a fixture running this close to
# pypy cannot clear on a host where pyre happens to land faster.
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

