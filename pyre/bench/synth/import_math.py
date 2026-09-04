# pyre-check: max-pypy-ratio=3
# The spread across hosts is wider than one number comfortably holds: over the
# 38 CI jobs of 2026-09-03 ubuntu reads 0.6x-1.2x and windows 0.9x-1.1x, while
# macos reads 1.7x-2.5x and crossed the old ceiling of 2.  Because
# `perf_gate_floor` derives the floor as ceiling/6, the two ends are pinned
# together: 3 leaves 20% over the widest reading and 20% under the narrowest,
# and no single value does better than that here.  A ceiling of 3.5 would put
# the floor at 0.583x, inside ubuntu's own band.
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
