# pyre-check: max-pypy-ratio=2
# pyre-check: skip-cpython
# A count cpython could finish inside its reference timeout leaves pypy back
# on the floor, so cpython is dropped deliberately rather than by spending
# the whole timeout to discover the same drop.
import math

# Sized so pypy's own execution clears Windows `FLOOR_GATE_MIN_BASELINE_S`
# (~0.16s).  Below that macos readings sat in the `?` band and crossed 2.
N = 152000000


def main():
    i = 0
    acc = 0
    while i < N:
        x = math.sqrt((i & 255) + 1)
        acc = acc + int(x * 1000.0)
        i = i + 1
    print(acc)


main()
