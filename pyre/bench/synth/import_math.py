# pyre-check: max-pypy-ratio=8
# The 291 this replaces was never fitted to an observation: pypy's execution
# sat on the startup-subtraction floor, so the ratio it bounded was pyre's time
# divided by a constant. With pypy's side now a measurement the loop reads
# 1.3-1.9x, and 8 is twice the slowest reading from either size. Keeping a
# ceiling at or above PERF_GATE_FLOOR_DIVISOR would also pin the derived floor
# to parity, which a fixture running this close to pypy cannot clear on a host
# where pyre happens to land faster.
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

