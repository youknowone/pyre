# pyre-check: max-pypy-ratio=96
# Pins virtual range construction for one-, two-, and three-bound calls while
# retaining correct residual behavior for exceptional, subclass, index, and
# escaping-object shapes.
#
# N is 400000 because pypy runs the body in 0.01s at 20000, which leaves its
# startup-subtracted time under the execution floor on every host and makes
# the ratio an artifact of that floor rather than a measurement. At 400000
# pypy spends 0.10s, clearing the floor even on the platform with the
# coarsest timer.
#
# `main` used to run interpreted end to end. The `try: range(0, 3, 0)` below
# puts an out-of-line handler after the trailing comprehension, and the loop
# region that gates the back edge grew across the gap between them and picked
# up that comprehension's call-bearing `FOR_ITER` -- an opcode this loop never
# reaches. With the region built from the exception table instead, the while
# loop and the three `for` loops compile, and this gate's own metric falls
# from 122x to 23.2x dynasm / 30.2x cranelift / 24.7x wasm. A separate
# min-of-five interleaved harness reads the same move as 158x to 35.2x /
# 38.3x, so the ceiling is set from the slower of the two readings: 2.5x of
# 38.3x, the slack every bench here carries against a runner 2.5x slower than
# an idle local box.
N = 400000


try:
    class RangeSubclass(range):
        pass
except TypeError:
    RangeSubclass = None


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def main():
    one = 0
    two = 0
    three = 0
    zero = 0
    subclass = 0
    indexed = 0
    escaped = []
    n = 5
    a = 2
    b = 7
    step = -2
    i = 0
    while i < N:
        for value in range(n):
            one += value
        for value in range(a, b):
            two += value
        for value in range(9, -1, step):
            three += value
        try:
            range(0, 3, 0)
        except ValueError:
            zero += 1
        if RangeSubclass is None:
            subclass += 3
        else:
            subclass += len(RangeSubclass(3))
        indexed += len(range(Index(4)))
        escaped.append(range(i, i + 3))
        i += 1
    print(one, two, three, zero, subclass, indexed)
    print(len(escaped))
    print(
        [
            (item.start, item.stop, item.step, len(item))
            for item in (escaped[0], escaped[10000], escaped[-1])
        ]
    )


main()
