# pyre-check: max-pypy-ratio=190
# pyre-check: min-pypy-ratio=20
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
# The ceiling rose from 50 because that bound was fitted to the floored
# denominator, not because anything got slower: the honest ratio here is
# 83x as a median of interleaved pairwise runs. It is dominated by the four
# deliberately residual shapes below rather than by the virtualized loops --
# each iteration also raises and catches a ValueError.
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
