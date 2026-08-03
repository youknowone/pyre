# pyre-check: max-pypy-ratio=50
# Pins virtual range construction for one-, two-, and three-bound calls while
# retaining correct residual behavior for exceptional, subclass, index, and
# escaping-object shapes.


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
    while i < 20000:
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
