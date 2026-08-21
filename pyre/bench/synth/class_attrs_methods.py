# pyre-check: max-pypy-ratio=8
# The ceiling sits between the two measured states: folded this runs 3.7x
# pypy, and with `builtin_type_getattr` suppressed about 87x.
# A hot `getattr(type, name)` loop rides along: the read resolves through the
# class MRO, and without the `builtin_type_getattr` fold it measures 13.1x on
# its own (0.091s -> 1.190s).
N = 600000


class Counter:
    def __init__(self, base):
        self.value = base

    def add(self, x):
        self.value = self.value + x
        return self.value


def main():
    c = Counter(3)
    i = 0
    acc = 0
    while i < N:
        acc = acc + c.add(i & 7)
        i = i + 1
    print(acc + c.value)


main()


class Holder:
    tag = 13


def hot_type_getattr(n):
    """Hot `getattr(type, name)`, the `builtin_type_getattr` fold."""
    s = 0
    i = 0
    while i < n:
        s += getattr(Holder, "tag")
        i += 1
    return s


print(hot_type_getattr(20000000))
