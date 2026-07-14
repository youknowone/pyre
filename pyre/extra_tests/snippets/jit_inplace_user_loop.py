"""Hot user ``__i*__`` dispatch must not drop the tracing-boundary item."""


class HotInPlace:
    def __init__(self, val):
        self.val = val
        self.calls = 0

    def __iadd__(self, other):
        self.calls += 1
        self.val += other
        return self

    def __isub__(self, other):
        self.calls += 1
        self.val -= other
        return self

    def __imul__(self, other):
        self.calls += 1
        self.val *= other
        return self


add = HotInPlace(0)
for n in range(1000):
    add += n
assert add.calls == 1000
assert add.val == 499500

sub = HotInPlace(499500)
for n in range(1000):
    sub -= n
assert sub.calls == 1000
assert sub.val == 0

mul = HotInPlace(1)
for n in range(1000):
    mul *= n
assert mul.calls == 1000


class HotInPlaceFresh:
    def __init__(self, val, calls=0):
        self.val = val
        self.calls = calls

    def __iadd__(self, other):
        return HotInPlaceFresh(self.val + other, self.calls + 1)


fresh = HotInPlaceFresh(0)
for n in range(1000):
    fresh += n
assert fresh.calls == 1000
assert fresh.val == 499500
