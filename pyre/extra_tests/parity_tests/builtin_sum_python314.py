import itertools
import math
import random


# `pypy/module/__builtin__/app_functional.py:54 _regular_sum` folds over the
# live iterator, so `next` and `__add__` interleave rather than the whole
# iterable being materialised first.
order = []


class Item:
    def __init__(self, value):
        self.value = value

    def __radd__(self, other):
        order.append("add%d" % self.value)
        return other + self.value


def items():
    for i in range(3):
        order.append("next%d" % i)
        yield Item(i)


assert sum(items()) == 3
assert order == ["next0", "add0", "next1", "add1", "next2", "add2"], order


# Streaming also means an unbounded iterable is folded rather than collected.
class Stop(Exception):
    pass


seen = 0


class Counted:
    def __radd__(self, other):
        global seen
        seen += 1
        if seen > 10_000:
            raise Stop
        return other


try:
    sum(Counted() for _ in itertools.count())
except Stop:
    pass
assert seen == 10_001, seen


assert repr(sum([-0.0])) == "0.0"
assert repr(sum([-0.0], -0.0)) == "-0.0"
assert repr(sum([], -0.0)) == "-0.0"

assert sum([0.1] * 10) == 1.0
assert math.isinf(sum([float("inf"), float("inf")]))
assert math.isinf(sum([1e308, 1e308]))

# An `int` wider than a machine word is folded into the compensated
# accumulators as a double, so the compensation term survives it; only a
# magnitude beyond f64 range raises.
assert sum([1.0, 10**100, 1.0, -(10**100)]) == 2.0
assert sum([2j, 1.0, 10**100, 1.0, -(10**100)]) == 2 + 2j
assert sum([1.0, 2**63]) == 1.0 + float(2**63)
for values in ([1.0, 10**1000], [1j, 10**1000]):
    try:
        sum(values)
    except OverflowError:
        pass
    else:
        raise AssertionError("expected OverflowError for %r" % (values,))

random.seed(0)
values = [
    complex(random.random() - 0.5, random.random() - 0.5)
    for _ in range(10_000)
]
assert sum(values) == complex(
    sum(value.real for value in values),
    sum(value.imag for value in values),
)

for values in (
    [complex(1, -0.0), 1],
    [1, complex(1, -0.0)],
    [complex(1, -0.0), 1.0],
    [1.0, complex(1, -0.0)],
):
    result = sum(values)
    assert result == complex(2, -0.0)
    assert math.copysign(1.0, result.imag) == -1.0

print("OK")
