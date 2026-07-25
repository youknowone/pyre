"""`math.lcm` converts every argument even after the result reaches zero."""

import math


def raises_type_error(operation):
    try:
        operation()
    except TypeError as exc:
        assert "cannot be interpreted as an integer" in str(exc), str(exc)
    else:
        raise AssertionError("lcm skipped an argument conversion")


assert math.lcm() == 1
assert math.lcm(-5) == 5
assert math.lcm(6, 10) == 30
assert math.lcm(6, 10, 14) == 210
assert math.lcm(0, 0) == 0
assert math.lcm(0, 3) == 0
assert math.lcm(3, 0) == 0
assert math.lcm(0, 2**70) == 0
assert math.lcm(2**70, 0, 5) == 0

# A zero short-circuits the arithmetic only; the remaining arguments still go
# through __index__, so a non-integer after a zero is still a TypeError.
raises_type_error(lambda: math.lcm(0, 1.5))
raises_type_error(lambda: math.lcm(4, 0, 1.5))
raises_type_error(lambda: math.lcm(0, 0, "a"))
raises_type_error(lambda: math.lcm(0, object()))


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


assert math.lcm(0, Index(4)) == 0
assert math.lcm(Index(6), Index(10)) == 30

calls = []


class Counting:
    def __index__(self):
        calls.append(1)
        return 0


assert math.lcm(0, Counting(), Counting()) == 0
assert len(calls) == 2, calls

print("OK")
