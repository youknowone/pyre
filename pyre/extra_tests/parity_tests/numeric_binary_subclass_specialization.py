"""Hot numeric binary specializers preserve reflected subclass dispatch."""

ROUNDS = 10000


class IntOperand(int):
    def __rfloordiv__(self, other):
        return ("rfloordiv", other, int(self))

    def __radd__(self, other):
        return ("radd", other, int(self))


class FloatOperand(float):
    def __rtruediv__(self, other):
        return ("rtruediv", other, float(self))


def int_floor_tail(divisor):
    result = None
    for _ in range(ROUNDS):
        result = 7 // 3
    return 7 // divisor


def int_add_tail(value):
    result = None
    for _ in range(ROUNDS):
        result = 7 + 3
    return 7 + value


def float_div_tail(divisor):
    result = None
    for _ in range(ROUNDS):
        result = 7.0 / 2.0
    return 7.0 / divisor


assert int_floor_tail(IntOperand(0)) == ("rfloordiv", 7, 0)
assert int_floor_tail(IntOperand(3)) == ("rfloordiv", 7, 3)
assert int_add_tail(IntOperand(3)) == ("radd", 7, 3)
assert float_div_tail(FloatOperand(0.0)) == ("rtruediv", 7.0, 0.0)
assert float_div_tail(FloatOperand(2.0)) == ("rtruediv", 7.0, 2.0)

print("OK")
