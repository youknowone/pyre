# A float subclass that overrides the arithmetic and comparison dunders, driven
# hot enough to compile. The walker's float specialization lowers BINARY_OP to
# `FloatAdd` / `FloatSub` / `FloatMul` / `FloatTrueDiv` and COMPARE_OP to
# `FloatLt` / `FloatEq` / ... , all of which bypass special-method dispatch.
#
# A numeric subclass keeps the builtin `ob_type` layout while its Python-visible
# class lives in `w_class`, so the `guard_class` those paths emit reads `ob_type`
# and cannot tell the subclass apart at runtime either. Only an exactness test on
# the concrete operands keeps a subclass out of the raw path, which is what
# `walker_float_specialization_operands` checks before returning its operands.
# Without it every line below silently loses the override and prints the raw
# IEEE result.
#
# The int subclass at the bottom is the control: the int specialization has
# carried that exactness test all along, so it must stay correct either way.
N = 20000


class MyFloat(float):
    def __add__(self, other):
        return float.__add__(self, other) + 1000.0

    def __radd__(self, other):
        return float.__radd__(self, other) + 2000.0

    def __sub__(self, other):
        return float.__sub__(self, other) - 1000.0

    def __mul__(self, other):
        return float.__mul__(self, other) * 2.0

    def __truediv__(self, other):
        return float.__truediv__(self, other) + 7.0

    def __lt__(self, other):
        return not float.__lt__(self, other)

    def __eq__(self, other):
        return not float.__eq__(self, other)

    __hash__ = float.__hash__


def add_hot(n):
    total = 0.0
    for i in range(n):
        total = MyFloat(1.5) + total
    return total


def radd_hot(n):
    total = 0.0
    for i in range(n):
        total = total + MyFloat(1.5)
    return total


def sub_hot(n):
    # Accumulating rather than alternating: `MyFloat(1.5) - total` oscillates
    # between two values and lands on the same result with or without the
    # override, so it would not discriminate.
    total = 0.0
    for i in range(n):
        total = MyFloat(total) - 1.0
    return total


def mul_hot(n):
    total = 0.0
    for i in range(n):
        total = MyFloat(2.0) * 1.5
    return total


def truediv_hot(n):
    total = 0.0
    for i in range(n):
        total = MyFloat(9.0) / 2.0
    return total


def lt_hot(n):
    hits = 0
    for i in range(n):
        if MyFloat(1.0) < 2.0:
            hits += 1
    return hits


def eq_hot(n):
    hits = 0
    for i in range(n):
        if MyFloat(1.0) == 1.0:
            hits += 1
    return hits


# Mixed operands: an exact float on one side, the subclass on the other, so the
# int/float coercion arm is exercised as well as the plain float/float arm.
def mixed_int_operand_hot(n):
    total = 0.0
    for i in range(n):
        total = MyFloat(3.0) + 2
    return total


class MyInt(int):
    def __add__(self, other):
        return int.__add__(self, other) + 1000


def int_control_hot(n):
    total = 0
    for i in range(n):
        total = MyInt(3) + total
    return total


print(add_hot(N))
print(radd_hot(N))
print(sub_hot(N))
print(mul_hot(N))
print(truediv_hot(N))
print(lt_hot(N))
print(eq_hot(N))
print(mixed_int_operand_hot(N))
print(int_control_hot(N))
