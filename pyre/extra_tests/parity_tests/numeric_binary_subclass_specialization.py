# CPython-suite gap: numeric tests do not alternate subclasses at a hot binary site.
# parity-tests reason: this guards pyre's JIT exact-class specialization guards.

"""Hot numeric binary specializers preserve reflected subclass dispatch.

The subclass operand has to arrive at the *same* BINARY_OP pc that went hot, so
each driver iterates a list whose tail holds the subclass.  Feeding it to a
separate tail expression instead exercises a pc that was never specialized and
the check passes whether or not the class guard exists.
"""

HOT = 20000
TAIL = 3


class IntOperand(int):
    def __radd__(self, other):
        return ("radd", other, int(self))

    def __rfloordiv__(self, other):
        return ("rfloordiv", other, int(self))

    def __rmul__(self, other):
        return ("rmul", other, int(self))

    def __floordiv__(self, other):
        return ("floordiv", int(self), other)


class FloatOperand(float):
    def __rtruediv__(self, other):
        return ("rtruediv", other, float(self))

    def __truediv__(self, other):
        return ("truediv", float(self), other)


def rhs_add(operands):
    out = None
    for operand in operands:
        out = 7 + operand
    return out


def rhs_floordiv(operands):
    out = None
    for operand in operands:
        try:
            out = 7 // operand
        except ZeroDivisionError:
            out = ("zero",)
    return out


def rhs_mul(operands):
    out = None
    for operand in operands:
        out = 7 * operand
    return out


def rhs_truediv(operands):
    out = None
    for operand in operands:
        try:
            out = 7.0 / operand
        except ZeroDivisionError:
            out = ("zero",)
    return out


def lhs_floordiv(operands):
    out = None
    for operand in operands:
        try:
            out = operand // 3
        except ZeroDivisionError:
            out = ("zero",)
    return out


def lhs_truediv(operands):
    out = None
    for operand in operands:
        out = operand / 2.0
    return out


def tail(hot_value, subclass_value):
    return [hot_value] * HOT + [subclass_value] * TAIL


# A non-zero divisor keeps the value arm live; a zero divisor takes the raising
# arm.  Both must still reach the subclass's reflected method.
assert rhs_add(tail(3, IntOperand(3))) == ("radd", 7, 3)
assert rhs_mul(tail(3, IntOperand(3))) == ("rmul", 7, 3)
assert rhs_floordiv(tail(3, IntOperand(3))) == ("rfloordiv", 7, 3)
assert rhs_floordiv(tail(0, IntOperand(0))) == ("rfloordiv", 7, 0)
assert rhs_truediv(tail(2.0, FloatOperand(2.0))) == ("rtruediv", 7.0, 2.0)
assert rhs_truediv(tail(0.0, FloatOperand(0.0))) == ("rtruediv", 7.0, 0.0)

# The left operand position needs its own guard.
assert lhs_floordiv(tail(7, IntOperand(7))) == ("floordiv", 7, 3)
assert lhs_floordiv(tail(0, IntOperand(0))) == ("floordiv", 0, 3)
assert lhs_truediv(tail(7.0, FloatOperand(7.0))) == ("truediv", 7.0, 2.0)

# A bool operand goes hot through the tagged/bool path, which skips the emitted
# class guard; the subclass tail must still divert.
assert rhs_add(tail(True, IntOperand(3))) == ("radd", 7, 3)
assert rhs_floordiv(tail(True, IntOperand(3))) == ("rfloordiv", 7, 3)


# ── `__float__` coercion at a hot site ──────────────────────────────────────
# The float coercions are the same exact-class question one layer down. Reading
# an int's payload is the answer only for an exact builtin: `PyNumber_Float`
# (the `float()` constructor) and `PyFloat_AsDouble` (the `math` entry points)
# both dispatch `nb_float` for a non-float, so a subclass override decides.
# The layout predicates read `ob_type`, which the subclass shares, so both the
# interpreter fast path and the trace guard admitted it and read the payload.
import math


class ToFloat(int):
    def __float__(self):
        return 99.0


class ToFloatFromFloat(float):
    def __float__(self):
        return 99.0


def coerce_hot(operands, fn):
    out = None
    for operand in operands:
        out = fn(operand)
    return out


# `float()` honors the override for both bases: `PyNumber_Float` short-circuits
# only `PyFloat_CheckExact`.
assert coerce_hot(tail(4, ToFloat(4)), float) == 99.0
assert coerce_hot(tail(4.0, ToFloatFromFloat(4.0)), float) == 99.0

# `math` uses `PyFloat_AsDouble`, whose short-circuit is `PyFloat_Check` — so a
# FLOAT subclass keeps its payload while an INT subclass takes the override.
assert coerce_hot(tail(4, ToFloat(4)), math.sqrt) == math.sqrt(99.0)
assert coerce_hot(tail(4.0, ToFloatFromFloat(4.0)), math.sqrt) == 2.0
assert coerce_hot(tail(4, ToFloat(4)), math.cos) == math.cos(99.0)
assert coerce_hot(tail(4, ToFloat(4)), math.frexp) == math.frexp(99.0)
assert coerce_hot(tail(4, ToFloat(4)), lambda a: math.ldexp(a, 1)) == 198.0

# `complex()` reaches the same real-number ladder.
assert coerce_hot(tail(4, ToFloat(4)), complex) == complex(99.0, 0.0)

# A float presentation code formats the `PyNumber_Float` conversion.
assert coerce_hot(tail(4, ToFloat(4)), lambda a: format(a, ".2f")) == "99.00"

# `loghelper` is the exception that proves the rule: it converts EVERY
# `PyLong_Check` operand from its payload, subclass included, for both the
# argument and the base. Routing either through the general coercion would
# answer a different logarithm.
assert coerce_hot(tail(4, ToFloat(4)), math.log) == math.log(4)
assert math.log(100, ToFloat(10)) == 2.0

# `__index__`-based entry points are likewise payload-only.
assert coerce_hot(tail(16, ToFloat(16)), math.isqrt) == 4
assert coerce_hot(tail(0, ToFloat(0)), lambda a: math.ldexp(1.0, a)) == 1.0

print("OK")
