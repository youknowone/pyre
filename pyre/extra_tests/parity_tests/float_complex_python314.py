"""float/complex TypeDef parity with Python 3.14's from_number API."""

import math
import operator
import warnings


assert {"__doc__", "__hash__", "__repr__", "from_number"} <= set(float.__dict__)
assert {"__doc__", "from_number"} <= set(complex.__dict__)
assert "__str__" not in complex.__dict__

for value in (0.0, -0.0, 1.5, float("inf")):
    assert float.__hash__(value) == hash(value)
    assert float.__repr__(value) == repr(value)

z = 1 + 2j
assert float.from_number(3) == 3.0
assert float.from_number(1.5) == 1.5
assert complex.from_number(3) == 3 + 0j
assert complex.from_number(z) == z


class Number:
    def __float__(self):
        return 2.5

    def __complex__(self):
        return 2 + 3j


assert float.from_number(Number()) == 2.5
assert complex.from_number(Number()) == 2 + 3j


class ComplexIndexOnly:
    def __index__(self):
        return 271


class ComplexSubclass(complex):
    pass


assert complex.from_number(ComplexIndexOnly()) == 271 + 0j
complex_subclass_value = ComplexSubclass.from_number(ComplexIndexOnly())
assert complex_subclass_value == 271 + 0j
assert type(complex_subclass_value) is ComplexSubclass

complex_nan = complex(float("nan"), float("nan"))
assert complex.from_number(complex_nan) is complex_nan
assert (1 - 2j).__getnewargs__() == (1.0, -2.0)

try:
    abs(complex(float.fromhex("0x1.fffffffffffffp+1023"),
                float.fromhex("0x1.fffffffffffffp+1023")))
except OverflowError as error:
    assert str(error) == "absolute value too large"
else:
    raise AssertionError("an overflowing complex magnitude must raise")

assert complex.__lt__(1 + 1j, 2 + 2j) is NotImplemented
assert complex(2**60, 0) != 2**60 + 1

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert complex(real=4.25 + 1.5j) == 4.25 + 1.5j
assert len(caught) == 1
assert caught[0].category is DeprecationWarning
assert str(caught[0].message) == (
    "complex() argument 'real' must be a real number, not complex"
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert complex(1 + 2j, 3 + 4j) == -3 + 5j
assert len(caught) == 2


class ComplexOnly:
    def __complex__(self):
        return 3 + 4j


try:
    complex(1, ComplexOnly())
except TypeError as error:
    assert str(error) == (
        "complex() argument 'imag' must be a real number, not ComplexOnly"
    )
else:
    raise AssertionError("the imaginary argument must use the real protocol")

assert format(1 + 2j, "+g") == "+1+2j"
assert format(1 + 2j, " g") == " 1+2j"
assert format(1 + 2j, "#g") == "1.00000+2.00000j"
assert format(complex(float("inf"), float("nan")), "+g") == "+inf+nanj"
assert format(1234.5 + 6789j, ",.2f") == "1,234.50+6,789.00j"

def assert_float_identical(value, expected):
    if math.isnan(expected):
        assert math.isnan(value)
    elif expected == 0.0:
        assert value == 0.0
        assert math.copysign(1.0, value) == math.copysign(1.0, expected)
    else:
        assert value == expected


def assert_complex_identical(value, expected):
    assert_float_identical(value.real, expected.real)
    assert_float_identical(value.imag, expected.imag)


assert_complex_identical(complex(-0.0, -0.0) + -0.0, complex(-0.0, -0.0))
assert_complex_identical(-0.0 + complex(-0.0, -0.0), complex(-0.0, -0.0))
assert_complex_identical(complex(-0.0, -0.0) - 0.0, complex(-0.0, -0.0))
assert_complex_identical(-0.0 - complex(0.0, 0.0), complex(-0.0, -0.0))
for operation in (operator.add, operator.sub):
    try:
        operation(1j, 10**1000)
    except OverflowError as error:
        assert str(error) == "int too large to convert to float"
    else:
        raise AssertionError("over-range int must fail complex arithmetic")

inf = float("inf")
nan = float("nan")
assert_complex_identical(complex(inf, 1.0) * 0.0, complex(nan, 0.0))
assert_complex_identical(
    complex(1e300, 1.0) * complex(nan, inf), complex(-inf, inf)
)
assert_complex_identical(
    (1.0 + 1.0j) / complex(-inf, inf), complex(0.0, -0.0)
)
assert_complex_identical(1.0 / complex(-inf, -inf), complex(-0.0, 0.0))

for base, exponent in (
    (1e200 + 1.0j, 1e200 + 1.0j),
    (1e200 + 1.0j, 5),
    (9.0j, 33.0j**3),
):
    try:
        pow(base, exponent)
    except OverflowError as error:
        assert str(error) == "complex exponentiation"
    else:
        raise AssertionError("overflowing complex power must raise")


class IndexOnly:
    def __index__(self):
        return 314


class FloatSubclass(float):
    pass


assert float.from_number(IndexOnly()) == 314.0
subclass_value = FloatSubclass.from_number(IndexOnly())
assert subclass_value == 314.0
assert type(subclass_value) is FloatSubclass

nan = float("nan")
assert float.from_number(nan) is nan
assert hash(nan) == object.__hash__(nan)


class NanSubclass(float):
    pass


nan_subclass = NanSubclass("nan")
assert hash(nan_subclass) == object.__hash__(nan_subclass)

for value in (1.6e308, -1.7e308):
    try:
        round(value, -308)
    except OverflowError as error:
        assert str(error) == "rounded value too large to represent"
    else:
        raise AssertionError("an infinite rounded result must overflow")


class FloatString(str):
    def __float__(self):
        return float(str(self)) + 1


assert float(FloatString("8")) == 9.0


class HugeIndex:
    def __index__(self):
        return 2**2000


try:
    float(HugeIndex())
except OverflowError:
    pass
else:
    raise AssertionError("an oversized __index__ result must overflow float")


class FromHexSubclass(float):
    def __new__(cls, value):
        return float.__new__(cls, value + 1)


class FromHexInitSubclass(float):
    def __init__(self, value):
        self.initialized = value


fromhex_value = FromHexSubclass.fromhex((1.5).hex())
assert type(fromhex_value) is FromHexSubclass
assert fromhex_value == 2.5

fromhex_init_value = FromHexInitSubclass.fromhex((1.5).hex())
assert type(fromhex_init_value) is FromHexInitSubclass
assert fromhex_init_value == 1.5
assert fromhex_init_value.initialized == 1.5

try:
    float("\t \n")
except ValueError as error:
    assert str(error) == "could not convert string to float: '\\t \\n'"
else:
    raise AssertionError("invalid float text must raise ValueError")

for cls in (float, complex):
    try:
        cls.from_number("3")
    except TypeError:
        pass
    else:
        raise AssertionError("from_number must not parse strings")

assert float.__doc__.startswith("Convert a string or number")
assert complex.__doc__.startswith("Create a complex number")

# Python 3.14 permits a second grouping option after the precision.  It
# groups digits to the right of the decimal point independently of the
# traditional integer-part grouping option.
x = 123_456.123_456
assert format(x, "._f") == "123456.123_456"
assert format(x, ".,f") == "123456.123,456"
assert format(x, "_._f") == "123_456.123_456"
assert format(x, ".10_f") == "123456.123_456_000_0"
assert format(x, "+.11_e") == "+1.234_561_234_56e+05"
assert format(x, ">21._f") == "       123456.123_456"
assert format(x, "021_._f") == "0_000_123_456.123_456"
assert format(x, "023_.10_f") == "0_123_456.123_456_000_0"

print("float/complex 3.14 surface: ok")
