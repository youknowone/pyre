"""float/complex TypeDef parity with Python 3.14's from_number API."""


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
