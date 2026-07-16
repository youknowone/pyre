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

for cls in (float, complex):
    try:
        cls.from_number("3")
    except TypeError:
        pass
    else:
        raise AssertionError("from_number must not parse strings")

assert float.__doc__.startswith("Convert a string or number")
assert complex.__doc__.startswith("Create a complex number")
print("float/complex 3.14 surface: ok")
