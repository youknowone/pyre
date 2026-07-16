"""PyPy integer implementation with the Python 3.14 public surface."""


assert {"__doc__", "__hash__", "__sizeof__", "is_integer"} <= set(int.__dict__)
assert "__str__" not in int.__dict__
assert int.__str__ is object.__str__

for value in (0, 1, -1, 2**30 - 1, 2**30, 2**60, -(2**60)):
    assert int.__hash__(value) == hash(value)
    assert int.__sizeof__(value) == value.__sizeof__()
    assert value.is_integer() is True
    assert str(value) == repr(value)

assert int.__doc__.startswith("int([x]) -> integer")
print("int 3.14 surface: ok")
