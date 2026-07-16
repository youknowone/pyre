"""bool surface where CPython 3.14 differs from the bundled PyPy source."""


assert "__doc__" in bool.__dict__
assert "__invert__" in bool.__dict__
assert "__str__" not in bool.__dict__
assert bool.__invert__(True) == -2
assert bool.__invert__(False) == -1
assert str(True) == "True"
assert str(False) == "False"
assert bool.__doc__.startswith("Returns True when the argument is true")
print("bool 3.14 surface: ok")
