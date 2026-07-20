"""bool surface where CPython 3.14 differs from the bundled PyPy source."""

import warnings


assert "__doc__" in bool.__dict__
assert "__invert__" in bool.__dict__
assert "__str__" not in bool.__dict__
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    assert bool.__invert__(True) == -2
    assert bool.__invert__(False) == -1
assert len(caught) == 2
assert all(item.category is DeprecationWarning for item in caught)
assert all("Bitwise inversion '~' on bool is deprecated" in str(item.message) for item in caught)

for args, message in [
    ((), "descriptor '__invert__' of 'bool' object needs an argument"),
    ((1,), "descriptor '__invert__' requires a 'bool' object but received a 'int'"),
    ((False, True), "expected 0 arguments, got 1"),
]:
    try:
        bool.__invert__(*args)
    except TypeError as error:
        assert str(error) == message
    else:
        raise AssertionError(args)
assert str(True) == "True"
assert str(False) == "False"
assert bool.__doc__.startswith("Returns True when the argument is true")
print("bool 3.14 surface: ok")
