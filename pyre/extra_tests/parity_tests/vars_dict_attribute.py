"""`vars` answers for every object carrying a `__dict__`, exceptions included.

`app_inspect.py:21-24 vars` reads `obj.__dict__` through the attribute protocol
and turns only the resulting AttributeError into a TypeError.
"""

import sys


class Plain:
    pass


instance = Plain()
instance.a = 1
assert vars(instance) == {"a": 1}
assert vars(instance) is instance.__dict__

assert vars(Plain)["__module__"] == __name__
assert vars(sys)["maxsize"] == sys.maxsize

error = ValueError("v")
assert vars(error) == {}
error.note = "n"
assert vars(error) == {"note": "n"}

group = ExceptionGroup("m", [ValueError("v")])
assert isinstance(vars(group), dict)

for argument in (1, [], "text", (), object()):
    try:
        vars(argument)
    except TypeError as failure:
        assert str(failure) == "vars() argument must have __dict__ attribute", failure
    else:
        raise AssertionError(f"vars({argument!r}) must reject a dict-less object")


class Raising:
    @property
    def __dict__(self):
        raise RuntimeError("boom")


try:
    vars(Raising())
except RuntimeError as failure:
    assert str(failure) == "boom", failure
else:
    raise AssertionError("vars() must not swallow a non-AttributeError")

print("OK")
