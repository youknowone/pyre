"""CPython 3.14 text signatures for slice descriptors."""

import inspect


EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "indices": "($self, object, /)",
    "__reduce__": "($self, /)",
}

for name, signature in EXPECTED.items():
    assert slice.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(slice.__eq__)) == "(self, value, /)"
assert str(inspect.signature(slice.indices)) == "(self, object, /)"
assert str(inspect.signature(slice.__reduce__)) == "(self, /)"

print("OK")
