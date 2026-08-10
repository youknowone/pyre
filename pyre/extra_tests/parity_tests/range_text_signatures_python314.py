"""CPython 3.14 text signatures for range descriptors."""

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
    "__iter__": "($self, /)",
    "__bool__": "($self, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__contains__": "($self, key, /)",
    "__reversed__": "($self, /)",
    "__reduce__": "($self, /)",
    "count": "($self, object, /)",
    "index": "($self, object, /)",
}

for name, signature in EXPECTED.items():
    assert range.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(range.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(range.__contains__)) == "(self, key, /)"
assert str(inspect.signature(range.count)) == "(self, object, /)"

print("range text signatures Python 3.14 parity: ok")
