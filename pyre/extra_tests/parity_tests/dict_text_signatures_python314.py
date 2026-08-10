"""CPython 3.14 text signatures for dict descriptors."""

import inspect


EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__ior__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "__contains__": "($self, key, /)",
    "__sizeof__": "($self, /)",
    "get": "($self, key, default=None, /)",
    "setdefault": "($self, key, default=None, /)",
    "pop": "($self, key, default=<unrepresentable>, /)",
    "popitem": "($self, /)",
    "keys": "($self, /)",
    "items": "($self, /)",
    "values": "($self, /)",
    "update": None,
    "fromkeys": "($type, iterable, value=None, /)",
    "clear": "($self, /)",
    "copy": "($self, /)",
    "__reversed__": "($self, /)",
    "__class_getitem__": "($type, object, /)",
}

for name, signature in EXPECTED.items():
    assert dict.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(dict.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(dict.get)) == "(self, key, default=None, /)"
assert str(inspect.signature(dict.fromkeys)) == "(iterable, value=None, /)"
assert str(inspect.signature(dict.__class_getitem__)) == "(object, /)"

print("dict text signatures Python 3.14 parity: ok")
