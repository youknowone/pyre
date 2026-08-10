"""CPython 3.14 text signatures for tuple descriptors."""

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
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__getnewargs__": "($self, /)",
    "index": "($self, value, start=0, stop=sys.maxsize, /)",
    "count": "($self, value, /)",
    "__class_getitem__": "($type, object, /)",
}

for name, signature in EXPECTED.items():
    assert tuple.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(tuple.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(tuple.index)) == (
    "(self, value, start=0, stop=9223372036854775807, /)"
)
assert str(inspect.signature(tuple.__class_getitem__)) == "(object, /)"

print("OK")
