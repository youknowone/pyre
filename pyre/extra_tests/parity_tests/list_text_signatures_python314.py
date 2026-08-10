"""CPython 3.14 text signatures for list descriptors."""

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
    "__len__": "($self, /)",
    "__getitem__": "($self, index, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__iadd__": "($self, value, /)",
    "__imul__": "($self, value, /)",
    "__reversed__": "($self, /)",
    "__sizeof__": "($self, /)",
    "clear": "($self, /)",
    "copy": "($self, /)",
    "append": "($self, object, /)",
    "insert": "($self, index, object, /)",
    "extend": "($self, iterable, /)",
    "pop": "($self, index=-1, /)",
    "remove": "($self, value, /)",
    "index": "($self, value, start=0, stop=sys.maxsize, /)",
    "count": "($self, value, /)",
    "reverse": "($self, /)",
    "sort": "($self, /, *, key=None, reverse=False)",
    "__class_getitem__": "($type, object, /)",
}

for name, signature in EXPECTED.items():
    assert list.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(list.__getitem__)) == "(self, index, /)"
assert str(inspect.signature(list.index)) == (
    "(self, value, start=0, stop=9223372036854775807, /)"
)
assert str(inspect.signature(list.sort)) == (
    "(self, /, *, key=None, reverse=False)"
)
assert str(inspect.signature(list.__class_getitem__)) == "(object, /)"

print("list text signatures Python 3.14 parity: ok")
