"""dict TypeDef surface: PyPy structure plus CPython 3.14 additions."""


surface = {
    "__class_getitem__",
    "__contains__",
    "__delitem__",
    "__doc__",
    "__eq__",
    "__ge__",
    "__getitem__",
    "__gt__",
    "__hash__",
    "__init__",
    "__ior__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__ne__",
    "__new__",
    "__or__",
    "__repr__",
    "__reversed__",
    "__ror__",
    "__setitem__",
    "__sizeof__",
    "clear",
    "copy",
    "fromkeys",
    "get",
    "items",
    "keys",
    "pop",
    "popitem",
    "setdefault",
    "update",
    "values",
}
assert surface <= set(dict.__dict__)
assert dict.__hash__ is None

for name in ("__lt__", "__le__", "__gt__", "__ge__"):
    assert getattr(dict, name)({}, {}) is NotImplemented

assert dict.__eq__({"x": 1}, {"x": 1}) is True
assert dict.__ne__({"x": 1}, {"x": 2}) is True
assert dict.__eq__({}, []) is NotImplemented
assert dict.__ne__({}, []) is NotImplemented

try:
    hash({})
except TypeError:
    pass
else:
    raise AssertionError("dict must remain unhashable")

assert {}.__sizeof__() > 0
assert {"x": 1}.__sizeof__() > {}.__sizeof__()
assert dict.__doc__.startswith("dict() -> new empty dictionary")
print("dict surface: ok")
