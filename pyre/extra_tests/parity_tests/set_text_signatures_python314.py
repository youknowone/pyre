"""CPython 3.14 text signatures for set and frozenset descriptors."""

import inspect


COMMON = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__sub__": "($self, value, /)",
    "__rsub__": "($self, value, /)",
    "__and__": "($self, value, /)",
    "__rand__": "($self, value, /)",
    "__xor__": "($self, value, /)",
    "__rxor__": "($self, value, /)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__len__": "($self, /)",
    "__contains__": "($self, object, /)",
    "copy": "($self, /)",
    "difference": "($self, /, *others)",
    "intersection": "($self, /, *others)",
    "isdisjoint": "($self, other, /)",
    "issubset": "($self, other, /)",
    "issuperset": "($self, other, /)",
    "__reduce__": "($self, /)",
    "__sizeof__": "($self, /)",
    "symmetric_difference": "($self, other, /)",
    "union": "($self, /, *others)",
    "__class_getitem__": "($type, object, /)",
}

SET_ONLY = {
    "__init__": "($self, /, *args, **kwargs)",
    "__isub__": "($self, value, /)",
    "__iand__": "($self, value, /)",
    "__ixor__": "($self, value, /)",
    "__ior__": "($self, value, /)",
    "add": "($self, object, /)",
    "clear": "($self, /)",
    "discard": "($self, object, /)",
    "difference_update": "($self, /, *others)",
    "intersection_update": "($self, /, *others)",
    "pop": "($self, /)",
    "remove": "($self, object, /)",
    "symmetric_difference_update": "($self, other, /)",
    "update": "($self, /, *others)",
}

for cls, expected in (
    (set, {**COMMON, **SET_ONLY}),
    (frozenset, {**COMMON, "__hash__": "($self, /)"}),
):
    for name, signature in expected.items():
        assert cls.__dict__[name].__text_signature__ == signature, (cls, name)

assert str(inspect.signature(set.difference)) == "(self, /, *others)"
assert str(inspect.signature(set.update)) == "(self, /, *others)"
assert str(inspect.signature(frozenset.symmetric_difference)) == "(self, other, /)"
assert str(inspect.signature(frozenset.__class_getitem__)) == "(object, /)"

print("OK")
