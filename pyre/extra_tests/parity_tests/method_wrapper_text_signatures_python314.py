"""CPython 3.14 signatures for staticmethod and classmethod wrappers."""

import inspect


EXPECTED = {
    staticmethod: {
        "__new__": "($type, *args, **kwargs)",
        "__repr__": "($self, /)",
        "__call__": "($self, /, *args, **kwargs)",
        "__get__": "($self, instance, owner=None, /)",
        "__init__": "($self, /, *args, **kwargs)",
        "__class_getitem__": "($type, object, /)",
    },
    classmethod: {
        "__new__": "($type, *args, **kwargs)",
        "__repr__": "($self, /)",
        "__get__": "($self, instance, owner=None, /)",
        "__init__": "($self, /, *args, **kwargs)",
        "__class_getitem__": "($type, object, /)",
    },
}

for typ, signatures in EXPECTED.items():
    for name, signature in signatures.items():
        assert typ.__dict__[name].__text_signature__ == signature, (typ, name)

assert str(inspect.signature(staticmethod.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(staticmethod.__call__)) == "(self, /, *args, **kwargs)"
assert str(inspect.signature(classmethod.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(classmethod.__class_getitem__)) == "(object, /)"

print("OK")
