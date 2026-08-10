"""CPython 3.14 text signatures for type descriptors."""

import inspect


EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__call__": "($self, /, *args, **kwargs)",
    "__getattribute__": "($self, name, /)",
    "__setattr__": "($self, name, value, /)",
    "__delattr__": "($self, name, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "mro": "($self, /)",
    "__subclasses__": "($self, /)",
    "__prepare__": "($cls, name, bases, /, **kwds)",
    "__instancecheck__": "($self, instance, /)",
    "__subclasscheck__": "($self, subclass, /)",
    "__dir__": "($self, /)",
    "__sizeof__": "($self, /)",
}

for name, signature in EXPECTED.items():
    assert type.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(type.__call__)) == "(self, /, *args, **kwargs)"
assert str(inspect.signature(type.__prepare__)) == "(name, bases, /, **kwds)"
assert str(inspect.signature(type.__instancecheck__)) == "(self, instance, /)"

print("OK")
