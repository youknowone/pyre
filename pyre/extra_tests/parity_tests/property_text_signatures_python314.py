"""CPython 3.14 text signatures for property descriptors."""

import inspect


EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__init__": "($self, /, *args, **kwargs)",
    "__get__": "($self, instance, owner=None, /)",
    "__set__": "($self, instance, value, /)",
    "__delete__": "($self, instance, /)",
    "getter": "($self, object, /)",
    "setter": "($self, object, /)",
    "deleter": "($self, object, /)",
    "__set_name__": "($self, owner, name, /)",
}

for name, signature in EXPECTED.items():
    assert property.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(property.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(property.setter)) == "(self, object, /)"
assert str(inspect.signature(property.__set_name__)) == "(self, owner, name, /)"

print("OK")
