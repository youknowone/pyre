"""CPython 3.14 text signatures for super descriptors."""

import inspect


EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__init__": "($self, /, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__getattribute__": "($self, name, /)",
    "__get__": "($self, instance, owner=None, /)",
}

for name, signature in EXPECTED.items():
    assert super.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(super.__repr__)) == "(self, /)"
assert str(inspect.signature(super.__getattribute__)) == "(self, name, /)"
assert str(inspect.signature(super.__get__)) == "(self, instance, owner=None, /)"

print("OK")
