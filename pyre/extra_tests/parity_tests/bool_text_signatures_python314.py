"""CPython 3.14 text signatures for bool's own descriptors."""

import inspect


EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__invert__": "($self, /)",
    "__and__": "($self, value, /)",
    "__rand__": "($self, value, /)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__xor__": "($self, value, /)",
    "__rxor__": "($self, value, /)",
}

for name, signature in EXPECTED.items():
    descriptor = bool.__dict__[name]
    assert descriptor.__text_signature__ == signature, name

assert str(inspect.signature(bool.__repr__)) == "(self, /)"
assert str(inspect.signature(bool.__and__)) == "(self, value, /)"

print("OK")
