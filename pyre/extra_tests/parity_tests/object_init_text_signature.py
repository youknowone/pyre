"""`object.__init__` is a slot wrapper, and a slot wrapper carries the text
signature `inspect.signature` parses. `object.__new__` and `int.__pow__` pin the
two neighbouring shapes that already worked.
"""

import inspect
import types

assert type(object.__dict__["__init__"]) is types.WrapperDescriptorType
assert type(object.__dict__["__new__"]) is types.BuiltinFunctionType

assert object.__init__.__text_signature__ == "($self, /, *args, **kwargs)"
assert object.__new__.__text_signature__ == "($type, *args, **kwargs)"
assert int.__dict__["__pow__"].__text_signature__ == "($self, value, mod=None, /)"

OBJECT_TEXT_SIGNATURES = {
    "__repr__": "($self, /)",
    "__str__": "($self, /)",
    "__hash__": "($self, /)",
    "__getattribute__": "($self, name, /)",
    "__setattr__": "($self, name, value, /)",
    "__delattr__": "($self, name, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__reduce__": "($self, /)",
    "__reduce_ex__": "($self, protocol, /)",
    "__getstate__": "($self, /)",
    "__format__": "($self, format_spec, /)",
    "__dir__": "($self, /)",
    # `__sizeof__` is deliberately absent here; the reference still carries it.
    "__init_subclass__": "($type, /)",
    "__subclasshook__": "($type, object, /)",
}

for name, text_signature in OBJECT_TEXT_SIGNATURES.items():
    assert object.__dict__[name].__text_signature__ == text_signature

assert str(inspect.signature(object.__init__)) == "(self, /, *args, **kwargs)"
# `object.__new__` is bound through its `staticmethod` shape, so `$type` is
# consumed and does not survive into the rendered signature.
assert str(inspect.signature(object.__new__)) == "(*args, **kwargs)"
assert str(inspect.signature(int.__dict__["__pow__"])) == "(self, value, mod=None, /)"

print("OK")
