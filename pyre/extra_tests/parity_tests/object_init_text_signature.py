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

assert str(inspect.signature(object.__init__)) == "(self, /, *args, **kwargs)"
# `object.__new__` is bound through its `staticmethod` shape, so `$type` is
# consumed and does not survive into the rendered signature.
assert str(inspect.signature(object.__new__)) == "(*args, **kwargs)"
assert str(inspect.signature(int.__dict__["__pow__"])) == "(self, value, mod=None, /)"
