"""CPython 3.14 exposes the native struct unpack iterator surface."""

import struct


iterator = struct.iter_unpack("b", b"ab")
typ = type(iterator)
assert typ.__module__ == "_struct"
assert typ.__name__ == "unpack_iterator"
assert "__getattribute__" in typ.__dict__
assert list(iterator) == [(97,), (98,)]
print("OK")
