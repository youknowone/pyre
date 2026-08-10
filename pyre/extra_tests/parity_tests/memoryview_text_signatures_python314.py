"""CPython 3.14 text signatures for memoryview descriptors."""

import inspect
import sys


EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__buffer__": "($self, flags, /)",
    "__release_buffer__": "($self, buffer, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "release": "($self, /)",
    "tobytes": "($self, /, order='C')",
    "hex": "($self, /, sep=<unrepresentable>, bytes_per_sep=1)",
    "tolist": "($self, /)",
    "cast": "($self, /, format, shape=<unrepresentable>)",
    "toreadonly": "($self, /)",
    "_from_flags": "($type, /, object, flags)",
    "count": "($self, value, /)",
    "index": "($self, value, start=0, stop=sys.maxsize, /)",
    "__enter__": "($self, /)",
    "__exit__": "($self, /, *exc_info)",
    "__class_getitem__": "($type, object, /)",
}

for name, signature in EXPECTED.items():
    assert memoryview.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(memoryview.tobytes)) == "(self, /, order='C')"
assert str(inspect.signature(memoryview.index)) == (
    f"(self, value, start=0, stop={sys.maxsize}, /)"
)
assert str(inspect.signature(memoryview._from_flags)) == "(object, flags)"

print("memoryview text signatures Python 3.14 parity: ok")
