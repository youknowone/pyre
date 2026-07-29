"""CPython 3.14 pickle BUILD and sys.intern share string identity."""

import pickle
import sys


class Value:
    pass


for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    original = Value()
    original.attribute_name = protocol
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    original_key = next(iter(original.__dict__))
    restored_key = next(iter(restored.__dict__))
    assert restored_key is original_key
    assert sys.intern(restored_key) is original_key

dynamic = "".join(("never-", "interned-", str(id(Value))))
assert sys.intern(dynamic) is dynamic
same_value = dynamic.swapcase().swapcase()
assert same_value == dynamic and same_value is not dynamic
assert sys.intern(same_value) is dynamic

try:
    sys.intern(b"bytes")
except TypeError:
    pass
else:
    raise AssertionError("sys.intern accepted bytes")


class StringSubclass(str):
    pass


try:
    sys.intern(StringSubclass("value"))
except TypeError as exc:
    assert str(exc) == "can't intern StringSubclass"
else:
    raise AssertionError("sys.intern accepted a str subclass")

print("OK")
