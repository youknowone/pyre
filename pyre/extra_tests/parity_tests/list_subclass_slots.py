"""Native list subclasses keep PyPy's instance-owned slot storage."""


class SlottedList(list):
    __slots__ = ("first", "second")


value = SlottedList([1, 2, 3])
assert not hasattr(value, "__dict__")

try:
    value.first
except AttributeError:
    pass
else:
    raise AssertionError("an unassigned list-subclass slot must be unbound")

value.first = "alpha"
value.second = ["beta"]
assert value.first == "alpha"
assert value.second == ["beta"]

del value.first
try:
    value.first
except AttributeError:
    pass
else:
    raise AssertionError("a deleted list-subclass slot must be unbound")

print("OK")
