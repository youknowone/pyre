import types


EXPECTED = {
    "__class_getitem__",
    "__contains__",
    "__doc__",
    "__eq__",
    "__ge__",
    "__getitem__",
    "__gt__",
    "__hash__",
    "__ior__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__ne__",
    "__new__",
    "__or__",
    "__repr__",
    "__reversed__",
    "__ror__",
    "__str__",
    "copy",
    "get",
    "items",
    "keys",
    "values",
}

assert set(types.MappingProxyType.__dict__) == EXPECTED
assert types.MappingProxyType.__doc__ == "Read-only proxy of a mapping."

proxy = types.MappingProxyType({"x": 1})
try:
    hash(proxy)
except TypeError as exc:
    assert str(exc) == "unhashable type: 'dict'"
else:
    raise AssertionError("mappingproxy(dict) was hashable")


class HashableMapping:
    def __getitem__(self, key):
        return key

    def __hash__(self):
        return 42


hashable_proxy = types.MappingProxyType(HashableMapping())
assert hash(hashable_proxy) == 42
assert types.MappingProxyType.__hash__(hashable_proxy) == 42

try:
    types.MappingProxyType.__hash__({})
except TypeError as exc:
    assert str(exc) == (
        "descriptor '__hash__' requires a 'mappingproxy' object "
        "but received a 'dict'"
    )
else:
    raise AssertionError("mappingproxy.__hash__ accepted dict")

print("OK")
