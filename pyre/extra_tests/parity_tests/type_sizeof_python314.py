class WithDict:
    pass


class WithoutDict:
    __slots__ = ()


class WithSlot:
    __slots__ = ("value",)


class ExplicitDict:
    __slots__ = ("__dict__",)


for static_type in (object, type, int, str, list, dict):
    assert type.__sizeof__(static_type) == 416

assert type.__sizeof__(WithDict) == 1704
assert type.__sizeof__(WithoutDict) == 936
assert type.__sizeof__(WithSlot) == 936
assert type.__sizeof__(ExplicitDict) == 1704
assert type.__sizeof__.__text_signature__ == "($self, /)"
assert type.__sizeof__.__doc__ == "Return memory consumption of the type object."

try:
    type.__sizeof__(1)
except TypeError:
    pass
else:
    raise AssertionError("type.__sizeof__ accepted a non-type receiver")

print("OK")
