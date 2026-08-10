"""CPython 3.14 text signatures for int descriptors."""

import inspect


VALUE_BINARY = {
    name: "($self, value, /)"
    for name in (
        "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__",
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
        "__mod__", "__rmod__", "__divmod__", "__rdivmod__", "__lshift__",
        "__rlshift__", "__rshift__", "__rrshift__", "__and__", "__rand__",
        "__xor__", "__rxor__", "__or__", "__ror__", "__floordiv__",
        "__rfloordiv__", "__truediv__", "__rtruediv__",
    )
}
SELF_ONLY = {
    name: "($self, /)"
    for name in (
        "__repr__", "__hash__", "__neg__", "__pos__", "__abs__", "__bool__",
        "__invert__", "__int__", "__float__", "__index__", "conjugate",
        "bit_length", "bit_count", "as_integer_ratio", "__trunc__", "__floor__",
        "__ceil__", "__getnewargs__", "__sizeof__", "is_integer",
    )
}
EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    **VALUE_BINARY,
    **SELF_ONLY,
    "__pow__": "($self, value, mod=None, /)",
    "__rpow__": "($self, value, mod=None, /)",
    "to_bytes": "($self, /, length=1, byteorder='big', *, signed=False)",
    "from_bytes": "($type, /, bytes, byteorder='big', *, signed=False)",
    "__round__": "($self, ndigits=None, /)",
    "__format__": "($self, format_spec, /)",
}

for name, signature in EXPECTED.items():
    assert int.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(int.to_bytes)) == (
    "(self, /, length=1, byteorder='big', *, signed=False)"
)
assert str(inspect.signature(int.from_bytes)) == (
    "(bytes, byteorder='big', *, signed=False)"
)
assert str(inspect.signature(int.__pow__)) == "(self, value, mod=None, /)"

print("OK")
