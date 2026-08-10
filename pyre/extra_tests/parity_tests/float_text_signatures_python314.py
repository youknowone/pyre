"""CPython 3.14 text signatures for float descriptors."""

import inspect


VALUE_BINARY = {
    name: "($self, value, /)"
    for name in (
        "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__",
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
        "__mod__", "__rmod__", "__divmod__", "__rdivmod__", "__floordiv__",
        "__rfloordiv__", "__truediv__", "__rtruediv__",
    )
}
SELF_ONLY = {
    name: "($self, /)"
    for name in (
        "__repr__", "__hash__", "__neg__", "__pos__", "__abs__", "__bool__",
        "__int__", "__float__", "conjugate", "__trunc__", "__floor__", "__ceil__",
        "as_integer_ratio", "hex", "is_integer", "__getnewargs__",
    )
}
EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    **VALUE_BINARY,
    **SELF_ONLY,
    "__pow__": "($self, value, mod=None, /)",
    "__rpow__": "($self, value, mod=None, /)",
    "from_number": "($type, number, /)",
    "__round__": "($self, ndigits=None, /)",
    "fromhex": "($type, string, /)",
    "__getformat__": "($type, typestr, /)",
    "__format__": "($self, format_spec, /)",
}

for name, signature in EXPECTED.items():
    assert float.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(float.from_number)) == "(number, /)"
assert str(inspect.signature(float.fromhex)) == "(string, /)"
assert str(inspect.signature(float.__pow__)) == "(self, value, mod=None, /)"

print("float text signatures Python 3.14 parity: ok")
