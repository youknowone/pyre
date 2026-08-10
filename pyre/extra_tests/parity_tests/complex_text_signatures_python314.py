"""CPython 3.14 text signatures for complex descriptors."""

import inspect


VALUE_BINARY = {
    name: "($self, value, /)"
    for name in (
        "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__",
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
        "__truediv__", "__rtruediv__",
    )
}
SELF_ONLY = {
    name: "($self, /)"
    for name in (
        "__repr__", "__hash__", "__neg__", "__pos__", "__abs__", "__bool__",
        "conjugate", "__complex__", "__getnewargs__",
    )
}
EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    **VALUE_BINARY,
    **SELF_ONLY,
    "__pow__": "($self, value, mod=None, /)",
    "__rpow__": "($self, value, mod=None, /)",
    "from_number": "($type, number, /)",
    "__format__": "($self, format_spec, /)",
}

for name, signature in EXPECTED.items():
    assert complex.__dict__[name].__text_signature__ == signature, name

assert str(inspect.signature(complex.from_number)) == "(number, /)"
assert str(inspect.signature(complex.__pow__)) == "(self, value, mod=None, /)"
assert str(inspect.signature(complex.conjugate)) == "(self, /)"

print("complex text signatures Python 3.14 parity: ok")
