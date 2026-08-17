# CPython-suite gap: exact builtin and descriptor text signatures are not asserted.
# parity-tests reason: guard pyre's CPython 3.14 signature metadata surface.

"""CPython 3.14 text signatures for builtins and builtin types.

These cases share one process and one descriptor-checking harness so adding a
type does not duplicate the test machinery or interpreter startup.
"""

import builtins
import inspect
import sys


def check_descriptors(owner, expected):
    for name, signature in expected.items():
        descriptor = owner.__dict__[name]
        actual = descriptor.__text_signature__
        assert actual == signature, (owner, name, actual)


# bool_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__invert__": "($self, /)",
    "__and__": "($self, value, /)",
    "__rand__": "($self, value, /)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__xor__": "($self, value, /)",
    "__rxor__": "($self, value, /)",
}

check_descriptors(bool, EXPECTED)

assert str(inspect.signature(bool.__repr__)) == "(self, /)"
assert str(inspect.signature(bool.__and__)) == "(self, value, /)"


# builtin_text_signatures_python314

EXPECTED = {
    "__import__": "($module, /, name, globals=None, locals=None, fromlist=(),\n           level=0)",
    "abs": "($module, x, /)",
    "aiter": "($module, async_iterable, /)",
    "all": "($module, iterable, /)",
    "anext": "($module, aiterator, default=<unrepresentable>, /)",
    "any": "($module, iterable, /)",
    "ascii": "($module, obj, /)",
    "bin": "($module, number, /)",
    "breakpoint": "($module, /, *args, **kws)",
    "callable": "($module, obj, /)",
    "chr": "($module, i, /)",
    "compile": "($module, /, source, filename, mode, flags=0,\n        dont_inherit=False, optimize=-1, *, _feature_version=-1)",
    "delattr": "($module, obj, name, /)",
    "divmod": "($module, x, y, /)",
    "eval": "($module, source, /, globals=None, locals=None)",
    "exec": "($module, source, /, globals=None, locals=None, *, closure=None)",
    "format": "($module, value, format_spec='', /)",
    "globals": "($module, /)",
    "hasattr": "($module, obj, name, /)",
    "hash": "($module, obj, /)",
    "hex": "($module, number, /)",
    "id": "($module, obj, /)",
    "input": "($module, prompt='', /)",
    "isinstance": "($module, obj, class_or_tuple, /)",
    "issubclass": "($module, cls, class_or_tuple, /)",
    "len": "($module, obj, /)",
    "locals": "($module, /)",
    "oct": "($module, number, /)",
    "open": "($module, /, file, mode='r', buffering=-1, encoding=None,\n     errors=None, newline=None, closefd=True, opener=None)",
    "ord": "($module, character, /)",
    "pow": "($module, /, base, exp, mod=None)",
    "print": "($module, /, *args, sep=' ', end='\\n', file=None, flush=False)",
    "repr": "($module, obj, /)",
    "round": "($module, /, number, ndigits=None)",
    "setattr": "($module, obj, name, value, /)",
    "sorted": "($module, iterable, /, *, key=None, reverse=False)",
    "sum": "($module, iterable, /, start=0)",
}

for name, signature in EXPECTED.items():
    assert getattr(builtins, name).__text_signature__ == signature, name

for name in ("__build_class__", "dir", "getattr", "iter", "max", "min", "next", "vars"):
    assert getattr(builtins, name).__text_signature__ is None, name

assert str(inspect.signature(len)) == "(obj, /)"
assert str(inspect.signature(sorted)) == (
    "(iterable, /, *, key=None, reverse=False)"
)
assert str(inspect.signature(open)) == (
    "(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, "
    "closefd=True, opener=None)"
)
assert str(inspect.signature(print)) == (
    "(*args, sep=' ', end='\\n', file=None, flush=False)"
)


# bytearray_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__str__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__buffer__": "($self, flags, /)",
    "__release_buffer__": "($self, buffer, /)",
    "__mod__": "($self, value, /)",
    "__rmod__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__iadd__": "($self, value, /)",
    "__imul__": "($self, value, /)",
    "__alloc__": "($self, /)",
    "__reduce__": "($self, /)",
    "__reduce_ex__": "($self, proto=0, /)",
    "__sizeof__": "($self, /)",
    "append": "($self, item, /)",
    "capitalize": "($self, /)",
    "center": "($self, width, fillchar=b' ', /)",
    "clear": "($self, /)",
    "copy": "($self, /)",
    "count": "($self, sub[, start[, end]], /)",
    "decode": "($self, /, encoding='utf-8', errors='strict')",
    "endswith": "($self, suffix[, start[, end]], /)",
    "expandtabs": "($self, /, tabsize=8)",
    "extend": "($self, iterable_of_ints, /)",
    "find": "($self, sub[, start[, end]], /)",
    "hex": "($self, /, sep=<unrepresentable>, bytes_per_sep=1)",
    "index": "($self, sub[, start[, end]], /)",
    "insert": "($self, index, item, /)",
    "isalnum": "($self, /)",
    "isalpha": "($self, /)",
    "isascii": "($self, /)",
    "isdigit": "($self, /)",
    "islower": "($self, /)",
    "isspace": "($self, /)",
    "istitle": "($self, /)",
    "isupper": "($self, /)",
    "join": "($self, iterable_of_bytes, /)",
    "ljust": "($self, width, fillchar=b' ', /)",
    "lower": "($self, /)",
    "lstrip": "($self, bytes=None, /)",
    "partition": "($self, sep, /)",
    "pop": "($self, index=-1, /)",
    "remove": "($self, value, /)",
    "replace": "($self, old, new, count=-1, /)",
    "removeprefix": "($self, prefix, /)",
    "removesuffix": "($self, suffix, /)",
    "resize": "($self, size, /)",
    "reverse": "($self, /)",
    "rfind": "($self, sub[, start[, end]], /)",
    "rindex": "($self, sub[, start[, end]], /)",
    "rjust": "($self, width, fillchar=b' ', /)",
    "rpartition": "($self, sep, /)",
    "rsplit": "($self, /, sep=None, maxsplit=-1)",
    "rstrip": "($self, bytes=None, /)",
    "split": "($self, /, sep=None, maxsplit=-1)",
    "splitlines": "($self, /, keepends=False)",
    "startswith": "($self, prefix[, start[, end]], /)",
    "strip": "($self, bytes=None, /)",
    "swapcase": "($self, /)",
    "title": "($self, /)",
    "translate": "($self, table, /, delete=b'')",
    "upper": "($self, /)",
    "zfill": "($self, width, /)",
}

check_descriptors(bytearray, EXPECTED)

raw_maketrans = bytearray.__dict__["maketrans"]
assert not hasattr(raw_maketrans, "__text_signature__")
assert bytearray.maketrans.__text_signature__ == "(frm, to, /)"
assert bytearray.__dict__["fromhex"].__text_signature__ == "($type, string, /)"

assert str(inspect.signature(bytearray.decode)) == (
    "(self, /, encoding='utf-8', errors='strict')"
)
assert str(inspect.signature(bytearray.resize)) == "(self, size, /)"
assert str(inspect.signature(bytearray.fromhex)) == "(string, /)"


# bytes_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__str__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__buffer__": "($self, flags, /)",
    "__mod__": "($self, value, /)",
    "__rmod__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__getnewargs__": "($self, /)",
    "__bytes__": "($self, /)",
    "capitalize": "($self, /)",
    "center": "($self, width, fillchar=b' ', /)",
    "count": "($self, sub[, start[, end]], /)",
    "decode": "($self, /, encoding='utf-8', errors='strict')",
    "endswith": "($self, suffix[, start[, end]], /)",
    "expandtabs": "($self, /, tabsize=8)",
    "find": "($self, sub[, start[, end]], /)",
    "hex": "($self, /, sep=<unrepresentable>, bytes_per_sep=1)",
    "index": "($self, sub[, start[, end]], /)",
    "isalnum": "($self, /)",
    "isalpha": "($self, /)",
    "isascii": "($self, /)",
    "isdigit": "($self, /)",
    "islower": "($self, /)",
    "isspace": "($self, /)",
    "istitle": "($self, /)",
    "isupper": "($self, /)",
    "join": "($self, iterable_of_bytes, /)",
    "ljust": "($self, width, fillchar=b' ', /)",
    "lower": "($self, /)",
    "lstrip": "($self, bytes=None, /)",
    "partition": "($self, sep, /)",
    "replace": "($self, old, new, count=-1, /)",
    "removeprefix": "($self, prefix, /)",
    "removesuffix": "($self, suffix, /)",
    "rfind": "($self, sub[, start[, end]], /)",
    "rindex": "($self, sub[, start[, end]], /)",
    "rjust": "($self, width, fillchar=b' ', /)",
    "rpartition": "($self, sep, /)",
    "rsplit": "($self, /, sep=None, maxsplit=-1)",
    "rstrip": "($self, bytes=None, /)",
    "split": "($self, /, sep=None, maxsplit=-1)",
    "splitlines": "($self, /, keepends=False)",
    "startswith": "($self, prefix[, start[, end]], /)",
    "strip": "($self, bytes=None, /)",
    "swapcase": "($self, /)",
    "title": "($self, /)",
    "translate": "($self, table, /, delete=b'')",
    "upper": "($self, /)",
    "zfill": "($self, width, /)",
}

check_descriptors(bytes, EXPECTED)

raw_maketrans = bytes.__dict__["maketrans"]
assert not hasattr(raw_maketrans, "__text_signature__")
assert bytes.maketrans.__text_signature__ == "(frm, to, /)"
assert bytes.__dict__["fromhex"].__text_signature__ == "($type, string, /)"

assert str(inspect.signature(bytes.decode)) == (
    "(self, /, encoding='utf-8', errors='strict')"
)
assert str(inspect.signature(bytes.replace)) == "(self, old, new, count=-1, /)"
assert str(inspect.signature(bytes.fromhex)) == "(string, /)"


# complex_text_signatures_python314

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

check_descriptors(complex, EXPECTED)

assert str(inspect.signature(complex.from_number)) == "(number, /)"
assert str(inspect.signature(complex.__pow__)) == "(self, value, mod=None, /)"
assert str(inspect.signature(complex.conjugate)) == "(self, /)"


# dict_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__ior__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "__contains__": "($self, key, /)",
    "__sizeof__": "($self, /)",
    "get": "($self, key, default=None, /)",
    "setdefault": "($self, key, default=None, /)",
    "pop": "($self, key, default=<unrepresentable>, /)",
    "popitem": "($self, /)",
    "keys": "($self, /)",
    "items": "($self, /)",
    "values": "($self, /)",
    "update": None,
    "fromkeys": "($type, iterable, value=None, /)",
    "clear": "($self, /)",
    "copy": "($self, /)",
    "__reversed__": "($self, /)",
    "__class_getitem__": "($type, object, /)",
}

check_descriptors(dict, EXPECTED)

assert str(inspect.signature(dict.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(dict.get)) == "(self, key, default=None, /)"
assert str(inspect.signature(dict.fromkeys)) == "(iterable, value=None, /)"
assert str(inspect.signature(dict.__class_getitem__)) == "(object, /)"


# float_text_signatures_python314

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

check_descriptors(float, EXPECTED)

assert str(inspect.signature(float.from_number)) == "(number, /)"
assert str(inspect.signature(float.fromhex)) == "(string, /)"
assert str(inspect.signature(float.__pow__)) == "(self, value, mod=None, /)"


# functional_iterator_text_signatures_python314

METHODS = {
    enumerate: ("__iter__", "__next__", "__reduce__", "__class_getitem__"),
    reversed: (
        "__iter__",
        "__next__",
        "__length_hint__",
        "__reduce__",
        "__setstate__",
    ),
    map: ("__iter__", "__next__", "__reduce__", "__setstate__"),
    filter: ("__iter__", "__next__", "__reduce__"),
    zip: ("__iter__", "__next__", "__reduce__", "__setstate__"),
}

for typ, names in METHODS.items():
    assert typ.__dict__["__new__"].__text_signature__ == "($type, *args, **kwargs)"
    for name in names:
        expected = (
            "($type, object, /)"
            if name == "__class_getitem__"
            else "($self, object, /)"
            if name == "__setstate__"
            else "($self, /)"
        )
        assert typ.__dict__[name].__text_signature__ == expected, (typ, name)

assert str(inspect.signature(enumerate.__next__)) == "(self, /)"
assert str(inspect.signature(reversed.__setstate__)) == "(self, object, /)"
assert str(inspect.signature(enumerate.__class_getitem__)) == "(object, /)"


# int_text_signatures_python314

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

check_descriptors(int, EXPECTED)

assert str(inspect.signature(int.to_bytes)) == (
    "(self, /, length=1, byteorder='big', *, signed=False)"
)
assert str(inspect.signature(int.from_bytes)) == (
    "(bytes, byteorder='big', *, signed=False)"
)
assert str(inspect.signature(int.__pow__)) == "(self, value, mod=None, /)"


# list_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__len__": "($self, /)",
    "__getitem__": "($self, index, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__iadd__": "($self, value, /)",
    "__imul__": "($self, value, /)",
    "__reversed__": "($self, /)",
    "__sizeof__": "($self, /)",
    "clear": "($self, /)",
    "copy": "($self, /)",
    "append": "($self, object, /)",
    "insert": "($self, index, object, /)",
    "extend": "($self, iterable, /)",
    "pop": "($self, index=-1, /)",
    "remove": "($self, value, /)",
    "index": "($self, value, start=0, stop=sys.maxsize, /)",
    "count": "($self, value, /)",
    "reverse": "($self, /)",
    "sort": "($self, /, *, key=None, reverse=False)",
    "__class_getitem__": "($type, object, /)",
}

check_descriptors(list, EXPECTED)

assert str(inspect.signature(list.__getitem__)) == "(self, index, /)"
assert str(inspect.signature(list.index)) == (
    "(self, value, start=0, stop=9223372036854775807, /)"
)
assert str(inspect.signature(list.sort)) == (
    "(self, /, *, key=None, reverse=False)"
)
assert str(inspect.signature(list.__class_getitem__)) == "(object, /)"


# memoryview_text_signatures_python314

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

check_descriptors(memoryview, EXPECTED)

assert str(inspect.signature(memoryview.tobytes)) == "(self, /, order='C')"
assert str(inspect.signature(memoryview.index)) == (
    f"(self, value, start=0, stop={sys.maxsize}, /)"
)
assert str(inspect.signature(memoryview._from_flags)) == "(object, flags)"


# method_wrapper_text_signatures_python314

EXPECTED = {
    staticmethod: {
        "__new__": "($type, *args, **kwargs)",
        "__repr__": "($self, /)",
        "__call__": "($self, /, *args, **kwargs)",
        "__get__": "($self, instance, owner=None, /)",
        "__init__": "($self, /, *args, **kwargs)",
        "__class_getitem__": "($type, object, /)",
    },
    classmethod: {
        "__new__": "($type, *args, **kwargs)",
        "__repr__": "($self, /)",
        "__get__": "($self, instance, owner=None, /)",
        "__init__": "($self, /, *args, **kwargs)",
        "__class_getitem__": "($type, object, /)",
    },
}

for typ, signatures in EXPECTED.items():
    for name, signature in signatures.items():
        assert typ.__dict__[name].__text_signature__ == signature, (typ, name)

assert str(inspect.signature(staticmethod.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(staticmethod.__call__)) == "(self, /, *args, **kwargs)"
assert str(inspect.signature(classmethod.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(classmethod.__class_getitem__)) == "(object, /)"


# property_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__init__": "($self, /, *args, **kwargs)",
    "__get__": "($self, instance, owner=None, /)",
    "__set__": "($self, instance, value, /)",
    "__delete__": "($self, instance, /)",
    "getter": "($self, object, /)",
    "setter": "($self, object, /)",
    "deleter": "($self, object, /)",
    "__set_name__": "($self, owner, name, /)",
}

check_descriptors(property, EXPECTED)

assert str(inspect.signature(property.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(property.setter)) == "(self, object, /)"
assert str(inspect.signature(property.__set_name__)) == "(self, owner, name, /)"


# range_text_signatures_python314

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
    "__bool__": "($self, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__contains__": "($self, key, /)",
    "__reversed__": "($self, /)",
    "__reduce__": "($self, /)",
    "count": "($self, object, /)",
    "index": "($self, object, /)",
}

check_descriptors(range, EXPECTED)

assert str(inspect.signature(range.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(range.__contains__)) == "(self, key, /)"
assert str(inspect.signature(range.count)) == "(self, object, /)"


# set_text_signatures_python314

COMMON = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__sub__": "($self, value, /)",
    "__rsub__": "($self, value, /)",
    "__and__": "($self, value, /)",
    "__rand__": "($self, value, /)",
    "__xor__": "($self, value, /)",
    "__rxor__": "($self, value, /)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__len__": "($self, /)",
    "__contains__": "($self, object, /)",
    "copy": "($self, /)",
    "difference": "($self, /, *others)",
    "intersection": "($self, /, *others)",
    "isdisjoint": "($self, other, /)",
    "issubset": "($self, other, /)",
    "issuperset": "($self, other, /)",
    "__reduce__": "($self, /)",
    "__sizeof__": "($self, /)",
    "symmetric_difference": "($self, other, /)",
    "union": "($self, /, *others)",
    "__class_getitem__": "($type, object, /)",
}

SET_ONLY = {
    "__init__": "($self, /, *args, **kwargs)",
    "__isub__": "($self, value, /)",
    "__iand__": "($self, value, /)",
    "__ixor__": "($self, value, /)",
    "__ior__": "($self, value, /)",
    "add": "($self, object, /)",
    "clear": "($self, /)",
    "discard": "($self, object, /)",
    "difference_update": "($self, /, *others)",
    "intersection_update": "($self, /, *others)",
    "pop": "($self, /)",
    "remove": "($self, object, /)",
    "symmetric_difference_update": "($self, other, /)",
    "update": "($self, /, *others)",
}

for cls, expected in (
    (set, {**COMMON, **SET_ONLY}),
    (frozenset, {**COMMON, "__hash__": "($self, /)"}),
):
    for name, signature in expected.items():
        assert cls.__dict__[name].__text_signature__ == signature, (cls, name)

assert str(inspect.signature(set.difference)) == "(self, /, *others)"
assert str(inspect.signature(set.update)) == "(self, /, *others)"
assert str(inspect.signature(frozenset.symmetric_difference)) == "(self, other, /)"
assert str(inspect.signature(frozenset.__class_getitem__)) == "(object, /)"


# slice_text_signatures_python314

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
    "indices": "($self, object, /)",
    "__reduce__": "($self, /)",
}

check_descriptors(slice, EXPECTED)

assert str(inspect.signature(slice.__eq__)) == "(self, value, /)"
assert str(inspect.signature(slice.indices)) == "(self, object, /)"
assert str(inspect.signature(slice.__reduce__)) == "(self, /)"


# str_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__str__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__mod__": "($self, value, /)",
    "__rmod__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "encode": "($self, /, encoding='utf-8', errors='strict')",
    "replace": "($self, old, new, /, count=-1)",
    "split": "($self, /, sep=None, maxsplit=-1)",
    "rsplit": "($self, /, sep=None, maxsplit=-1)",
    "join": "($self, iterable, /)",
    "capitalize": "($self, /)",
    "casefold": "($self, /)",
    "title": "($self, /)",
    "center": "($self, width, fillchar=' ', /)",
    "count": "($self, sub[, start[, end]], /)",
    "expandtabs": "($self, /, tabsize=8)",
    "find": "($self, sub[, start[, end]], /)",
    "partition": "($self, sep, /)",
    "index": "($self, sub[, start[, end]], /)",
    "ljust": "($self, width, fillchar=' ', /)",
    "lower": "($self, /)",
    "lstrip": "($self, chars=None, /)",
    "rfind": "($self, sub[, start[, end]], /)",
    "rindex": "($self, sub[, start[, end]], /)",
    "rjust": "($self, width, fillchar=' ', /)",
    "rstrip": "($self, chars=None, /)",
    "rpartition": "($self, sep, /)",
    "splitlines": "($self, /, keepends=False)",
    "strip": "($self, chars=None, /)",
    "swapcase": "($self, /)",
    "translate": "($self, table, /)",
    "upper": "($self, /)",
    "startswith": "($self, prefix[, start[, end]], /)",
    "endswith": "($self, suffix[, start[, end]], /)",
    "removeprefix": "($self, prefix, /)",
    "removesuffix": "($self, suffix, /)",
    "isascii": "($self, /)",
    "islower": "($self, /)",
    "isupper": "($self, /)",
    "istitle": "($self, /)",
    "isspace": "($self, /)",
    "isdecimal": "($self, /)",
    "isdigit": "($self, /)",
    "isnumeric": "($self, /)",
    "isalpha": "($self, /)",
    "isalnum": "($self, /)",
    "isidentifier": "($self, /)",
    "isprintable": "($self, /)",
    "zfill": "($self, width, /)",
    "format": "($self, /, *args, **kwargs)",
    "format_map": "($self, mapping, /)",
    "__format__": "($self, format_spec, /)",
    "__sizeof__": "($self, /)",
    "__getnewargs__": "($self, /)",
}

check_descriptors(str, EXPECTED)

raw_maketrans = str.__dict__["maketrans"]
assert not hasattr(raw_maketrans, "__text_signature__")
assert str.maketrans.__text_signature__ == (
    "(x, y=<unrepresentable>, z=<unrepresentable>, /)"
)

assert str(inspect.signature(str.encode)) == (
    "(self, /, encoding='utf-8', errors='strict')"
)
assert str(inspect.signature(str.replace)) == "(self, old, new, /, count=-1)"
assert str(inspect.signature(str.format)) == "(self, /, *args, **kwargs)"


# super_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__init__": "($self, /, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__getattribute__": "($self, name, /)",
    "__get__": "($self, instance, owner=None, /)",
}

check_descriptors(super, EXPECTED)

assert str(inspect.signature(super.__repr__)) == "(self, /)"
assert str(inspect.signature(super.__getattribute__)) == "(self, name, /)"
assert str(inspect.signature(super.__get__)) == "(self, instance, owner=None, /)"


# tuple_text_signatures_python314

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
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__getnewargs__": "($self, /)",
    "index": "($self, value, start=0, stop=sys.maxsize, /)",
    "count": "($self, value, /)",
    "__class_getitem__": "($type, object, /)",
}

check_descriptors(tuple, EXPECTED)

assert str(inspect.signature(tuple.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(tuple.index)) == (
    "(self, value, start=0, stop=9223372036854775807, /)"
)
assert str(inspect.signature(tuple.__class_getitem__)) == "(object, /)"


# type_text_signatures_python314

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__call__": "($self, /, *args, **kwargs)",
    "__getattribute__": "($self, name, /)",
    "__setattr__": "($self, name, value, /)",
    "__delattr__": "($self, name, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "mro": "($self, /)",
    "__subclasses__": "($self, /)",
    "__prepare__": "($cls, name, bases, /, **kwds)",
    "__instancecheck__": "($self, instance, /)",
    "__subclasscheck__": "($self, subclass, /)",
    "__dir__": "($self, /)",
    "__sizeof__": "($self, /)",
}

check_descriptors(type, EXPECTED)

assert str(inspect.signature(type.__call__)) == "(self, /, *args, **kwargs)"
assert str(inspect.signature(type.__prepare__)) == "(name, bases, /, **kwds)"
assert str(inspect.signature(type.__instancecheck__)) == "(self, instance, /)"

print("OK")
