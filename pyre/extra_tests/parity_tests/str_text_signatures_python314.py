"""CPython 3.14 text signatures for str descriptors."""

import inspect


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

for name, signature in EXPECTED.items():
    assert str.__dict__[name].__text_signature__ == signature, name

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

print("OK")
