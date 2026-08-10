"""CPython 3.14 text signatures for bytearray descriptors."""

import inspect


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

for name, signature in EXPECTED.items():
    assert bytearray.__dict__[name].__text_signature__ == signature, name

raw_maketrans = bytearray.__dict__["maketrans"]
assert not hasattr(raw_maketrans, "__text_signature__")
assert bytearray.maketrans.__text_signature__ == "(frm, to, /)"
assert bytearray.__dict__["fromhex"].__text_signature__ == "($type, string, /)"

assert str(inspect.signature(bytearray.decode)) == (
    "(self, /, encoding='utf-8', errors='strict')"
)
assert str(inspect.signature(bytearray.resize)) == "(self, size, /)"
assert str(inspect.signature(bytearray.fromhex)) == "(string, /)"

print("OK")
