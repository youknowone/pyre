"""PyPy W_BytesObject TypeDef plus Python 3.14's buffer slot."""


assert {"__buffer__", "__doc__", "__hash__"} <= set(bytes.__dict__)
value = b"abc\x00\xff"
assert bytes.__hash__(value) == hash(value)
assert bytes.__doc__.startswith("bytes(iterable_of_ints) -> bytes")

view = bytes.__buffer__(value, 0)
assert type(view) is memoryview
assert view.readonly is True
assert view.tobytes() == value

try:
    view[0] = 0
except TypeError:
    pass
else:
    raise AssertionError("bytes buffer must be read-only")

print("bytes surface: ok")
