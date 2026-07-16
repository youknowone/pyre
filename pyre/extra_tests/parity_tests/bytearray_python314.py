"""PyPy W_BytearrayObject TypeDef with Python 3.14 additions."""


required = {
    "__alloc__",
    "__buffer__",
    "__doc__",
    "__hash__",
    "__init__",
    "__mod__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__rmod__",
    "__sizeof__",
    "__str__",
    "resize",
}
assert required <= set(bytearray.__dict__)
assert bytearray.__hash__ is None

b = bytearray(b"abc")
assert repr(b) == "bytearray(b'abc')"
assert str(b) == repr(b)
assert b"%s" % b"x" == b"x"
assert bytearray(b"%s") % b"x" == bytearray(b"x")
assert bytearray.__rmod__(bytearray(b"x"), b"%s") is NotImplemented

assert bytearray().__alloc__() == 0
assert b.__alloc__() >= len(b) + 1
assert b.__sizeof__() >= b.__alloc__()

assert bytearray().__reduce__() == (bytearray, (), None)
assert b.__reduce__() == (bytearray, ("abc", "latin-1"), None)
assert b.__reduce_ex__(2) == (bytearray, ("abc", "latin-1"), None)
assert b.__reduce_ex__(4) == (bytearray, (b"abc",), None)

b.resize(5)
assert b == bytearray(b"abc\x00\x00")
b.resize(2)
assert b == bytearray(b"ab")
try:
    b.resize(-1)
except ValueError:
    pass
else:
    raise AssertionError("negative resize must fail")

b.__init__(b"reset")
assert b == bytearray(b"reset")
b.__init__()
assert b == bytearray()

b = bytearray(b"live")
view = bytearray.__buffer__(b, 0)
assert view.readonly is False
view[0] = ord("L")
assert b == bytearray(b"Live")
try:
    b.resize(1)
except BufferError:
    pass
else:
    raise AssertionError("resize with a live export must fail")
view.release()
b.resize(1)
assert b == bytearray(b"L")


class Child(bytearray):
    pass


child = Child(b"x")
child.attr = 1
assert repr(child) == "Child(b'x')"
assert child.__reduce__() == (Child, ("x", "latin-1"), {"attr": 1})
assert bytearray.__doc__.startswith("bytearray(iterable_of_ints)")
print("bytearray 3.14 surface: ok")
