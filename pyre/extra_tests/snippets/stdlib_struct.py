import struct

from testutils import assert_raises

data = struct.pack("IH", 14, 12)
assert data == bytes([14, 0, 0, 0, 12, 0])

v1, v2 = struct.unpack("IH", data)
assert v1 == 14
assert v2 == 12

data = struct.pack("<IH", 14, 12)
assert data == bytes([14, 0, 0, 0, 12, 0])

v1, v2 = struct.unpack("<IH", data)
assert v1 == 14
assert v2 == 12

data = struct.pack(">IH", 14, 12)
assert data == bytes([0, 0, 0, 14, 0, 12])

v1, v2 = struct.unpack(">IH", data)
assert v1 == 14
assert v2 == 12

data = struct.pack("3B", 65, 66, 67)
assert data == bytes([65, 66, 67])

v1, v2, v3 = struct.unpack("3B", data)
assert v1 == 65
assert v2 == 66
assert v3 == 67

with assert_raises(Exception):
    data = struct.pack("B0B", 65, 66)

with assert_raises(Exception):
    data = struct.pack("B2B", 65, 66)

data = struct.pack("B1B", 65, 66)

with assert_raises(Exception):
    struct.pack("<IH", "14", 12)

assert struct.calcsize("B") == 1
# assert struct.calcsize("<L4B") == 12

assert struct.Struct("3B").pack(65, 66, 67) == bytes([65, 66, 67])


class Indexable(object):
    def __init__(self, value):
        self._value = value

    def __index__(self):
        return self._value


data = struct.pack("B", Indexable(65))
assert data == bytes([65])

data = struct.pack("5s", b"test1")
assert data == b"test1"

data = struct.pack("3s", b"test2")
assert data == b"tes"

data = struct.pack("7s", b"test3")
assert data == b"test3\0\0"

data = struct.pack("?", True)
assert data == b"\1"

data = struct.pack("?", [])
assert data == b"\0"

assert struct.error.__module__ == "struct"
assert struct.error.__name__ == "error"

# Non-ASCII format string: error type matches CPython.
# str → UnicodeEncodeError (encoding='ascii')
# bytes → struct.error
try:
    struct.Struct("\udc00")
except UnicodeEncodeError as e:
    assert e.encoding == "ascii"
else:
    raise AssertionError("expected UnicodeEncodeError")

with assert_raises(UnicodeEncodeError):
    struct.Struct("한")

with assert_raises(struct.error):
    struct.Struct(b"\xff")

# CPython 3.14 accepts a zero-width Pascal field.  It consumes one pack
# argument and contributes one empty bytes result while occupying no bytes.
assert struct.calcsize("0p") == 0
assert struct.pack("0p", b"payload") == b""
assert struct.unpack("0p", b"") == (b"",)

unpack_iterator_type = type(struct.iter_unpack("B", b""))
with assert_raises(TypeError):
    unpack_iterator_type()

# unpack_from accepts buffer / offset positionally or by keyword.
_buf = struct.pack("ii", 111, 222)
assert struct.unpack_from("ii", _buf, offset=0) == (111, 222)
assert struct.unpack_from("ii", buffer=_buf, offset=0) == (111, 222)
_s = struct.Struct("ii")
assert _s.unpack_from(_buf, offset=0) == (111, 222)
assert _s.unpack_from(buffer=_buf) == (111, 222)

# pack / pack_into (module and method) reject keyword arguments.
with assert_raises(TypeError):
    struct.pack(format="ii")
with assert_raises(TypeError):
    struct.pack_into("ii", bytearray(8), 0, 1, 2, extra=3)
with assert_raises(TypeError):
    _s.pack_into(bytearray(8), 0, 1, 2, extra=3)

# CPython 3.14 Struct.__sizeof__: the fixed seven-word object prefix plus a
# four-word compiled-format entry for every code and the terminal sentinel.
word = struct.calcsize("P")
expected_codes = {
    "": 0,
    "i": 1,
    "10i": 1,
    "2s": 1,
    "0i": 0,
    "  i  h": 2,
    "@i": 1,
    "100x": 0,
    "2p": 1,
}
for format_string, code_count in expected_codes.items():
    expected_size = 7 * word + (code_count + 1) * 4 * word
    assert struct.Struct(format_string).__sizeof__() == expected_size


class StructWithSlot(struct.Struct):
    __slots__ = ("extra",)


assert StructWithSlot("").__sizeof__() == 12 * word
