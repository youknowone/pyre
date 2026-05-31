# int.from_bytes is a classmethod (bytes, byteorder='big', *, signed=False).
# byteorder is positional-or-keyword; signed is keyword-only.

assert int.from_bytes(b'\x01\x00', 'big') == 256
assert int.from_bytes(b'\x01\x00', 'little') == 1
assert int.from_bytes(b'\x01\x00') == 256  # byteorder defaults to 'big'
assert int.from_bytes(b'\x01\x00', byteorder='little') == 1
assert int.from_bytes(b'\xff', 'big', signed=True) == -1
assert int.from_bytes(b'\xff', 'big', signed=False) == 255
assert int.from_bytes([1, 0], 'big') == 256  # iterable of ints
assert int.from_bytes(bytearray(b'\x01\x00'), 'big') == 256

# byteorder must be 'little' or 'big'.
try:
    int.from_bytes(b'\x01', 'middle')
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "byteorder must be either 'little' or 'big'" in str(e), str(e)

# Calling on an instance still binds the type (classmethod).
assert (5).from_bytes(b'\x02', 'big') == 2

print("int_from_bytes_kwargs ok")

# signed is keyword-only: a third positional argument is an error.
try:
    int.from_bytes(b'\xff', 'big', True)
    raise AssertionError("expected TypeError")
except TypeError as e:
    assert "at most 2 positional arguments" in str(e), str(e)

print("int_from_bytes_kwargs positional-signed rejected")

# Gateway signature enforcement (byteorder='text', signed=bool).
def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}")
assert _raises(TypeError, lambda: int.from_bytes(b'\x01', 'big', foo=1))      # unknown kw
assert _raises(TypeError, lambda: int.from_bytes(b'\x01', 'big', byteorder='little'))  # dup
assert _raises(TypeError, lambda: int.from_bytes(b'\x01', 123))               # non-str byteorder
assert _raises(TypeError, lambda: int.from_bytes(b'\x01', byteorder=123))     # non-str kw
assert _raises(ValueError, lambda: int.from_bytes(b'\x01', 'mid'))            # bad str stays ValueError

# str.encode signature enforcement (encoding=None, errors=None).
assert _raises(TypeError, lambda: "ab".encode(foo=1))                          # unknown kw
assert _raises(TypeError, lambda: "ab".encode("utf-8", encoding="ascii"))      # dup
assert _raises(TypeError, lambda: "ab".encode(123))                            # non-str
assert "ab".encode(encoding="ascii") == b"ab"

print("int_from_bytes_kwargs enforcement ok")
