# bytes/bytearray.fromhex is a classmethod; .hex(sep, bytes_per_sep=0)
# disables separators; slice.indices() arity errors raise TypeError.

# fromhex works on an instance (classmethod binds the type, not self).
assert b''.fromhex('aa') == b'\xaa'
assert b'x'.fromhex('B9 01EF') == b'\xb9\x01\xef'
assert bytes.fromhex('aabb') == b'\xaa\xbb'
assert bytearray().fromhex('aa') == bytearray(b'\xaa')
assert bytearray.fromhex('aabb') == bytearray(b'\xaa\xbb')

# fromhex on a subclass routes through cls(value); the resulting value
# is correct (subclass identity is blocked on the bytes-subclass
# constructor, a separate pre-existing limitation).
class MyBytes(bytes):
    pass
assert MyBytes.fromhex('aa') == b'\xaa'

# hex(sep, bytes_per_sep): 0 disables separators entirely.
assert b'ab'.hex('-', 0) == '6162'
assert b'ab'.hex('-', 1) == '61-62'
assert b'abcd'.hex('-', 2) == '6162-6364'
assert b'abcd'.hex('_', -2) == '6162_6364'
assert bytearray(b'ab').hex(':', 0) == '6162'

# slice.indices() with the wrong number of arguments raises TypeError,
# not a Rust panic.
try:
    slice(1).indices()
    raise AssertionError("expected TypeError")
except TypeError:
    pass
assert slice(1, 10, 2).indices(100) == (1, 10, 2)

print("bytes_fromhex_hex_sep ok")
