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

# CPython 3.14 asks a separator subclass for its Python-visible length before
# reading the raw payload (gh-143195).  A reported length of one accepts a
# longer payload and uses its first unit.
class BytesSep(bytes):
    calls = 0

    def __len__(self):
        type(self).calls += 1
        return 1


class StrSep(str):
    calls = 0

    def __len__(self):
        type(self).calls += 1
        return 1


assert b'ab'.hex(BytesSep(b'::')) == '61:62'
assert BytesSep.calls == 1
assert b'ab'.hex(StrSep('::')) == '61:62'
assert StrSep.calls == 1

# bytes_per_sep coercion can mutate a bytearray receiver.  Its payload must be
# read after __index__ returns, not kept as a stale pre-call slice.
class ClearReceiver:
    def __init__(self, receiver):
        self.receiver = receiver

    def __index__(self):
        self.receiver.clear()
        return 1


receiver = bytearray(b'ab')
assert receiver.hex(':', ClearReceiver(receiver)) == ''
assert receiver == bytearray()

# slice.indices() with the wrong number of arguments raises TypeError,
# not a Rust panic.
try:
    slice(1).indices()
    raise AssertionError("expected TypeError")
except TypeError:
    pass
assert slice(1, 10, 2).indices(100) == (1, 10, 2)

print("bytes_fromhex_hex_sep ok")
