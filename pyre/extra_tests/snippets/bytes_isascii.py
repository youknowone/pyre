# bytes/bytearray.isascii() — True when every byte is <= 0x7F (empty is True).

assert b'abc'.isascii() is True
assert b''.isascii() is True
assert b'\x7f'.isascii() is True
assert b'\x80'.isascii() is False
assert bytes([0, 127]).isascii() is True
assert bytes([0, 128]).isascii() is False

assert bytearray(b'abc').isascii() is True
assert bytearray(b'').isascii() is True
assert bytearray([200]).isascii() is False

print("bytes_isascii ok")
