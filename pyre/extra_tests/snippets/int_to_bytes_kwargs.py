# int.to_bytes accepts `length` and `byteorder` as keyword arguments,
# not only positionally.  Self-contained (no testutils import).

# length as a keyword.
assert (1024).to_bytes(length=2, byteorder="little") == b"\x00\x04"
assert (255).to_bytes(length=4, byteorder="big") == b"\x00\x00\x00\xff"

# byteorder as a keyword with a positional length.
assert (1024).to_bytes(2, byteorder="big") == b"\x04\x00"

# byteorder defaults to "big" when only length is given by keyword.
assert (5).to_bytes(length=1) == b"\x05"

# Positional form still works.
assert (1024).to_bytes(2, "little") == b"\x00\x04"
assert (258).to_bytes(2, "big") == b"\x01\x02"
assert (5).to_bytes() == b"\x05"

# signed keyword is accepted.
assert (1).to_bytes(length=1, byteorder="big", signed=False) == b"\x01"

print("int_to_bytes_kwargs: OK")
