# pyre-check: gate=1
step = 2**63 - 1
b = bytearray(b'abc')
b[1::step] = b'x'
assigned = b == bytearray(b'axc')
del b[1::step]
result = (
    b'abc'[1::step] == b'b'
    and bytearray(b'abc')[1::step] == bytearray(b'b')
    and assigned
    and b == bytearray(b'ac')
)

assert result
