# pyre-check: gate=1
value = bytearray(b'old')
seen = []

def items():
    for item in range(1, 5):
        yield item
        seen.append((list(value), value.__alloc__()))

value.__init__(items())
prefixes_visible = all(
    prefix == list(range(1, index + 1)) and allocation > len(prefix)
    for index, (prefix, allocation) in enumerate(seen, 1)
)

partial = bytearray(b'old')
def broken():
    yield 7
    raise RuntimeError

try:
    partial.__init__(broken())
except RuntimeError:
    failure_preserves_prefix = partial == bytearray(b'\x07')
else:
    failure_preserves_prefix = False

result = (
    value == bytearray(b'\x01\x02\x03\x04')
    and prefixes_visible
    and failure_preserves_prefix
)

assert result
