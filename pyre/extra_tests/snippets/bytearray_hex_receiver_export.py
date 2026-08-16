# pyre-check: gate=1
value = bytearray(b'\xaa')

class Separator(bytes):
    def __len__(self):
        value.clear()
        return 1

try:
    value.hex(Separator(b':'))
except BufferError:
    preserved = value == bytearray(b'\xaa')
    value.clear()
    result = preserved and value == bytearray()
else:
    result = False

assert result
